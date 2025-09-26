#!/usr/bin/env python3
"""
Syscall Security Analyzer
============================================

- Uses a strict denylist for read detections: only known-secret paths are flagged (low noise).
- Supports strace -y/-yy inline fd annotations (e.g., 6</real/path>, 7</dev/urandom<char 1:9>>) and normalizes them.
- FD table tracking: open/close/dup/socket/pipe/memfd; learns from -y/-yy annotations seen in args/retval.
- Severity Enum: INFO < SUSPICIOUS < DANGEROUS < CRITICAL
- Threats = severity >= DANGEROUS; total_threats in metadata
- Family-aware network policy (AF_INET/AF_INET6 -> CRITICAL; AF_UNIX -> INFO; AF_NETLINK/AF_PACKET -> SUSPICIOUS)
- Writes: stdout/stderr -> SUSPICIOUS; others DANGEROUS (escalate to CRITICAL on sensitive paths)
- File modifications: DANGEROUS (escalate to CRITICAL on sensitive paths)
- Read-like attribution also for mmap(PROT_READ), sendfile, copy_file_range, splice
- copy_file_range dest index correct (fd_in, off_in, fd_out, off_out, len, flags)
- CONNECT(): parse fd, fd_path, IPv4/IPv6 address/port or AF_UNIX path;

Usage:
  analyzer = SyscallSecurityAnalyzer(trace, model_path="/abs/path/to/model")
  analyzer.print_report()
  analyzer.save_report("analysis.json")
"""

import ast
import json
import logging
import os
import re
from datetime import datetime
from enum import IntEnum
from typing import List, Dict, Any, Union, Optional, Tuple

from ._types import Syscall, Signal

logger = logging.getLogger("SyscallSecurityAnalyzer")


# ----------------------------
# Helpers
# ----------------------------

class Severity(IntEnum):
    INFO = 0
    SUSPICIOUS = 1
    DANGEROUS = 2
    CRITICAL = 3


_TMP_REMOTE_RE = re.compile(r"^/tmp/[^/\n]+/_remote_module_non_scriptable\.py$")
_PYC_RE = re.compile(r"(?:^|[\\/])__pycache__[\\/][^/\\]+\.cpython-3\d{1,2}(?:\.opt-[12])?\.pyc(?:\.[^/\\]+)?$")


def _is_allowed_write(path: str) -> bool:
    if not path:
        return False
    return bool(_TMP_REMOTE_RE.match(path) or _PYC_RE.search(path))


def _path_is_under(path: str, root: str) -> bool:
    """Return True if path is under root (prefix match with path boundaries), using realpath on both sides."""
    if not path or not root:
        return False
    try:
        path = os.path.realpath(path)
        root = os.path.realpath(root)
        if not root.endswith(os.sep):
            root += os.sep
        return path.startswith(root)
    except Exception:
        return False


def _looks_site_or_dist_packages(p: str) -> bool:
    return (
            "/site-packages/" in p
            or "/dist-packages/" in p
            or re.search(r"/lib/python3\.\d+/", p) is not None
    )


def _pretty_call(name: str, args: str, retval: Optional[str]) -> str:
    r = (retval or "").strip()
    return f"{name}({args}) = {r}" if r else f"{name}({args})"


# ----------------------------
# Annotation normalization
# ----------------------------

# Matches tokens like: '6</real/path>', '7</dev/urandom<char 1:9>>', '3<socket:[12345]>'
_FD_ANN_RE = re.compile(r"^\s*(?P<fd>-?\d+)\s*<(?P<ann>.+)>\s*$")


def _normalize_annotation(ann: str) -> Tuple[str, str]:
    """
    Normalize strace -y/-yy inline annotations to (value, kind).

    Examples:
      '/dev/urandom<char 1:9>'     -> ('/dev/urandom', 'file')
      'socket:[12345]'             -> ('[socket]', 'socket')
      'pipe:[67890]'               -> ('[pipe]', 'pipe')
      'anon_inode:[eventfd]'       -> ('[eventfd]', 'eventfd')
      'anon_inode:[signalfd]'      -> ('[anon]', 'anon')
      'memfd:torch_graph (deleted)'-> ('[memfd]', 'memfd')
    """
    s = ann.strip()

    if "<" in s:
        s = s.split("<", 1)[0].rstrip()

    s = s.replace(" (deleted)", "").strip()
    low = s.lower()

    if low.startswith("socket:[") or low.startswith("socket:"):
        return "[socket]", "socket"
    if low.startswith("pipe:[") or low.startswith("pipe:"):
        return "[pipe]", "pipe"
    if low.startswith("anon_inode:"):
        if "eventfd" in low:
            return "[eventfd]", "eventfd"
        return "[anon]", "anon"
    if low.startswith("memfd:"):
        return "[memfd]", "memfd"

    if not s.startswith("/") and "/" in s and not s.startswith("["):
        s = "/" + s

    return s, "file"


# ----------------------------
# Allowlist for benign reads
# ----------------------------
ALLOWED_READ_PATTERNS = [
    # POSIX: lib or lib64 + site/dist-packages
    re.compile(r'(?:^|[\\/])(lib|lib64)[\\/]python3(?:\.\d{1,2})?[\\/](?:site|dist)-packages(?:[\\/]|$)'),
    # Debian/Ubuntu dist-packages variants (covers /usr/lib/python3/dist-packages and /usr/local/...)
    re.compile(r'(?:^|[\\/])python3(?:\.\d{1,2})?[\\/]?dist-packages(?:[\\/]|$)'),
    re.compile(r'(?:^|[\\/])dist-packages(?:[\\/]|$)'),
    # Generic site-packages (covers venvs and case variations)
    re.compile(r'(?:^|[\\/])site-packages(?:[\\/]|$)', re.IGNORECASE),
    # Windows: ...\Lib\site-packages
    re.compile(r'(?:^|[\\/])Lib[\\/]site-packages(?:[\\/]|$)'),
    # Standard library tree (lib or lib64)
    re.compile(r'(?:^|[\\/])(lib|lib64)[\\/]python3(?:\.\d{1,2})?(?:[\\/]|$)'),
]


def _is_benign_lib_read(path: str) -> bool:
    if not path:
        return False
    candidates = [path]
    try:
        candidates.append(os.path.realpath(path))
    except Exception:
        pass
    for p in candidates:
        for rx in ALLOWED_READ_PATTERNS:
            if rx.search(p):
                return True
    return False


# ----------------------------
# Strict denylist for reads
# ----------------------------

SENSITIVE_CRITICAL_PATTERNS = [
    # System password files / root secrets
    re.compile(r"^/etc/shadow(?:$|/)"),
    re.compile(r"^/root(?:$|/).*"),

    # Private TLS keys
    re.compile(r"^/etc/letsencrypt/live/[^/]+/privkey\.pem$"),
    re.compile(r"^/etc/ssl/private/"),

    # SSH private keys (user and host)
    re.compile(r"^/(home|Users)/[^/]+/.ssh/(id_rsa|id_dsa|id_ecdsa|id_ed25519)$"),
    re.compile(r"^/etc/ssh/ssh_host_(rsa|dsa|ecdsa|ed25519)_key$"),

    # GPG private keyrings
    re.compile(r"^/(home|Users)/[^/]+/.gnupg/"),

    # Cloud & API tokens (high-value)
    re.compile(r"^/(home|Users)/[^/]+/\.aws/(credentials|config)$"),
    re.compile(r"^/(home|Users)/[^/]+/\.config/gcloud/application_default_credentials\.json$"),
    re.compile(r"^/(home|Users)/[^/]+/\.docker/config\.json$"),
    re.compile(r"^/(home|Users)/[^/]+/\.vault-token$"),
    re.compile(r"^/(home|Users)/[^/]+/\.azure/(accessTokens\.json|azureProfile\.json|tokenCache\.json)$"),
    re.compile(r"^/(home|Users)/[^/]+/\.huggingface/token$"),
    re.compile(r"^/(home|Users)/[^/]+/\.config/gh/hosts\.yml$"),
    re.compile(r"^/(home|Users)/[^/]+/\.netrc$"),

    # Platform service account secrets (k8s)
    re.compile(r"^/var/run/secrets/kubernetes\.io/serviceaccount/(token|ca\.crt|namespace)$"),
    re.compile(r"^/run/secrets/([^/]+)$"),

    # Any file named like common private key filenames anywhere
    re.compile(r".*/(?:private[-_]?key|key\.pem|id_rsa|id_dsa|id_ed25519|keyfile)\b"),

    # /proc leaks (process env can contain secrets)
    re.compile(r"^/proc/\d+/environ$"),
]

SENSITIVE_DANGEROUS_PATTERNS = [
    # VCS / credential stores
    re.compile(r"^/(home|Users)/[^/]+/\.git-credentials$"),
    re.compile(r"^/(home|Users)/[^/]+/\.netrc$"),
    re.compile(r"^/(home|Users)/[^/]+/\.config/gh/"),

    # Browser and profile data (cookies, tokens, saved passwords)
    re.compile(r"^/(home|Users)/[^/]+/\.config/(google-chrome|chromium|BraveSoftware)/"),
    re.compile(r"^/(home|Users)/[^/]+/\.mozilla/"),
    re.compile(r"^/(home|Users)/[^/]+/\.local/share/BraveSoftware/"),
    re.compile(r"^/(home|Users)/[^/]+/Library/Application Support/Google/Chrome/"),

    # Cloud SDK credentials and config (GCP, AWS, Azure, OCI)
    re.compile(r"^/(home|Users)/[^/]+/\.config/gcloud/"),
    re.compile(r"^/(home|Users)/[^/]+/\.aws/"),
    re.compile(r"^/(home|Users)/[^/]+/\.azure/"),
    re.compile(r"^/(home|Users)/[^/]+/\.oci/"),

    # Keyrings and OS-managed credential stores
    re.compile(r"^/(home|Users)/[^/]+/\.local/share/keyrings/"),
    re.compile(r"^/run/user/\d+/keyring/"),
    re.compile(r"^/(home|Users)/[^/]+/Library/Keychains/"),

    # Docker / container sensitive files
    re.compile(r"^/var/lib/docker/containers/"),
    re.compile(r"^/etc/docker/daemon\.json$"),
    re.compile(r"^/var/lib/kubelet/(?:pods|pki)/"),

    # Database config files that may include credentials
    re.compile(r"^/etc/mysql/"),
    re.compile(r"^/etc/postgresql/"),
    re.compile(r"^/(home|Users)/[^/]+/\.pgpass$"),

    # Generic .env and config files often containing secrets
    re.compile(r"^/(home|Users)/[^/]+/(\.env|\.env\..*|config\.env)$"),
    re.compile(r"^/(home|Users)/[^/]+/\.config/.*credentials.*$"),

    # SSH auxiliary files (configs, authorized_keys, agent sockets)
    re.compile(r"^/(home|Users)/[^/]+/\.ssh/(config|authorized_keys)$"),
    re.compile(r"^/tmp/ssh-.+/agent\.\d+$"),
    re.compile(r"^/run/ssh-agent/"),

    # Sudoers / privilege escalation configs
    re.compile(r"^/etc/sudoers(?:$|/)"),
    re.compile(r"^/etc/sudoers\.d/"),

    # System configs that reveal service accounts / keys
    re.compile(r"^/etc/krb5\.keytab$"),
    re.compile(r"^/etc/ssh/"),
    re.compile(r"^/etc/ssl/"),

    # Logs that may contain secrets (app dependent)
    re.compile(r"^/var/log/.*(auth|secure|aws|gcloud|kube).*"),

    # Common credentials in shell histories
    re.compile(r"^/(home|Users)/[^/]+/\.(bash_history|zsh_history)$"),
    re.compile(r"^/(home|Users)/[^/]+/\.ssh/history$"),
]


def _classify_denylisted_read(path: Optional[str]) -> Optional[Severity]:
    """Return severity if path is denylisted; otherwise None (ignore)."""
    if not path:
        return None
    try:
        p = os.path.realpath(path)
    except Exception:
        p = path
    for rx in SENSITIVE_CRITICAL_PATTERNS:
        if rx.match(p):
            return Severity.CRITICAL
    for rx in SENSITIVE_DANGEROUS_PATTERNS:
        if rx.match(p):
            return Severity.DANGEROUS
    return None


class SyscallSecurityAnalyzer:
    """
    Analyze a single syscall trace for security-relevant behavior.

    One instance = one analyzed trace.
    Create a new instance for each new trace you want to analyze.
    """

    CONSOLE_FDS = {"0", "1", "2"}  # stdin, stdout, stderr

    DEST_FD_INDEX = {
        "write": 0,
        "pwrite": 0,
        "pwrite64": 0,
        "writev": 0,
        "pwritev": 0,
        "sendfile": 0,  # out_fd, in_fd, offset, count
        "copy_file_range": 2,  # fd_in, off_in, fd_out, off_out, len, flags
        "splice": 2,  # fd_in, off_in, fd_out, off_out, len, flags
    }

    READ_SRC_FD_INDEX = {
        "read": 0,
        "readv": 0,
        "pread64": 0,
        "sendfile": 1,  # out_fd, in_fd, ...
        "copy_file_range": 0,  # fd_in, ...
        "splice": 0,  # fd_in, ...
    }

    CRITICAL_SIGNALS = {"SIGSEGV", "SIGILL", "SIGBUS", "SIGABRT", "SIGKILL", "SIGSYS", "SIGTERM"}
    BENIGN_SIGNALS = {"SIGCHLD", "SIGCONT", "SIGWINCH", "EXIT"}
    SUS_SIGNALS = {"SIGPIPE", "SIGHUP", "SIGTRAP"}

    MAX_PREVIEW_LENGTH = 200

    def __init__(self, trace: List[Union[Syscall, Signal]], model_path: Optional[str] = None):
        logging.getLogger("SyscallSecurityAnalyzer")
        self.model_path = model_path
        self.model_dir = None
        self._trace_len = len(trace)
        self._signals_ignored = 0
        self.fd_table: Dict[tuple[str, str], Dict[str, Any]] = {}
        self.detections: List[Dict[str, Any]] = []

        self._counts_by_category: Dict[str, int] = {
            "execution": 0,
            "network": 0,
            "console_write": 0,
            "file_write": 0,
            "file_modification": 0,
            "read": 0,
            "signal": 0,
        }
        self._counts_by_severity: Dict[str, int] = {s.name: 0 for s in Severity}

        self._run_analysis(trace)

    # ----------------------------
    # Utility helpers
    # ----------------------------

    def _retval_success(self, retval: str | None) -> bool:
        return bool(retval and not retval.strip().startswith("-") and retval.strip() != "?")

    def _parse_fd_and_annot_at_index(self, args: str, idx: int) -> tuple[Optional[str], Optional[str]]:
        try:
            parts = [p.strip() for p in args.split(",")]
            if idx < len(parts):
                token = parts[idx]
                m = _FD_ANN_RE.match(token)
                if m:
                    return m.group("fd"), m.group("ann")
                m2 = re.match(r"(-?\d+)", token)
                return (m2.group(1) if m2 else None), None
        except Exception:
            pass
        return None, None

    def _retval_fd_and_annot(self, retval: str | None) -> tuple[Optional[str], Optional[str]]:
        if not retval:
            return None, None
        rv = retval.strip()
        m = _FD_ANN_RE.match(rv)
        if m:
            return m.group("fd"), m.group("ann")
        if rv.isdigit():
            return rv, None
        return None, None

    def _maybe_learn_fd_from_annotation(self, pid: str, fd: Optional[str], ann: Optional[str]):
        if not fd or not ann:
            return
        norm_value, kind = _normalize_annotation(ann)
        info = self._fd_get(pid, fd)
        if info.get("path") in (None, "[unknown]", "[socket]", "[pipe]", "[memfd]", "[anon]") or info.get(
                "kind") == "unknown":
            self._fd_set(pid, fd, norm_value, kind=kind, flags=info.get("flags", ""),
                         cloexec=info.get("cloexec", False))

    def _record(self, category: str, severity: Severity, include_info: bool = False, **details):
        if not include_info and severity == Severity.INFO:
            return
        self.detections.append({
            "category": category,
            "severity": severity.name,
            "severity_level": int(severity),
            "details": details,
        })
        if category in self._counts_by_category:
            self._counts_by_category[category] += 1
        self._counts_by_severity[severity.name] = self._counts_by_severity.get(severity.name, 0) + 1

    def _fd_set(self, pid: str, fd: str, path_or_tag: str, kind: str = "file", flags: str = "", cloexec: bool = False):
        self.fd_table[(pid, fd)] = {"path": path_or_tag, "kind": kind, "flags": flags, "cloexec": cloexec}

    def _fd_del(self, pid: str, fd: str):
        self.fd_table.pop((pid, fd), None)

    def _fd_get(self, pid: str, fd: str) -> Dict[str, Any]:
        return self.fd_table.get((pid, fd), {"path": "[unknown]", "kind": "unknown", "flags": "", "cloexec": False})

    # ----------------------------
    # Classification helpers
    # ----------------------------

    def _classify_write_path(self, path: Optional[str]) -> Severity:
        if not path:
            return Severity.SUSPICIOUS
        if _is_allowed_write(path):
            return Severity.INFO
        # Consider some paths inherently sensitive for writes (reuse existing heuristics)
        if path and ("/.ssh/" in path or "/.gnupg/" in path or path.startswith("/etc/")):
            return Severity.CRITICAL
        return Severity.DANGEROUS

    # ----------------------------
    # FD tracking (per-trace)
    # ----------------------------

    def _track_fd_syscalls(self, sc: Syscall):
        pid, name, args, retval = sc.pid, sc.name, sc.args, (sc.retval or "").strip()
        try:
            if name in ("open", "openat", "openat2", "creat"):
                flags = ""
                cloexec = False
                mflags = re.search(r'(?:(O_[A-Z\|\d_]+))', args or "")
                if mflags:
                    flags = mflags.group(1)
                    cloexec = "O_CLOEXEC" in flags

                fd, ann = self._retval_fd_and_annot(retval)
                if fd:
                    if ann:
                        self._maybe_learn_fd_from_annotation(pid, fd, ann)
                    else:
                        m = re.search(r'"([^"]+)"', args)
                        path = m.group(1) if m else None
                        self._fd_set(pid, fd, path or "[unknown]", kind="file", flags=flags, cloexec=cloexec)

            elif name == "close":
                m = re.match(r"(\d+)", args.strip())
                if m:
                    self._fd_del(pid, m.group(1))

            elif name in ("dup", "dup2", "dup3"):
                parts = [a.strip() for a in args.split(",")]
                if self._retval_success(retval) and parts:
                    new_fd, _ = self._retval_fd_and_annot(retval)
                    old_fd = parts[0]
                    if new_fd:
                        info = self._fd_get(pid, old_fd).copy()
                        self._fd_set(pid, new_fd, info.get("path", "[unknown]"), info.get("kind", "unknown"),
                                     info.get("flags", ""), info.get("cloexec", False))

            elif name == "socket":
                newfd, ann = self._retval_fd_and_annot(retval)
                if newfd:
                    if ann:
                        self._maybe_learn_fd_from_annotation(pid, newfd, ann)
                    else:
                        self._fd_set(pid, newfd, "[socket]", kind="socket", flags="", cloexec=False)

            elif name in ("pipe", "pipe2"):
                fds = re.findall(r"\[(\d+),\s*(\d+)\]", retval)
                if fds:
                    fd1, fd2 = fds[0]
                    self._fd_set(pid, fd1, "[pipe]", kind="pipe")
                    self._fd_set(pid, fd2, "[pipe]", kind="pipe")

            elif name == "memfd_create":
                newfd, ann = self._retval_fd_and_annot(retval)
                if newfd:
                    if ann:
                        self._maybe_learn_fd_from_annotation(pid, newfd, ann)
                    else:
                        self._fd_set(pid, newfd, "[memfd]", kind="memfd")

            elif name == "mmap":
                fd, ann = self._parse_fd_and_annot_at_index(args or "", 4)
                self._maybe_learn_fd_from_annotation(pid, fd, ann)

        except Exception as e:
            logger.debug("FD tracking error on %s %r", sc, e, exc_info=True)

    # ----------------------------
    # Detection logic
    # ----------------------------

    def _analyze_exec(self, sc: Syscall) -> bool:
        if sc.name not in {"execve", "execveat"}:
            return False

        exe_match = re.match(r'^\s*"([^"]+)"', sc.args)
        executable = exe_match.group(1) if exe_match else "unknown"

        argv = []
        try:
            start = sc.args.index('[')
            end = sc.args.index(']', start) + 1
            raw_list = sc.args[start:end]
            argv = ast.literal_eval(raw_list)
        except Exception:
            pass

        summary = " ".join(argv) if argv else executable
        self._record(
            "execution",
            Severity.CRITICAL,
            name=sc.name,
            call=_pretty_call(sc.name, sc.args, sc.retval),
            pid=sc.pid,
            executable=executable,
            argv=argv,
            retval=sc.retval,
            timestamp=sc.timestamp,
            summary=summary,
        )
        return True

    def _network_family_from_args(self, args: str) -> Optional[str]:
        m = re.search(r"sa_family=([A-Z0-9_]+)", args or "")
        if m:
            return m.group(1)
        m = re.search(r"AF_[A-Z0-9_]+", args or "")
        if m:
            return m.group(0)
        return None

    def _classify_network_family(self, fam: Optional[str]) -> Severity:
        if fam in {"AF_INET", "AF_INET6"}:
            return Severity.CRITICAL
        if fam in {"AF_UNIX", "AF_LOCAL"}:
            return Severity.INFO
        if fam in {"AF_NETLINK", "AF_PACKET"}:
            return Severity.SUSPICIOUS
        return Severity.SUSPICIOUS

    def _parse_connect_endpoint(self, args: str) -> Dict[str, Optional[str]]:
        fam = self._network_family_from_args(args) or ""
        ip = port = unix_path = None

        mport = re.search(r"sin6?_port=htons\((\d+)\)", args)
        if mport:
            port = mport.group(1)

        if fam == "AF_UNIX":
            m = re.search(r'sun_path="([^"]+)"', args)
            if m:
                unix_path = m.group(1)
        elif fam == "AF_INET":
            m4 = re.search(r'sin_addr=inet_(?:addr|aton)\("([^"]+)"\)', args)
            if m4:
                ip = m4.group(1)
        elif fam == "AF_INET6":
            m6 = re.search(r'sin6_addr=inet_pton\([^,]+,\s*"([^"]+)"\)', args)
            if m6:
                ip = m6.group(1)
            else:
                m6b = re.search(r"sin6_addr=\{[^}]*\}", args)
                if m6b:
                    ip = "[sin6_addr]"
        return {"family": fam or None, "ip": ip, "port": port, "unix_path": unix_path}

    def _analyze_network(self, sc: Syscall) -> bool:
        net_calls = {"socket", "connect", "send", "sendto", "sendmsg", "recv", "recvfrom", "recvmsg"}
        if sc.name not in net_calls:
            return False

        if sc.name == "connect":
            fd, ann = self._parse_fd_and_annot_at_index(sc.args or "", 0)
            self._maybe_learn_fd_from_annotation(sc.pid, fd, ann)
            info = self._fd_get(sc.pid, fd or "?")
            endpoint = self._parse_connect_endpoint(sc.args or "")
            fam = endpoint.get("family")
            sev = self._classify_network_family(fam)

            dest = ""
            if fam in {"AF_INET", "AF_INET6"}:
                ip = endpoint.get("ip") or "[ip?]"
                port = endpoint.get("port") or "?"
                dest = f"{ip}:{port}"
            elif fam == "AF_UNIX":
                dest = endpoint.get("unix_path") or info.get("path") or "[unix?]"

            self._record(
                "network",
                sev,
                name=sc.name,
                call=_pretty_call(sc.name, sc.args, sc.retval),
                pid=sc.pid,
                fd=fd or "?",
                fd_path=info.get("path"),
                family=fam or "[unknown]",
                ip=endpoint.get("ip"),
                port=endpoint.get("port"),
                unix_path=endpoint.get("unix_path"),
                retval=sc.retval,
                timestamp=sc.timestamp,
                summary=f"connect {fam or ''} {dest}".strip(),
            )
            return True

        fam = self._network_family_from_args(sc.args)
        sev = self._classify_network_family(fam)

        self._record(
            "network",
            sev,
            name=sc.name,
            call=_pretty_call(sc.name, sc.args, sc.retval),
            pid=sc.pid,
            family=fam or "[unknown]",
            retval=sc.retval,
            timestamp=sc.timestamp,
            summary=f"{sc.name} {fam or ''}".strip(),
        )
        return True

    def _analyze_console_or_file_write(self, sc: Syscall) -> bool:
        write_calls = {"write", "pwrite", "pwrite64", "writev", "pwritev", "sendfile", "copy_file_range", "splice"}
        if sc.name not in write_calls:
            return False

        pid = sc.pid
        dest_idx = self.DEST_FD_INDEX.get(sc.name, 0)
        dest_fd, dest_ann = self._parse_fd_and_annot_at_index(sc.args, dest_idx)
        dest_fd = dest_fd or "?"
        self._maybe_learn_fd_from_annotation(sc.pid, dest_fd, dest_ann)
        dest_info = self._fd_get(pid, dest_fd)
        dest_path = dest_info.get("path")

        if dest_fd in {"1", "2"}:
            preview = ""
            try:
                parts = [p.strip() for p in sc.args.split(',')]
                for part in parts:
                    if '"' in part:
                        m = re.search(r'"([^"]*)"', part)
                        if m:
                            preview = m.group(1)
                            if len(preview) > self.MAX_PREVIEW_LENGTH:
                                preview = preview[: self.MAX_PREVIEW_LENGTH] + "..."
                            break
            except Exception:
                pass

            self._record(
                "console_write",
                Severity.SUSPICIOUS,  # keep noise low; still visible if desired
                name=sc.name,
                call=_pretty_call(sc.name, sc.args, sc.retval),
                pid=pid,
                fd=dest_fd,
                fd_path="stdout/stderr",
                retval=sc.retval,
                timestamp=sc.timestamp,
                summary=f"print → {preview}",
            )
            return True

        if dest_info.get("kind") in ("pipe", "socket", "eventfd", "memfd", "anon", "unknown"):
            return False
        if dest_path and _is_allowed_write(dest_path):
            return False

        sev = self._classify_write_path(dest_path)

        bytes_written = None
        try:
            m = re.match(r"^(\d+)", (sc.retval or "").strip())
            if m:
                bytes_written = int(m.group(1))
        except Exception:
            pass

        self._record(
            "file_write",
            sev,
            name=sc.name,
            call=_pretty_call(sc.name, sc.args, sc.retval),
            pid=pid,
            fd=dest_fd,
            fd_path=dest_path or "[unknown]",
            bytes=bytes_written,
            retval=sc.retval,
            timestamp=sc.timestamp,
            summary=f"{dest_path or '[unknown]'} ({bytes_written if bytes_written is not None else '?'}B)",
        )
        return True

    def _analyze_file_modification(self, sc: Syscall) -> bool:
        mods = {
            "unlink", "unlinkat", "rmdir",
            "rename", "renameat", "renameat2",
            "chmod", "fchmod", "chown", "fchown",
            "truncate", "ftruncate",
            "symlink", "symlinkat",
            "setxattr", "removexattr",
        }
        if sc.name not in mods:
            return False

        path = None
        try:
            m = re.search(r'"([^"]+)"', sc.args)
            path = m.group(1) if m else None
            if path and _is_allowed_write(path):
                return False
        except Exception:
            pass

        sev = Severity.CRITICAL if (path and (
                "/.ssh/" in path or "/.gnupg/" in path or path.startswith("/etc/"))) else Severity.DANGEROUS

        self._record(
            "file_modification",
            sev,
            name=sc.name,
            call=_pretty_call(sc.name, sc.retval, sc.retval),
            pid=sc.pid,
            syscall=sc.name,
            file_path=path or "[unknown]",
            retval=sc.retval,
            timestamp=sc.timestamp,
            summary=f"{sc.name} {path}",
        )
        return True

    def _analyze_read_like(self, sc: Syscall) -> bool:
        """Denylist-only read detection: only flag reads of paths matching sensitive patterns."""
        name = sc.name

        def _emit_if_denylisted(fd: Optional[str], op_label: str) -> bool:
            if not fd or not fd.isdigit():
                return False
            info = self._fd_get(sc.pid, fd)
            if info.get("kind") != "file":
                return False  # ignore non-files
            path = info.get("path")
            if _is_benign_lib_read(path):
                return False
            sev = _classify_denylisted_read(path)
            if not sev:
                return False
            self._record(
                "read",
                sev,
                name=sc.name,
                call=_pretty_call(sc.name, sc.args, sc.retval),
                pid=sc.pid,
                fd=fd,
                fd_path=path or "[unknown]",
                retval=sc.retval,
                timestamp=sc.timestamp,
                summary=f"{op_label} ← {path or '[unknown]'}",
            )
            return True

        if name == "mmap":
            args = sc.args or ""
            prot = ("PROT_" in args) and ("PROT_READ" in args)
            fd, ann = self._parse_fd_and_annot_at_index(args, 4)  # 5th arg
            self._maybe_learn_fd_from_annotation(sc.pid, fd, ann)
            is_anon = ("MAP_ANONYMOUS" in args) or (fd == "-1")
            if prot and fd and fd.isdigit() and not is_anon:
                return _emit_if_denylisted(fd, "mmap(PROT_READ)")
            return False

        if name in self.READ_SRC_FD_INDEX:
            src_idx = self.READ_SRC_FD_INDEX[name]
            src_fd, src_ann = self._parse_fd_and_annot_at_index(sc.args or "", src_idx)
            src_fd = src_fd or "?"
            self._maybe_learn_fd_from_annotation(sc.pid, src_fd, src_ann)
            return _emit_if_denylisted(src_fd, name)

        return False

    def _analyze_signal(self, sg: Signal) -> bool:
        name = (sg.name or "").upper()
        details = sg.details or ""

        if name in self.BENIGN_SIGNALS:
            return False
        if name in self.CRITICAL_SIGNALS:
            severity = Severity.CRITICAL
        elif name in self.SUS_SIGNALS:
            severity = Severity.SUSPICIOUS
        else:
            severity = Severity.SUSPICIOUS

        if "killed" in details.lower() and name not in {"SIGCHLD", "EXIT"}:
            severity = Severity.CRITICAL

        self._record(
            "signal",
            severity,
            name=name,
            call=f"{name} {details}".strip(),
            pid=sg.pid,
            signal=name,
            details=details,
            timestamp=sg.timestamp,
            summary=f"{name} {details}".strip(),
        )
        return True

    # ----------------------------
    # Core per-trace analysis
    # ----------------------------

    def _analyze_syscall(self, sc: Syscall):
        self._track_fd_syscalls(sc)
        if self._analyze_exec(sc):
            return
        if self._analyze_network(sc):
            return
        if self._analyze_read_like(sc):
            return
        if self._analyze_console_or_file_write(sc):
            return
        if self._analyze_file_modification(sc):
            return

    def _run_analysis(self, events: List[Union[Syscall, Signal]]):
        logger.info(f"Analyzing {len(events)} events...")
        for ev in events:
            try:
                if isinstance(ev, Syscall):
                    self._analyze_syscall(ev)
                elif isinstance(ev, Signal):
                    self._analyze_signal(ev)
                else:
                    self._signals_ignored += 1
            except Exception as e:
                logger.error(f"Analyzer error on %s %r", ev, e, exc_info=logger.isEnabledFor(logging.DEBUG))
        logger.info("Analysis complete.")

    # ----------------------------
    # Reporting
    # ----------------------------

    def number_of_threats(self) -> int:
        return sum(1 for d in self.detections if d.get("severity") in (Severity.DANGEROUS.name, Severity.CRITICAL.name))

    def get_report(self) -> Dict[str, Any]:
        total_threats = self.number_of_threats()
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "events_total": self._trace_len,
                "signals_ignored": self._signals_ignored,
                "detections_total": len(self.detections),
                "detections_by_category": dict(self._counts_by_category),
                "detections_by_severity": dict(self._counts_by_severity),
                "total_threats": total_threats,
                "threat_threshold": "DANGEROUS",
            },
            "detections": self.detections,
        }

    def save_report(self, filename: str):
        try:
            with open(filename, "w") as f:
                json.dump(self.get_report(), f, indent=4)
            logger.info(f"Report saved to {filename}")
        except Exception as e:
            logger.error("Failed to save report: %r", e, exc_info=logger.isEnabledFor(logging.DEBUG))

    def print_report(self, include_info: bool = False):
        report = self.get_report()
        md = report["metadata"]
        bar = "=" * self.MAX_PREVIEW_LENGTH
        print("\n" + bar)
        print("🔒 SYSCALL SECURITY ANALYSIS REPORT")
        print(bar)
        print(f"Timestamp           : {md['timestamp']}")
        print(f"Events (total)      : {md['events_total']}")
        print(f"Signals ignored     : {md['signals_ignored']}")
        print(f"Detections (total)  : {md['detections_total']}")
        print(f"Total THREATS (>=DANGEROUS): {md['total_threats']}")
        print("Counts by category  :")
        for k, v in md["detections_by_category"].items():
            print(f"  - {k}: {v}")
        print("Counts by severity  :")
        for k, v in md["detections_by_severity"].items():
            print(f"  - {k}: {v}")

        print("\nDetections:")
        for det in report["detections"]:
            sev_name = det["severity"]
            if not include_info and sev_name == Severity.INFO.name:
                continue
            cat = det["category"]
            summary = det["details"].get("summary", "")
            print(f"  • [{sev_name}] {cat.upper()} → {summary}")
        print(bar)
