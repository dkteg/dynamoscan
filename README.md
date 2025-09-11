# dynamoscan

A dynamic scanner to detect suspicious system calls in ML artifacts with support for `tensorflow SavedModel`,
`keras`, `torch`, `pickle` and other pickle derivatives  `dill` and `joblib`.

---

## Prerequisites

- The tool is tested and confirmed to run on Python 3.11 only.
- Due to dependencies (e.g., tensorflow) with specific versions, other Python versions may be incompatible.

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e .
```

## 🖥️ CLI Overview

Dynamoscan provides a simple command-line interface to run syscall-based scans against machine learning model artifacts.

### Usage

```bash
dynamoscan MODEL_PATH [options]
```

* **`MODEL_PATH`** (required)
  Path to a model file (e.g. `.pt`, `.pkl`, `.keras`, `.npy`) or a TensorFlow SavedModel directory.
  The target is scanned as a **single unit**.

### Options

| Option               | Description                                                                                                   |
|----------------------|---------------------------------------------------------------------------------------------------------------|
| `--cwd PATH`         | Working directory for the scan. Defaults to the file’s parent directory (or the SavedModel directory itself). |
| `--trace-file PATH`  | Write the raw syscall trace to a text file.                                                                   |
| `--report-file PATH` | Write the JSON scan report to this file.                                                                      |
| `--print-trace`      | Print the syscall trace to stdout.                                                                            |
| `--no-print-res`     | Suppress the human-readable report output (enabled by default).                                               |

### Exit Codes

* `0` → Scan completed successfully, **no detections**.
* `1` → One or more detections found, or an error occurred.

### Examples

```bash
# Scan a PyTorch model and print a human-readable report
dynamoscan ./models/model.pt

# Save both a JSON report and the syscall trace
dynamoscan ./models/model.pt \
  --report-file model.report.json \
  --trace-file model.trace.txt

# Scan a TensorFlow SavedModel directory
dynamoscan ./exported/saved_model --print-trace
```

---

Would you like me to also include in the README a **sample report JSON snippet** (with `metadata` + `detections`) so
users know what to expect from `--report-file`?

---

## Scanning prerequisites

`dynamoscan` traces model loading with **strace** and analyzes syscalls. For security reasons, many Linux systems
restrict
attaching to processes. The tool performs a strict preflight and will **exit with an error** unless both conditions are
met:

1. **strace is installed** and available in `PATH`
2. **Yama ptrace policy** allows same‑UID attaches:  
   `/proc/sys/kernel/yama/ptrace_scope` must equal **`0`**

### How to set it

Install strace (Debian/Ubuntu):

```bash
sudo apt-get update && sudo apt-get install -y strace
```

Temporarily relax Yama (until reboot):

```bash
sudo sysctl -w kernel.yama.ptrace_scope=0
```

Persist across reboots:

```bash
echo 'kernel.yama.ptrace_scope=0' | sudo tee /etc/sysctl.d/10-ptrace.conf
sudo sysctl --system
```

> **Security note:** `ptrace_scope=0` allows same‑UID processes to trace each other. Revert to `1` when not scanning if
> desired:
> ```bash
> sudo sysctl -w kernel.yama.ptrace_scope=1
> ```

If these prerequisites are not met, `dynamoscan` logs a clear error describing what to install/change and exits with a
non‑zero code.

---

## ⚠️ Security and Isolation Warning

`dynamoscan` executes machine learning artifacts in a traced environment to monitor their system
calls. While the scanner attempts to mitigate risks by using `strace` for observation, **loading untrusted model files
is inherently dangerous**:

- Malicious artifacts **can execute arbitrary code** during deserialization or model loading.
- Code inside artifacts may attempt **file system modification, network exfiltration, privilege escalation, or
  persistence**.
- The relaxed `ptrace_scope=0` setting further increases exposure by allowing same‑UID tracing.

👉 **For safety, always run `dynamoscan` in a controlled or isolated environment**, such as:

- A **disposable virtual machine** (VM)
- A **container sandbox** (Docker/Podman) with restrictive seccomp/apparmor profiles
- A **dedicated test machine** with no sensitive data or credentials

Never run the dynamic scanner directly on your personal workstation, production host, or any system containing sensitive
information. Treat the scan environment as **potentially compromised** once an artifact has been executed.

If in doubt, use the **static scanners only** (`modelscan`, `picklescan`, `fickling`) when analyzing untrusted files
outside of a sandboxed setup.

---
