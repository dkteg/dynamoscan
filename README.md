# _dynamoscan_

A dynamic scanner to detect suspicious system calls in ML artifacts with support for `tensorflow SavedModel`, `keras`,
`torch`, `pickle` and other pickle derivatives `dill` and `joblib`.

***

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
  Path to a model file (e.g. `.pt`, `.pkl`, `.keras`, `.npy`) or a TensorFlow SavedModel directory. The target is
  scanned as a **single unit**.

### Options

| Option                 | Description                                                                                                   |
|------------------------|---------------------------------------------------------------------------------------------------------------|
| `--cwd PATH`           | Working directory for the scan. Defaults to the file's parent directory (or the SavedModel directory itself). |
| `--trace-file PATH`    | Write the raw syscall trace to a text file.                                                                   |
| `--report-file PATH`   | Write the JSON scan report to this file.                                                                      |
| `--print-trace`        | Print the syscall trace to stdout.                                                                            |
| `--print-res`          | Print human-readable report output (enabled by default).                                                      |
| `--no-print-res`       | Suppress the human-readable report output.                                                                    |
| `--docker`             | Run scan in isolated Docker container (recommended for untrusted models).                                     |
| `--docker-rebuild`     | Force rebuild of the Docker image.                                                                            |
| `--docker-config PATH` | Use custom Docker configuration YAML.                                                                         |

### Exit Codes

* **`0`** → Scan completed successfully, **no detections**.
* **`1`** → Scan completed, **threats detected** (check report for details).
* **`2`** → Scan failed or encountered an error.

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

# Scan untrusted model in Docker isolation (recommended)
dynamoscan suspicious.pkl --docker --report-file scan_results.json
```

***

## Scanning Prerequisites (Local Mode)

`dynamoscan` traces model loading with **strace** and analyzes syscalls. For security reasons, many Linux systems
restrict attaching to processes. The tool performs a strict preflight and will **exit with an error** unless both
conditions are met:

1. **strace is installed** and available in `PATH`
2. **Yama ptrace policy** allows same‑UID attaches: `/proc/sys/kernel/yama/ptrace_scope` must equal **`0`**

> **Note:** These prerequisites are **only required for local mode** (without `--docker`). When using Docker isolation,
> strace and ptrace configuration are handled automatically inside the container.

### How to Set It Up

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

***

## ⚠️ Security and Isolation Warning

`dynamoscan` executes machine learning artifacts in a traced environment to monitor their system calls. While the
scanner attempts to mitigate risks by using `strace` for observation, **loading untrusted model files is inherently
dangerous**:

- Malicious artifacts **can execute arbitrary code** during deserialization or model loading.
- Code inside artifacts may attempt **file system modification, network exfiltration, privilege escalation, or
  persistence**.
- The relaxed `ptrace_scope=0` setting further increases exposure by allowing same‑UID tracing.

👉 **For safety, always run `dynamoscan` in a controlled or isolated environment**, such as:

- A **disposable virtual machine** (VM)
- A **container sandbox** (Docker/Podman) using the `--docker` flag
- A **dedicated test machine** with no sensitive data or credentials
- An **already-isolated CI/CD environment** or sandbox

Never run the dynamic scanner directly on your personal workstation, production host, or any system containing sensitive
information. Treat the scan environment as **potentially compromised** once an artifact has been executed.

If in doubt, use **static scanners only** (`modelscan`, `picklescan`, `fickling`) when analyzing untrusted files outside
a sandboxed setup.

***

## 🐳 Docker Isolation Mode (Recommended)

For maximum safety when scanning untrusted models, `dynamoscan` can run scans inside an isolated Docker container. This
approach provides strong security boundaries and eliminates the need to modify system-level ptrace settings on your host
machine.

### Prerequisites

**1. Install docker dependencies**

```shell
pip install -e .[docker]
```

**2. Install Docker**

```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y docker.io

# Or follow official Docker installation guide for your distribution
```

**3. Configure User Permissions**

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Apply the group membership (or log out and back in)
newgrp docker

# Verify Docker access
docker ps
```

### Optional installation

### Usage

Simply add the `--docker` flag to any scan command:

```bash
# Basic Docker scan
dynamoscan ./models/model.pt --docker

# With report output
dynamoscan ./models/model.pt --docker --report-file report.json

# Force rebuild the Docker image (e.g., after dynamoscan updates)
dynamoscan ./models/model.pt --docker --docker-rebuild
```

### First Run: Automatic Image Build

On first use, `dynamoscan` will automatically build a Docker image containing all necessary dependencies.

The build process:

- Takes up to 10 minutes depending on your system
- Only happens once (the image is reused for subsequent scans)
- Can be forced to rebuild with `--docker-rebuild`

### Docker Mode Options

| Option                 | Description                                     |
|------------------------|-------------------------------------------------|
| `--docker`             | Enable Docker isolation mode                    |
| `--docker-rebuild`     | Force rebuild of the Docker image               |
| `--docker-config PATH` | Use custom Docker configuration YAML (optional) |

### Advanced Configuration

Create a custom `docker_config.yaml` file to customize resource limits, network settings, or add ML framework
dependencies:

```yaml
docker:
  image_name: dynamoscan-runtime
  image_tag: latest
  resource_limits:
    memory: '16g'      # Default: 8g
    cpus: '8.0'        # Default: 4.0
  network_mode: none   # Default: none (no network access)
  extra_packages:
    - onnx             # Add additional ML frameworks if needed
    - transformers
```

Then use it:

```bash
dynamoscan model.pt --docker --docker-config /path/to/docker_config.yaml
```

### Troubleshooting

**Permission Denied Error:**

```
Failed to connect to Docker daemon: Permission denied
```

→ Make sure you're in the `docker` group and have logged out/in or run `newgrp docker`

**Image Build Fails:**  
→ Ensure you have sufficient disk space (at least 2GB free) and a stable internet connection

**Container Cleanup:**  
Remove stopped containers:

```bash
docker container prune -f
```

Remove the dynamoscan image to rebuild from scratch:

```bash
docker rmi dynamoscan-runtime:latest
dynamoscan model.pt --docker  # Will rebuild automatically
```

### Docker vs Local Mode Comparison

| Feature        | Local Mode                                    | Docker Mode                                             |
|----------------|-----------------------------------------------|---------------------------------------------------------|
| Security       | ⚠️ Requires host trust or external isolation  | ✅ Strong container isolation                            |
| Setup          | Requires strace + ptrace config               | Only Docker needed                                      |
| Performance    | Faster (native execution)                     | Slightly slower (containerization overhead)             |
| Network access | ⚠️ Full host network                          | ✅ Isolated (no network by default)                      |
| Use case       | Trusted models, already-isolated environments | **Untrusted models, production scans on regular hosts** |

**Recommendation:**

- Use `--docker` when scanning models from unknown or untrusted sources **on your regular host machine**.
- Use local mode (without `--docker`) when already running in an isolated environment like a dedicated VM, CI/CD
  sandbox, or disposable test system where the scanning prerequisites are met.

