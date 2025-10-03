"""Docker container manager for isolated model scanning."""

from __future__ import annotations

import docker
import logging
import shutil
import uuid
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger("docker_manager")


class DockerManager:
    """Manages Docker container lifecycle for isolated ML model scanning."""
    MODEL_DIR = "/model"
    OUTPUT_DIR_PREFIX = "/output"

    FALLBACK_DEFAULTS = {
        'image_name': 'dynamoscan-runtime',
        'image_tag': 'latest',
        'extra_packages': [],
        'resource_limits': {
            'memory': '8g',
            'cpus': '4.0'
        },
        'network_mode': 'none',
        'enable_gpu': False
    }

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize Docker manager."""
        try:
            self.client = docker.from_env()
        except ImportError:
            raise ImportError(
                "docker package not installed. Install with: pip install docker"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Docker daemon: {e}")

        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config(config_path)
        self.container_name = "dynamoscan-" + uuid.uuid4().hex

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load Docker configuration from YAML file."""
        if config_path:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            return self.FALLBACK_DEFAULTS.copy()

        logger.info(f"Loading Docker config from: {config_file}")
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f)

        if not user_config or 'docker' not in user_config:
            logger.warning("Config file has no 'docker' section. Using defaults.")
            return self.FALLBACK_DEFAULTS.copy()

        config = user_config['docker']

        for key, default_val in self.FALLBACK_DEFAULTS.items():
            if key not in config:
                config[key] = default_val
            elif key == 'resource_limits' and isinstance(default_val, dict):
                config[key] = {**default_val, **config.get(key, {})}

        return config

    def image_exists(self) -> bool:
        """Check if Docker image exists locally."""
        image_tag = f"{self.config['image_name']}:{self.config['image_tag']}"
        try:
            self.client.images.get(image_tag)
            return True
        except Exception:
            return False

    def build_image(self, force_rebuild: bool = False) -> str:
        """Build Docker image with ML dependencies."""
        image_tag = f"{self.config['image_name']}:{self.config['image_tag']}"

        if self.image_exists() and not force_rebuild:
            logger.info(
                f"Docker image {image_tag} already exists "
                f"(use --docker-rebuild to force rebuild)"
            )
            return image_tag

        logger.info("Building Docker image...")

        docker_dir = self.project_root / "docker"
        dockerfile_path = docker_dir / "Dockerfile"

        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            build_context = Path(tmpdir)

            shutil.copytree(
                self.project_root / "dynamoscan",
                build_context / "dynamoscan"
            )

            shutil.copy(
                self.project_root / "pyproject.toml",
                build_context / "pyproject.toml"
            )

            shutil.copy(dockerfile_path, build_context / "Dockerfile")

            extra_packages = self.config.get('extra_packages', [])
            if extra_packages:
                logger.info(f"Extra packages: {', '.join(extra_packages)}")
                with open(build_context / "extra_requirements.txt", 'w') as f:
                    f.write('\n'.join(extra_packages))
            else:
                with open(build_context / "extra_requirements.txt", 'w') as f:
                    f.write('')

            try:
                logger.info("Building image (this may take several minutes)...")

                build_logs = self.client.api.build(
                    path=str(build_context),
                    tag=image_tag,
                    rm=True,
                    decode=True
                )

                for chunk in build_logs:
                    if 'stream' in chunk:
                        print(chunk['stream'].strip())
                    elif 'error' in chunk:
                        raise RuntimeError(f"Docker build error: {chunk['error']}")

                logger.info(f"Successfully built image: {image_tag}")
                return image_tag

            except Exception as e:
                logger.error(f"Failed to build Docker image: {e}")
                raise

    def run_isolated_scan(
            self,
            model_path: str,
            trace_file: Optional[str] = None,
            report_file: Optional[str] = None,
            cwd: Optional[str] = None,
            print_trace: bool = False,
            print_res: bool = False,
            overwrite: bool = False
    ) -> int:
        """
        Execute full scan inside isolated Docker container.

        Args:
            model_path: Path to model file on host
            trace_file: Optional trace file path on host
            report_file: Optional report file path on host
            cwd: Working directory
            print_trace: Print trace output
            print_res: Print results
            overwrite: Overwrite existing trace file

        Returns:
            Tuple of (exit_code, stdout)
        """

        # Resolve all paths
        model_path = Path(model_path).resolve()

        # Collect output file paths and ensure parent directories exist
        output_files = {}
        if trace_file:
            trace_file = Path(trace_file).resolve()
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            output_files['trace'] = trace_file

        if report_file:
            report_file = Path(report_file).resolve()
            report_file.parent.mkdir(parents=True, exist_ok=True)
            output_files['report'] = report_file

        # Setup volumes
        volumes = self._setup_volumes(model_path, output_files, cwd)

        # Build container command
        container_cmd = self._build_container_command(
            model_path, output_files, cwd, print_trace, print_res, overwrite, volumes
        )

        # Container configuration
        container_config = {
            'image': f"{self.config['image_name']}:{self.config['image_tag']}",
            'name': self.container_name,
            'command': container_cmd,
            'volumes': volumes,
            'working_dir': self.MODEL_DIR,
            'cap_add': ['SYS_PTRACE'],
            'security_opt': ['seccomp=unconfined', 'apparmor=unconfined'],
            'network_mode': self.config.get('network_mode', 'none'),
            'mem_limit': self.config.get('resource_limits', {}).get('memory', '8g'),
            'detach': False,
            'remove': False,
            'user': 'scanner',
            'stderr': True,
            'stdout': True
        }

        cpus = self.config.get('resource_limits', {}).get('cpus', '4.0')
        container_config['nano_cpus'] = int(float(cpus) * 1e9)

        logger.debug("Starting isolated scan in Docker container")
        logger.debug(f"Network: {self.config.get('network_mode', 'none')}")
        logger.info(f"Command: dynamoscan {' '.join(container_cmd)}")

        try:
            output = self.client.containers.run(**container_config)
            stdout = _decode(output)
            print(stdout, end='')

            return 0

        except docker.errors.ContainerError as e:
            exit_code = e.exit_status
            output = e.container.logs(stdout=True, stderr=True)
            output_str = _decode(output)
            print(output_str, end='')

            return exit_code

        except Exception as e:
            logger.error(f"Failed to run container: {e}")
            raise

        finally:
            self.remove_container()

    def _setup_volumes(
            self,
            model_path: Path,
            output_files: Dict[str, Path],
            cwd: Optional[str]
    ) -> Dict[str, Dict]:
        """Setup volume mounts for container."""
        volumes = {
            str(model_path.parent): {
                'bind': self.MODEL_DIR,
                'mode': 'ro'
            }
        }

        # Collect unique output directories
        output_dirs: Set[Path] = set()
        for file_path in output_files.values():
            output_dirs.add(file_path.parent)

        # Mount each unique output directory
        for idx, output_dir in enumerate(output_dirs):
            output_dir_str = str(output_dir)
            if output_dir_str in volumes:
                volumes[output_dir_str]['mode'] = "rw"
                continue

            mount_point = f'{self.OUTPUT_DIR_PREFIX}{idx}' if idx > 0 else f'{self.OUTPUT_DIR_PREFIX}'
            volumes[str(output_dir)] = {
                'bind': mount_point,
                'mode': 'rw'
            }

        # Add cwd if specified
        if cwd:
            cwd_path = Path(cwd).resolve()
            volumes[str(cwd_path)] = {
                'bind': '/workspace',
                'mode': 'rw'
            }

        return volumes

    def _build_container_command(
            self,
            model_path: Path,
            output_files: Dict[str, Path],
            cwd: Optional[str],
            print_trace: bool,
            print_res: bool,
            overwrite: bool,
            volumes: Dict[str, Dict]
    ) -> List[str]:
        """Build command to run inside container with path translation."""
        cmd = [f'{self.MODEL_DIR}/{model_path.name}']

        # Translate trace file path
        if 'trace' in output_files:
            trace_file = output_files['trace']
            container_dir = volumes[str(trace_file.parent)]['bind']
            cmd.extend(['--trace-file', f'{container_dir}/{trace_file.name}'])

        # Translate report file path
        if 'report' in output_files:
            report_file = output_files['report']
            container_dir = volumes[str(report_file.parent)]['bind']
            cmd.extend(['--report-file', f'{container_dir}/{report_file.name}'])

        if cwd:
            cmd.extend(['--cwd', '/workspace'])

        if print_trace:
            cmd.append('--print-trace')

        if not print_res:
            cmd.append('--no-print')

        if overwrite:
            cmd.append('--overwrite')

        return cmd

    def remove_image(self) -> None:
        """Remove Docker image from local system."""
        image_tag = f"{self.config['image_name']}:{self.config['image_tag']}"
        try:
            self.client.images.remove(image_tag, force=True)
            logger.info(f"Removed image: {image_tag}")
        except Exception as e:
            logger.error(f"Failed to remove image: {e}")
            raise

    def remove_container(self) -> None:
        try:
            container = self.client.containers.get(self.container_name)
            container.remove(force=True)  # Forces removal even if running
        except docker.errors.NotFound:
            pass


def _decode(output: str) -> str:
    return output.decode('utf-8') if isinstance(output, bytes) else output
