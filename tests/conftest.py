# tests/conftest.py
import os
import random
import shutil
import string

import pytest

BASE_TARGET_FILES = ["/tmp/testfile", "/tmp/existing_file.txt", "/tmp/source"]
BASE_TARGET_DIRS = ["/tmp/malicious_dir"]
GENERATED_PATHS = [
    "/tmp/moved_file",
    "/tmp/passwd_link",
    "/tmp/malicious_write.txt",
    "/tmp/hacked_shell.txt",
    "/tmp/spawned.txt",
    "/tmp/network_scan.txt",
    "/tmp/malware",
    "/tmp/malicious.txt",
    "/tmp/eval_test.txt",
    "/tmp/HACKED.txt",
]


def create_base_state():
    for p in BASE_TARGET_FILES:
        os.makedirs(os.path.dirname(p) or "/", exist_ok=True)
        with open(p, "w") as f:
            f.write("test content")
    for d in BASE_TARGET_DIRS:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "test_file.txt"), "w") as f:
            f.write("test")


def clean_paths():
    # clean both generated artifacts and base targets/dirs
    for p in GENERATED_PATHS + BASE_TARGET_FILES + BASE_TARGET_DIRS:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
        except (FileNotFoundError, PermissionError):
            pass


@pytest.fixture
def fresh_env():
    # Set up once for this test file
    create_base_state()
    yield
    # Tear down after all tests in this file finish
    clean_paths()


@pytest.fixture
def rnd_name(length=8):
    letters = string.ascii_letters
    return ''.join(random.choices(letters, k=length))
