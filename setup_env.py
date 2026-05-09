"""Create local virtual environment and install project dependencies."""

import os
import subprocess
from pathlib import Path


# Resolve all project paths relative to this script so it works from any cwd.
ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


def run(command):
    # Reuse repository root as the working directory for every subprocess call.
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_venv():
    # Skip recreation if the virtual environment is already present.
    if VENV_DIR.exists():
        print(".venv already exists, skipping creation.")
        return

    print(".venv not found, creating it now...")
    run(["py", "-m", "venv", ".venv"])


def install_requirements():
    # Pick the venv's Python executable based on the current operating system.
    if os.name == "nt":
        python_exe = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_exe = VENV_DIR / "bin" / "python" # check for macOS/Linux

    print("Installing requirements...")
    run([str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def main():
    # Create the environment first, then install packages into that environment.
    ensure_venv()
    install_requirements()


if __name__ == "__main__":
    main()