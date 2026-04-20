"""Create .venv if needed, then install requirements."""

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_venv():
    if VENV_DIR.exists():
        print(".venv already exists, skipping creation.")
        return

    print(".venv not found, creating it now...")
    run(["py", "-m", "venv", ".venv"])


def install_requirements():
    if os.name == "nt":
        python_exe = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_exe = VENV_DIR / "bin" / "python" # check for macOS/Linux

    print("Installing requirements...")
    run([str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def main():
    ensure_venv()
    install_requirements()


if __name__ == "__main__":
    main()