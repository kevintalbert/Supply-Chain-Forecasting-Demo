#!/usr/bin/env python3
# ###########################################################################
# CML “Install Dependencies” job — pip installs project ``requirements.txt``.
# Model builds use the same file via ``cdsw-build.sh``.
# ###########################################################################
import subprocess
import sys


def main() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    )
    print("Installed requirements.txt")


if __name__ == "__main__":
    main()
