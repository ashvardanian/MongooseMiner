import subprocess
import sys


def install_package(package_pypi_name: str) -> None:
    try:
        subprocess.check_call([sys.executable, "-m", "uv", "pip", "install", package_pypi_name])
    except subprocess.CalledProcessError:
        raise InstallError(package_pypi_name)
    
    
class InstallError(Exception):
    def __init__(self, package_pypi_name: str) -> None:
        super.__init__(f"Error: {package_pypi_name} could not be installed.")
    
    
if __name__ == "__main__":
    PACKAGE = "numpy"
    install_package(PACKAGE)