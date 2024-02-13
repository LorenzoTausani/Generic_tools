import subprocess

def pip_install_library(libname: str) -> None:
    """
    Install a Python library using pip.

    Parameters:
        libname (str): The name of the library to install.

    Returns:
        None
    """
    try:
        # Execute the pip install command
        subprocess.check_call(["pip", "install", libname])
        # Print success message
        print(libname + " installed successfully!")
    except subprocess.CalledProcessError as e:
        # Print error message if installation fails
        print(f"Failed to install {libname}: {e}")