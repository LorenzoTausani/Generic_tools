from typing import Union, List
import os
import shutil

def remove_dirs(root: str, folders_to_remove: Union[str, List[str]]) -> None:
    """
    Remove specified directories in the given root path.

    Parameters:
    - root (str): The root path where directories exist.
    - folders_to_remove (Union[str, List[str]]): A single folder name or a list of folder names to be removed.

    Returns:
    - None
    """
    if not isinstance(folders_to_remove, list):
        folders_to_remove = [folders_to_remove]

    for f in folders_to_remove:
        if os.path.isdir(os.path.join(root, f)):
            shutil.rmtree(os.path.join(root,f+'/'))
