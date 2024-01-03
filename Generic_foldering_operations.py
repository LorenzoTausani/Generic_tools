from typing import Union, List
import os
import shutil
import glob

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

def find_files_by_extension(directory: str, extension: str, recursive: bool = False) -> list:
    """
    Trova tutti i file con una determinata estensione all'interno di una directory (inclusi i sottodirectory).

    Args:
    - directory (str): Il percorso della directory in cui cercare i file.
    - extension (str): L'estensione dei file da cercare (ad esempio: '.txt', '.xlsx', ecc.).
    - recursive (bool): Se True, la ricerca viene estesa anche ai sottodirectory.

    Returns:
    - list: Una lista di percorsi dei file che corrispondono all'estensione specificata nella directory.
    """
    if recursive:
        pattern = os.path.join(directory, f"**/*{extension}")
    else:
        pattern = os.path.join(directory, f"*{extension}")

    files = glob.glob(pattern, recursive=recursive)
    return files

