from typing import Union, List
import os
import shutil
import glob
from Generic_downloader import *
pip_install_library('gitpython')
import git

def clone_repo_code(gituser='LorenzoTausani', repo_name='XDREAM_custom_LT'):
    repo_address = f'https://github.com/{gituser}/{repo_name}.git'
    try:
        git.Repo.clone_from(repo_address, f'./{repo_name}')
        print(repo_name+" cloned successfully!")
    except git.exc.GitCommandError as e:
        print(f"Failed to clone repository {repo_name}: {e}")

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


