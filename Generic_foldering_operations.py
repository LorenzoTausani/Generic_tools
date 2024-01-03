import os
import shutil

def remove_dirs(root,folders_to_remove):
    if not isinstance(folders_to_remove, list):
        folders_to_remove = [folders_to_remove]
    for f in folders_to_remove:
        if os.path.isdir(os.path.join(root, f)):
            shutil.rmtree(os.path.join(root,f+'/'))
