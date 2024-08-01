import os
import shutil
import stat
from utility import Utility

def set_readonly_except_pycache(path):
    """
    Set the given path (file or directory) to read-only, except for __pycache__ directories.
    """
    for root, dirs, files in os.walk(path):
        if "__pycache__" in root:
            continue
        for dir in dirs:
            if dir != "__pycache__":
                dir_path = os.path.join(root, dir)
                os.chmod(dir_path, stat.S_IRUSR | stat.S_IXUSR)
        for file in files:
            file_path = os.path.join(root, file)
            os.chmod(file_path, stat.S_IRUSR)

def copy_directory(source_dir, target_dir):
    """
    Copy a directory to a target location, replacing it if it already exists,
    and set it to read-only except for __pycache__.
    """
    try:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        set_readonly_except_pycache(target_dir)
        print(f"Copied {source_dir} to {target_dir} and set to read-only (except __pycache__)")
    except Exception as e:
        print(f"Error copying {source_dir} to {target_dir}: {str(e)}")

def copy_utils_to_directories(source_dir, target_directories):
    """
    Copy the utils directory to specified directories and set to read-only (except __pycache__).
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return
    
    for directory in target_directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist. Creating it.")
            os.makedirs(directory)
        
        destination = os.path.join(directory, "utils")
        copy_directory(source_dir, destination)
        
        copy_utils_path = os.path.join(destination, "copy_utils.py")
        if os.path.exists(copy_utils_path):
            os.remove(copy_utils_path)

if __name__ == "__main__":
    utility = Utility()
    
    # Path to utils directory
    utils_dir = os.path.dirname(__file__)
    
    target_dirs = [
    ]
    
    # Convert relative paths to absolute paths
    target_dirs = [os.path.join(utility.repo_root, dir) for dir in target_dirs]
    
    copy_utils_to_directories(utils_dir, target_dirs)