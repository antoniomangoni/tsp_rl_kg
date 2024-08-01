import os
from typing import Optional

class Utility:
    def __init__(self, repo_name: Optional[str] = None):
        self.repo_name = repo_name or "vpp_forecasting"
        self.repo_root = self._find_git_root_path()
    
    def _find_git_root_path(self) -> str:
        current_dir = os.getcwd()
        original_dir = current_dir
        
        while current_dir != '/':
            if os.path.basename(current_dir) == self.repo_name:
                return current_dir
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root directory
                break
            current_dir = parent_dir
        
        raise ValueError(f'Repository "{self.repo_name}" not found in the path hierarchy starting from {original_dir}')
    
    def is_in_repo(self, path: str) -> bool:
        """Check if the given path is inside the specified repository.
        Args:
            path (str): The path to check.
        Returns:
            bool: True if the path is inside the repository, False otherwise.
        """
        absolute_path = os.path.abspath(path)
        return os.path.commonpath([self.repo_root, absolute_path]) == self.repo_root

    def get_abs_with_relative_path(self, *path_parts: str) -> str:
        """
        Takes a list of path parts
        Returns: 
            - The full path.
        """
        path = os.path.join(self.repo_root, *path_parts)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def get_print_path(self, path: str) -> str:
        if self.is_in_repo(path):
            return os.path.relpath(path, self.repo_root)
        else:
            return os.path.abspath(path)
        
