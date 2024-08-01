import os
import sys
import logging
from logging.handlers import RotatingFileHandler

try:
    from utils.utility import Utility
except ImportError:
    try:
        from .utility import Utility
    except ImportError:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utility import Utility

class CustomLogger:
    def __init__(self, logger_name, log_file='app.log'):
        self.utility = Utility()
        sys.path.append(self.utility.repo_root)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Create logs directory at the root of the repository if it doesn't exist
        log_dir = os.path.join(self.utility.repo_root, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File Handler
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, log_file),
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger

# Example usage
if __name__ == "__main__":
    # Create a logger for a specific module
    logger = CustomLogger("my_module").get_logger()
    
    # Use the logger
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")