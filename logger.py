import logging
import warnings

class Logger:
    def __init__(self, base_log_file='training.log'):
        self.loggers = {}
        self.base_log_file = base_log_file
        self._setup_logger('main', base_log_file)
        self._setup_logger('training', 'training.log')
        self._setup_logger('eval', 'eval.log')
        
        warnings.filterwarnings("always")

    def _setup_logger(self, name, log_file):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.loggers[name] = logger

    def get_logger(self, name):
        if name not in self.loggers:
            log_file = f"{name}_{self.base_log_file}"
            self._setup_logger(name, log_file)
        return self.loggers[name]

    def info(self, message, logger_name='main'):
        self.get_logger(logger_name).info(message)

    def warning(self, message, logger_name='main'):
        self.get_logger(logger_name).warning(message)

    def error(self, message, logger_name='main'):
        self.get_logger(logger_name).error(message)