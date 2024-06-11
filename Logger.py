import logging
import os
from logging.handlers import RotatingFileHandler


class AppLogger:
    def __init__(self, log_file='app.log', log_dir='./logs', logger_name='app_logger'):
        self.log_file = log_file
        self.log_dir = log_dir
        self.logger_name = logger_name
        self.logger = None
        self.setup_logger()

    def setup_logger(self):
        """Set up the logger with rotation, formatting, and console output."""
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Ensure the log directory is secure
        os.makedirs(self.log_dir, exist_ok=True)
        os.chmod(self.log_dir, 0o700)

        log_path = os.path.join(self.log_dir, self.log_file)

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)

        # Check if the logger already has handlers to avoid duplication
        if not self.logger.hasHandlers():
            # Use a rotating file handler for file logging
            file_handler = RotatingFileHandler(log_path, maxBytes=1000000, backupCount=5)
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)

            # Use a stream handler for console logging
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

logger = AppLogger().get_logger()