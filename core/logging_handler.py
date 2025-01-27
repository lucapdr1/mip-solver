# core/logging_handler.py

import logging
import os
from datetime import datetime

LOGGING_LEVEL = logging.INFO

class LoggingHandler:
    def __init__(self, log_dir='experiments', file_path=None):
        """
        Initialize the logging handler.

        Args:
            log_dir (str): Directory to store experiment logs.
            file_path (str): Path to the input file for customization in log file name.
        """
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Extract the base file name from file_path (e.g., "example-min" from "input/example-min.mps")
        base_file_name = os.path.basename(file_path) if file_path else "experiment"
        base_file_name_no_ext = os.path.splitext(base_file_name)[0]
        
        # Generate log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'{base_file_name_no_ext}_{timestamp}.log')
        
        # Get the logger instance
        self.logger = logging.getLogger(__name__)
        
        # Avoid duplicate handlers by checking if handlers are already set
        if not self.logger.hasHandlers():
            # Configure logging
            self.logger.setLevel(LOGGING_LEVEL)
            file_handler = logging.FileHandler(log_file)
            stream_handler = logging.StreamHandler()
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    def get_logger(self):
        """
        Return the logger instance.

        Returns:
            logging.Logger: Logger instance
        """
        return self.logger
