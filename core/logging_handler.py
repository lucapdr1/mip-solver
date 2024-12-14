# core/logging_handler.py

import logging
import os
from datetime import datetime

class LoggingHandler:
    def __init__(self, log_dir='experiments'):
        """
        Initialize the logging handler.
        
        Args:
            log_dir (str): Directory to store experiment logs
        """
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def get_logger(self):
        """
        Return the logger instance.
        
        Returns:
            logging.Logger: Logger instance
        """
        return self.logger
