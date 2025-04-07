
import logging
import os
import atexit
import numpy as np
import pandas as pd
from datetime import datetime
from utils.config import LOG_LEVEL, PRODUCTION, BUCKET_NAME, OUTPUT_DIR, INPUT_PROBLEM, MATRICES_TO_CSV, LOG_MATRIX

class LoggingHandler:
    _instance = None  # Singleton instance tracker
    _production_initialized = False  # Track production setup

    def __new__(cls, *args, **kwargs):
        """Enforce singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, file_path=None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Initialization code (keep your existing __init__ logic)
        self.log_dir = OUTPUT_DIR
        self.file_path = file_path or "experiment"
        self.logger = None
        self.log_file = None
        self._initialize_logging_core()

        if PRODUCTION and not self._production_initialized:
            import boto3
            self._setup_production_features()
            LoggingHandler._production_initialized = True

        self._initialized = True  # Mark as initialized

    def _setup_production_features(self):
        """Production setup that runs only once"""
        atexit.register(self._s3_upload_wrapper)
        self.logger.debug("Initialized S3 logging capabilities")

    # Keep other methods unchanged...

    def _initialize_logging_core(self):
        """Core logging setup that always runs"""
        # Ensure directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate filename components
        base_name = os.path.basename(self.file_path)
        base_name = os.path.splitext(base_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log file path
        self.log_file = os.path.join(
            self.log_dir, 
            f'{base_name}_{timestamp}_{INPUT_PROBLEM}.log'
        )

        # Configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(LOG_LEVEL)

        # Clear existing handlers to prevent duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File handler (always present)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s: %(message)s'
        ))
        self.logger.addHandler(file_handler)

        # Stream handler (always present)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s: %(message)s'
        ))
        self.logger.addHandler(stream_handler)

        # Ensure final flush
        atexit.register(self._flush_handlers)

    def _s3_upload_wrapper(self):
        """Safe wrapper for S3 upload with error handling"""
        try:
            self._upload_to_s3()
        except Exception as e:
            self.logger.error(f"S3 upload failed: {str(e)}")

    def _upload_to_s3(self):
        """Perform actual S3 upload"""
        if not PRODUCTION:
            return

        s3 = boto3.client('s3')
        bucket_name = BUCKET_NAME
        s3_path = f"{OUTPUT_DIR}{os.path.basename(self.log_file)}"

        with open(self.log_file, 'rb') as f:
            s3.put_object(
                Bucket=bucket_name,
                Key=s3_path,
                Body=f,
                Metadata={
                    'generated_at': datetime.now().isoformat(),
                    'source': self.file_path
                }
            )
        self.logger.info(f"Uploaded logs to s3://{bucket_name}/{s3_path}")

    def _flush_handlers(self):
        """Ensure all handlers are flushed"""
        for handler in self.logger.handlers:
            handler.flush()

    def get_logger(self):
        """Public accessor for the logger"""
        return self
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def lazy_debug(self, msg, *args, **kwargs):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    @staticmethod
    def log_model_differences(logger, model1, model2):
        """Log detailed differences between all parameters of two models"""
        tolerance = 1e-6  # Define a tolerance for floating-point comparisons
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Compare objective coefficients
        obj1 = [v.Obj for v in model1.getVars()]
        obj2 = [v.Obj for v in model2.getVars()]
        if obj1 != obj2:
            logger.lazy_debug("Objective coefficient differences found:")
            for i, (o1, o2) in enumerate(zip(obj1, obj2)):
                if abs(o1 - o2) > tolerance:
                    logger.lazy_debug(f"Var {i}: {o1} vs {o2}")

        # Compare variable bounds
        for i, (v1, v2) in enumerate(zip(model1.getVars(), model2.getVars())):
            if abs(v1.LB - v2.LB) > tolerance or abs(v1.UB - v2.UB) > tolerance:
                logger.lazy_debug(f"Variable bounds differ for Var {i}: LB={v1.LB} vs {v2.LB}, UB={v1.UB} vs {v2.UB}")

        # Compare constraint matrix structure and coefficients
        A1 = model1.getA().toarray()
        A2 = model2.getA().toarray()

        if A1.shape != A2.shape:
            logger.lazy_debug(f"Matrix shape differs: {A1.shape} vs {A2.shape}")
        else:
            diff_matrix = np.abs(A1 - A2) > tolerance
            if LOG_MATRIX and diff_matrix.any():
                logger.lazy_debug("Matrix coefficient differences found:")
                df_A1 = pd.DataFrame(A1, columns=[f"x{i}" for i in range(A1.shape[1])])
                df_A2 = pd.DataFrame(A2, columns=[f"x{i}" for i in range(A2.shape[1])])

                df_combined = pd.concat([df_A1, df_A2], axis=1, keys=["Model 1", "Model 2"])

                if MATRICES_TO_CSV:
                    df_combined.to_csv(f"constraint_matrix_model_{timestamp}.csv", index=False)
                    logger.lazy_debug(f"Constraint matrices saved as 'constraint_matrix_model_{timestamp}.csv'")

                logger.lazy_debug("\n" + df_combined.to_string())

        # Compare RHS
        rhs1 = [c.RHS for c in model1.getConstrs()]
        rhs2 = [c.RHS for c in model2.getConstrs()]
        if rhs1 != rhs2:
            logger.lazy_debug("RHS differences found:")
            df_rhs = pd.DataFrame({"Model 1": rhs1, "Model 2": rhs2})
            logger.lazy_debug("\n" + df_rhs.to_string())

        # Compare constraint senses
        senses1 = [c.Sense for c in model1.getConstrs()]
        senses2 = [c.Sense for c in model2.getConstrs()]
        if senses1 != senses2:
            logger.lazy_debug("Constraint sense differences found:")
            df_sense = pd.DataFrame({"Model 1": senses1, "Model 2": senses2})
            logger.lazy_debug("\n" + df_sense.to_string())

        # Compare variable types
        vtype1 = [v.VType for v in model1.getVars()]
        vtype2 = [v.VType for v in model2.getVars()]
        if vtype1 != vtype2:
            logger.lazy_debug("Variable type differences found:")
            df_vtype = pd.DataFrame({"Model 1": vtype1, "Model 2": vtype2})
            logger.lazy_debug("\n" + df_vtype.to_string())
