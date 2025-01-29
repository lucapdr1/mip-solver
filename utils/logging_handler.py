
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from utils.config import LOG_LEVEL, MATRICES_TO_CSV

class LoggingHandler:
    def __init__(self, log_dir='experiments', file_path=None):
        """
        Initialize the logging handler.

        Args:
            log_dir (str): Directory to store experiment logs.
            file_path (str): Path to the input file for customization in log file name.
        """
        os.makedirs(log_dir, exist_ok=True)

        base_file_name = os.path.basename(file_path) if file_path else "experiment"
        base_file_name_no_ext = os.path.splitext(base_file_name)[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'{base_file_name_no_ext}_{timestamp}.log')

        self.logger = logging.getLogger(__name__)

        if not self.logger.hasHandlers():
            self.logger.setLevel(LOG_LEVEL)
            file_handler = logging.FileHandler(log_file)
            stream_handler = logging.StreamHandler()

            formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    def get_logger(self):
        """Return the logger instance."""
        return self.logger

    @staticmethod
    def log_model_differences(logger, model1, model2):
        """Log detailed differences between all parameters of two models"""
        tolerance = 1e-6  # Define a tolerance for floating-point comparisons
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Compare objective coefficients
        obj1 = [v.Obj for v in model1.getVars()]
        obj2 = [v.Obj for v in model2.getVars()]
        if obj1 != obj2:
            logger.debug("Objective coefficient differences found:")
            for i, (o1, o2) in enumerate(zip(obj1, obj2)):
                if abs(o1 - o2) > tolerance:
                    logger.debug(f"Var {i}: {o1} vs {o2}")

        # Compare variable bounds
        for i, (v1, v2) in enumerate(zip(model1.getVars(), model2.getVars())):
            if abs(v1.LB - v2.LB) > tolerance or abs(v1.UB - v2.UB) > tolerance:
                logger.debug(f"Variable bounds differ for Var {i}: LB={v1.LB} vs {v2.LB}, UB={v1.UB} vs {v2.UB}")

        # Compare constraint matrix structure and coefficients
        A1 = model1.getA().toarray()
        A2 = model2.getA().toarray()

        if A1.shape != A2.shape:
            logger.debug(f"Matrix shape differs: {A1.shape} vs {A2.shape}")
        else:
            diff_matrix = np.abs(A1 - A2) > tolerance
            if diff_matrix.any():
                logger.debug("Matrix coefficient differences found:")
                df_A1 = pd.DataFrame(A1, columns=[f"x{i}" for i in range(A1.shape[1])])
                df_A2 = pd.DataFrame(A2, columns=[f"x{i}" for i in range(A2.shape[1])])

                df_combined = pd.concat([df_A1, df_A2], axis=1, keys=["Model 1", "Model 2"])

                if MATRICES_TO_CSV:
                    df_combined.to_csv(f"constraint_matrix_model_{timestamp}.csv", index=False)
                    logger.debug(f"Constraint matrices saved as 'constraint_matrix_model_{timestamp}.csv'")

                logger.debug("\n" + df_combined.to_string())

        # Compare RHS
        rhs1 = [c.RHS for c in model1.getConstrs()]
        rhs2 = [c.RHS for c in model2.getConstrs()]
        if rhs1 != rhs2:
            logger.debug("RHS differences found:")
            df_rhs = pd.DataFrame({"Model 1": rhs1, "Model 2": rhs2})
            logger.debug("\n" + df_rhs.to_string())

        # Compare constraint senses
        senses1 = [c.Sense for c in model1.getConstrs()]
        senses2 = [c.Sense for c in model2.getConstrs()]
        if senses1 != senses2:
            logger.debug("Constraint sense differences found:")
            df_sense = pd.DataFrame({"Model 1": senses1, "Model 2": senses2})
            logger.debug("\n" + df_sense.to_string())

        # Compare variable types
        vtype1 = [v.VType for v in model1.getVars()]
        vtype2 = [v.VType for v in model2.getVars()]
        if vtype1 != vtype2:
            logger.debug("Variable type differences found:")
            df_vtype = pd.DataFrame({"Model 1": vtype1, "Model 2": vtype2})
            logger.debug("\n" + df_vtype.to_string())
