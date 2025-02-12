import numpy as np
import scipy.sparse as sp
from utils.logging_handler import LoggingHandler


class Normalizer:
    def __init__(self):
        """
        Initialize the Normalizer with a logger for debugging purposes.

        Args:
            logger: Logging handler for detailed debug messages.
        """
        self.logger = LoggingHandler().get_logger()

    def normalize_matrix_consistently(self, A, obj_coeffs, rhs, bounds):
        """
        Consistently normalize both rows and columns of matrix A.

        Args:
            A (scipy.sparse.csr_matrix): Constraint matrix.
            obj_coeffs (np.ndarray): Objective coefficients.
            rhs (np.ndarray): Right-hand side values.
            bounds (list of tuples): List of (lower_bound, upper_bound) for variables.

        Returns:
            tuple: Normalized (A, obj_coeffs, rhs, bounds) and scaling factors.
        """
        A_csr = A.tocsr()

        # Compute row and column norms
        row_norms = np.sqrt(A_csr.power(2).sum(axis=1).A.flatten())  # Row norms
        col_norms = np.sqrt(A_csr.power(2).sum(axis=0).A.flatten())  # Column norms

        self.logger.debug("Column norms before scaling:")
        for i, norm in enumerate(col_norms):
            self.logger.debug(f"Col {i}: {norm}")

        # Avoid division by zero
        row_norms[row_norms == 0] = 1
        col_norms[col_norms == 0] = 1

        # Normalize the matrix
        row_scaling = np.reciprocal(row_norms)
        col_scaling = np.reciprocal(col_norms)
        D_row = sp.diags(row_scaling)  # Diagonal matrix for rows
        D_col = sp.diags(col_scaling)  # Diagonal matrix for columns

        A_normalized = D_row @ A_csr @ D_col  # Apply row and column normalization

        # Normalize objective coefficients and RHS
        obj_coeffs = obj_coeffs * col_norms
        rhs = rhs * row_scaling

        # Normalize bounds
        normalized_bounds = []
        for i, (lb, ub) in enumerate(bounds):
            scale = col_scaling[i]
            normalized_bounds.append((lb * scale, ub * scale))

        return A_normalized, obj_coeffs, rhs, normalized_bounds

    def normalize(self, A, obj_coeffs, rhs, bounds):
        """
        Public method to normalize the inputs and log the results.

        Args:
            A (scipy.sparse.csr_matrix): Constraint matrix.
            obj_coeffs (np.ndarray): Objective coefficients.
            rhs (np.ndarray): Right-hand side values.
            bounds (list of tuples): List of (lower_bound, upper_bound) for variables.

        Returns:
            tuple: Normalized (A, obj_coeffs, rhs, bounds) and scaling factors.
        """
        self.logger.debug("Starting normalization...")
        normalized_A, normalized_obj_coeffs, normalized_rhs, normalized_bounds = self.normalize_matrix_consistently(
            A, obj_coeffs, rhs, bounds
        )
        self.logger.debug("Normalization completed.")
        self.logger.debug(f"Normalized matrix shape: {normalized_A.shape}, nnz: {normalized_A.nnz}")
        return normalized_A, normalized_obj_coeffs, normalized_rhs, normalized_bounds
