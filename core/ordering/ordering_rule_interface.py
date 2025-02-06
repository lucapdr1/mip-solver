from abc import ABC, abstractmethod

class OrderingRule(ABC):
    @abstractmethod
    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Score variables based on the ordering rule.
        Args:
            vars: List of variables.
            obj_coeffs: Objective coefficients of variables.
            bounds: Variable bounds (LB, UB).

        Returns:
            List of scores for each variable.
        """
        pass

    @abstractmethod
    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Score constraints based on the ordering rule.
        Args:
            constraints: List of constraints.
            A: Constraint matrix.
            rhs: Right-hand side values.

        Returns:
            List of scores for each constraint.
        """
        pass
