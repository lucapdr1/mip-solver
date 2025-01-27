import gurobipy as gp
from gurobipy import GRB

class ProblemPrinter:
    @staticmethod
    def format_model(model):
        """
        Formats the Gurobi model into a human-readable string.
        
        Args:
            model (gurobipy.Model): Gurobi model to be formatted.

        Returns:
            str: Formatted representation of the model.
        """
        lines = []

        # Objective
        lines.append("\nObjective:")
        obj_terms = []
        for var in model.getVars():
            coeff = var.Obj
            if coeff != 0:
                obj_terms.append(f"{coeff:+g} {var.VarName}")
        objective = " + ".join(obj_terms).replace(" + -", " - ")
        lines.append(f"    {objective} {'(minimize)' if model.ModelSense == GRB.MINIMIZE else '(maximize)'}")

        # Variables
        lines.append("\nVariables:")
        for var in model.getVars():
            lb = f"[{var.LB:g}" if var.LB > -GRB.INFINITY else "[-inf"
            ub = f", {var.UB:g}]" if var.UB < GRB.INFINITY else ", inf]"
            lines.append(f"    {var.VarName} {lb}{ub} ({'Continuous' if var.VType == GRB.CONTINUOUS else 'Integer'})")

        # Constraints
        lines.append("\nConstraints:")
        for constr in model.getConstrs():
            lhs_terms = []
            row = model.getRow(constr)
            for i in range(row.size()):
                coeff = row.getCoeff(i)
                lhs_terms.append(f"{coeff:+g} {row.getVar(i).VarName}")
            lhs = " + ".join(lhs_terms).replace(" + -", " - ")
            sense = {"<": "<=", ">": ">=", "=": "="}[constr.Sense]
            lines.append(f"    {lhs} {sense} {constr.RHS:g}")

        return "\n".join(lines)

    @staticmethod
    def log_model(model, logger, level="INFO"):
        """
        Logs the Gurobi model using the provided logger.
        
        Args:
            model (gurobipy.Model): Gurobi model to be logged.
            logger (logging.Logger): Logger instance.
            level (str): Logging level (e.g., "DEBUG", "INFO").
        """
        model_str = ProblemPrinter.format_model(model)
        log_method = getattr(logger, level.lower())  # Default to logger.info if level is invalid
        log_method(model_str)
