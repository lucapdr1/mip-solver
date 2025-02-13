import gurobipy as gp
from gurobipy import GRB
from utils.gurobi_utils import get_Input_problem, init_gurobi_env

from core.ordering.rule_combination import RuleComposition
from core.ordering.variables.objective_coefficient_rule import ObjectiveCoefficientRule
from utils.problem_printer import ProblemPrinter
from utils.logging_handler import LoggingHandler
rules = [
    ObjectiveCoefficientRule(),
]
ordering_rule = RuleComposition(rules)


# Example of usage with your CanonicalFormGenerator
if __name__ == "__main__":
    # Create the dummy model
    gp_env = init_gurobi_env()
    input_problem = get_Input_problem()
    model = gp.read(input_problem, env=gp_env)
    logger = LoggingHandler().get_logger()
    ProblemPrinter.log_model(model,logger)
