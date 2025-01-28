import gurobipy as gp
from gurobipy import GRB
from core.canonical_form_generator import CanonicalFormGenerator
from utils.config import INPUT_PROBLEM

from core.ordering.rule_combination import RuleComposition
from core.ordering.objective_coefficient_rule import ObjectiveCoefficientRule
from utils.problem_printer import ProblemPrinter
from utils.logging_handler import LoggingHandler

rules = [
    ObjectiveCoefficientRule(),
]
ordering_rule = RuleComposition(rules)


# Example of usage with your CanonicalFormGenerator
if __name__ == "__main__":
    # Create the dummy model
    model = gp.read(INPUT_PROBLEM)
    logger = LoggingHandler().get_logger()
    ProblemPrinter.log_model(model,logger)
