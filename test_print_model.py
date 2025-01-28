import gurobipy as gp
from gurobipy import GRB
from core.canonical_form_generator import CanonicalFormGenerator
from utils.config import INPUT_PROBLEM

from core.ordering.rule_combination import RuleComposition
from core.ordering.objective_coefficient_rule import ObjectiveCoefficientRule

rules = [
    ObjectiveCoefficientRule(),
]
ordering_rule = RuleComposition(rules)

# Example of usage with your CanonicalFormGenerator
if __name__ == "__main__":
    # Create the dummy model
    model = gp.read(INPUT_PROBLEM)

    # Generate the canonical form
    generator = CanonicalFormGenerator(model, ordering_rule)
    canonical_model = generator.get_canonical_form()
