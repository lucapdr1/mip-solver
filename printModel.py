import gurobipy as gp
from gurobipy import GRB
from core.canonical_form_generator import CanonicalFormGenerator
from core.problem_printer import ProblemPrinter

# Example of usage with your CanonicalFormGenerator
if __name__ == "__main__":
    # Create the dummy model
    model = gp.read("input/dummy.mps")

    # Generate the canonical form
    generator = CanonicalFormGenerator(model)
    canonical_model = generator.get_canonical_form()
