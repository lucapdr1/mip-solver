from core.ordering.blocks.block_rules import HierarchicalBlockOrderingRule, SizeBlockOrderingRule
from core.ordering.blocks.rules_adapter import OrderingRuleBlockAdapter

class BlockOrderingFactory:
    """
    Factory class for creating block ordering rules from existing ordering rules.
    
    This class provides methods to:
    1. Convert any ordering rule to a block ordering rule
    2. Create composite block ordering rules (lexicographic ordering)
    3. Configure common block ordering scenarios
    """
    
    @staticmethod
    def from_ordering_rule(ordering_rule, aggregation='sum', component='both', descending=True):
        """
        Create a block ordering rule from any ordering rule.
        
        :param ordering_rule: Any OrderingRule implementation
        :param aggregation: How to aggregate scores - 'average', 'max', 'sum', or 'median'
        :param component: Which component to score - 'variables', 'constraints', or 'both'
        :param descending: If True, higher scores come first
        :return: A BlockOrderingRule that wraps the given ordering rule
        """
        return OrderingRuleBlockAdapter(
            ordering_rule=ordering_rule,
            aggregation=aggregation,
            component=component,
            descending=descending
        )
    
    @staticmethod
    def lexicographic(*rules):
        """
        Create a hierarchical (lexicographic) block ordering rule from multiple rules.
        
        :param rules: Any number of BlockOrderingRule objects or OrderingRule objects
        :return: A HierarchicalBlockOrderingRule combining all given rules
        """
        block_rules = []
        
        for rule in rules:
            # If the rule is already a BlockOrderingRule, use it directly
            if hasattr(rule, 'score_blocks'):
                block_rules.append(rule)
            # Otherwise, assume it's an OrderingRule and wrap it
            else:
                block_rules.append(BlockOrderingFactory.from_ordering_rule(rule))
        
        return HierarchicalBlockOrderingRule(block_rules)
    
    @staticmethod
    def with_size_tiebreaker(ordering_rule, **kwargs):
        """
        Create a block ordering rule that uses the given ordering rule as primary
        criterion and block size as a tiebreaker.
        
        :param ordering_rule: Any OrderingRule implementation
        :param kwargs: Additional parameters for from_ordering_rule
        :return: A HierarchicalBlockOrderingRule
        """
        primary_rule = BlockOrderingFactory.from_ordering_rule(ordering_rule, **kwargs)
        size_rule = SizeBlockOrderingRule(descending=True)
        
        return HierarchicalBlockOrderingRule([primary_rule, size_rule])