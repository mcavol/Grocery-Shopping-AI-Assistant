"""
Budgeting Agent - Checks budget constraints and suggests optimizations.
"""

from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langsmith import traceable
from .base_agent import BaseAgent
from state import ShoppingItem


class BudgetingAgent(BaseAgent):
    """Agent responsible for budget analysis and optimization."""
    
    def __init__(self, llm):
        super().__init__(llm, "budgeting")
        self.prompt_template = PromptTemplate(
            input_variables=["total_cost", "budget", "items_list"],
            template="""
            You are a budget optimization expert. Analyze the shopping list against the budget.
            
            Total Cost: ${total_cost:.2f}
            Budget: ${budget}
            Items: {items_list}
            
            If over budget:
            1. Suggest specific items to remove or substitute
            2. Recommend cheaper alternatives
            3. Explain the savings
            
            If within budget:
            1. Confirm the list fits the budget
            2. Mention remaining budget
            3. Suggest optional additions if significant budget remains
            
            Provide actionable recommendations.
            """
        )
    
    @traceable
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze budget and optimize shopping list."""
        try:
            self.log_execution(state, "Analyzing budget constraints")
            
            total_cost = state.get("total_cost", 0)
            budget = state.get("budget")
            shopping_items = state.get("shopping_items", [])
            
            if not budget:
                # No budget specified, proceed without optimization
                state["next_agent"] = "finalizer"
                self.mark_completed(state)
                self.log_execution(state, "No budget specified, proceeding without optimization")
                return state
            
            # Check budget constraints
            budget_analysis = self._analyze_budget(total_cost, budget, shopping_items)
            
            if total_cost > budget:
                # Over budget - optimize list
                optimized_items = self._optimize_for_budget(shopping_items, budget)
                state["shopping_items"] = optimized_items
                state["total_cost"] = sum(item.estimated_price for item in optimized_items)
                
                self.log_execution(state, f"Optimized list: ${state['total_cost']:.2f} (was ${total_cost:.2f})")
            else:
                # Within budget
                remaining = budget - total_cost
                self.log_execution(state, f"Within budget, ${remaining:.2f} remaining")
            
            # Generate budget report using LLM
            items_summary = ", ".join([f"{item.name} (${item.estimated_price:.2f})" 
                                     for item in state["shopping_items"]])
            
            prompt = self.prompt_template.format(
                total_cost=state["total_cost"],
                budget=budget,
                items_list=items_summary
            )
            
            budget_report = self.llm.invoke(prompt).strip()
            state["messages"].append(f"Budget Analysis: {budget_report}")
            
            state["next_agent"] = "finalizer"
            self.mark_completed(state)
            
        except Exception as e:
            self.handle_error(state, f"Budget analysis failed: {str(e)}")
            state["next_agent"] = "finalizer"  # Continue to finalizer even if budget analysis fails
            
        return state
    
    def _analyze_budget(self, total_cost: float, budget: float, items: List[ShoppingItem]) -> Dict[str, Any]:
        """Analyze budget vs actual cost."""
        return {
            "total_cost": total_cost,
            "budget": budget,
            "over_budget": total_cost > budget,
            "difference": abs(total_cost - budget),
            "percentage": (total_cost / budget) * 100 if budget > 0 else 0
        }
    
    def _optimize_for_budget(self, items: List[ShoppingItem], budget: float) -> List[ShoppingItem]:
        """Optimize shopping list to fit within budget."""
        # Sort items by price (highest first) for removal priority
        sorted_items = sorted(items, key=lambda x: x.estimated_price, reverse=True)
        optimized_items = sorted_items.copy()
        current_total = sum(item.estimated_price for item in optimized_items)
        
        # Remove most expensive items until within budget
        while current_total > budget and optimized_items:
            # Remove the most expensive item that would bring us closer to budget
            for i, item in enumerate(optimized_items):
                new_total = current_total - item.estimated_price
                if new_total <= budget:
                    optimized_items.pop(i)
                    current_total = new_total
                    break
            else:
                # If no single item removal helps, remove the most expensive
                if optimized_items:
                    removed = optimized_items.pop(0)
                    current_total -= removed.estimated_price
        
        # Try to substitute with cheaper alternatives
        optimized_items = self._substitute_cheaper_alternatives(optimized_items, budget - current_total)
        
        return optimized_items
    
    def _substitute_cheaper_alternatives(self, items: List[ShoppingItem], remaining_budget: float) -> List[ShoppingItem]:
        """Substitute with cheaper alternatives where possible."""
        # Simple substitution rules
        substitutions = {
            "beef": ("chicken", 5.99),
            "organic": ("regular", lambda price: price * 0.7),
            "premium": ("standard", lambda price: price * 0.8)
        }
        
        for item in items:
            for expensive, (cheaper, new_price) in substitutions.items():
                if expensive.lower() in item.name.lower():
                    if callable(new_price):
                        item.estimated_price = new_price(item.estimated_price)
                    else:
                        item.estimated_price = new_price
                    item.name = item.name.replace(expensive.title(), cheaper.title())
                    break
        
        return items