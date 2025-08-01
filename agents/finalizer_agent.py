"""
Finalizer Agent - Aggregates everything into final shopping list.
"""

from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langsmith import traceable
from .base_agent import BaseAgent


class FinalizerAgent(BaseAgent):
    """Agent responsible for creating the final shopping list."""
    
    def __init__(self, llm):
        super().__init__(llm, "finalizer")
        self.prompt_template = PromptTemplate(
            input_variables=["recipe_name", "items_list", "total_cost", "budget", "people_count"],
            template="""
            Create a comprehensive final shopping list summary.
            
            Recipe: {recipe_name}
            Serves: {people_count} people
            Total Cost: ${total_cost:.2f}
            Budget: {budget}
            
            Shopping Items:
            {items_list}
            
            Create a well-formatted final shopping list that includes:
            1. Header with recipe name and cost summary
            2. Items organized by store category
            3. Total cost and budget status
            4. Any additional notes or recommendations
            
            Make it ready for printing or mobile use.
            """
        )
    
    @traceable
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create final shopping list."""
        try:
            self.log_execution(state, "Creating final shopping list")
            
            shopping_items = state.get("shopping_items", [])
            recipe = state.get("recipe")
            total_cost = state.get("total_cost", 0)
            budget = state.get("budget")
            people_count = state.get("people_count", 4)
            
            if not shopping_items:
                final_list = "No items found for shopping list."
            else:
                # Organize items by category
                categorized_items = self._organize_by_category(shopping_items)
                
                # Format items list for LLM
                items_text = self._format_items_for_prompt(categorized_items)
                
                # Generate final list using LLM
                budget_text = f"${budget:.2f}" if budget else "No budget specified"
                recipe_name = recipe.name if recipe else "Custom meal"
                
                prompt = self.prompt_template.format(
                    recipe_name=recipe_name,
                    items_list=items_text,
                    total_cost=total_cost,
                    budget=budget_text,
                    people_count=people_count
                )
                
                final_list = self.llm.invoke(prompt).strip()
            
            # Update state
            state["final_list"] = final_list
            state["next_agent"] = "complete"
            
            self.mark_completed(state)
            self.log_execution(state, "Final shopping list created")
            
        except Exception as e:
            self.handle_error(state, f"Failed to create final list: {str(e)}")
            # Create a basic fallback list
            state["final_list"] = self._create_fallback_list(state)
            state["next_agent"] = "complete"
            
        return state
    
    def _organize_by_category(self, items):
        """Organize shopping items by store category."""
        categories = {}
        for item in items:
            category = item.category
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        return categories
    
    def _format_items_for_prompt(self, categorized_items):
        """Format categorized items for LLM prompt."""
        formatted = []
        for category, items in categorized_items.items():
            formatted.append(f"\n{category.upper()}:")
            for item in items:
                formatted.append(f"  - {item.name} ({item.quantity}) - ${item.estimated_price:.2f}")
        return "\n".join(formatted)
    
    def _create_fallback_list(self, state):
        """Create a basic fallback shopping list."""
        items = state.get("shopping_items", [])
        total_cost = state.get("total_cost", 0)
        
        fallback = "SHOPPING LIST\n" + "="*50 + "\n\n"
        
        if items:
            for i, item in enumerate(items, 1):
                fallback += f"{i}. {item.name} ({item.quantity}) - ${item.estimated_price:.2f}\n"
            
            fallback += f"\nTOTAL: ${total_cost:.2f}\n"
            
            if state.get("budget"):
                budget = state["budget"]
                if total_cost <= budget:
                    fallback += f"✓ Within budget (${budget:.2f})\n"
                else:
                    fallback += f"⚠ Over budget by ${total_cost - budget:.2f}\n"
        else:
            fallback += "No items to display.\n"
        
        return fallback