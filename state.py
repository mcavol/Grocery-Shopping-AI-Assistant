"""
State management for the grocery shopping assistant.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from typing_extensions import TypedDict


class ShoppingItem(BaseModel):
    """Individual shopping item."""
    name: str
    quantity: str
    estimated_price: float
    category: str = "general"


class Recipe(BaseModel):
    """Recipe information."""
    name: str
    ingredients: List[str]
    servings: int
    instructions: Optional[str] = None
    
    @validator('instructions', pre=True)
    def convert_instructions_to_string(cls, v):
        """Convert instructions from list to string if needed."""
        if v is None:
            return "No instructions provided."
        elif isinstance(v, list):
            # Join list items with newlines or numbered steps
            if len(v) > 0:
                # Check if items are already numbered
                if any(item.strip().startswith(('1.', '2.', '3.', 'Step 1', 'Step 2')) for item in v[:3]):
                    return '\n'.join(v)
                else:
                    # Add numbering
                    return '\n'.join(f"{i+1}. {step}" for i, step in enumerate(v))
            else:
                return "No instructions provided."
        elif isinstance(v, str):
            return v
        else:
            return str(v)


class ShoppingState(TypedDict):
    """State shared across all agents."""
    # User input
    user_request: str
    budget: Optional[float]
    people_count: int
    
    # Agent outputs
    plan: Optional[str]
    recipe: Optional[Recipe]
    ingredients: List[str]
    shopping_items: List[ShoppingItem]
    total_cost: float
    final_list: Optional[str]
    
    # Control flow
    next_agent: str
    messages: List[str]
    errors: List[str]
    completed_agents: List[str]


def create_initial_state(user_request: str, budget: Optional[float] = None, people_count: int = 4) -> ShoppingState:
    """Create initial state from user input."""
    return ShoppingState(
        user_request=user_request,
        budget=budget,
        people_count=people_count,
        plan=None,
        recipe=None,
        ingredients=[],
        shopping_items=[],
        total_cost=0.0,
        final_list=None,
        next_agent="planner",
        messages=[],
        errors=[],
        completed_agents=[]
    )