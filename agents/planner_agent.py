"""
Planner Agent - Interprets user intent and creates execution plan.
"""

from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langsmith import traceable
from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """Agent responsible for interpreting user requests and creating execution plans."""
    
    def __init__(self, llm):
        super().__init__(llm, "planner")
        self.prompt_template = PromptTemplate(
            input_variables=["user_request", "budget", "people_count"],
            template="""
            You are a grocery shopping planner. Analyze the user's request and create a detailed plan.
            
            User Request: {user_request}
            Budget: ${budget} (if specified)
            People Count: {people_count}
            
            Create a plan that includes:
            1. What type of meal/items are needed
            2. Key considerations (budget, dietary needs, etc.)
            3. Next steps for other agents
            
            Be specific and actionable. Return only the plan text.
            """
        )
    
    @traceable
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan from user request."""
        try:
            self.log_execution(state, "Creating execution plan")
            
            # Format budget for display
            budget_str = f"{state['budget']}" if state['budget'] else "Not specified"
            
            # Generate plan using LLM
            prompt = self.prompt_template.format(
                user_request=state["user_request"],
                budget=budget_str,
                people_count=state["people_count"]
            )
            
            plan = self.llm.invoke(prompt).strip()
            
            # Update state
            state["plan"] = plan
            state["next_agent"] = "recipe"
            
            self.mark_completed(state)
            self.log_execution(state, f"Plan created: {plan[:100]}...")
            
        except Exception as e:
            self.handle_error(state, f"Failed to create plan: {str(e)}")
            state["next_agent"] = "error"
            
        return state