"""
Supervisor Agent - Orchestrates the execution flow and manages agent interactions.
"""

from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langsmith import traceable
from .base_agent import BaseAgent


class SupervisorAgent(BaseAgent):
    """Supervisor agent that manages the execution flow between other agents."""
    
    def __init__(self, llm):
        super().__init__(llm, "supervisor")
        self.agent_sequence = ["planner", "recipe", "product_finder", "budgeting", "finalizer"]
        self.prompt_template = PromptTemplate(
            input_variables=["current_state", "completed_agents", "errors"],
            template="""
            You are the supervisor of a grocery shopping assistant system.
            
            Current State Summary:
            {current_state}
            
            Completed Agents: {completed_agents}
            Errors: {errors}
            
            Based on the current state, determine:
            1. Which agent should execute next
            2. Whether to retry a failed agent
            3. Whether to skip an agent due to errors
            4. Whether the process is complete
            
            Return only the name of the next agent to execute, or "complete" if finished.
            Valid agents: planner, recipe, product_finder, budgeting, finalizer, complete, error
            """
        )
    
    @traceable
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Determine next agent in the execution flow."""
        try:
            self.log_execution(state, "Determining next agent")
            
            current_agent = state.get("next_agent", "planner")
            completed_agents = state.get("completed_agents", [])
            errors = state.get("errors", [])
            
            # Check for completion
            if current_agent == "complete" or len(completed_agents) >= len(self.agent_sequence):
                state["next_agent"] = "complete"
                self.log_execution(state, "Process completed")
                return state
            
            # Check for critical errors
            if len(errors) > 3:
                state["next_agent"] = "error"
                self.log_execution(state, "Too many errors, stopping process")
                return state
            
            # Determine next agent using LLM
            next_agent = self._determine_next_agent(state)
            
            if next_agent not in ["planner", "recipe", "product_finder", "budgeting", "finalizer", "complete", "error"]:
                # Fallback to sequential execution
                next_agent = self._get_next_sequential_agent(completed_agents)
            
            state["next_agent"] = next_agent
            self.log_execution(state, f"Next agent: {next_agent}")
            
        except Exception as e:
            self.handle_error(state, f"Supervisor decision failed: {str(e)}")
            # Fallback to sequential execution
            state["next_agent"] = self._get_next_sequential_agent(state.get("completed_agents", []))
            
        return state
    
    def _determine_next_agent(self, state: Dict[str, Any]) -> str:
        """Use LLM to determine next agent."""
        try:
            current_state_summary = self._create_state_summary(state)
            
            prompt = self.prompt_template.format(
                current_state=current_state_summary,
                completed_agents=", ".join(state.get("completed_agents", [])),
                errors="; ".join(state.get("errors", []))
            )
            
            response = self.llm.invoke(prompt).strip().lower()
            
            # Extract agent name from response
            for agent in self.agent_sequence + ["complete", "error"]:
                if agent in response:
                    return agent
            
            # Fallback
            return self._get_next_sequential_agent(state.get("completed_agents", []))
            
        except Exception:
            return self._get_next_sequential_agent(state.get("completed_agents", []))
    
    def _get_next_sequential_agent(self, completed_agents: List[str]) -> str:
        """Get next agent in sequential order."""
        for agent in self.agent_sequence:
            if agent not in completed_agents:
                return agent
        return "complete"
    
    def _create_state_summary(self, state: Dict[str, Any]) -> str:
        """Create a summary of current state."""
        summary_parts = []
        
        if state.get("user_request"):
            summary_parts.append(f"User Request: {state['user_request']}")
        
        if state.get("plan"):
            summary_parts.append(f"Plan: {state['plan'][:100]}...")
        
        if state.get("recipe"):
            summary_parts.append(f"Recipe: {state['recipe'].name}")
        
        if state.get("shopping_items"):
            summary_parts.append(f"Items Found: {len(state['shopping_items'])}")
        
        if state.get("total_cost"):
            summary_parts.append(f"Total Cost: ${state['total_cost']:.2f}")
        
        if state.get("budget"):
            summary_parts.append(f"Budget: ${state['budget']:.2f}")
        
        return "; ".join(summary_parts) if summary_parts else "No state information available"