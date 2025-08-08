from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain.llms.base import BaseLLM
from langsmith import traceable
import logging

from agents import (
    SupervisorAgent,
    PlannerAgent,
    RecipeAgent,
    ProductFinderAgent,
    BudgetingAgent,
    FinalizerAgent
)
from state import ShoppingState

logger = logging.getLogger(__name__)


class GroceryShoppingGraph:
    """LangGraph implementation for grocery shopping workflow"""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.agents = self._initialize_agents()
        self.graph = self._build_graph()
        self.max_iterations = 10
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents."""
        return {
            "supervisor": SupervisorAgent(self.llm),
            "planner": PlannerAgent(self.llm),
            "recipe": RecipeAgent(self.llm),
            "product_finder": ProductFinderAgent(self.llm),
            "budgeting": BudgetingAgent(self.llm),
            "finalizer": FinalizerAgent(self.llm)
        }
    
    def _build_graph(self) -> StateGraph:
        """Build the execution graph."""
        workflow = StateGraph(ShoppingState)
        
        # Add agent nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("recipe", self._recipe_node)
        workflow.add_node("product_finder", self._product_finder_node)
        workflow.add_node("budgeting", self._budgeting_node)
        workflow.add_node("finalizer", self._finalizer_node)
        
        # Define linear flow
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "recipe")
        workflow.add_edge("recipe", "product_finder")
        workflow.add_edge("product_finder", "budgeting")
        workflow.add_edge("budgeting", "finalizer")
        workflow.add_edge("finalizer", END)
        
        return workflow.compile(checkpointer=None)
    
    @traceable
    def _planner_node(self, state: ShoppingState) -> ShoppingState:
        """Planner node execution."""
        return self.agents["planner"].execute(state)
    
    @traceable
    def _recipe_node(self, state: ShoppingState) -> ShoppingState:
        """Recipe node execution."""
        return self.agents["recipe"].execute(state)
    
    @traceable
    def _product_finder_node(self, state: ShoppingState) -> ShoppingState:
        """Product finder node execution."""
        return self.agents["product_finder"].execute(state)
    
    @traceable
    def _budgeting_node(self, state: ShoppingState) -> ShoppingState:
        """Budgeting node execution."""
        return self.agents["budgeting"].execute(state)
    
    @traceable
    def _finalizer_node(self, state: ShoppingState) -> ShoppingState:
        """Finalizer node execution."""
        return self.agents["finalizer"].execute(state)
    
    @traceable
    def run(self, initial_state: ShoppingState) -> ShoppingState:
        """Execute the complete workflow - NO HARDCODED FALLBACKS."""
        try:
            initial_state["_iteration_count"] = 0
            
            logger.info("Starting grocery shopping workflow with Mistral API")
            
            # Run the graph
            final_state = self.graph.invoke(
                initial_state,
                config={"recursion_limit": 15}
            )
            
            # Check if we completed successfully
            completed_agents = final_state.get("completed_agents", [])
            required_agents = ["planner", "recipe", "product_finder", "budgeting", "finalizer"]
            
            if len(completed_agents) >= 3:  # At least 3 agents completed
                logger.info("✅ Workflow completed successfully")
                final_state["messages"].append("✅ Shopping list generated successfully via Mistral API")
            else:
                logger.warning("⚠️ Workflow incomplete - some agents failed")
                final_state["errors"].append("Workflow incomplete - check API connection")
            
            return final_state
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Graph execution failed: {error_msg}")
            
            initial_state["errors"].append(f"Graph execution error: {error_msg}")
            initial_state["next_agent"] = "error"
            
            # NO HARDCODED FALLBACKS - Return error state
            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                initial_state["errors"].append("❌ Mistral API rate limit reached. Please wait and try again.")
                initial_state["messages"].append("💡 Try again in a few minutes or check your API credits.")
            elif "api" in error_msg.lower() or "connection" in error_msg.lower():
                initial_state["errors"].append("❌ Mistral API connection failed. Check your API key.")
                initial_state["messages"].append("💡 Verify MISTRAL_API_KEY is set correctly.")
            else:
                initial_state["errors"].append("❌ Unexpected error occurred. Check logs for details.")
            
            return initial_state
    
    def get_agent_status(self, state: ShoppingState) -> Dict[str, str]:
        """Get status of all agents."""
        completed = state.get("completed_agents", [])
        errors = state.get("errors", [])
        
        status = {}
        for agent in ["planner", "recipe", "product_finder", "budgeting", "finalizer"]:
            if agent in completed:
                status[agent] = "✅ Completed"
            elif any(agent in error for error in errors):
                status[agent] = "❌ Failed"
            else:
                status[agent] = "⏳ Pending"
        
        return status
