"""
Base agent class for the grocery shopping system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain.llms.base import BaseLLM
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, llm: BaseLLM, name: str):
        self.llm = llm
        self.name = name
        
    @abstractmethod
    @traceable
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task and return updated state."""
        pass
    
    def log_execution(self, state: Dict[str, Any], action: str):
        """Log agent execution."""
        logger.info(f"Agent {self.name} executing {action}")
        state["messages"].append(f"{self.name}: {action}")
        
    def handle_error(self, state: Dict[str, Any], error: str):
        """Handle and log errors."""
        logger.error(f"Agent {self.name} error: {error}")
        state["errors"].append(f"{self.name}: {error}")
        
    def mark_completed(self, state: Dict[str, Any]):
        """Mark agent as completed."""
        if self.name not in state["completed_agents"]:
            state["completed_agents"].append(self.name)