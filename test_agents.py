"""
Basic tests for the grocery shopping agents.
"""

import unittest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import PlannerAgent, RecipeAgent, ProductFinderAgent, BudgetingAgent, FinalizerAgent
from state import create_initial_state, ShoppingItem, Recipe


class TestAgents(unittest.TestCase):
    """Test cases for grocery shopping agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_llm.invoke = Mock()
        
        # Sample state
        self.test_state = create_initial_state(
            "I want pizza for 4 people under $25",
            budget=25.0,
            people_count=4
        )
    
    def test_planner_agent(self):
        """Test PlannerAgent execution."""
        # Mock LLM response
        self.mock_llm.invoke.return_value = "Plan to make pizza for 4 people within $25 budget"
        
        agent = PlannerAgent(self.mock_llm)
        result_state = agent.execute(self.test_state.copy())
        
        # Assertions
        self.assertIn("planner", result_state["completed_agents"])
        self.assertIsNotNone(result_state["plan"])
        self.assertEqual(result_state["next_agent"], "recipe")
    
    def test_recipe_agent(self):
        """Test RecipeAgent execution."""
        # Mock LLM response with valid JSON
        mock_recipe = """{
            "name": "Simple Pizza",
            "ingredients": ["pizza dough", "tomato sauce", "mozzarella cheese"],
            "servings": 4,
            "instructions": "Bake at 450F for 15 minutes"
        }"""
        self.mock_llm.invoke.return_value = mock_recipe
        
        # Set up state with plan
        test_state = self.test_state.copy()
        test_state["plan"] = "Make pizza for 4 people"
        
        agent = RecipeAgent(self.mock_llm)
        result_state = agent.execute(test_state)
        
        # Assertions
        self.assertIn("recipe", result_state["completed_agents"])
        self.assertIsNotNone(result_state["recipe"])
        self.assertEqual(result_state["recipe"].name, "Simple Pizza")
        self.assertEqual(len(result_state["ingredients"]), 3)
        self.assertEqual(result_state["next_agent"], "product_finder")
    
    def test_recipe_agent_fallback(self):
        """Test RecipeAgent fallback when LLM returns invalid JSON."""
        # Mock LLM response with invalid JSON
        self.mock_llm.invoke.return_value = "This is not valid JSON for pizza recipe"
        
        test_state = self.test_state.copy()
        test_state["plan"] = "Make pizza for 4 people"
        
        agent = RecipeAgent(self.mock_llm)
        result_state = agent.execute(test_state)
        
        # Should use fallback recipe
        self.assertIn("recipe", result_state["completed_agents"])
        self.assertIsNotNone(result_state["recipe"])
        self.assertIn("pizza", result_state["recipe"].name.lower())
    
    def test_product_finder_agent(self):
        """Test ProductFinderAgent execution."""
        # Mock LLM response with product mapping
        mock_products = """[
            {"name": "Pizza Dough", "quantity": "2 packages", "estimated_price": 5.98, "category": "bakery"},
            {"name": "Tomato Sauce", "quantity": "1 jar", "estimated_price": 1.99, "category": "pantry"}
        ]"""
        self.mock_llm.invoke.return_value = mock_products
        
        # Set up state with ingredients
        test_state = self.test_state.copy()
        test_state["ingredients"] = ["pizza dough", "tomato sauce"]
        
        agent = ProductFinderAgent(self.mock_llm)
        result_state = agent.execute(test_state)
        
        # Assertions
        self.assertIn("product_finder", result_state["completed_agents"])
        self.assertEqual(len(result_state["shopping_items"]), 2)
        self.assertGreater(result_state["total_cost"], 0)
        self.assertEqual(result_state["next_agent"], "budgeting")
    
    def test_product_finder_rule_based_fallback(self):
        """Test ProductFinderAgent rule-based fallback."""
        # Mock LLM to return invalid JSON
        self.mock_llm.invoke.return_value = "Invalid JSON response"
        
        # Set up state with known ingredients
        test_state = self.test_state.copy()
        test_state["ingredients"] = ["pizza dough", "mozzarella cheese", "unknown ingredient"]
        
        agent = ProductFinderAgent(self.mock_llm)
        result_state = agent.execute(test_state)
        
        # Should use rule-based mapping
        self.assertIn("product_finder", result_state["completed_agents"])
        self.assertEqual(len(result_state["shopping_items"]), 3)
        self.assertGreater(result_state["total_cost"], 0)
    
    def test_budgeting_agent_within_budget(self):
        """Test BudgetingAgent when within budget."""
        self.mock_llm.invoke.return_value = "Shopping list is within budget with $5 remaining"
        
        # Set up state with items under budget
        test_state = self.test_state.copy()
        test_state["shopping_items"] = [
            ShoppingItem(name="Item 1", quantity="1 unit", estimated_price=10.0, category="test"),
            ShoppingItem(name="Item 2", quantity="1 unit", estimated_price=8.0, category="test")
        ]
        test_state["total_cost"] = 18.0
        test_state["budget"] = 25.0
        
        agent = BudgetingAgent(self.mock_llm)
        result_state = agent.execute(test_state)
        
        # Assertions
        self.assertIn("budgeting", result_state["completed_agents"])
        self.assertEqual(result_state["next_agent"], "finalizer")
        self.assertEqual(result_state["total_cost"], 18.0)  # Should remain unchanged
    
    def test_budgeting_agent_over_budget(self):
        """Test BudgetingAgent when over budget."""
        self.mock_llm.invoke.return_value = "Optimized shopping list to fit budget"
        
        # Set up state with items over budget
        test_state = self.test_state.copy()
        test_state["shopping_items"] = [
            ShoppingItem(name="Expensive Item", quantity="1 unit", estimated_price=20.0, category="test"),
            ShoppingItem(name="Another Item", quantity="1 unit", estimated_price=10.0, category="test")
        ]
        test_state["total_cost"] = 30.0
        test_state["budget"] = 25.0
        
        agent = BudgetingAgent(self.mock_llm)
        result_state = agent.execute(test_state)
        
        # Assertions
        self.assertIn("budgeting", result_state["completed_agents"])
        self.assertLessEqual(result_state["total_cost"], 25.0)  # Should be optimized
        self.assertLess(len(result_state["shopping_items"]), 2)  # Some items removed
    
    def test_budgeting_agent_no_budget(self):
        """Test BudgetingAgent when no budget is specified."""
        # Set up state without budget
        test_state = self.test_state.copy()
        test_state["budget"] = None
        test_state["shopping_items"] = [
            ShoppingItem(name="Item 1", quantity="1 unit", estimated_price=10.0, category="test")
        ]
        test_state["total_cost"] = 10.0
        
        agent = BudgetingAgent(self.mock_llm)
        result_state = agent.execute(test_state)
        
        # Should proceed without optimization
        self.assertIn("budgeting", result_state["completed_agents"])
        self.assertEqual(result_state["next_agent"], "finalizer")
        self.assertEqual(result_state["total_cost"], 10.0)
    
    def test_finalizer_agent(self):
        """Test FinalizerAgent execution."""
        mock_final_list = """SHOPPING LIST
========================
Recipe: Test Pizza
Total: $20.00
Budget: $25.00 ✓

ITEMS:
- Pizza Dough: $5.99
- Sauce: $1.99
- Cheese: $4.99

TOTAL: $12.97"""
        
        self.mock_llm.invoke.return_value = mock_final_list
        
        # Set up complete state
        test_state = self.test_state.copy()
        test_state["recipe"] = Recipe(
            name="Test Pizza",
            ingredients=["dough", "sauce", "cheese"],
            servings=4
        )
        test_state["shopping_items"] = [
            ShoppingItem(name="Pizza Dough", quantity="1 package", estimated_price=5.99, category="bakery"),
            ShoppingItem(name="Sauce", quantity="1 jar", estimated_price=1.99, category="pantry"),
            ShoppingItem(name="Cheese", quantity="1 package", estimated_price=4.99, category="dairy")
        ]
        test_state["total_cost"] = 12.97
        
        agent = FinalizerAgent(self.mock_llm)
        result_state = agent.execute(test_state)
        
        # Assertions
        self.assertIn("finalizer", result_state["completed_agents"])
        self.assertIsNotNone(result_state["final_list"])
        self.assertEqual(result_state["next_agent"], "complete")
    
    def test_finalizer_agent_fallback(self):
        """Test FinalizerAgent fallback when LLM fails."""
        # Mock LLM to raise exception
        self.mock_llm.invoke.side_effect = Exception("LLM error")
        
        # Set up state with shopping items
        test_state = self.test_state.copy()
        test_state["shopping_items"] = [
            ShoppingItem(name="Test Item", quantity="1 unit", estimated_price=5.99, category="test")
        ]
        test_state["total_cost"] = 5.99
        
        agent = FinalizerAgent(self.mock_llm)
        result_state = agent.execute(test_state)
        
        # Should create fallback list
        self.assertIn("finalizer", result_state["completed_agents"])
        self.assertIsNotNone(result_state["final_list"])
        self.assertIn("SHOPPING LIST", result_state["final_list"])
        self.assertIn("Test Item", result_state["final_list"])


class TestStateManagement(unittest.TestCase):
    """Test state management functionality."""
    
    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state(
            "Test request",
            budget=50.0,
            people_count=6
        )
        
        self.assertEqual(state["user_request"], "Test request")
        self.assertEqual(state["budget"], 50.0)
        self.assertEqual(state["people_count"], 6)
        self.assertEqual(state["next_agent"], "planner")
        self.assertEqual(len(state["completed_agents"]), 0)
        self.assertEqual(len(state["messages"]), 0)
        self.assertEqual(len(state["errors"]), 0)
    
    def test_shopping_item_model(self):
        """Test ShoppingItem model."""
        item = ShoppingItem(
            name="Test Item",
            quantity="2 lbs",
            estimated_price=4.99,
            category="produce"
        )
        
        self.assertEqual(item.name, "Test Item")
        self.assertEqual(item.quantity, "2 lbs")
        self.assertEqual(item.estimated_price, 4.99)
        self.assertEqual(item.category, "produce")
    
    def test_recipe_model(self):
        """Test Recipe model."""
        recipe = Recipe(
            name="Test Recipe",
            ingredients=["ingredient1", "ingredient2"],
            servings=4,
            instructions="Test instructions"
        )
        
        self.assertEqual(recipe.name, "Test Recipe")
        self.assertEqual(len(recipe.ingredients), 2)
        self.assertEqual(recipe.servings, 4)
        self.assertEqual(recipe.instructions, "Test instructions")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)