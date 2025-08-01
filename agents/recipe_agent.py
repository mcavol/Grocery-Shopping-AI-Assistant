"""
Recipe Agent - Finds suitable recipes and extracts ingredients.
NO HARDCODED RECIPES - Only uses Mistral API
"""

from typing import Dict, Any, List, Union
from langchain.prompts import PromptTemplate
from langsmith import traceable
from .base_agent import BaseAgent
from state import Recipe
import json
import logging

logger = logging.getLogger(__name__)


class RecipeAgent(BaseAgent):
    """Agent responsible for finding recipes and extracting ingredients - NO HARDCODING."""
    
    def __init__(self, llm):
        super().__init__(llm, "recipe")
        self.prompt_template = PromptTemplate(
            input_variables=["user_request", "people_count", "plan"],
            template="""
            You are a recipe expert. Based on the user's request and plan, suggest a suitable recipe.
            
            User Request: {user_request}
            People Count: {people_count}
            Plan: {plan}
            
            Provide a recipe in VALID JSON format with:
            {{
                "name": "Recipe Name",
                "ingredients": ["2 cups all-purpose flour", "1 lb ground beef", "2 tbsp olive oil"],
                "servings": {people_count},
                "instructions": "Step by step cooking instructions as a single string"
            }}
            
            IMPORTANT RULES:
            1. Make ingredients SPECIFIC with quantities (e.g., "2 cups flour" not just "flour")
            2. Base recipe on the user's actual request
            3. Scale ingredients for {people_count} people
            4. Return ONLY valid JSON, no extra text
            5. If user mentions budget, suggest affordable ingredients
            6. Instructions must be a SINGLE STRING, not an array
            
            JSON Response:
            """
        )
    
    @traceable
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Find recipe and extract ingredients using ONLY Mistral API."""
        try:
            self.log_execution(state, "Finding suitable recipe via Mistral API")
            
            # Generate recipe using LLM - NO FALLBACKS
            prompt = self.prompt_template.format(
                user_request=state["user_request"],
                people_count=state["people_count"],
                plan=state["plan"] or "General meal planning"
            )
            
            # Call Mistral API
            recipe_response = self.llm.invoke(prompt).strip()
            logger.info(f"Mistral API response: {recipe_response[:200]}...")
            
            # Parse JSON response
            try:
                # Clean response if it has extra text
                if "```json" in recipe_response:
                    recipe_response = recipe_response.split("```json")[1].split("```")[0].strip()
                elif "```" in recipe_response:
                    json_start = recipe_response.find('{')
                    json_end = recipe_response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        recipe_response = recipe_response[json_start:json_end]
                
                recipe_data = json.loads(recipe_response)
                
                # Fix instructions if it's a list
                if isinstance(recipe_data.get('instructions'), list):
                    recipe_data['instructions'] = '\n'.join(recipe_data['instructions'])
                
                # Ensure instructions is a string
                if 'instructions' in recipe_data and recipe_data['instructions'] is None:
                    recipe_data['instructions'] = "No specific instructions provided."
                
                recipe = Recipe(**recipe_data)
                
                self.log_execution(state, f"Successfully parsed recipe: {recipe.name}")
                
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                # If JSON parsing fails, make another API call for simpler format
                logger.warning(f"JSON parsing failed: {e}, making simpler API call")
                simple_recipe = self._get_simple_recipe_format(state)
                recipe = simple_recipe
            
            # Validate recipe has required fields
            if not recipe.name or not recipe.ingredients:
                raise ValueError("Recipe missing required fields")
            
            # Update state
            state["recipe"] = recipe
            state["ingredients"] = recipe.ingredients
            state["next_agent"] = "product_finder"
            
            self.mark_completed(state)
            self.log_execution(state, f"Recipe found: {recipe.name} with {len(recipe.ingredients)} ingredients")
            
        except Exception as e:
            # NO FALLBACK - Fail properly and let user know API is required
            error_msg = f"Failed to find recipe via Mistral API: {str(e)}"
            self.handle_error(state, error_msg)
            logger.error(error_msg)
            
            # Set error state
            state["next_agent"] = "error"
            state["errors"].append("Recipe agent requires working Mistral API connection")
            
        return state
    
    def _get_simple_recipe_format(self, state: Dict[str, Any]) -> Recipe:
        """Make a simpler API call if JSON parsing fails."""
        simple_prompt = f"""
        Create a simple recipe for: {state['user_request']}
        For {state['people_count']} people.
        
        Respond in this EXACT format:
        RECIPE_NAME: [name here]
        INGREDIENTS: [ingredient 1 with quantity], [ingredient 2 with quantity], [ingredient 3 with quantity]
        INSTRUCTIONS: [brief instructions as single line]
        
        Example format:
        RECIPE_NAME: Spaghetti Carbonara
        INGREDIENTS: 1 lb spaghetti pasta, 4 large eggs, 1 cup grated parmesan cheese, 8 oz pancetta
        INSTRUCTIONS: Cook pasta, fry pancetta, mix eggs and cheese, combine everything while hot.
        """
        
        try:
            response = self.llm.invoke(simple_prompt).strip()
            
            # Parse the response
            lines = response.split('\n')
            recipe_name = ""
            ingredients = []
            instructions = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("RECIPE_NAME:"):
                    recipe_name = line.replace("RECIPE_NAME:", "").strip()
                elif line.startswith("INGREDIENTS:"):
                    ingredients_text = line.replace("INGREDIENTS:", "").strip()
                    ingredients = [ing.strip() for ing in ingredients_text.split(',')]
                elif line.startswith("INSTRUCTIONS:"):
                    instructions = line.replace("INSTRUCTIONS:", "").strip()
            
            if not recipe_name or not ingredients:
                raise ValueError("Failed to parse simple recipe format")
            
            # Ensure instructions is a string
            if not instructions:
                instructions = "No specific instructions provided."
            
            return Recipe(
                name=recipe_name,
                ingredients=ingredients,
                servings=state["people_count"],
                instructions=instructions
            )
            
        except Exception as e:
            logger.error(f"Simple recipe format also failed: {e}")
            raise ValueError("Mistral API connection required for recipe generation")