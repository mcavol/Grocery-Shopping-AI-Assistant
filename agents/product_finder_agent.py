"""
Product Finder Agent - Maps ingredients to store products with prices.
"""

from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langsmith import traceable
from .base_agent import BaseAgent
from state import ShoppingItem
import json
import logging

logger = logging.getLogger(__name__)


class ProductFinderAgent(BaseAgent):
    """Agent responsible for mapping ingredients to store products - NO HARDCODING."""
    
    def __init__(self, llm):
        super().__init__(llm, "product_finder")
        
        self.prompt_template = PromptTemplate(
            input_variables=["ingredients"],
            template="""
            You are a grocery store product expert. Convert these recipe ingredients into specific store products with realistic quantities and current prices.
            
            Ingredients: {ingredients}
            
            For each ingredient, create a JSON array with realistic grocery store products:
            [
                {{
                    "name": "Specific product name as sold in store",
                    "quantity": "Realistic store package size (e.g., '1 lb package', '16 oz container', '2 liter bottle')",
                    "estimated_price": 4.99,
                    "category": "produce/dairy/meat/pantry/frozen/bakery/deli"
                }}
            ]
            
            IMPORTANT RULES:
            1. Use realistic current grocery store prices (2024-2025)
            2. Use actual package sizes stores sell (don't make up weird sizes)
            3. Choose appropriate store categories
            4. If ingredient needs multiple products, include all
            5. Return ONLY valid JSON array, no extra text
            
            JSON Response:
            """
        )
    
    @traceable
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Map ingredients to store products using ONLY Mistral API."""
        try:
            self.log_execution(state, "Mapping ingredients to store products via Mistral API")
            
            ingredients = state.get("ingredients", [])
            if not ingredients:
                self.handle_error(state, "No ingredients found to map")
                state["next_agent"] = "error"
                return state
            
            # Use Mistral API to map products - NO FALLBACKS
            shopping_items = self._api_based_mapping(ingredients)
            
            if not shopping_items:
                raise ValueError("Failed to map any products via Mistral API")
            
            # Update state
            state["shopping_items"] = shopping_items
            state["total_cost"] = sum(item.estimated_price for item in shopping_items)
            state["next_agent"] = "budgeting"
            
            self.mark_completed(state)
            self.log_execution(state, f"Mapped {len(shopping_items)} products, total: ${state['total_cost']:.2f}")
            
        except Exception as e:
            error_msg = f"Failed to map products via Mistral API: {str(e)}"
            self.handle_error(state, error_msg)
            logger.error(error_msg)
            
            # Set error state - NO FALLBACK
            state["next_agent"] = "error"
            state["errors"].append("Product finder requires working Mistral API connection")
            
        return state
    
    def _api_based_mapping(self, ingredients: List[str]) -> List[ShoppingItem]:
        """Use ONLY Mistral API to map ingredients to products."""
        try:
            # Create prompt with all ingredients
            ingredients_text = ", ".join(ingredients)
            prompt = self.prompt_template.format(ingredients=ingredients_text)
            
            # Call Mistral API
            response = self.llm.invoke(prompt).strip()
            logger.info(f"Mistral API response: {response[:200]}...")
            
            # Clean response if it has extra text
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    response = response[json_start:json_end]
            
            # Parse JSON response
            products_data = json.loads(response)
            
            if not isinstance(products_data, list):
                raise ValueError("API response is not a list")
            
            shopping_items = []
            for item_data in products_data:
                try:
                    shopping_item = ShoppingItem(**item_data)
                    shopping_items.append(shopping_item)
                except Exception as e:
                    logger.warning(f"Failed to create shopping item from {item_data}: {e}")
                    continue
            
            if not shopping_items:
                raise ValueError("No valid shopping items created from API response")
            
            return shopping_items
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            # Try simpler format
            return self._try_simple_format(ingredients)
        except Exception as e:
            logger.error(f"API mapping failed: {e}")
            raise ValueError(f"Mistral API required for product mapping: {str(e)}")
    
    def _try_simple_format(self, ingredients: List[str]) -> List[ShoppingItem]:
        """Try a simpler format if JSON parsing fails."""
        simple_prompt = f"""
        List grocery store products for these ingredients: {', '.join(ingredients)}
        
        Format each product as:
        PRODUCT: [name] | QUANTITY: [package size] | PRICE: [realistic price] | CATEGORY: [store section]
        
        Example:
        PRODUCT: Ground Beef | QUANTITY: 1 lb package | PRICE: 6.99 | CATEGORY: meat
        PRODUCT: Whole Milk | QUANTITY: 1 gallon | PRICE: 3.79 | CATEGORY: dairy
        """
        
        try:
            response = self.llm.invoke(simple_prompt).strip()
            lines = response.split('\n')
            
            shopping_items = []
            for line in lines:
                if 'PRODUCT:' in line and 'PRICE:' in line:
                    try:
                        parts = line.split('|')
                        name = parts[0].replace('PRODUCT:', '').strip()
                        quantity = parts[1].replace('QUANTITY:', '').strip()
                        price_text = parts[2].replace('PRICE:', '').strip()
                        category = parts[3].replace('CATEGORY:', '').strip()
                        
                        # Extract price number
                        price = float(price_text.replace('$', '').strip())
                        
                        shopping_item = ShoppingItem(
                            name=name,
                            quantity=quantity,
                            estimated_price=price,
                            category=category
                        )
                        shopping_items.append(shopping_item)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse line: {line}, error: {e}")
                        continue
            
            if not shopping_items:
                raise ValueError("No products parsed from simple format")
                
            return shopping_items
            
        except Exception as e:
            raise ValueError(f"Both JSON and simple format failed: {str(e)}")
