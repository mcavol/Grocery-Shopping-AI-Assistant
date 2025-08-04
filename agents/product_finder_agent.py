"""
Product Finder Agent - Maps ingredients to store products with prices.
NOW WITH REAL WALMART INTEGRATION via SerpAPI + Mistral fallback
"""

from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate
from langsmith import traceable
from .base_agent import BaseAgent
from state import ShoppingItem
import json
import logging
import requests
import os
import time

logger = logging.getLogger(__name__)


class ProductFinderAgent(BaseAgent):
    """Agent responsible for mapping ingredients to store products - REAL WALMART + MISTRAL FALLBACK."""
    
    def __init__(self, llm):
        super().__init__(llm, "product_finder")
        
        # SerpAPI configuration
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.use_real_store = bool(self.serpapi_key)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between SerpAPI calls
        
        # Log SerpAPI status
        if self.use_real_store:
            logger.info("✅ SerpAPI key found - will use real Walmart data")
        else:
            logger.warning("⚠️ No SerpAPI key - using Mistral AI estimates only")
        
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
        """Map ingredients to store products using REAL WALMART + Mistral fallback."""
        try:
            self.log_execution(state, f"Mapping ingredients to products (SerpAPI: {self.use_real_store})")
            
            ingredients = state.get("ingredients", [])
            if not ingredients:
                self.handle_error(state, "No ingredients found to map")
                state["next_agent"] = "error"
                return state
            
            shopping_items = []
            
            if self.use_real_store:
                # Try SerpAPI first for real Walmart data
                try:
                    shopping_items = self._get_walmart_products(ingredients)
                    if shopping_items:
                        self.log_execution(state, f"Found {len(shopping_items)} products via Walmart")
                    else:
                        raise ValueError("No products found via SerpAPI")
                        
                except Exception as e:
                    logger.warning(f"SerpAPI failed: {e}, falling back to Mistral estimates")
                    shopping_items = self._api_based_mapping(ingredients)
            else:
                # Use Mistral API estimates only
                shopping_items = self._api_based_mapping(ingredients)
            
            if not shopping_items:
                raise ValueError("Failed to map any products")
            
            # Update state
            state["shopping_items"] = shopping_items
            state["total_cost"] = sum(item.estimated_price for item in shopping_items)
            state["next_agent"] = "budgeting"
            
            self.mark_completed(state)
            data_source = "Walmart (SerpAPI)" if self.use_real_store and any(hasattr(item, '_from_walmart') for item in shopping_items) else "Mistral AI estimates"
            self.log_execution(state, f"Mapped {len(shopping_items)} products via {data_source}, total: ${state['total_cost']:.2f}")
            
        except Exception as e:
            error_msg = f"Failed to map products: {str(e)}"
            self.handle_error(state, error_msg)
            logger.error(error_msg)
            
            state["next_agent"] = "error"
            state["errors"].append("Product finder failed - check API connections")
            
        return state
    
    def _get_walmart_products(self, ingredients: List[str]) -> List[ShoppingItem]:
        """Get real products from Walmart via SerpAPI."""
        shopping_items = []
        
        for ingredient in ingredients:
            try:
                # Rate limiting
                self._rate_limit_wait()
                
                # Search Walmart for this ingredient
                walmart_results = self._search_walmart_product(ingredient)
                
                if walmart_results:
                    # Convert to ShoppingItem
                    item = self._walmart_result_to_shopping_item(walmart_results[0], ingredient)
                    if item:
                        item._from_walmart = True  # Mark as real data
                        shopping_items.append(item)
                        logger.info(f"✅ Found Walmart product for {ingredient}: {item.name} - ${item.estimated_price}")
                    else:
                        # Fallback to Mistral for this ingredient
                        fallback_item = self._get_mistral_fallback_item(ingredient)
                        if fallback_item:
                            shopping_items.append(fallback_item)
                else:
                    # No Walmart results, use Mistral fallback
                    fallback_item = self._get_mistral_fallback_item(ingredient)
                    if fallback_item:
                        shopping_items.append(fallback_item)
                        
            except Exception as e:
                logger.warning(f"Failed to get Walmart data for {ingredient}: {e}")
                # Fallback to Mistral for this ingredient
                fallback_item = self._get_mistral_fallback_item(ingredient)
                if fallback_item:
                    shopping_items.append(fallback_item)
        
        return shopping_items
    
    def _search_walmart_product(self, query: str) -> Optional[List[Dict]]:
        """Search for a product on Walmart using SerpAPI."""
        try:
            params = {
                "engine": "walmart",
                "query": query,
                "api_key": self.serpapi_key
            }
            
            logger.info(f"🔍 Searching Walmart for: {query}")
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            
            if response.status_code == 200:
                results = response.json()
                
                # Check for API errors
                if "error" in results:
                    logger.error(f"SerpAPI error: {results['error']}")
                    return None
                
                # Extract organic results
                organic_results = results.get("organic_results", [])
                if organic_results:
                    return organic_results[:3]  # Return top 3 results
                else:
                    logger.warning(f"No organic results for {query}")
                    return None
                    
            elif response.status_code == 401:
                logger.error("❌ SerpAPI: Invalid API key")
                self.use_real_store = False  # Disable for remaining calls
                return None
            elif response.status_code == 429:
                logger.warning("⚠️ SerpAPI: Rate limit exceeded")
                time.sleep(2)  # Wait a bit
                return None
            else:
                logger.error(f"SerpAPI HTTP error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning("SerpAPI request timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"SerpAPI request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected SerpAPI error: {e}")
            return None
    
    def _walmart_result_to_shopping_item(self, walmart_item: Dict, original_ingredient: str) -> Optional[ShoppingItem]:
        """Convert Walmart search result to ShoppingItem."""
        try:
            # Extract data from Walmart result
            name = walmart_item.get("title", "").strip()
            if not name:
                return None
            
            # Extract price
            price = 0.0
            if "primary_offer" in walmart_item and "offer_price" in walmart_item["primary_offer"]:
                price_str = str(walmart_item["primary_offer"]["offer_price"])
                # Remove $ and convert to float
                price = float(price_str.replace("$", "").replace(",", ""))
            elif "price" in walmart_item:
                price_str = str(walmart_item["price"])
                price = float(price_str.replace("$", "").replace(",", ""))
            
            # If no price found, skip this item
            if price <= 0:
                logger.warning(f"No valid price found for {name}")
                return None
            
            # Determine category based on ingredient
            category = self._categorize_ingredient(original_ingredient)
            
            # Determine quantity (use size info if available)
            quantity = "1 unit"
            if "variants" in walmart_item and walmart_item["variants"]:
                size_info = walmart_item["variants"][0].get("size", "")
                if size_info:
                    quantity = size_info
            
            return ShoppingItem(
                name=name,
                quantity=quantity,
                estimated_price=price,
                category=category
            )
            
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse Walmart item {walmart_item}: {e}")
            return None
    
    def _get_mistral_fallback_item(self, ingredient: str) -> Optional[ShoppingItem]:
        """Get fallback item from Mistral for a single ingredient."""
        try:
            # Use simplified prompt for single ingredient
            single_prompt = f"""
            Create a grocery store product for this ingredient: {ingredient}
            
            Respond in this EXACT format:
            PRODUCT: [name] | QUANTITY: [package size] | PRICE: [realistic price] | CATEGORY: [store section]
            
            Example:
            PRODUCT: Ground Beef | QUANTITY: 1 lb package | PRICE: 6.99 | CATEGORY: meat
            """
            
            response = self.llm.invoke(single_prompt).strip()
            
            # Parse the response
            if 'PRODUCT:' in response and 'PRICE:' in response:
                parts = response.split('|')
                name = parts[0].replace('PRODUCT:', '').strip()
                quantity = parts[1].replace('QUANTITY:', '').strip()
                price_text = parts[2].replace('PRICE:', '').strip()
                category = parts[3].replace('CATEGORY:', '').strip()
                
                # Extract price number
                price = float(price_text.replace('$', '').strip())
                
                return ShoppingItem(
                    name=name,
                    quantity=quantity,
                    estimated_price=price,
                    category=category
                )
            
        except Exception as e:
            logger.warning(f"Mistral fallback failed for {ingredient}: {e}")
        
        return None
    
    def _categorize_ingredient(self, ingredient: str) -> str:
        """Categorize ingredient for store section."""
        ingredient_lower = ingredient.lower()
        
        if any(word in ingredient_lower for word in ['beef', 'chicken', 'pork', 'fish', 'turkey', 'meat', 'salmon']):
            return 'meat'
        elif any(word in ingredient_lower for word in ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'dairy']):
            return 'dairy'
        elif any(word in ingredient_lower for word in ['apple', 'banana', 'tomato', 'onion', 'lettuce', 'carrot', 'potato']):
            return 'produce'
        elif any(word in ingredient_lower for word in ['bread', 'bagel', 'roll', 'bun', 'bakery']):
            return 'bakery'
        elif any(word in ingredient_lower for word in ['frozen', 'ice cream', 'popsicle']):
            return 'frozen'
        elif any(word in ingredient_lower for word in ['deli', 'sliced', 'ham', 'turkey slice']):
            return 'deli'
        else:
            return 'pantry'
    
    def _rate_limit_wait(self):
        """Rate limiting for SerpAPI calls."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
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
