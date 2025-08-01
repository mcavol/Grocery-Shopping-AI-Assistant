"""
Main Streamlit application for the Grocery Shopping AI Assistant.
NO HARDCODING - Requires proper Mistral API connection
"""

import streamlit as st
import os
import logging
from typing import Optional
import re
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangSmith configuration
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "grocery-shopping-assistant")

# These imports are now safe after env vars are set
from graph import GroceryShoppingGraph
from state import create_initial_state
from llm_config import create_llm, validate_api_key


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_state' not in st.session_state:
        st.session_state.current_state = None
    if 'api_validated' not in st.session_state:
        st.session_state.api_validated = False
    # FIX: Initialize the user input key used by the text_area
    if 'user_input_area' not in st.session_state:
        st.session_state.user_input_area = ""


def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.header("🛒 Shopping Assistant Config")
    
    env_api_key = os.getenv("MISTRAL_API_KEY")
    
    api_key = st.sidebar.text_input(
        "Mistral API Key (Required)",
        type="password",
        value=env_api_key if env_api_key else "",
        help="Enter your Mistral API key. Get one from https://console.mistral.ai/",
        placeholder="Enter your Mistral API key..."
    )
    
    if api_key:
        if len(api_key) < 20:
            st.sidebar.error("❌ API key appears to be invalid (too short)")
            return None
        
        if not st.session_state.get('api_validated') or st.session_state.get('last_api_key') != api_key:
            with st.spinner("Validating API key..."):
                if validate_api_key(api_key):
                    st.session_state.api_validated = True
                    st.session_state.last_api_key = api_key
                    st.sidebar.success("✅ API key validated")
                else:
                    st.sidebar.error("❌ API key validation failed")
                    return None
        else:
            st.sidebar.success("✅ API key validated")
        
        os.environ["MISTRAL_API_KEY"] = api_key
        st.sidebar.info("💡 Free tier has rate limits. The app includes automatic rate limiting (1.2s between calls).")
        
    elif env_api_key:
        api_key = env_api_key
        st.sidebar.success("✅ API key loaded from environment")
    else:
        st.sidebar.error("❌ Mistral API key is required")
        st.sidebar.info("💡 Get your free API key from: https://console.mistral.ai/")
        return None
    
    st.sidebar.subheader("🔍 LangSmith Tracing")
    langsmith_key = st.sidebar.text_input(
        "LangSmith API Key (Optional)",
        type="password",
        value=os.getenv("LANGCHAIN_API_KEY", ""),
        help="Optional: Enter LangSmith API key for tracing",
        placeholder="Enter LangSmith key..."
    )
    
    if langsmith_key:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        st.sidebar.success("✅ LangSmith tracing enabled")
    
    return api_key


def parse_user_input(user_input: str) -> tuple:
    """
    FIX: More robustly parse user input to extract budget and people count.
    """
    budget = None
    people_count = 4  # default

    # --- FIX: Robust Budget Extraction ---
    # Patterns to look for: $25, 25 dollars, budget of 25.50, under 30 eur, etc.
    budget_patterns = [
        r'[\$€£](\d+(?:\.\d{1,2})?)',  # e.g., $25.50
        r'(\d+(?:\.\d{1,2})?)\s*(?:dollars|dollar|eur|euros|gbp|pounds)',  # e.g., 25 dollars
        r'(?:budget|under|limit of)\s*[\$€£]?\s*(\d+(?:\.\d{1,2})?)' # e.g., budget of 25
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            # The capturing group is the first one that's not None
            budget_str = next((g for g in match.groups() if g is not None), None)
            if budget_str:
                budget = float(budget_str)
                break  # Stop after first match

    # People count extraction (remains the same, was already quite good)
    people_patterns = [
        r'(\d+)\s+people', r'for\s+(\d+)', r'serves?\s+(\d+)', r'(\d+)\s+person'
    ]
    for pattern in people_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            people_count = int(match.group(1))
            break
            
    return budget, people_count


def display_agent_status(state):
    """Display the status of all agents."""
    if not state:
        return
    st.subheader("🤖 Agent Status")
    agents = ["planner", "recipe", "product_finder", "budgeting", "finalizer"]
    completed = state.get("completed_agents", [])
    errors = state.get("errors", [])
    cols = st.columns(len(agents))
    for i, agent in enumerate(agents):
        with cols[i]:
            if agent in completed:
                status, help_text = "✅", "Completed successfully"
            elif any(agent in error.lower() for error in errors):
                status, help_text = "❌", "Failed - check errors below"
            else:
                status, help_text = "⏳", "Pending execution"
            st.metric(label=agent.replace("_", " ").title(), value=status, help=help_text)


def display_shopping_results(state):
    """Display the shopping results."""
    if not state:
        return
    recipe = state.get("recipe")
    if recipe:
        st.subheader("🍽️ Recipe")
        st.write(f"**{recipe.name}** (Serves {recipe.servings})")
        if recipe.instructions:
            with st.expander("📋 Cooking Instructions"):
                st.write(recipe.instructions)
    shopping_items = state.get("shopping_items", [])
    total_cost = state.get("total_cost", 0)
    budget = state.get("budget")
    if shopping_items:
        st.subheader("🛍️ Shopping List")
        categories = {}
        for item in shopping_items:
            categories.setdefault(item.category, []).append(item)
        for category, items in categories.items():
            st.write(f"**{category.upper()}:**")
            for item in items:
                st.write(f"  • {item.name} ({item.quantity}) - ${item.estimated_price:.2f}")
            st.write("")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cost", f"${total_cost:.2f}")
        with col2:
            if budget:
                st.metric("Budget", f"${budget:.2f}")
        with col3:
            if budget:
                remaining = budget - total_cost
                st.metric("Remaining", f"${remaining:.2f}", delta=remaining, delta_color="normal" if remaining >= 0 else "inverse")
    final_list = state.get("final_list")
    if final_list:
        st.subheader("📋 Final Shopping List")
        st.text_area("Copy this list for shopping:", value=final_list, height=300)


def display_messages_and_errors(state):
    """Display system messages and errors."""
    if not state:
        return
    messages, errors = state.get("messages", []), state.get("errors", [])
    if errors:
        st.subheader("⚠️ Issues")
        for error in errors:
            if "rate limit" in error.lower():
                st.warning(f"🐌 {error}")
            elif "api" in error.lower():
                st.error(f"🔌 {error}")
            else:
                st.error(f"❌ {error}")
    if messages:
        with st.expander("🔍 System Messages", expanded=False):
            for msg in messages:
                st.info(msg)


def create_graph_with_api_key(api_key: str):
    """Create graph with proper API key validation."""
    try:
        os.environ["MISTRAL_API_KEY"] = api_key
        llm = create_llm(api_key)
        test_response = llm.invoke("Respond with only 'System ready' if you can read this.")
        if "ready" not in str(test_response).lower():
            raise ValueError("LLM test failed")
        logger.info(f"✅ LLM test successful.")
        graph = GroceryShoppingGraph(llm)
        return graph
    except Exception as e:
        logger.error(f"Failed to create graph: {str(e)}")
        raise e


def clear_results_callback():
    """
    This function is called when the 'Clear Results' button is clicked.
    It resets the relevant session state variables.
    """
    st.session_state.current_state = None
    st.session_state.user_input_area = ""


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Grocery Shopping AI Assistant", page_icon="🛒", layout="wide")
    
    initialize_session_state()
    
    st.title("🛒 Grocery Shopping AI Assistant")
    st.markdown("*Multi-Agent System powered by LangGraph & Mistral AI*")
    st.markdown("**⚠️ Note: This app requires a valid Mistral API key.**")
    
    api_key = setup_sidebar()
    
    if not api_key:
        st.error("❌ **Mistral API Key Required**")
        st.info("Please enter a valid Mistral API key in the sidebar to begin.")
        return
    
    try:
        if not st.session_state.graph or st.session_state.get('graph_api_key') != api_key:
            with st.spinner("🚀 Initializing system with Mistral API..."):
                st.session_state.graph = create_graph_with_api_key(api_key)
                st.session_state.graph_api_key = api_key
            st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        return
    
    st.subheader("💬 What would you like to shop for?")
    
    # --- FIX: Simplified Demo Button Logic ---
    if st.button("🍽️ Dinner for 4 people under $25"):
        # Set the text_area's session state key directly and rerun
        st.session_state.user_input_area = "Prepare a shopping list for dinner for 4 people within a $25 budget"
        st.rerun()

    user_input = st.text_area(
        "Describe what you want to shop for:",
        placeholder="e.g., I want to buy ingredients for chicken curry for 5 people under 30 dollars",
        height=100,
        key="user_input_area"  # This key links the widget to st.session_state.user_input_area
    )
    
    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("🔄 Generate Shopping List", type="primary", disabled=not user_input.strip())
    with col2:
        st.button("🗑️ Clear Results", on_click=clear_results_callback)
    
    if process_button and user_input.strip():
        budget, people_count = parse_user_input(user_input)
        
        st.info(f"📊 Processing: **{people_count} people**" + (f", with a budget of **${budget:.2f}**" if budget else ", with **no budget limit**"))
        
        with st.spinner("🤖 Agents are working with Mistral AI... (This may take 30-60 seconds)"):
            try:
                initial_state = create_initial_state(user_input.strip(), budget, people_count)
                final_state = st.session_state.graph.run(initial_state)
                st.session_state.current_state = final_state
                
                errors = final_state.get("errors", [])
                completed = final_state.get("completed_agents", [])
                
                if len(completed) >= 3:
                    st.success("✅ Shopping list generated successfully!")
                elif errors:
                    st.warning("⚠️ Some issues occurred, but we have partial results.")
                else:
                    st.error("❌ Failed to generate shopping list.")
            except Exception as e:
                st.error(f"❌ Error processing request: {str(e)}")
                if "rate limit" in str(e).lower():
                    st.info("🐌 Rate limit hit. Please wait a minute and try again.")
                elif "api key" in str(e).lower():
                    st.info("🔑 Please verify your Mistral API key is correct.")
    
    if st.session_state.current_state:
        st.markdown("---")
        display_agent_status(st.session_state.current_state)
        display_shopping_results(st.session_state.current_state)
        display_messages_and_errors(st.session_state.current_state)
    
    st.markdown("---")
    st.markdown("*Built with LangChain, LangGraph, Streamlit, and Mistral AI*")


if __name__ == "__main__":
    main()