"""
Main Streamlit application for the Grocery Shopping AI Assistant.
NOW WITH WORKING SPEECH RECOGNITION + REAL WALMART INTEGRATION
FIXED AUDIO RECORDING IMPLEMENTATION WITH AUTOMATIC TRANSCRIPTION
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
from speech_utils import (
    get_whisper_model_info, 
    create_best_voice_input,
    check_audio_packages,
    show_audio_setup_instructions,
    install_audio_packages_button
)


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
    if 'user_input_area' not in st.session_state:
        st.session_state.user_input_area = ""
    if 'speech_enabled' not in st.session_state:
        st.session_state.speech_enabled = True
    if 'whisper_model' not in st.session_state:
        st.session_state.whisper_model = "base"
    if 'audio_packages_checked' not in st.session_state:
        st.session_state.audio_packages_checked = False
    if 'voice_input_result' not in st.session_state:
        st.session_state.voice_input_result = None


def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.header("🛒 Shopping Assistant Config")
    
    # MISTRAL API KEY SECTION
    st.sidebar.subheader("🤖 Mistral AI (Required)")
    env_api_key = os.getenv("MISTRAL_API_KEY")
    
    api_key = st.sidebar.text_input(
        "Mistral API Key",
        type="password",
        value=env_api_key if env_api_key else "",
        help="Enter your Mistral API key. Get one from https://console.mistral.ai/",
        placeholder="Enter your Mistral API key..."
    )
    
    if api_key:
        if len(api_key) < 20:
            st.sidebar.error("❌ API key appears to be invalid (too short)")
            return None, None
        
        if not st.session_state.get('api_validated') or st.session_state.get('last_api_key') != api_key:
            with st.spinner("Validating Mistral API key..."):
                if validate_api_key(api_key):
                    st.session_state.api_validated = True
                    st.session_state.last_api_key = api_key
                    st.sidebar.success("✅ Mistral API key validated")
                else:
                    st.sidebar.error("❌ Mistral API key validation failed")
                    return None, None
        else:
            st.sidebar.success("✅ Mistral API key validated")
        
        os.environ["MISTRAL_API_KEY"] = api_key
        
    elif env_api_key:
        api_key = env_api_key
        st.sidebar.success("✅ Mistral API key loaded from environment")
    else:
        st.sidebar.error("❌ Mistral API key is required")
        st.sidebar.info("💡 Get your free API key from: https://console.mistral.ai/")
        return None, None
    
    # SERPAPI SECTION
    st.sidebar.subheader("🏪 Real Store Data (Optional)")
    env_serpapi_key = os.getenv("SERPAPI_KEY")
    
    serpapi_key = st.sidebar.text_input(
        "SerpAPI Key",
        type="password",
        value=env_serpapi_key if env_serpapi_key else "",
        help="Optional: Get real Walmart prices. Get your key from https://serpapi.com/",
        placeholder="Enter your SerpAPI key for real prices..."
    )
    
    if serpapi_key and serpapi_key != "your_serpapi_key_here":
        os.environ["SERPAPI_KEY"] = serpapi_key
        st.sidebar.success("✅ SerpAPI key configured - Real Walmart data enabled!")
        st.sidebar.info("🛍️ The app will fetch actual prices from Walmart")
    elif env_serpapi_key and env_serpapi_key != "your_serpapi_key_here":
        os.environ["SERPAPI_KEY"] = env_serpapi_key
        st.sidebar.success("✅ SerpAPI key loaded from environment")
        st.sidebar.info("🛍️ Real Walmart data enabled!")
    else:
        st.sidebar.warning("⚠️ No SerpAPI key - Using AI estimates only")
        st.sidebar.info("💡 Get real Walmart prices at: https://serpapi.com/")
        if "SERPAPI_KEY" in os.environ:
            del os.environ["SERPAPI_KEY"]
    
    # SPEECH RECOGNITION SECTION
    st.sidebar.subheader("🎤 Speech Recognition")
    
    # Check audio packages status
    if not st.session_state.audio_packages_checked:
        st.session_state.audio_packages_status = check_audio_packages()
        st.session_state.audio_packages_checked = True
    
    packages_ok = st.session_state.audio_packages_status.get("audio_recorder_streamlit", False) and \
                  st.session_state.audio_packages_status.get("whisper", False)
    
    if packages_ok:
        st.session_state.speech_enabled = st.sidebar.checkbox(
            "Enable Voice Input",
            value=st.session_state.speech_enabled,
            help="Use microphone to speak your shopping requests"
        )
        
        if st.session_state.speech_enabled:
            # Whisper model selection
            model_options = get_whisper_model_info()
            selected_model = st.sidebar.selectbox(
                "Speech Recognition Quality",
                options=list(model_options.keys()),
                index=list(model_options.keys()).index(st.session_state.whisper_model),
                format_func=lambda x: f"{x.title()} - {model_options[x]}",
                help="Higher quality models are more accurate but slower"
            )
            st.session_state.whisper_model = selected_model
            
            st.sidebar.success("🎤 Voice input ready!")
            if st.session_state.whisper_model in ["medium", "large"]:
                st.sidebar.warning("⚠️ Large models may take time to load initially")
        else:
            st.sidebar.info("💡 Voice input disabled")
    else:
        st.sidebar.error("❌ Audio packages missing")
        missing_packages = []
        if not st.session_state.audio_packages_status.get("audio_recorder_streamlit"):
            missing_packages.append("audio-recorder-streamlit")
        if not st.session_state.audio_packages_status.get("whisper"):
            missing_packages.append("openai-whisper")
        
        st.sidebar.code(f"pip install {' '.join(missing_packages)}")
        st.session_state.speech_enabled = False
    
    # Rate limiting info
    st.sidebar.info("💡 Free tier includes rate limiting for both APIs")
    
    # LANGSMITH SECTION
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
    
    return api_key, serpapi_key


def parse_user_input(user_input: str) -> tuple:
    """More robustly parse user input to extract budget and people count."""
    budget = None
    people_count = 4  # default

    # Budget extraction patterns
    budget_patterns = [
        r'[\$€£](\d+(?:\.\d{1,2})?)',  # e.g., $25.50
        r'(\d+(?:\.\d{1,2})?)\s*(?:dollars|dollar|eur|euros|gbp|pounds)',  # e.g., 25 dollars
        r'(?:budget|under|limit of)\s*[\$€£]?\s*(\d+(?:\.\d{1,2})?)' # e.g., budget of 25
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            budget_str = next((g for g in match.groups() if g is not None), None)
            if budget_str:
                budget = float(budget_str)
                break

    # People count extraction
    people_patterns = [
        r'(\d+)\s+people', r'for\s+(\d+)', r'serves?\s+(\d+)', r'(\d+)\s+person'
    ]
    for pattern in people_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            people_count = int(match.group(1))
            break
            
    return budget, people_count


def display_data_source_info():
    """Display information about data sources being used."""
    serpapi_enabled = bool(os.getenv("SERPAPI_KEY")) and os.getenv("SERPAPI_KEY") != "your_serpapi_key_here"
    
    if serpapi_enabled:
        st.info("🛍️ **Real Store Data**: Using actual Walmart prices via SerpAPI + AI estimates as fallback")
    else:
        st.info("🤖 **AI Estimates**: Using Mistral AI for price estimates (enable SerpAPI for real Walmart data)")


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
    """Display the shopping results with data source indicators."""
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
        
        # Show data source breakdown
        walmart_items = [item for item in shopping_items if hasattr(item, '_from_walmart')]
        ai_items = [item for item in shopping_items if not hasattr(item, '_from_walmart')]
        
        if walmart_items and ai_items:
            st.info(f"📊 **Data Sources**: {len(walmart_items)} from Walmart, {len(ai_items)} from AI estimates")
        elif walmart_items:
            st.success(f"🛍️ **All {len(walmart_items)} items** from real Walmart data!")
        else:
            st.info(f"🤖 **All {len(ai_items)} items** from AI estimates")
        
        categories = {}
        for item in shopping_items:
            categories.setdefault(item.category, []).append(item)
        
        for category, items in categories.items():
            st.write(f"**{category.upper()}:**")
            for item in items:
                # Add indicator for data source
                source_indicator = " 🛍️" if hasattr(item, '_from_walmart') else " 🤖"
                st.write(f"  • {item.name} ({item.quantity}) - ${item.estimated_price:.2f}{source_indicator}")
            st.write("")
        
        # Add legend
        if walmart_items and ai_items:
            st.caption("🛍️ = Real Walmart price | 🤖 = AI estimate")
        
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
            elif "serpapi" in error.lower():
                st.warning(f"🏪 {error}")
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
    """This function is called when the 'Clear Results' button is clicked."""
    st.session_state.current_state = None
    st.session_state.user_input_area = ""
    st.session_state.voice_input_result = None


def handle_voice_input():
    """Handle voice input from microphone with automatic transcription and text injection."""
    if not st.session_state.speech_enabled:
        return None
    
    # Check if audio packages are available
    packages_status = check_audio_packages()
    
    if not (packages_status.get("audio_recorder_streamlit") and packages_status.get("whisper")):
        st.error("❌ Audio packages not available")
        with st.expander("📦 Install Audio Packages", expanded=True):
            show_audio_setup_instructions()
            install_audio_packages_button()
        return None
    
    st.subheader("🎤 Voice Input")
    st.info("💡 **Browser Permissions**: Allow microphone access when prompted")
    
    # Create the voice input interface with automatic transcription
    try:
        transcribed_text = create_best_voice_input(st.session_state.whisper_model)
        
        # If we got transcribed text, automatically inject it into the text area
        if transcribed_text and transcribed_text.strip():
            # Update the session state for the text area
            st.session_state.user_input_area = transcribed_text.strip()
            st.session_state.voice_input_result = transcribed_text.strip()
            
            # Show success message and rerun to update the text area
            st.success(f"✅ **Voice input captured**: {transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}")
            st.info("📝 Text has been automatically filled in the input box below!")
            
            # Force a rerun to update the text area
            st.rerun()
        
        return transcribed_text
        
    except Exception as e:
        st.error(f"❌ Voice input error: {e}")
        st.info("💡 Make sure microphone permissions are enabled in your browser")
        return None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Grocery Shopping AI Assistant", 
        page_icon="🛒", 
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🛒 Grocery Shopping AI Assistant")
    st.markdown("*Multi-Agent System powered by LangGraph, Mistral AI & Real Walmart Data*")
    st.markdown("**⚠️ Note: Mistral API key required. SerpAPI key optional for real Walmart prices.**")
    
    # Check audio packages status and show info
    if not st.session_state.audio_packages_checked:
        packages_status = check_audio_packages()
        st.session_state.audio_packages_status = packages_status
        st.session_state.audio_packages_checked = True
    
    audio_available = (st.session_state.audio_packages_status.get("audio_recorder_streamlit", False) and 
                      st.session_state.audio_packages_status.get("whisper", False))
    
    if not audio_available:
        with st.expander("📦 Audio Setup (Optional)", expanded=False):
            st.warning("🎤 Audio recording packages not found.")
            show_audio_setup_instructions()
            install_audio_packages_button()
            st.info("💡 Voice input will be available after installing packages and restarting the app")
    
    api_keys = setup_sidebar()
    if not api_keys or not api_keys[0]:
        st.error("❌ **Mistral API Key Required**")
        st.info("Please enter a valid Mistral API key in the sidebar to begin.")
        return
    
    api_key, serpapi_key = api_keys
    
    try:
        if not st.session_state.graph or st.session_state.get('graph_api_key') != api_key:
            with st.spinner("🚀 Initializing system with Mistral AI..."):
                st.session_state.graph = create_graph_with_api_key(api_key)
                st.session_state.graph_api_key = api_key
            st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        return
    
    # Display data source information
    display_data_source_info()
    
    st.subheader("💬 What would you like to shop for?")
    
    # Demo button
    if st.button("🍽️ Try Demo: Dinner for 4 people under $25"):
        st.session_state.user_input_area = "Prepare a shopping list for dinner for 4 people within a $25 budget"
        st.rerun()

    # Voice input section (only if packages are available)
    if st.session_state.speech_enabled and audio_available:
        with st.expander("🎤 Voice Input", expanded=False):
            handle_voice_input()
    elif st.session_state.speech_enabled and not audio_available:
        with st.expander("🎤 Voice Input - Setup Required", expanded=False):
            st.warning("📦 Audio packages required for voice input")
            show_audio_setup_instructions()

    # Text input - use the value from session state
    user_input = st.text_area(
        "Describe what you want to shop for:",
        placeholder="e.g., I want to buy ingredients for chicken curry for 5 people under 30 dollars",
        height=100,
        value=st.session_state.user_input_area,  # This will be updated by voice input
        key="text_input_main",
        help="💡 Type your request or use voice input above (if available)"
    )
    
    # Update session state when text area changes
    if user_input != st.session_state.user_input_area:
        st.session_state.user_input_area = user_input
    
    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("🔄 Generate Shopping List", type="primary", disabled=not user_input.strip())
    with col2:
        st.button("🗑️ Clear Results", on_click=clear_results_callback)
    
    if process_button and user_input.strip():
        budget, people_count = parse_user_input(user_input)
        
        # Show processing info with data source
        serpapi_enabled = bool(os.getenv("SERPAPI_KEY")) and os.getenv("SERPAPI_KEY") != "your_serpapi_key_here"
        data_source_msg = "real Walmart data + AI estimates" if serpapi_enabled else "AI estimates"
        
        st.info(f"📊 Processing: **{people_count} people**" + (f", budget **${budget:.2f}**" if budget else ", **no budget limit**") + f" using {data_source_msg}")
        
        with st.spinner("🤖 Agents working with real store data... (May take 45-90 seconds for real prices)"):
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
                    st.info("🔑 Please verify your API keys are correct.")
                elif "serpapi" in str(e).lower():
                    st.info("🏪 SerpAPI issue - falling back to AI estimates.")
    
    if st.session_state.current_state:
        st.markdown("---")
        display_agent_status(st.session_state.current_state)
        display_shopping_results(st.session_state.current_state)
        display_messages_and_errors(st.session_state.current_state)
    
    # Footer with package status
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("*Built with LangChain, LangGraph, Streamlit, Mistral AI, SerpAPI & OpenAI Whisper*")
    with col2:
        if audio_available:
            st.success("🎤 Voice input: Ready")
        else:
            st.warning("🎤 Voice input: Setup required")


if __name__ == "__main__":
    main()