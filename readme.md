# 🛒 Grocery Shopping AI Assistant

A sophisticated multi-agent system that helps users create shopping lists and place grocery orders using LangGraph, LangChain, and Streamlit. The system utilizes a Supervisor Agent pattern with specialized agents for different tasks.

## 🏗️ Architecture

The system employs a **Supervisor Multi-Agent Architecture** with the following components:

- **Supervisor Agent**: Orchestrates execution flow and manages agent interactions
- **Planner Agent**: Interprets user intent and creates execution plans
- **Recipe Agent**: Finds suitable recipes and extracts ingredients
- **Product Finder Agent**: Maps ingredients to store products with estimated prices
- **Budgeting Agent**: Analyzes budget constraints and optimizes shopping lists
- **Finalizer Agent**: Aggregates everything into a final shopping list

## 🚀 Features

- ✅ Natural language input processing
- ✅ Recipe recommendations based on user preferences
- ✅ Budget-aware shopping list optimization
- ✅ Multi-category product mapping
- ✅ Real-time agent status tracking
- ✅ LangSmith integration for tracing and monitoring
- ✅ Streamlit web interface
- ✅ Demo mode for testing without API keys

## 🛠️ Technical Stack

- **Python 3.10+**
- **LangChain**: LLM orchestration and chaining
- **LangGraph**: Multi-agent workflow management
- **LangSmith**: Tracing and monitoring
- **Streamlit**: Web interface
- **Mistral AI**: Large Language Model
- **Pydantic**: Data validation and serialization

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd grocery-shopping-assistant
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env
# Edit .env with your API keys
```

## 🔑 API Keys Setup

### Required:
- **Mistral API Key**: Get from [Mistral Console](https://console.mistral.ai/)

### Optional:
- **LangSmith API Key**: Get from [LangSmith](https://langsmith.langchain.com/) for tracing

## 🏃‍♂️ Running the Application

### Start the Streamlit App:
```bash
streamlit run main.py
```

### Demo Mode:
The application includes a demo mode that works without API keys using fallback responses.

## 📝 Usage Examples

### Example 1: Basic Request
```
Input: "I want to buy ingredients for pizza"
Output: Complete shopping list with pizza ingredients
```

### Example 2: Budget-Constrained Request
```
Input: "Prepare a shopping list for dinner for 4 people within a $25 budget"
Output: Optimized shopping list under $25 for 4 people
```

### Example 3: Specific Cuisine
```
Input: "Italian pasta dinner for 6 people, budget $35"
Output: Italian-themed shopping list optimized for budget
```

## 🧪 Testing

Run the test suite:
```bash
python test_agents.py
```

The tests cover:
- Individual agent functionality
- State management
- Error handling
- Fallback mechanisms
- Budget optimization

## 📁 Project Structure

```
grocery-shopping-assistant/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class
│   ├── planner_agent.py       # User intent interpretation
│   ├── recipe_agent.py        # Recipe finding and ingredient extraction
│   ├── product_finder_agent.py # Ingredient to product mapping
│   ├── budgeting_agent.py     # Budget analysis and optimization
│   ├── finalizer_agent.py     # Final list generation
│   └── supervisor_agent.py    # Execution flow management
├── graph.py                   # LangGraph workflow construction
├── state.py                   # State management and data models
├── llm_config.py             # Mistral AI LLM configuration
├── main.py                   # Streamlit application entry point
├── test_agents.py            # Test suite
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 🔄 Workflow Process

1. **User Input**: Natural language request processing
2. **Planning**: Intent analysis and execution plan creation
3. **Recipe Finding**: Recipe selection and ingredient extraction
4. **Product Mapping**: Convert ingredients to store products with prices
5. **Budget Analysis**: Check constraints and optimize if needed
6. **Finalization**: Generate formatted shopping list

## 🎯 Key Features in Detail

### Multi-Agent Coordination
- Supervisor manages execution flow
- Agents communicate through shared state
- Error handling and recovery mechanisms

### Budget Intelligence
- Automatic budget constraint checking
- Smart optimization when over budget
- Alternative product suggestions

### Product Database
- Simulated store inventory with realistic prices
- Category-based organization
- Fallback mapping for unknown ingredients

### User Experience
- Real-time agent status updates
- Comprehensive error reporting
- Printable shopping list format

## 🚨 Error Handling

The system includes robust error handling:
- **LLM API failures**: Fallback responses for demo purposes
- **Invalid JSON**: Parsing error recovery
- **Missing data**: Default value provision
- **Budget optimization**: Graceful degradation

## 🔍 Monitoring with LangSmith

When LangSmith is configured, the system provides:
- Agent execution tracing
- Performance monitoring  
- Debug information
- Cost tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🔮 Future Enhancements

- [ ] Real grocery store API integration
- [ ] Nutritional analysis
- [ ] Meal planning across multiple days
- [ ] Shopping list sharing
- [ ] Mobile app development
- [ ] Voice interface
- [ ] Inventory management

## 📞 Support

For issues and questions:
1. Check the test suite for examples
2. Review the agent implementations
3. Enable LangSmith tracing for debugging
4. Use demo mode for testing

---

**Built with ❤️ using LangChain, LangGraph, and Streamlit**