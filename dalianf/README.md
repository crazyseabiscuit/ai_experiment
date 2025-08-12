# Multi-Agent System with Memory, Tools, and Human Feedback

A comprehensive multi-agent system that integrates memory management, dynamic tool integration, human feedback mechanisms, and RAGAS validation for coordinated AI agent workflows.

## 🚀 Features

### Core Architecture
- **Multi-Agent Coordination**: Coordinator agent manages specialist agents (Researcher, Analyst, Executor, Validator)
- **Memory Management**: Short-term, long-term, and working memory with SQLite persistence
- **Tool Integration**: Dynamic tool registry with categorized tools for various tasks
- **Human Feedback**: Interactive feedback system with learning capabilities
- **Message Bus**: Asynchronous communication system for agent coordination
- **RAGAS Validation**: Quality validation for agent outputs

### Agent Types
1. **Coordinator Agent**: Orchestrates workflows and task distribution
2. **Researcher Agent**: Information gathering and analysis
3. **Analyst Agent**: Data interpretation and insight generation
4. **Executor Agent**: Task implementation and workflow execution
5. **Validator Agent**: Quality assurance and validation

### Tool Categories
- **Search & Information**: Web search, stock prices, market news
- **Data Analysis**: Financial analysis, portfolio metrics, investment reports
- **Visualization**: Chart creation and data visualization
- **Communication**: Email and notification systems
- **Task Management**: Task creation and status tracking
- **Quality & Validation**: Data quality checks and compliance validation
- **Risk Assessment**: Risk analysis and mitigation strategies

## 📦 Installation

### Prerequisites
```bash
pip install langchain langchain-community langgraph
pip install pandas numpy matplotlib seaborn
pip install requests
pip install dashscope
```

### Setup
1. Clone the repository
2. Set up your DashScope API key in `core_system.py`
3. Install dependencies
4. Run the demo or CLI

## 🎯 Quick Start

### Basic Usage
```python
import asyncio
from multi_agent_system.cli import MultiAgentCLI

async def main():
    cli = MultiAgentCLI()
    await cli.initialize_system()
    await cli.interactive_loop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Run Demo
```bash
python multi_agent_demo.py
```

### Run CLI
```bash
python -m multi_agent_system.cli
```

## 📖 Usage Examples

### 1. Investment Research
```python
# Research market trends
response = await coordinator.process_user_input(
    "Research current market trends in technology sector"
)

# Analyze financial data
response = await coordinator.process_user_input(
    "Analyze the performance of AI companies in Q1 2024"
)
```

### 2. Tool Integration
```python
# Use available tools
tools = tool_registry.get_tools_by_category("analysis")
for tool in tools:
    result = tool.func("sample data")
```

### 3. Human Feedback
```python
# Request feedback
feedback_interface.request_feedback(
    session_id="session_123",
    agent_id="researcher",
    phase="analysis",
    content="Analysis results...",
    context={"task": "market_research"}
)
```

## 🛠️ System Architecture

### Memory Management
- **Short-term Memory**: ConversationBufferMemory for recent interactions
- **Long-term Memory**: ConversationSummaryMemory for persistent storage
- **Working Memory**: Dictionary-based temporary storage
- **Database**: SQLite for persistent storage of conversations and feedback

### Tool System
- **Dynamic Registration**: Tools can be registered at runtime
- **Categorization**: Tools organized by category for easy access
- **Usage Tracking**: Monitor tool usage patterns
- **Extensibility**: Easy to add new tools

### Feedback System
- **Interactive Interface**: Console-based feedback collection
- **Pattern Analysis**: Analyze feedback trends and patterns
- **Learning System**: Adapt agent behavior based on feedback
- **Auto-Feedback**: Automated feedback generation for testing

### Validation Framework
- **RAGAS Integration**: Quality metrics for agent outputs
- **Multi-metric Evaluation**: Faithfulness, relevance, correctness
- **Continuous Improvement**: Feedback-driven quality enhancement

## 📊 Demo Modes

### 1. Comprehensive Demo
Demonstrates all system capabilities through predefined scenarios:
- Investment Research
- Financial Analysis  
- Market Intelligence

### 2. Interactive Demo
Interactive command-line interface for real-time interaction:
- Natural language commands
- Dynamic task assignment
- Real-time feedback collection

### 3. Stress Test
Performance evaluation under load:
- Concurrent task execution
- Response time analysis
- Success rate monitoring

## 🔧 Configuration

### Environment Variables
```python
# API Configuration
DASHSCOPE_API_KEY = 'your-api-key-here'

# Database Configuration
DATABASE_PATH = 'multi_agent_memory.db'

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'multi_agent_system.log'
```

### Agent Configuration
```python
# Customize agent prompts and capabilities
agent_config = AgentConfig(
    name="Custom Agent",
    agent_type=AgentType.CUSTOM,
    system_prompt="Your custom prompt here",
    description="Agent description",
    tools=["tool1", "tool2"],
    max_iterations=10
)
```

## 📈 Monitoring & Analytics

### Performance Metrics
- Task completion rates
- Response time analysis
- Feedback quality scores
- Tool usage patterns

### Memory Analytics
- Conversation history analysis
- Feedback pattern recognition
- Learning progress tracking

### System Health
- Agent status monitoring
- Tool availability checks
- Memory usage statistics

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python multi_agent_demo.py --mode=stress_test
```

### Performance Tests
```bash
python multi_agent_demo.py --mode=performance_test
```

## 🚀 Deployment

### Local Development
```bash
# Run development server
python multi_agent_demo.py

# Run CLI interface
python -m multi_agent_system.cli
```

### Production Deployment
1. Set up environment variables
2. Configure database
3. Set up monitoring
4. Deploy with proper scaling

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings
- Write comprehensive tests

## 📋 TODO

### High Priority
- [ ] Add more sophisticated tool integrations
- [ ] Implement advanced learning algorithms
- [ ] Add web interface
- [ ] Improve error handling

### Medium Priority
- [ ] Add more agent types
- [ ] Implement distributed processing
- [ ] Add API endpoints
- [ ] Improve documentation

### Low Priority
- [ ] Add visualization dashboard
- [ ] Implement plugin system
- [ ] Add multi-language support
- [ ] Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the framework
- DashScope for LLM services
- RAGAS for validation metrics
- OpenAI for agent architecture inspiration

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Join our discussion forums
- Contact the development team

---

**Built with ❤️ using Python, LangChain, and modern AI technologies**