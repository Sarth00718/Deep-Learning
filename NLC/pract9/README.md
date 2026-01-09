# Practical 9: Build a Conversational Agent/Chatbot using LangChain

## Overview
This practical demonstrates how to build conversational agents using LangChain framework with different levels of complexity.

## Files

### 1. `langchain-chatbot.py`
Basic conversational chatbot with:
- Conversation memory
- Custom prompts
- Simple Q&A interface

### 2. `langchain-chatbot-advanced.py`
Advanced chatbot with:
- Windowed conversation memory (remembers last 5 exchanges)
- Custom tools (time, calculator)
- Agent-based architecture
- Statistics tracking
- Memory management

## Requirements

```bash
pip install langchain langchain-groq python-dotenv
```

## Setup

1. Create a `.env` file in the `pract9` directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

2. Get your Groq API key from: https://console.groq.com/

## Usage

### Basic Chatbot
```bash
python langchain-chatbot.py
```

### Advanced Chatbot
```bash
python langchain-chatbot-advanced.py
```

## Features Demonstrated

1. **Conversation Memory**: Maintains context across multiple turns
2. **Custom Prompts**: Defines chatbot personality and behavior
3. **Tool Integration**: Extends capabilities with custom functions
4. **Agent Architecture**: Uses ReAct pattern for reasoning
5. **Error Handling**: Graceful error management

## Example Interactions

### Basic Chatbot
```
You: Hello!
Chatbot: Hello! How can I help you today?

You: What's your name?
Chatbot: I'm an AI assistant created to help you with various tasks.
```

### Advanced Chatbot
```
You: What time is it?
Chatbot: The current time is 2024-01-15 14:30:45

You: Calculate 25 * 4 + 10
Chatbot: The result is: 110

You: stats
Conversation Statistics:
  Total messages: 2
  Memory size: 4 messages
```

## Key Concepts

- **LangChain**: Framework for building LLM applications
- **Memory**: Stores conversation history
- **Agents**: Autonomous entities that use tools
- **Tools**: Functions the agent can call
- **Prompts**: Templates for structuring inputs

## Troubleshooting

1. **API Key Error**: Ensure `.env` file exists with valid GROQ_API_KEY
2. **Import Error**: Install required packages: `pip install langchain langchain-groq python-dotenv`
3. **Connection Error**: Check internet connection and API key validity
