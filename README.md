# RAG_toxic-interviewer

A RAG (Retrieval-Augmented Generation) based interviewer system powered by LangChain and OpenAI.

## Features

- Interactive chat interface using Gradio
- Document upload and processing for knowledge base
- Conversational memory for context-aware responses
- Vector-based document retrieval using FAISS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xustacy/RAG_toxic-interviewer.git
cd RAG_toxic-interviewer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

Run the application:
```bash
python app.py
```

The Gradio interface will be available at `http://localhost:7860`

## Features

- Upload text documents to build a knowledge base
- Chat with the AI interviewer
- Conversation history is maintained during the session
- Reset conversation when needed

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls