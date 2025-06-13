# Deepgram Chat Voice Assistant Demo

This is a RAG-based chat application for Deepgram documentation and SDKs, featuring both text-based and voice-enabled interfaces. The purpose of this demo is the show the power of voice with Deepgram and demo integrations with RAG applications. Examples where you could see this type of application is when onboarding new employees and helping investors understand your product and codebase through a conversation. 

# Issue Reporting
If you have found a bug or if you have a feature request, please report them at this repository issues section. Please do not report security vulnerabilities on the public GitHub issue tracker.

Check out our KNOWN ISSUES before reporting.

# Demo Features
Capture streaming audio using Deepgram Nova-3 Streaming Speech to Text. </br>
Natural Language responses using an OpenAI LLM.</br>
Speech to Text conversion using Deepgram Aura-2 Text to Speech.</br>

# What is Deepgram?
Deepgram is a foundational AI company providing speech-to-text and language understanding capabilities to make data readable and actionable by human or machines.

# Sign-Up for Deepgram
Want to start building using this project? [Sign-up now for Deepgram and create an API key]([url](https://console.deepgram.com/signup?jump=keys)).

## Quickstart

### Chat Interface (chat.py)
A text-based interface that uses RAG (Retrieval Augmented Generation) to provide accurate, document-backed responses about Deepgram's APIs and SDKs.

### Voice Agent (Agent.py)
A voice-enabled interface that combines Deepgram's voice capabilities with the RAG system, allowing users to:
- Interact with the chatbot using voice
- Receive both voice and text responses
- Get document-backed answers through natural conversation
- Access the same semantic understanding as the text interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a .env file with your API keys:
```bash
OPENAI_API_KEY=your_openai_key
DEEPGRAM_API_KEY=your_deepgram_key
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4-turbo-preview
```

3. Set up Git LFS and pull vector stores:
```bash
# Install Git LFS if you haven't already
git lfs install

# Pull the vector store files
git lfs pull
```

4. Run the application:

For text-based chat:
```bash
python chat.py
```

For voice-enabled interface:
```bash
python Agent.py
```
Then open your browser to `http://localhost:3000`

## Vector Stores

The application uses FAISS vector stores for efficient document retrieval. These are tracked using Git LFS:

```bash
# Track FAISS index files
git lfs track "*.index"
git lfs track "*.faiss"
git lfs track "data/vector_db/**/*"

# Make sure .gitattributes is tracked
git add .gitattributes
```

The vector stores contain:
- Documentation embeddings
- SDK code embeddings
- Semantic search indices

## Features

- Real-time voice interaction
- Document-backed responses
- Semantic caching
- Source tracking
- Both voice and text output
- Natural conversation flow

## Architecture

- `Agent.py`: Voice agent implementation using Deepgram's WebSocket API
- `chat.py`: RAG-powered chatbot implementation
- `templates/index.html`: Frontend interface
- `data/vector_db/`: FAISS vector stores for document retrieval
- `data/cache/`: Semantic cache for similar queries

# Getting Help
We love to hear from you so if you have questions, comments or find a bug in the project, let us know! You can either:
- Open an issue in this repository
- [Join the Deepgram Github Discussions Community]([url](https://github.com/orgs/deepgram/discussions))
- [Join the Deepgram Discord Community]([url](https://discord.com/invite/xWRaCDBtW4))

# Author
[Deepgram]([url](https://deepgram.com/))


