# Wikipedia RAG Application

A web-based Retrieval-Augmented Generation (RAG) application that answers user questions using curated Wikipedia content.

## Features
- Ingests and processes selected Wikipedia pages related to AI and machine learning topics.
- Builds a vector store index with LlamaIndex embeddings for efficient semantic search.
- Integrates OpenAI's GPT-4o-mini to generate natural language answers based on retrieved context.
- Interactive and user-friendly interface built with Streamlit.
- Displays both the AI-generated answer and the retrieved Wikipedia content for transparency.
- Caches index and query results to optimize performance.
- Securely manages API keys with dotenv.

## Technologies Used
- Python
- Streamlit
- LlamaIndex (GPT Index)
- OpenAI API
- Wikipedia API
- dotenv

## Installation
1. Clone the repository
2. create a .env file with your openAi api key saved as     OPENAI_API_KEY="<insert_key_here>"
3. download dependencies
       pip install streamlit llama-index python-dotenv
4. run using streamlit run main.py
