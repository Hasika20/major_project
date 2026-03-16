#!/usr/bin/env python3
"""
RAG-LLM Healthcare Insurance Assistant

Users upload a healthcare insurance PDF directly in the app and ask questions.
No admin pre-processing step is required.

Usage:
    streamlit run User/app.py --server.port 8502

Pre-requisites:
    pip install -r requirements.txt
    ollama pull llama3.2
"""

def print_usage():
    print(__doc__)

if __name__ == "__main__":
    print_usage()
