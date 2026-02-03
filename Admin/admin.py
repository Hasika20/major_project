"""
FREE Admin Interface for Healthcare Insurance PDF Processing System.

100% FREE VERSION - No cloud costs!
Uses local AI models and storage.

Features:
- Free sentence-transformers embeddings (no AWS Bedrock)
- Local Ollama LLM (no API costs)
- Local filesystem storage (no S3 costs)
"""

import streamlit as st
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui_components import ui_components

def main():
    """Main function for the FREE admin interface."""
    st.title("🏥 FREE Healthcare Insurance PDF Processor")
    st.info("💰 100% Free - No cloud costs! Uses local AI models.")
    st.write("Process healthcare insurance documents to create searchable vector stores.")
    
    # Create tabs for different processing options
    tab1, tab2 = st.tabs(["📄 Single File Upload", "📚 Bulk Process All PDFs"])
    
    with tab1:
        ui_components.render_single_file_upload_tab()
    
    with tab2:
        ui_components.render_bulk_processing_tab()

if __name__ == '__main__':
    main()
