"""
Local storage operations module (FREE VERSION).
Handles all file operations using local filesystem instead of S3.
No cloud costs!
"""

import streamlit as st
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .config import config
except ImportError:
    from config import config

class LocalStorageManager:
    """Manages local file operations (replaces S3 for FREE usage)."""
    
    def __init__(self):
        self.storage_dir = config.vector_store_dir
    
    def check_file_exists(self, file_prefix: str) -> bool:
        """
        Check if a vector store folder exists locally.
        
        Args:
            file_prefix (str): The folder name to check
            
        Returns:
            bool: True if folder exists, False otherwise
        """
        folder_path = self.storage_dir / file_prefix
        return folder_path.exists() and folder_path.is_dir()
    
    def get_existing_files(self) -> list:
        """
        Get list of all existing vector store folders.
        
        Returns:
            list: List of folder names
        """
        try:
            if not self.storage_dir.exists():
                return []
            return [item.name for item in self.storage_dir.iterdir() if item.is_dir()]
        except Exception as e:
            st.error(f"Error listing local files: {str(e)}")
            return []
    
    def check_pdf_already_processed(self, file_prefix: str) -> tuple:
        """
        Check if vector store exists for a PDF.
        
        Args:
            file_prefix (str): The file prefix to check
            
        Returns:
            tuple: (exists, local_path, local_path)
        """
        exists = self.check_file_exists(file_prefix)
        local_path = str(self.storage_dir / file_prefix) if exists else None
        return exists, local_path, local_path
    
    def upload_vector_store(self, local_faiss_path: str, local_pkl_path: str, 
                           s3_faiss_key: str, s3_pkl_key: str) -> bool:
        """
        Compatibility function (no upload needed for local storage).
        Files are already saved locally.
        
        Returns:
            bool: Always True (files already saved)
        """
        return True

# Global storage manager instance
s3_manager = LocalStorageManager()
