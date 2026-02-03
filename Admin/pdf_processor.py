"""
PDF processing module.
Handles PDF loading, text extraction, chunking, and vector store creation.
"""

import os
import uuid
from pathlib import Path
from typing import List, Tuple, Callable, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Handle both relative and absolute imports
try:
    from .config import config
    from .s3_operations import s3_manager
except ImportError:
    from config import config
    from s3_operations import s3_manager

class PDFProcessor:
    """Handles PDF processing operations."""
    
    def __init__(self):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.bedrock_embeddings = config.bedrock_embeddings
    
    def split_text(self, documents: List, original_filename: str) -> List:
        """
        Split documents into chunks and enrich metadata.
        
        Args:
            documents (List): List of document objects from PDF loader
            original_filename (str): Original filename for metadata
            
        Returns:
            List: List of text chunks with enriched metadata
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        
        # Enrich metadata with original filename
        for chunk in chunks:
            if hasattr(chunk, 'metadata'):
                chunk.metadata['original_filename'] = original_filename
            else:
                chunk.metadata = {'original_filename': original_filename}
        
        return chunks
    
    def create_vector_store(self, request_id: str, chunks: List, file_prefix: str = "my_faiss") -> Tuple[bool, str, str]:
        """
        Create vector store and save to local filesystem (FREE VERSION).
        
        Args:
            request_id (str): Unique request identifier
            chunks (List): Text chunks to vectorize
            file_prefix (str): Prefix for local folder naming
            
        Returns:
            Tuple[bool, str, str]: (success, local_path, local_path)
        """
        try:
            # Create FAISS vector store with FREE embeddings
            vector_store_faiss = FAISS.from_documents(chunks, self.bedrock_embeddings)
            
            # Save to local data directory (replaces S3)
            save_path = config.vector_store_dir / file_prefix
            save_path.mkdir(exist_ok=True)
            
            # Save vector store
            vector_store_faiss.save_local(str(save_path))
            
            local_path = str(save_path)
            return True, local_path, local_path
            
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def process_pdf_file(self, pdf_path: str, progress_callback: Optional[Callable] = None, 
                        skip_existing: bool = True) -> Tuple:
        """
        Process a single PDF file and create vector store.
        
        Args:
            pdf_path (str): Path to the PDF file
            progress_callback (Callable, optional): Callback function for progress updates
            skip_existing (bool): Whether to skip files that already exist in S3
            
        Returns:
            Tuple: (success, filename, pages, chunks, faiss_key, pkl_key, status)
        """
        try:
            # Get original filename for metadata and S3 naming
            original_filename = os.path.basename(pdf_path)
            file_prefix = self._clean_filename_for_s3(original_filename)
            
            if progress_callback:
                progress_callback(f"📄 Processing: {original_filename}")
            
            # Check if file already exists in S3
            if skip_existing:
                already_processed, existing_faiss_key, existing_pkl_key = s3_manager.check_pdf_already_processed(file_prefix)
                if already_processed:
                    if progress_callback:
                        progress_callback(f"⏭️ Skipping {original_filename} - already exists in S3")
                        progress_callback(f"📁 Existing files: {existing_faiss_key}, {existing_pkl_key}")
                    return True, original_filename, "N/A", "N/A", existing_faiss_key, existing_pkl_key, "skipped"
            
            # Generate unique request ID for this file
            request_id = str(uuid.uuid4())
            
            # Load the PDF file
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            
            if progress_callback:
                progress_callback(f"📖 Loaded {len(pages)} pages from {original_filename}")
            
            # Split text into chunks
            chunks = self.split_text(pages, original_filename)
            
            if progress_callback:
                progress_callback(f"✂️ Created {len(chunks)} chunks from {original_filename}")
            
            # Create vector store
            success, faiss_key, pkl_key = self.create_vector_store(request_id, chunks, file_prefix)
            
            if success:
                if progress_callback:
                    progress_callback(f"✅ Vector store created for {original_filename}")
                    progress_callback(f"� Saved locally to: {faiss_key}")
                return True, original_filename, len(pages), len(chunks), faiss_key, pkl_key, "processed"
            else:
                if progress_callback:
                    progress_callback(f"❌ Failed to create vector store for {original_filename}")
                return False, original_filename, len(pages), len(chunks), None, None, "failed"
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error processing {original_filename}: {str(e)}")
            return False, original_filename, 0, 0, None, str(e), "error"
    
    def _clean_filename_for_s3(self, filename: str) -> str:
        """
        Clean filename for S3 key naming.
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Cleaned filename suitable for S3 keys
        """
        # Remove extension and clean special characters
        name_without_ext = os.path.splitext(filename)[0]
        cleaned = name_without_ext.replace(" ", "_").replace("(", "").replace(")", "")
        return cleaned

# Global PDF processor instance
pdf_processor = PDFProcessor()
