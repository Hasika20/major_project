"""
Bulk processing module.
Handles bulk processing of multiple PDF files with progress tracking and result management.
"""

import time
import uuid
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# Handle both relative and absolute imports
try:
    from .pdf_processor import pdf_processor
    from .s3_operations import s3_manager
except ImportError:
    from pdf_processor import pdf_processor
    from s3_operations import s3_manager

class BulkProcessor:
    """Manages bulk processing operations for multiple PDF files."""
    
    def __init__(self):
        self.pdf_processor = pdf_processor
        self.s3_manager = s3_manager
    
    def find_pdf_files(self, pdf_sources_path: Path) -> List[Path]:
        """
        Find all PDF files in the specified directory.
        
        Args:
            pdf_sources_path (Path): Path to the PDF sources directory
            
        Returns:
            List[Path]: List of PDF file paths
        """
        return list(pdf_sources_path.glob("*.pdf"))
    
    def process_all_pdfs(self, skip_existing: bool = True) -> List[Dict[str, Any]]:
        """
        Process all PDF files in the pdf-sources folder.
        
        Args:
            skip_existing (bool): Whether to skip files that already exist in S3
            
        Returns:
            List[Dict]: List of processing results for each file
        """
        # Get the project root directory (one level up from Admin folder)
        project_root = Path(__file__).parent.parent
        pdf_sources_path = project_root / "pdf-sources"
        
        # Find all PDF files
        pdf_files = self.find_pdf_files(pdf_sources_path)
        
        if not pdf_files:
            st.error("No PDF files found in pdf-sources folder!")
            return []
        
        # Check existing files in S3 if skipping duplicates
        if skip_existing:
            st.info("🔍 Checking for existing files in S3...")
            existing_files = self.s3_manager.get_existing_files()
            st.write(f"Found {len(existing_files)} existing files in S3 bucket")
        
        st.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Create UI components for progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create scrollable container for real-time updates
        st.subheader("📝 Processing Log")
        log_container = st.container()
        with log_container:
            log_placeholder = st.empty()
        
        # Initialize log content
        log_messages = []
        # Generate unique session ID for this processing session
        session_id = str(uuid.uuid4())[:8]
        update_counter = [0]  # Use list to allow modification in nested function
        
        def update_progress(message: str):
            """Update progress display and log."""
            status_text.text(message)
            log_messages.append(message)
            # Keep only last 50 messages to prevent UI overload
            if len(log_messages) > 50:
                log_messages.pop(0)
            
            # Update scrollable log with latest messages
            # Use counter to ensure truly unique keys
            update_counter[0] += 1
            unique_key = f"log_area_{session_id}_{update_counter[0]}"
            with log_placeholder.container():
                st.text_area(
                    "Real-time Processing Updates", 
                    value="\n".join(log_messages),
                    height=300,
                    disabled=True,
                    key=unique_key
                )
        
        results = []
        
        # Process each PDF file
        for i, pdf_path in enumerate(pdf_files):
            try:
                update_progress(f"🔄 Starting {i+1}/{len(pdf_files)}: {pdf_path.name}")
                
                # Process with duplicate checking
                process_result = self.pdf_processor.process_pdf_file(
                    str(pdf_path), 
                    progress_callback=update_progress,
                    skip_existing=skip_existing
                )
                
                # Handle the extended return format
                if len(process_result) == 7:
                    success, filename, pages, chunks, faiss_key, pkl_key, status = process_result
                else:
                    # Fallback for old format
                    success, filename, pages, chunks, faiss_key, pkl_key = process_result
                    status = "processed" if success else "failed"
                
                results.append({
                    'filename': filename,
                    'success': success,
                    'pages': pages,
                    'chunks': chunks,
                    'faiss_key': faiss_key,
                    'pkl_key': pkl_key,
                    'status': status
                })
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(pdf_files))
                
                # Add a small delay to prevent overwhelming the system
                time.sleep(0.3)
                
            except Exception as e:
                error_msg = f"❌ Failed to process {pdf_path.name}: {str(e)}"
                update_progress(error_msg)
                results.append({
                    'filename': pdf_path.name,
                    'success': False,
                    'pages': 0,
                    'chunks': 0,
                    'faiss_key': None,
                    'pkl_key': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Display final results
        update_progress("🎉 Bulk processing completed!")
        status_text.text("🎉 Bulk processing completed!")
        
        return results
    
    def display_results_summary(self, results: List[Dict[str, Any]]):
        """
        Display processing results summary.
        
        Args:
            results (List[Dict]): List of processing results
        """
        # Summary metrics
        st.subheader("📊 Processing Summary")
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        skipped = sum(1 for r in results if r.get('status') == 'skipped')
        processed = sum(1 for r in results if r.get('status') == 'processed')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(results))
        with col2:
            st.metric("Processed", processed, delta=f"+{processed}")
        with col3:
            st.metric("Skipped", skipped, delta=f"↻{skipped}")
        with col4:
            st.metric("Failed", failed, delta=f"-{failed}" if failed > 0 else None)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Create tabs for different result types
        if skipped > 0:
            tab1, tab2, tab3 = st.tabs(["✅ All Results", "📁 Skipped Files", "🔄 Processed Files"])
        else:
            tab1, tab2 = st.tabs(["✅ All Results", "🔄 Processed Files"])
            tab3 = None
        
        with tab1:
            self._display_all_results(results)
        
        if tab3:  # Skipped files tab
            with tab3:
                self._display_skipped_results(results)
        
        with tab2:  # Processed files tab
            self._display_processed_results(results)
    
    def _display_all_results(self, results: List[Dict[str, Any]]):
        """Display all processing results."""
        for result in results:
            if result['success']:
                if result.get('status') == 'skipped':
                    st.info(f"⏭️ {result['filename']} (Skipped - already exists)")
                    st.write(f"   📁 Existing S3 files: {result['faiss_key']}, {result['pkl_key']}")
                else:
                    st.success(f"✅ {result['filename']} (Newly processed)")
                    st.write(f"   📖 Pages: {result['pages']} | ✂️ Chunks: {result['chunks']}")
                    st.write(f"   📤 S3 Keys: {result['faiss_key']}, {result['pkl_key']}")
            else:
                st.error(f"❌ {result['filename']}")
                if 'error' in result:
                    st.write(f"   Error: {result['error']}")
    
    def _display_skipped_results(self, results: List[Dict[str, Any]]):
        """Display skipped files results."""
        skipped_files = [r for r in results if r.get('status') == 'skipped']
        if skipped_files:
            st.write(f"**{len(skipped_files)} files were skipped because they already exist in S3:**")
            for result in skipped_files:
                st.info(f"📁 {result['filename']}")
                st.write(f"   Existing files: {result['faiss_key']}, {result['pkl_key']}")
        else:
            st.write("No files were skipped.")
    
    def _display_processed_results(self, results: List[Dict[str, Any]]):
        """Display processed files results."""
        processed_files = [r for r in results if r.get('status') == 'processed']
        if processed_files:
            st.write(f"**{len(processed_files)} files were newly processed:**")
            for result in processed_files:
                st.success(f"✅ {result['filename']}")
                st.write(f"   📖 Pages: {result['pages']} | ✂️ Chunks: {result['chunks']}")
                st.write(f"   📤 S3 Keys: {result['faiss_key']}, {result['pkl_key']}")
        else:
            st.write("No files were newly processed.")

# Global bulk processor instance
bulk_processor = BulkProcessor()
