#!/usr/bin/env python3
"""
Bulk Processing Demo Script - FREE VERSION

This script demonstrates the bulk PDF processing functionality without using Streamlit.
Uses FREE local AI models - no cloud costs!
"""

import sys
import os
from pathlib import Path

# Add Admin directory to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "Admin"))

def main():
    """Demo the bulk processing functionality with FREE local AI"""
    print("🚀 RAG-LLM Healthcare Insurance - FREE Bulk Processing Demo")
    print("💰 100% Free - No cloud costs!")
    print("=" * 60)
    
    try:
        # Import from the modular structure
        from Admin.bulk_processor import bulk_processor
        from Admin.s3_operations import s3_manager  # Now local storage manager
        print("✅ Successfully imported processing functions")
    except ImportError as e:
        print(f"❌ Failed to import functions: {e}")
        print("Make sure you're running this from the project root directory.")
        return False
    
    # Check PDF sources
    pdf_sources_path = project_root / "pdf-sources"
    pdf_files = list(pdf_sources_path.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in pdf-sources folder!")
        return False
    
    print(f"📁 Found {len(pdf_files)} PDF files in pdf-sources:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"   {i}. {pdf_file.name}")
    
    print("\n" + "=" * 60)
    
    # Check existing files in local storage
    print("\n🔍 Checking for existing processed files locally...")
    
    # Check existing files in local storage
    print("\n🔍 Checking for existing processed files locally...")
    
    existing_files = s3_manager.get_existing_files()
    print(f"Found {len(existing_files)} existing processed documents locally")
    
    # Check which PDFs are already processed
    already_processed = []
    needs_processing = []
    
    for pdf_file in pdf_files:
        filename = pdf_file.name
        file_prefix = os.path.splitext(filename)[0].replace(" ", "_").replace("(", "").replace(")", "")
        already_exists, local_path, _ = s3_manager.check_pdf_already_processed(file_prefix)
        
        if already_exists:
            already_processed.append((filename, local_path))
        else:
            needs_processing.append(filename)
    
    if already_processed:
        print(f"\n📁 {len(already_processed)} files already processed:")
        for filename, local_path in already_processed[:3]:  # Show first 3
            print(f"   ✅ {filename}")
        if len(already_processed) > 3:
            print(f"   ... and {len(already_processed) - 3} more")
    
    if needs_processing:
        print(f"\n🔄 {len(needs_processing)} files need processing:")
        for filename in needs_processing[:5]:  # Show first 5
            print(f"   📄 {filename}")
        if len(needs_processing) > 5:
            print(f"   ... and {len(needs_processing) - 5} more")
    
    # Ask user if they want to proceed
    print("\n" + "=" * 60)
    skip_existing = input("Skip files that already exist locally? (Y/n): ").lower() != 'n'
    
    if skip_existing and not needs_processing:
        print("✅ All files already processed locally! No processing needed.")
        return True
    
    files_to_process = needs_processing if skip_existing else [f.name for f in pdf_files]
    
    if files_to_process:
        response = input(f"Process {len(files_to_process)} files? This may take several minutes. (y/N): ")
        if response.lower() != 'y':
            print("❌ Processing cancelled by user.")
            return False
    else:
        print("✅ No files need processing.")
        return True
    
    print("\n🔄 Starting FREE bulk processing with local AI...")
    print("💰 Using Sentence Transformers + Ollama (no cloud costs!)")
    print("=" * 60)
    
    # Use the bulk processor
    try:
        print("\n📝 Processing all PDFs...")
        results = bulk_processor.process_all_pdfs(skip_existing=skip_existing)
        
        # Display summary
        if results:
            print("\n" + "=" * 60)
            print("📊 Processing Summary:")
            print("=" * 60)
            
            processed_count = sum(1 for r in results if r.get('status') == 'processed')
            skipped_count = sum(1 for r in results if r.get('status') == 'skipped')
            failed_count = sum(1 for r in results if r.get('status') in ['failed', 'error'])
            
            print(f"✅ Processed: {processed_count}")
            print(f"⏭️  Skipped: {skipped_count}")
            print(f"❌ Failed: {failed_count}")
            print(f"📁 Total: {len(results)}")
            
            print("\n📄 Detailed Results:")
            for result in results:
                status_icon = "✅" if result['success'] else "❌"
                status_text = result.get('status', 'unknown')
                print(f"{status_icon} {result['filename']} - {status_text}")
            
            print("\n💾 Files saved to: data/vector_stores/")
            print("=" * 60)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 BULK PROCESSING SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    skipped = sum(1 for r in results if r.get('status') == 'skipped')
    processed = sum(1 for r in results if r.get('status') == 'processed')
    
    # Calculate totals only for processed files (not skipped ones)
    total_pages = sum(r['pages'] for r in results if r.get('status') == 'processed' and isinstance(r['pages'], int))
    total_chunks = sum(r['chunks'] for r in results if r.get('status') == 'processed' and isinstance(r['chunks'], int))
    
            
            print("\n🎉 Processing complete!")
            print("\n📝 Next Steps:")
            print("1. Start User Interface: streamlit run User/app.py --server.port 8502")
            print("2. Select a document and ask questions")
            print("3. All data stored locally (no cloud costs!)")
            
            return True
        else:
            print("❌ No results returned from processing")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during bulk processing: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("- Make sure Ollama is installed and running")
        print("- Check that llama3.2 model is downloaded: ollama list")
        print("- Verify Python packages are installed: pip list")
        return False

if __name__ == "__main__":
    try:
        print("\n💡 Tip: This script uses FREE local AI - no cloud costs!")
        print("   Make sure Ollama is running and models are downloaded.\n")
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
