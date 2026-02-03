import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# Add Admin directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent / "Admin"))
from config import config

# Load environment variables (though not needed for free version)
load_dotenv()

def get_available_vector_stores():
    """Get list of processed documents from local storage."""
    stores = []
    storage_dir = config.vector_store_dir
    
    if not storage_dir.exists():
        return []
    
    for store_dir in storage_dir.iterdir():
        if store_dir.is_dir():
            stores.append({
                'name': store_dir.name,
                'display_name': store_dir.name.replace("_", " ").title(),
                'path': str(store_dir)
            })
    
    return sorted(stores, key=lambda x: x['display_name'])

def load_vector_store(store_path):
    """Load FAISS vector store from local storage."""
    try:
        # Try with allow_dangerous_deserialization parameter (for newer versions)
        try:
            vector_store = FAISS.load_local(
                store_path,
                config.bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
        except TypeError:
            # Fall back to older version without the parameter
            vector_store = FAISS.load_local(
                store_path,
                config.bedrock_embeddings
            )
        return vector_store
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None

def _build_context_sections(docs):
    """Format retrieved docs into numbered sections with source hints for citations."""
    sections = []
    for idx, doc in enumerate(docs, start=1):
        metadata = getattr(doc, 'metadata', {})
        
        # Get original filename or fallback to source path
        original_filename = metadata.get('original_filename', metadata.get('source', 'unknown'))
        page = metadata.get('page', 'unknown')
        
        # Clean up source display
        if original_filename != 'unknown' and '/' in str(original_filename):
            original_filename = str(original_filename).split('/')[-1]  # Get just filename
        
        content = (doc.page_content or '').strip()
        sections.append(
            f"[S{idx}]\n{content}\n(Source: {original_filename}, page {page})"
        )
    return "\n\n".join(sections)

def _build_converse_messages(question, docs):
    """Create safety-guided user message for Nova Converse (no system role supported)."""
    context_text = _build_context_sections(docs)

    system_instructions = (
        "You are a helpful assistant for healthcare insurance documents. "
        "Follow these rules strictly: \n"
        "- Ground every answer ONLY in the provided Context sections.\n"
        "- If the answer is not in context, say you don't know and suggest checking the policy documents.\n"
        "- Include inline citations using the section IDs like [S1], [S2] wherever specific facts are used.\n"
        "- Be concise, neutral, and precise. Avoid speculation or fabrication.\n"
        "- Do not provide legal, medical, or financial advice. Provide informational guidance only.\n"
        "- Do not output secrets, credentials, or personal data.\n"
        "- If the user asks for actions that could cause harm or are outside scope, refuse briefly and provide safer alternatives.\n"
        "- Prefer bullet lists for multi-part answers.\n"
    )

    user_prompt = (
        f"{system_instructions}\n\n"
        f"Context sections (use for grounding and citations):\n\n"
        f"{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer with citations like [S1], [S2]. If unknown, say you don't know."
    )

    return [
        {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
    ]

def _extract_sources_from_docs(docs):
    """Extract source information from retrieved documents"""
    sources = []
    for idx, doc in enumerate(docs, start=1):
        metadata = getattr(doc, 'metadata', {})
        original_filename = metadata.get('original_filename', metadata.get('source', 'unknown'))
        page = metadata.get('page', 'unknown')
        
        # Clean up source display
        if original_filename != 'unknown' and '/' in str(original_filename):
            original_filename = str(original_filename).split('/')[-1]
            
        sources.append({
            'id': f'S{idx}',
            'filename': original_filename,
            'page': page,
            'content_preview': (doc.page_content or '')[:150] + '...' if len(doc.page_content or '') > 150 else doc.page_content or ''
        })
    return sources

def query_documents(question, vector_store):
    """Query the documents using FREE local RAG."""
    try:
        # Search for relevant documents
        docs = vector_store.similarity_search(question, k=3)
        
        if not docs:
            return "No relevant documents found.", []
        
        # Extract source information for later display
        sources = _extract_sources_from_docs(docs)
        
        # Build context with safety guidelines
        context_text = _build_context_sections(docs)
        
        # Create prompt for FREE local Ollama LLM
        system_instructions = (
            "You are a helpful assistant for healthcare insurance documents. "
            "Follow these rules strictly: \n"
            "- Ground every answer ONLY in the provided Context sections.\n"
            "- If the answer is not in context, say you don't know and suggest checking the policy documents.\n"
            "- Include inline citations using the section IDs like [S1], [S2] wherever specific facts are used.\n"
            "- Be concise, neutral, and precise. Avoid speculation or fabrication.\n"
            "- Do not provide legal, medical, or financial advice. Provide informational guidance only.\n"
            "- Prefer bullet lists for multi-part answers.\n"
        )
        
        prompt = (
            f"{system_instructions}\n\n"
            f"Context sections (use for grounding and citations):\n\n"
            f"{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer with citations like [S1], [S2]. If unknown, say you don't know."
        )
        
        # Generate response using FREE Ollama
        answer = config.generate_response(prompt, max_tokens=512, temperature=0.2)
        
        return answer, sources
    
    except Exception as e:
        return f"Error querying documents: {str(e)}", []

def main():
    st.title("🏥 FREE Healthcare Insurance Assistant")
    st.info("💰 100% Free - Powered by Local AI (Ollama + Sentence Transformers)")
    st.write("Ask questions about healthcare insurance documents")
    
    # Get available documents from local storage
    stores = get_available_vector_stores()
    
    if not stores:
        st.warning("⚠️ No processed documents found.")
        st.info("📝 **Next Steps:**\n1. Run the Admin interface: `streamlit run Admin/admin.py --server.port 8501`\n2. Process some PDF files\n3. Come back here to query them!")
        st.stop()
    
    # Document selection
    selected_store = st.selectbox(
        "📚 Select a document to query:",
        options=stores,
        format_func=lambda x: x['display_name']
    )
    
    # Load vector store (with caching)
    if 'vector_store' not in st.session_state or st.session_state.get('current_doc') != selected_store['name']:
        with st.spinner(f"Loading {selected_store['display_name']}..."):
            st.session_state.vector_store = load_vector_store(selected_store['path'])
            st.session_state.current_doc = selected_store['name']
        
        if st.session_state.vector_store:
            st.success(f"✅ Loaded: {selected_store['display_name']}")
        else:
            st.error("❌ Failed to load document. Please try reprocessing it.")
            st.stop()
    
    # Question input
    question = st.text_input("💬 Enter your question about this document:")
    
    if st.button("Ask Question", type="primary") and question:
        with st.spinner("🔍 Searching documents and generating answer..."):
            answer, sources = query_documents(question, st.session_state.vector_store)
            
            st.subheader("Answer:")
            st.write(answer)
            
            # Display source information if available
            if sources:
                st.subheader("📚 Sources:")
                for source in sources:
                    with st.expander(f"{source['id']}: {source['filename']} (Page {source['page']})"):
                        st.write("**Content Preview:**")
                        st.write(f"_{source['content_preview']}_")
                        st.write(f"**Full Reference:** {source['filename']}, Page {source['page']}")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.write("""
        - What is a deductible?
        - What services are covered under preventive care?
        - How does copayment work?
        - What is the difference between in-network and out-of-network providers?
        - What are essential health benefits?
        - Explain the difference between AI/AN limited cost sharing and Zero Cost Sharing coverage
        """)
    
    # System info
    with st.expander("ℹ️ System Information"):
        st.write(f"""
        **FREE Local AI Stack:**
        - Embeddings: {config.embedding_model_name}
        - LLM: {config.llm_model_name}
        - Storage: Local filesystem (data/vector_stores/)
        - Processed documents: {len(stores)}
        
        **No cloud costs!** Everything runs on your computer.
        """)

if __name__ == '__main__':
    main()
