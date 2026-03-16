import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Import shared config from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

# Load environment variables (optional for local mode)
load_dotenv()


def _build_context_sections(docs):
    """Format retrieved docs into numbered sections with source hints for citations."""
    sections = []
    for idx, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {})
        original_filename = metadata.get("original_filename", metadata.get("source", "unknown"))
        page = metadata.get("page", "unknown")

        if original_filename != "unknown" and "/" in str(original_filename):
            original_filename = str(original_filename).split("/")[-1]

        content = (doc.page_content or "").strip()
        sections.append(f"[S{idx}]\n{content}\n(Source: {original_filename}, page {page})")
    return "\n\n".join(sections)


def _extract_sources_from_docs(docs):
    """Extract source references from retrieved chunks."""
    sources = []
    for idx, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {})
        original_filename = metadata.get("original_filename", metadata.get("source", "unknown"))
        page = metadata.get("page", "unknown")

        if original_filename != "unknown" and "/" in str(original_filename):
            original_filename = str(original_filename).split("/")[-1]

        content = doc.page_content or ""
        sources.append(
            {
                "id": f"S{idx}",
                "filename": original_filename,
                "page": page,
                "content_preview": (content[:150] + "...") if len(content) > 150 else content,
            }
        )
    return sources


def _create_vector_store_from_upload(uploaded_file):
    """Build an in-memory vector store from an uploaded PDF."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        pages = loader.load_and_split()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["original_filename"] = uploaded_file.name

        vector_store = FAISS.from_documents(chunks, config.bedrock_embeddings)
        return vector_store, len(pages), len(chunks), None
    except Exception as exc:
        return None, 0, 0, str(exc)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _query_uploaded_document(question, vector_store):
    """Query uploaded PDF via RAG and return answer with sources."""
    try:
        docs = vector_store.similarity_search(question, k=3)
        if not docs:
            return "No relevant content was found in the uploaded PDF.", []

        sources = _extract_sources_from_docs(docs)
        context_text = _build_context_sections(docs)

        prompt = (
            "You are a healthcare insurance document assistant.\n"
            "Rules:\n"
            "- Use only the provided context for factual claims.\n"
            "- Include citations like [S1], [S2] for all key facts.\n"
            "- If the answer is not in context, clearly say it is not available in this document.\n"
            "- Be concise and precise.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer with citations."
        )

        answer = config.generate_response(prompt, max_tokens=512, temperature=0.2)
        return answer, sources
    except Exception as exc:
        return f"Error querying document: {str(exc)}", []

def main():
    # Bold, editorial-style CSS styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:wght@500;600;700&display=swap');

        :root {
            --ink: #1b1d1f;
            --ink-soft: #4a4f57;
            --paper: #f4f1ea;
            --sunset: #f2a65a;
            --clay: #e07a5f;
            --sage: #81b29a;
            --ocean: #3d5a80;
            --card: #fffdf7;
            --line: #e6e0d6;
            --shadow: 0 16px 40px rgba(27, 29, 31, 0.12);
        }

        * {
            font-family: 'Space Grotesk', sans-serif;
        }

        .main {
            background: radial-gradient(circle at 10% 5%, rgba(242, 166, 90, 0.25), transparent 42%),
                        radial-gradient(circle at 90% 0%, rgba(129, 178, 154, 0.3), transparent 38%),
                        var(--paper);
        }

        /* Header */
        .main-header {
            background: linear-gradient(120deg, #1b1d1f 0%, #2f3340 55%, #3d5a80 100%);
            padding: 3rem 2.5rem 2.75rem 2.5rem;
            margin: -1rem -1rem 2.5rem -1rem;
            position: relative;
            overflow: hidden;
        }

        .main-header::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 85% 15%, rgba(242, 166, 90, 0.35), transparent 50%);
            opacity: 0.85;
        }

        .main-title {
            color: #fdfbf7;
            font-size: clamp(2.1rem, 3vw, 2.9rem);
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
            position: relative;
            z-index: 1;
            font-family: 'Fraunces', serif;
        }

        .main-subtitle {
            color: rgba(253, 251, 247, 0.88);
            font-size: 1.05rem;
            margin-top: 0.75rem;
            font-weight: 500;
            position: relative;
            z-index: 1;
        }

        /* Description card */
        .info-section {
            background: var(--card);
            padding: 2rem 2.25rem;
            border-radius: 18px;
            margin: 1.5rem 0 2rem 0;
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
        }

        .info-title {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0 0 0.8rem 0;
        }

        .info-text {
            color: var(--ink-soft);
            font-size: 1rem;
            line-height: 1.7;
            margin: 0.6rem 0;
        }

        /* Content cards */
        .content-box, .content-card {
            background: var(--card);
            padding: 2rem 2.25rem;
            border-radius: 18px;
            margin: 1.25rem 0;
            border: 1px solid var(--line);
            box-shadow: 0 10px 24px rgba(27, 29, 31, 0.08);
            animation: rise 0.45s ease-out;
        }

        .section-heading, .section-title {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0 0 1.2rem 0;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        /* Answer display */
        .answer-container {
            background: #fff4e3;
            padding: 1.6rem 1.8rem;
            border-radius: 14px;
            margin: 1.25rem 0;
            border: 1px solid rgba(224, 122, 95, 0.35);
            box-shadow: inset 0 0 0 1px rgba(224, 122, 95, 0.08);
        }

        .answer-text {
            color: var(--ink);
            font-size: 1.02rem;
            line-height: 1.8;
            white-space: pre-wrap;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background: #fff;
            border: 2px dashed rgba(61, 90, 128, 0.5);
            border-radius: 14px;
            padding: 1.2rem;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: rgba(61, 90, 128, 0.9);
            background: #f8f5ee;
        }

        /* Input styling */
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 1px solid #d7cfc4;
            padding: 0.85rem 1rem;
            font-size: 1rem;
            background: #fff;
        }

        .stTextInput>div>div>input:focus {
            border-color: var(--ocean);
            box-shadow: 0 0 0 3px rgba(61, 90, 128, 0.15);
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, var(--ocean), #23324f);
            color: #fdfbf7;
            border: none;
            border-radius: 999px;
            padding: 0.8rem 1.8rem;
            font-weight: 600;
            font-size: 1rem;
            letter-spacing: 0.02em;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 20px rgba(35, 50, 79, 0.25);
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 26px rgba(35, 50, 79, 0.3);
        }

        /* Expander */
        .streamlit-expanderHeader {
            background: #fff;
            border-radius: 12px;
            border: 1px solid var(--line);
            font-weight: 600;
        }

        /* Success messages */
        .stSuccess {
            background: #e9f4ef;
            border-radius: 10px;
            border: 1px solid rgba(129, 178, 154, 0.6);
        }

        @keyframes rise {
            from { transform: translateY(12px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        /* Hide branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="main-header">
            <div class="main-title">Healthcare Insurance Assistant</div>
            <div class="main-subtitle">Upload a policy PDF. Ask real questions. Get cited answers.</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Information section
    st.markdown("""
        <div class="info-section">
            <div class="info-title">How It Works</div>
            <p class="info-text">
                Drop in a healthcare insurance PDF. We build a private index in your session and answer
                using only that document, with clear citations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Upload and question input in single card
    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="section-title">Upload PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a healthcare insurance PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    upload_signature = None
    if uploaded_file is not None:
        upload_signature = (uploaded_file.name, uploaded_file.size)

    if upload_signature and st.session_state.get("upload_signature") != upload_signature:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            vector_store, pages, chunks, error = _create_vector_store_from_upload(uploaded_file)

        if error:
            st.error(f"Failed to process PDF: {error}")
            st.session_state.pop("vector_store", None)
            st.session_state.pop("upload_signature", None)
        else:
            st.session_state["vector_store"] = vector_store
            st.session_state["upload_signature"] = upload_signature
            st.success(f"Ready: {uploaded_file.name} ({pages} pages, {chunks} chunks)")
    
    # Add divider
    st.markdown('<div style="margin: 1.5rem 0; border-top: 1px solid #e9ecef;"></div>', unsafe_allow_html=True)
    
    question = ""
    ask_button = False
    if "vector_store" in st.session_state:
        # Question input section
        st.markdown('<div class="section-title">Ask Your Question</div>', unsafe_allow_html=True)
        question = st.text_input(
            "Enter your question about the uploaded document",
            placeholder="e.g., What services are covered under preventive care?",
            label_visibility="collapsed"
        )
        
        ask_button = st.button("Get Answer", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if ask_button and question:

        with st.spinner("Analyzing documents and generating response..."):
            answer, sources = _query_uploaded_document(question, st.session_state["vector_store"])
            
            # Display answer
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-container"><div class="answer-text">{answer}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display sources
            if sources:
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Source References</div>', unsafe_allow_html=True)
                for source in sources:
                    with st.expander(f"{source['id']}: {source['filename']} - Page {source['page']}", expanded=False):
                        st.markdown("**Document Excerpt:**")
                        st.write(f"_{source['content_preview']}_")
                        st.markdown("---")
                        st.markdown(f"**Reference:** {source['filename']}, Page {source['page']}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Example questions
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    with st.expander("Example Questions", expanded=False):
        st.markdown("""
        **Coverage & Benefits:**
        - What services are covered under preventive care?
        - What are essential health benefits?
        - What is the difference between in-network and out-of-network providers?
        
        **Costs & Payments:**
        - What is a deductible?
        - How does copayment work?
        - What are the differences between cost-sharing options?
        
        **Specific Programs:**
        - Explain AI/AN limited cost sharing coverage
        - What is Zero Cost Sharing coverage?
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System information
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    with st.expander("System Information", expanded=False):
        st.markdown(f"""
        **Technology Stack:**
        - Embedding Model: {config.embedding_model_name}
        - Language Model: {config.llm_model_name}
        - Storage: In-session FAISS index from uploaded PDF
        - Admin Page Required: No
        
        **Privacy & Security:**
        - All processing occurs locally on your machine
        - No data is sent to external servers
        - No subscription or cloud costs required
        """)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
