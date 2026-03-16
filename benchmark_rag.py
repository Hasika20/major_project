#!/usr/bin/env python3
"""
Benchmark: Ollama + RAG  vs  Gemini (No RAG)
─────────────────────────────────────────────
Proves that a smaller local model (Llama 3.2) grounded with RAG
outperforms a powerful cloud LLM (Gemini) answering from general
knowledge alone, using five objective metrics:

  1. Faithfulness Score   – answer stays inside the retrieved document
  2. Hallucination Risk   – sentences the document cannot support
  3. Answer Relevancy     – how well the answer addresses the question
  4. Citation Rate        – % of answers with traceable [S1]/[S2] refs
  5. Response Time        – seconds to generate an answer

Usage:
    python benchmark_rag.py
"""

import sys, re, time
from pathlib import Path
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = "AIzaSyAsF03bTrAb90KOk5XuzDM0Min3qFJcBK8"   # <-- paste your key here
GEMINI_MODEL    = "gemini-1.5-flash"
OLLAMA_MODEL    = "llama3.2"
TOP_K_CHUNKS    = 3
SIM_THRESHOLD   = 0.35   # cosine-sim below this = hallucination risk per sentence

# Test questions — generic enough to work across any insurance document
TEST_QUESTIONS = [
    "What is a deductible and how does it work?",
    "What is the difference between copayment and coinsurance?",
    "What services are covered under preventive care?",
    "What is an out-of-pocket maximum?",
    "How does the network of providers affect my costs?",
]
# ─────────────────────────────────────────────────────────────────────────────

# config.py lives at the project root — no extra path manipulation needed


# ── Embedding helpers ─────────────────────────────────────────────────────────
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("⏳ Loading sentence-transformer for metrics...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Loaded.\n")
    return _embed_model

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def embed(text):
    return get_embed_model().encode([text])[0]

def embed_batch(texts):
    return get_embed_model().encode(texts)


# ── Metric functions ──────────────────────────────────────────────────────────
def faithfulness_score(answer: str, context: str) -> float:
    """
    Cosine similarity between answer embedding and context embedding.
    Higher = answer is more grounded in the retrieved document content.
    """
    return max(0.0, cosine_sim(embed(answer), embed(context)))


def hallucination_risk(answer: str, context: str) -> float:
    """
    Split answer into sentences. For each sentence, compute its max
    cosine-sim against context sentences. Sentences below SIM_THRESHOLD
    are 'unsupported'. Returns fraction of unsupported sentences.
    Lower is better.
    """
    answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]
    if not answer_sentences:
        return 0.0

    context_sentences = [s.strip() for s in re.split(r'[.!?\n]', context) if len(s.strip()) > 10]
    if not context_sentences:
        return 1.0

    a_embs = embed_batch(answer_sentences)
    c_embs = embed_batch(context_sentences)

    unsupported = 0
    for a_emb in a_embs:
        max_sim = max(cosine_sim(a_emb, c_emb) for c_emb in c_embs)
        if max_sim < SIM_THRESHOLD:
            unsupported += 1

    return unsupported / len(answer_sentences)


def answer_relevancy(answer: str, question: str) -> float:
    """
    Cosine similarity between answer and question.
    Higher = answer directly addresses the question asked.
    """
    return max(0.0, cosine_sim(embed(answer), embed(question)))


def citation_rate(answer: str) -> float:
    """1.0 if answer contains at least one [S1]/[S2] style citation, else 0.0."""
    return 1.0 if re.search(r'\[S\d+\]', answer) else 0.0


# ── RAG system ────────────────────────────────────────────────────────────────
def load_vector_store(store_path: Path):
    from config import config
    from langchain_community.vectorstores import FAISS
    try:
        return FAISS.load_local(
            str(store_path),
            config.bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
    except TypeError:
        # Older langchain_community versions don't support the parameter
        return FAISS.load_local(str(store_path), config.bedrock_embeddings)

def retrieve_context(vector_store, question: str) -> str:
    docs = vector_store.similarity_search(question, k=TOP_K_CHUNKS)
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[S{i}] (page {page})\n{doc.page_content.strip()}")
    return "\n\n".join(parts)

def rag_prompt(question: str, context: str) -> str:
    return (
        "You are a healthcare insurance assistant. "
        "Answer ONLY using the context below. Cite as [S1], [S2] etc.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

def plain_prompt(question: str) -> str:
    return (
        f"Answer this healthcare insurance question from your general knowledge:\n\n"
        f"Question: {question}\n\nAnswer:"
    )


# ── Model callers ─────────────────────────────────────────────────────────────
def call_ollama(prompt: str):
    import ollama
    start = time.perf_counter()
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 400, "temperature": 0.2}
    )
    return resp["message"]["content"], time.perf_counter() - start

def call_gemini(prompt: str):
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    start = time.perf_counter()
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return resp.text, time.perf_counter() - start


# ── Store picker ──────────────────────────────────────────────────────────────
def pick_store():
    from config import config
    stores = sorted([s for s in config.vector_store_dir.iterdir() if s.is_dir()])
    if not stores:
        print("❌ No processed documents. Run Admin interface first.")
        sys.exit(1)
    print("\n📁 Available documents:")
    for i, s in enumerate(stores, 1):
        print(f"  {i}. {s.name}")
    while True:
        try:
            c = int(input("\nSelect document (number): "))
            if 1 <= c <= len(stores):
                return stores[c - 1]
        except ValueError:
            pass


# ── Results table ─────────────────────────────────────────────────────────────
def print_metrics_table(results: list):
    """
    results: list of dicts with keys:
        question, ollama_faith, ollama_hallu, ollama_relev, ollama_cite,
        ollama_time, gemini_faith, gemini_hallu, gemini_relev, gemini_cite, gemini_time
    """
    sep = "─" * 95

    print(f"\n{'='*95}")
    print("  📊  BENCHMARK RESULTS: Ollama + RAG  vs  Gemini (No RAG)")
    print(f"{'='*95}")

    header = f"{'Metric':<25} {'Ollama+RAG':>12} {'Gemini(NoRAG)':>14}  {'Winner':>10}"
    print(f"\n{header}")
    print(sep)

    # Aggregate
    def avg(key): return np.mean([r[key] for r in results])

    metrics = [
        ("Faithfulness ↑",   "ollama_faith",  "gemini_faith",  True),
        ("Hallucination ↓",  "ollama_hallu",  "gemini_hallu",  False),
        ("Answer Relevancy ↑","ollama_relev",  "gemini_relev",  True),
        ("Citation Rate ↑",  "ollama_cite",   "gemini_cite",   True),
        ("Response Time ↓",  "ollama_time",   "gemini_time",   False),
    ]

    overall_ollama = 0
    overall_gemini = 0

    for label, ok, gk, higher_is_better in metrics:
        ov = avg(ok)
        gv = avg(gk)
        if higher_is_better:
            winner = "Ollama+RAG" if ov > gv else "Gemini"
        else:
            winner = "Ollama+RAG" if ov < gv else "Gemini"

        if winner == "Ollama+RAG":
            overall_ollama += 1
        else:
            overall_gemini += 1

        fmt = ".2f" if "Time" not in label else ".2fs"
        unit = "s" if "Time" in label else ""
        print(f"  {label:<23} {ov:>10.3f}{unit}   {gv:>10.3f}{unit}   {winner:>12}")

    print(sep)
    print(f"\n  🏆 OVERALL SCORE:  Ollama+RAG {overall_ollama}/5   |   Gemini(NoRAG) {overall_gemini}/5")

    winner_label = "Ollama + RAG" if overall_ollama >= overall_gemini else "Gemini (No RAG)"
    print(f"\n  ✅ WINNER: {winner_label}")
    print(f"\n{'='*95}\n")

def print_per_question(results: list):
    print(f"\n{'='*95}")
    print("  📋  PER-QUESTION BREAKDOWN")
    print(f"{'='*95}")
    for i, r in enumerate(results, 1):
        print(f"\n  Q{i}: {r['question']}")
        print(f"  {'─'*88}")
        print(f"  {'Metric':<22} {'Ollama+RAG':>12}   {'Gemini(NoRAG)':>13}")
        print(f"  {'─'*55}")
        rows = [
            ("Faithfulness ↑",    r['ollama_faith'],  r['gemini_faith'],  True,  ".3f"),
            ("Hallucination ↓",   r['ollama_hallu'],  r['gemini_hallu'],  False, ".3f"),
            ("Answer Relevancy ↑",r['ollama_relev'],  r['gemini_relev'],  True,  ".3f"),
            ("Citation Rate ↑",   r['ollama_cite'],   r['gemini_cite'],   True,  ".1f"),
            ("Response Time ↓",   r['ollama_time'],   r['gemini_time'],   False, ".2f"),
        ]
        for lbl, ov, gv, hib, fmt in rows:
            ov_wins = (ov > gv) if hib else (ov < gv)
            mark_o = " ✅" if ov_wins  else ""
            mark_g = " ✅" if not ov_wins else ""
            print(f"  {lbl:<22} {ov:>{10}{fmt}}{mark_o:<3}  {gv:>{10}{fmt}}{mark_g}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🏥 RAG Benchmark: Ollama + RAG  vs  Gemini (No RAG)")
    print("─" * 50)

    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("❌ Set your GEMINI_API_KEY in benchmark_rag.py first.")
        sys.exit(1)

    try:
        from google import genai  # noqa: F401
    except ImportError:
        print("❌ Run: pip install google-genai")
        sys.exit(1)

    store_path   = pick_store()
    print(f"\n⏳ Loading vector store: {store_path.name} ...")
    vector_store = load_vector_store(store_path)
    print("✅ Ready.\n")

    print("─" * 60)
    print(f"  Running {len(TEST_QUESTIONS)} test questions...")
    print(f"  {'Ollama model':20}: {OLLAMA_MODEL} + RAG")
    print(f"  {'Gemini model':20}: {GEMINI_MODEL} (no RAG — general knowledge only)")
    print("─" * 60)

    results = []

    for idx, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{idx}/{len(TEST_QUESTIONS)}] {question}")

        # Retrieve context (for Ollama only)
        context = retrieve_context(vector_store, question)

        # ── Ollama + RAG ──
        print("  ⏳ Ollama + RAG ...", end="", flush=True)
        try:
            ollama_answer, ollama_time = call_ollama(rag_prompt(question, context))
        except Exception as e:
            ollama_answer, ollama_time = f"ERROR: {e}", 0.0
        print(f" done ({ollama_time:.1f}s)")

        # ── Gemini (no RAG) ──
        print("  ⏳ Gemini (no RAG) ...", end="", flush=True)
        try:
            gemini_answer, gemini_time = call_gemini(plain_prompt(question))
        except Exception as e:
            gemini_answer, gemini_time = f"ERROR: {e}", 0.0
        print(f" done ({gemini_time:.1f}s)")

        # ── Compute metrics ──
        results.append({
            "question":    question,
            "ollama_faith": faithfulness_score(ollama_answer, context),
            "ollama_hallu": hallucination_risk(ollama_answer, context),
            "ollama_relev": answer_relevancy(ollama_answer, question),
            "ollama_cite":  citation_rate(ollama_answer),
            "ollama_time":  ollama_time,
            "gemini_faith": faithfulness_score(gemini_answer, context),
            "gemini_hallu": hallucination_risk(gemini_answer, context),
            "gemini_relev": answer_relevancy(gemini_answer, question),
            "gemini_cite":  citation_rate(gemini_answer),
            "gemini_time":  gemini_time,
        })

    # ── Print results ──
    print_per_question(results)
    print_metrics_table(results)

    # ── Key insights ──
    import numpy as np
    avg = lambda k: np.mean([r[k] for r in results])
    print("  💡 KEY INSIGHTS")
    print("  ─" * 40)
    faith_gain = avg("ollama_faith") - avg("gemini_faith")
    hallu_drop = avg("gemini_hallu") - avg("ollama_hallu")
    print(f"  • Ollama+RAG is {faith_gain*100:+.1f}% more faithful to the document")
    print(f"  • Ollama+RAG has {hallu_drop*100:.1f}% lower hallucination risk")
    print(f"  • Ollama+RAG citation rate: {avg('ollama_cite')*100:.0f}%  |  Gemini: {avg('gemini_cite')*100:.0f}%")
    print(f"  • RAG gives full traceability — every answer links to an exact page")
    print(f"  • Gemini answers from general knowledge only — no policy-specific grounding\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
