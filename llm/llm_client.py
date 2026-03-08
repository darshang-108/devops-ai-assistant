"""
LLM answer generation module – retrieves relevant context from the vector
store and sends it alongside the user question to a local Ollama Mistral model.

Uses the ``ollama`` Python package for direct communication.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import ollama
from dotenv import load_dotenv
from retrieval.retriever import retrieve_documents, format_context

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


# ── Prompt builder ───────────────────────────────────────────────────────────
def _build_prompt(question: str, context: str) -> str:
    """Build a single prompt string with instructions, context, and question."""
    return f"""You are a senior DevOps and software engineering assistant \
with deep expertise in FastAPI, Docker, Kubernetes, CI/CD pipelines, \
and cloud infrastructure.

## CONTEXT CHUNKS
The following chunks were retrieved from engineering repositories. \
Each chunk is labeled with its source file and repo:

{context}

## QUESTION
{question}

## REASONING STRATEGY
1. Read ALL context chunks carefully — the answer may span multiple chunks
2. Synthesize across chunks into one coherent response
3. Include relevant code examples or YAML/config snippets from the context
4. If context is partial, supplement with your DevOps expertise and clearly \
label it: "Note: supplemented with general DevOps knowledge:"
5. Only say context is insufficient if the topic is completely absent

## ANSWER FORMAT — STRICT RULES
You MUST always structure your answer using this format:

- Start with one short summary sentence (max 20 words)
- Then a blank line
- Then use bullet points OR numbered steps for all details
- Use **bold** for key terms, commands, and file names
- Use backtick code blocks for any code, YAML, commands, or file paths
- Every 2-3 bullet points add a blank line for breathing room
- If the answer has multiple aspects (what it is, how to use it, example) \
use ## subheadings to separate them
- Maximum 2 sentences in a row without a bullet, number, or code block
- NEVER write more than 3 sentences as a plain paragraph
- End with: "**Sources:** file1, file2" on its own line

Answer:"""


# ── Ollama call ──────────────────────────────────────────────────────────
def _ask_ollama(question: str, context: str) -> str:
    """Send the question + context to the local Ollama Mistral model."""
    prompt = _build_prompt(question, context)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response["message"]["content"]


# ── Public API ───────────────────────────────────────────────────────────────
def generate_answer(query: str, top_k: int = 15) -> str:
    """
    End-to-end: retrieve relevant chunks for *query* and generate an
    LLM-grounded answer via local Ollama Mistral.

    Parameters
    -----------
    query : str
        The user's natural-language question.
    top_k : int
        Number of context chunks to retrieve.

    Returns
    -------
    str
        The generated answer with sources.
    """
    # 1. Retrieve relevant chunks
    results = retrieve_documents(query, top_k=top_k)

    if not results:
        return "No relevant documents found. Please ingest documents first."

    # 2. Debug logging – show which documents were retrieved
    sources = list(dict.fromkeys(r["source"] for r in results))
    print("\nRetrieved Documents:")
    for i, src in enumerate(sources, 1):
        print(f"  {i}. {src}")
    print()

    # 3. Format context
    context = format_context(results)

    # 4. Generate answer
    print(f"Sending query to Ollama ({OLLAMA_MODEL}) with {len(results)} context chunks...")
    answer = _ask_ollama(query, context)

    # 5. Append unique sources to the response
    sources_section = "\n\nSources:\n" + "\n".join(sources)

    return answer + sources_section


# ── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    query = "How does authentication work?"
    print(f"Query: {query}\n")

    answer = generate_answer(query)
    print(f"Answer:\n{answer}")
