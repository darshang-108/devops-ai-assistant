"""
Multi-repo index builder – loads documents from local markdown files,
multiple GitHub repositories, and their local clones, then chunks
everything and builds (or updates) the FAISS vector index.

Usage
-----
    # Full rebuild from all configured repos
    python -m ingestion.build_index

    # Add a single new repo without full rebuild
    python -m ingestion.build_index --add-repo https://github.com/owner/repo \
                                     --repo-path ./repos/myrepo \
                                     --label myrepo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# Ensure project root is on sys.path so relative imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.load_documents import load_documents
from ingestion.chunk_documents import chunk_documents
from ingestion.github_loader import load_github_repo
from ingestion.code_loader import load_codebase
from vector_store.vector_db import create_vector_store, load_vector_store

RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"

# ── Multi-repo configuration ────────────────────────────────────────────────
GITHUB_REPOS = [
    {"url": "https://github.com/tiangolo/fastapi",     "max_issues": 20},
    {"url": "https://github.com/docker/compose",        "max_issues": 20},
    {"url": "https://github.com/kubernetes/kubernetes", "max_issues": 20},
]

LOCAL_REPOS = [
    {"path": "./repos/fastapi",    "label": "fastapi",       "allowed_dirs": None},
    {"path": "./repos/compose",    "label": "docker-compose", "allowed_dirs": None},
    {"path": "./repos/kubernetes", "label": "kubernetes",
     "allowed_dirs": ["pkg/api", "staging/src/k8s.io", "cmd"]},
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _tag_docs(docs: List[Document], repo_label: str) -> None:
    """Stamp every document's metadata with a ``repo`` label."""
    for doc in docs:
        doc.metadata["repo"] = repo_label


def _load_github_repos() -> List[Document]:
    """Iterate over GITHUB_REPOS, collecting docs with error isolation."""
    all_docs: List[Document] = []
    for repo_cfg in GITHUB_REPOS:
        url = repo_cfg["url"]
        max_issues = repo_cfg.get("max_issues", 20)
        label = url.rstrip("/").split("/")[-1]
        print(f"\n📡  Fetching GitHub data for {label} ({url}) ...")
        try:
            docs = load_github_repo(url, max_issues=max_issues)
            _tag_docs(docs, label)
            all_docs.extend(docs)
            print(f"✅  {label}: loaded {len(docs)} GitHub documents")
        except Exception as exc:
            print(f"⚠️  {label}: GitHub fetch failed — {exc}")
    return all_docs


def _load_local_repos() -> List[Document]:
    """Iterate over LOCAL_REPOS, collecting code docs with error isolation."""
    all_docs: List[Document] = []
    for repo_cfg in LOCAL_REPOS:
        label = repo_cfg["label"]
        repo_path = Path(repo_cfg["path"])
        allowed_dirs = repo_cfg.get("allowed_dirs")

        if not repo_path.exists():
            print(f"⚠️  {label}: path {repo_path} not found — skipping")
            continue

        print(f"\n📂  Loading code from {label} ({repo_path}) ...")
        try:
            if allowed_dirs:
                docs: List[Document] = []
                for sub in allowed_dirs:
                    sub_path = repo_path / sub
                    if sub_path.exists():
                        docs.extend(load_codebase(sub_path))
                    else:
                        print(f"   ↳ sub-dir {sub} not found — skipped")
            else:
                docs = load_codebase(repo_path)

            _tag_docs(docs, label)
            all_docs.extend(docs)
            print(f"✅  {label}: loaded {len(docs)} code files")
        except Exception as exc:
            print(f"⚠️  {label}: code loading failed — {exc}")
    return all_docs


# ── Full rebuild ─────────────────────────────────────────────────────────────
def build_index() -> None:
    """
    Build (or rebuild) the FAISS vector index from **all** configured sources.

    Steps:
      1. Local markdown/txt docs from ``data/raw_docs/``
      2. GitHub docs (README, issues, PRs, top-level .md, docs/) per repo
      3. Local code files per repo (scoped to allowed_dirs when set)
      4. Chunk all documents
      5. Build FAISS index
    """
    print("=" * 60)
    print("🔨  FULL INDEX BUILD — starting")
    print("=" * 60)

    # 1. Local docs
    print(f"\n📄  Loading local documents from {RAW_DOCS_DIR} ...")
    try:
        local_docs = load_documents(RAW_DOCS_DIR)
        print(f"✅  Loaded {len(local_docs)} local documents")
    except Exception as exc:
        print(f"⚠️  Failed to load local docs — {exc}")
        local_docs = []

    # 2. GitHub repos
    github_docs = _load_github_repos()

    # 3. Local code repos
    code_docs = _load_local_repos()

    # 4. Combine
    all_docs = local_docs + github_docs + code_docs
    print(f"\n📊  Total documents collected: {len(all_docs)}")

    if not all_docs:
        print("❌  No documents found — aborting index build")
        return

    # 5. Chunk
    print("\n✂️  Chunking documents ...")
    chunks = chunk_documents(all_docs)
    print(f"✅  Created {len(chunks)} chunks")

    # 6. Build FAISS index
    print("\n🗄️  Building FAISS vector index ...")
    create_vector_store(chunks)

    print("\n" + "=" * 60)
    print("🎉  Index rebuilt successfully")
    print("=" * 60)


# ── Incremental add ─────────────────────────────────────────────────────────
def add_repo(
    github_url: str,
    repo_path: str,
    label: str,
    max_issues: int = 20,
) -> None:
    """
    Add a **single** new repository to the existing FAISS index without
    performing a full rebuild.

    Parameters
    ----------
    github_url : str
        GitHub URL of the new repo.
    repo_path : str
        Local clone path for code indexing.
    label : str
        Short label used in metadata tagging.
    max_issues : int
        Max issues/PRs to fetch from GitHub.
    """
    print("=" * 60)
    print(f"➕  INCREMENTAL ADD — {label}")
    print("=" * 60)

    new_docs: List[Document] = []

    # GitHub docs
    print(f"\n📡  Fetching GitHub data from {github_url} ...")
    try:
        gh_docs = load_github_repo(github_url, max_issues=max_issues)
        _tag_docs(gh_docs, label)
        new_docs.extend(gh_docs)
        print(f"✅  Loaded {len(gh_docs)} GitHub documents")
    except Exception as exc:
        print(f"⚠️  GitHub fetch failed — {exc}")

    # Local code
    rp = Path(repo_path)
    if rp.exists():
        print(f"\n📂  Loading code from {repo_path} ...")
        try:
            code_docs = load_codebase(rp)
            _tag_docs(code_docs, label)
            new_docs.extend(code_docs)
            print(f"✅  Loaded {len(code_docs)} code files")
        except Exception as exc:
            print(f"⚠️  Code loading failed — {exc}")
    else:
        print(f"⚠️  Repo path {repo_path} not found — skipping code")

    if not new_docs:
        print("❌  No new documents loaded — nothing to add")
        return

    # Chunk new docs
    print("\n✂️  Chunking new documents ...")
    new_chunks = chunk_documents(new_docs)
    print(f"✅  Created {len(new_chunks)} new chunks")

    # Load existing index and append
    print("\n🗄️  Loading existing FAISS index ...")
    try:
        import json
        import numpy as np
        import faiss
        from embeddings.embedder import generate_embeddings
        from vector_store.vector_db import INDEX_DIR

        index, metadata = load_vector_store()
        print(f"   Existing index: {index.ntotal} vectors")

        # Generate embeddings for new chunks
        new_embeddings = generate_embeddings(new_chunks)

        # Append to index
        index.add(new_embeddings.astype(np.float32))

        # Append metadata
        for c in new_chunks:
            metadata.append({
                "text": c.page_content,
                "source": c.metadata.get("source"),
                "file_path": c.metadata.get("file_path"),
            })

        # Save updated index
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
        (INDEX_DIR / "index_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )
        print(f"✅  Index updated: {index.ntotal} total vectors")
    except FileNotFoundError:
        print("⚠️  No existing index found — performing full build for this repo")
        create_vector_store(new_chunks)

    print("\n" + "=" * 60)
    print(f"🎉  {label} added to index successfully")
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or update the FAISS vector index from multiple repos",
    )
    parser.add_argument(
        "--add-repo",
        type=str,
        default=None,
        help="GitHub URL of a single repo to add incrementally",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=None,
        help="Local clone path for the repo being added",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Short label for the repo being added",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        help="Max issues/PRs to fetch from GitHub (default: 20)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Clear the embedding cache before rebuilding (forces full re-embed)",
    )
    args = parser.parse_args()

    if args.clear_cache:
        from embeddings.embedding_cache import clear_cache
        clear_cache()

    if args.add_repo:
        if not args.repo_path or not args.label:
            parser.error("--add-repo requires --repo-path and --label")
        add_repo(
            github_url=args.add_repo,
            repo_path=args.repo_path,
            label=args.label,
            max_issues=args.max_issues,
        )
    else:
        build_index()
