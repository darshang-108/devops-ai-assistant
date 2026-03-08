"""
GitHub loader – fetches README, issues, and pull requests from a public
GitHub repository and returns LangChain ``Document`` objects compatible
with the rest of the ingestion pipeline.

Requires PyGithub: ``pip install PyGithub``
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from github import Github
from langchain_core.documents import Document

load_dotenv()

token = os.getenv("GITHUB_TOKEN")

if token:
    g = Github(token)
    print("Using authenticated GitHub API")
else:
    g = Github()
    print("Using unauthenticated GitHub API (low rate limit)")


def _parse_repo_url(repo_url: str) -> tuple[str, str]:
    """
    Extract (owner, repo_name) from a GitHub URL.

    Supports formats like:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - github.com/owner/repo
    """
    match = re.search(r"github\.com/([^/]+)/([^/.]+)", repo_url)
    if not match:
        raise ValueError(
            f"Could not parse GitHub owner/repo from URL: {repo_url}"
        )
    return match.group(1), match.group(2)


def load_github_repo(
    repo_url: str,
    max_issues: int = 20,
    token: Optional[str] = None,
) -> List[Document]:
    """
    Fetch documents from a public GitHub repository.

    Collects:
      A) README content
      B) Issue titles + descriptions (up to *max_issues*)
      C) Pull-request titles + descriptions (up to *max_issues*)

    Parameters
    ----------
    repo_url : str
        Full GitHub URL, e.g. ``https://github.com/tiangolo/fastapi``.
    max_issues : int
        Maximum number of issues *and* PRs to fetch (default 20 each).
    token : str | None
        GitHub personal-access token.  Falls back to the
        ``GITHUB_TOKEN`` environment variable if not provided.

    Returns
    -------
    List[Document]
        LangChain Document objects ready for chunking / embedding.
    """
    owner, repo_name = _parse_repo_url(repo_url)

    gh_token = token or os.getenv("GITHUB_TOKEN")
    gh = Github(gh_token) if gh_token else g

    repo = gh.get_repo(f"{owner}/{repo_name}")
    documents: List[Document] = []

    # ── A) README ────────────────────────────────────────────────────────
    try:
        readme = repo.get_readme()
        readme_text = readme.decoded_content.decode("utf-8", errors="replace")
        if readme_text.strip():
            documents.append(
                Document(
                    page_content=readme_text,
                    metadata={
                        "source": "github_readme",
                        "repo": repo_name,
                        "filename": readme.name,
                        "title": "README",
                    },
                )
            )
            print(f"  Loaded README ({len(readme_text)} chars)")
    except Exception as exc:
        print(f"  Could not fetch README: {exc}")

    # ── B) Top-level .md files (CONTRIBUTING, CHANGELOG, etc.) ─────────
    try:
        contents = repo.get_contents("")
        md_count = 0
        for item in contents:
            if (
                item.type == "file"
                and item.name.lower().endswith(".md")
                and item.name.upper() != "README.MD"
            ):
                try:
                    text = item.decoded_content.decode("utf-8", errors="replace")
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": "github_docs",
                                    "repo": repo_name,
                                    "filename": item.name,
                                    "title": item.name,
                                },
                            )
                        )
                        md_count += 1
                except Exception:
                    pass
        print(f"  Loaded {md_count} top-level .md files")
    except Exception as exc:
        print(f"  Could not fetch top-level files: {exc}")

    # ── C) docs/ or documentation/ folder (.md files, up to 30) ──────
    docs_loaded = 0
    for docs_dir_name in ("docs", "documentation"):
        try:
            dir_contents = repo.get_contents(docs_dir_name)
            queue = list(dir_contents)
            while queue and docs_loaded < 30:
                item = queue.pop(0)
                if item.type == "dir":
                    try:
                        queue.extend(repo.get_contents(item.path))
                    except Exception:
                        pass
                elif item.type == "file" and item.name.lower().endswith(".md"):
                    try:
                        text = item.decoded_content.decode("utf-8", errors="replace")
                        if text.strip():
                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": "github_docs",
                                        "repo": repo_name,
                                        "filename": item.path,
                                        "title": item.name,
                                    },
                                )
                            )
                            docs_loaded += 1
                    except Exception:
                        pass
        except Exception:
            pass  # folder doesn't exist, skip
    if docs_loaded:
        print(f"  Loaded {docs_loaded} docs from docs/documentation folder")

    # ── D) Issues ────────────────────────────────────────────────────────
    try:
        issues = repo.get_issues(state="open", sort="created", direction="desc")
    except Exception as exc:
        print(f"  Could not fetch issues: {exc}")
        issues = []
    issue_count = 0
    for issue in issues:
        if issue.pull_request is not None:
            continue  # skip PRs listed under issues
        if issue_count >= max_issues:
            break
        body = issue.body or ""
        content = f"Issue #{issue.number}: {issue.title}\n\n{body}"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "github_issue",
                    "repo": repo_name,
                    "filename": f"issue_{issue.number}.md",
                    "title": issue.title,
                },
            )
        )
        issue_count += 1
    print(f"  Loaded {issue_count} issues")

    # ── E) Pull Requests ─────────────────────────────────────────────────
    try:
        pulls = repo.get_pulls(state="open", sort="created", direction="desc")
    except Exception as exc:
        print(f"  Could not fetch pull requests: {exc}")
        pulls = []
    pr_count = 0
    for pr in pulls:
        if pr_count >= max_issues:
            break
        body = pr.body or ""
        content = f"PR #{pr.number}: {pr.title}\n\n{body}"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "github_pr",
                    "repo": repo_name,
                    "filename": f"pr_{pr.number}.md",
                    "title": pr.title,
                },
            )
        )
        pr_count += 1
    print(f"  Loaded {pr_count} pull requests")

    print(f"\nTotal GitHub documents: {len(documents)}")
    return documents


# ── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import argparse

    parser = argparse.ArgumentParser(description="Load docs from a GitHub repo")
    parser.add_argument("url", help="GitHub repo URL")
    parser.add_argument("--max-issues", type=int, default=20)
    args = parser.parse_args()

    print(f"Loading from: {args.url}\n")
    docs = load_github_repo(args.url, max_issues=args.max_issues)

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"{i}. [{meta['source']}] {meta.get('title', meta['filename'])}")
        print(f"   {doc.page_content[:120]}...")
        print()
