"""
Multi-repo configuration – centralises all repository definitions, supported
file types, and directory exclusions for the ingestion pipeline.
"""

from __future__ import annotations

# ── DevOps repositories to ingest ────────────────────────────────────────────
DEVOPS_REPOS = [
    {
        "url": "https://github.com/tiangolo/fastapi",
        "max_issues": 20,
        "local_path": "./repos/fastapi",
        "label": "fastapi",
        "allowed_dirs": None,
    },
    {
        "url": "https://github.com/docker/compose",
        "max_issues": 20,
        "local_path": "./repos/compose",
        "label": "docker-compose",
        "allowed_dirs": None,
    },
    {
        "url": "https://github.com/kubernetes/kubernetes",
        "max_issues": 20,
        "local_path": "./repos/kubernetes",
        "label": "kubernetes",
        "allowed_dirs": ["pkg/api", "staging/src/k8s.io", "cmd"],
    },
]

# ── Supported source-code / config extensions ────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".go", ".rs",
    ".yaml", ".yml", ".json", ".sh",
    ".dockerfile",
}

# Filenames that should be treated as supported even without an extension
SUPPORTED_FILENAMES = {"Dockerfile"}

# ── Directories to skip during code traversal ────────────────────────────────
IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", "tests",
    ".terraform", ".kube", "vendor", "dist", "build",
}
