"""Sphinx configuration for repository documentation."""

from __future__ import annotations

project = "IRP Meta-learning"
author = "Francesco Giuseppe Remondi"

extensions: list[str] = []
templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
