# RagMind — shared core used by the CLI (rag_cli.py) and the web app (web/app/main.py)
# Copyright (C) 2026  Seb Szmytka
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# See <https://www.gnu.org/licenses/> for details.
"""Shared RAG core: settings, Ollama client, text extraction, chunking,
prompt building and the retrieval pipeline. Storage lives with each front end
(ChromaDB in the CLI, a numpy index in the web app)."""

__version__ = "3.0.7"
