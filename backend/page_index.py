"""DocuMind — PageIndex (Hierarchical Document Tree)
Owner: Akhil (RAG Architecture)
Purpose: Builds a lightweight hierarchical tree from the section/heading metadata
         already stored on every chunk in the vector store.  The tree is used by
         retrieve_context() to let the LLM pick relevant sections BEFORE running
         the more expensive FAISS+BM25 search — mimicking how a human skims
         headings before reading pages.

Architecture:
  Document
    └── SectionNode  (one per unique section+heading within a source file)
          ├── summary    — first 400 chars of combined chunk text
          └── chunk_ids  — references back into the FAISS/BM25 index

The index is persisted to vector_data/page_index.json so it survives restarts
without needing to be rebuilt from scratch every time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SectionNode:
    source: str
    section: str       # classified type: "methodology", "results", "general", …
    heading: str       # raw heading text from the chunk (may be empty)
    summary: str       # first 400 chars of combined text for this section
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "section": self.section,
            "heading": self.heading,
            "summary": self.summary,
            "chunk_ids": self.chunk_ids,
        }

    @staticmethod
    def from_dict(d: dict) -> "SectionNode":
        return SectionNode(
            source=d["source"],
            section=d["section"],
            heading=d.get("heading", ""),
            summary=d.get("summary", ""),
            chunk_ids=d.get("chunk_ids", []),
        )


@dataclass
class DocumentNode:
    source: str
    # section_key → SectionNode
    # key = f"{section}|{heading[:40]}" for uniqueness within a document
    sections: dict[str, SectionNode] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """First 400 chars of the first section for a quick document overview."""
        for node in self.sections.values():
            if node.summary:
                return node.summary[:400]
        return ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "sections": {k: v.to_dict() for k, v in self.sections.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "DocumentNode":
        node = DocumentNode(source=d["source"])
        for k, v in d.get("sections", {}).items():
            node.sections[k] = SectionNode.from_dict(v)
        return node


# ---------------------------------------------------------------------------
# PageIndex
# ---------------------------------------------------------------------------

class PageIndex:
    """Hierarchical index: source → section → chunk_ids."""

    def __init__(self) -> None:
        self._docs: dict[str, DocumentNode] = {}   # source → DocumentNode

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_documents(self, documents: list[dict]) -> None:
        """Rebuild the entire tree from a flat list of chunk dicts.

        Called by rag.py after every ingest or delete.  Each chunk must have at
        minimum: source, text.  Optional but used: section, heading, chunk_id.
        """
        self._docs.clear()

        for chunk in documents:
            source  = chunk.get("source", "unknown")
            section = chunk.get("section") or "general"
            heading = chunk.get("heading") or ""
            text    = chunk.get("text", "")
            cid     = chunk.get("chunk_id") or (source + "_" + str(chunk.get("page", 0)))

            if source not in self._docs:
                self._docs[source] = DocumentNode(source=source)

            doc_node = self._docs[source]
            sec_key  = f"{section}|{heading[:40]}"

            if sec_key not in doc_node.sections:
                doc_node.sections[sec_key] = SectionNode(
                    source=source,
                    section=section,
                    heading=heading,
                    summary="",
                    chunk_ids=[],
                )

            sec_node = doc_node.sections[sec_key]
            sec_node.chunk_ids.append(cid)

            # Accumulate summary text (first 400 chars total)
            if len(sec_node.summary) < 400:
                remaining = 400 - len(sec_node.summary)
                sec_node.summary += text[:remaining]

        logger.info(
            "PageIndex built: %d documents, %d section nodes",
            len(self._docs), sum(len(d.sections) for d in self._docs.values()),
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_section_summaries(self, chat_id: str = None,
                               documents: list[dict] | None = None) -> list[dict]:
        """Return a flat list of section summary dicts for LLM-guided selection.

        Format: [{source, section, heading, summary}, ...]

        chat_id isolation: when provided, only sections from chunks that belong
        to that chat (or to global / un-tagged docs) are returned.  Because the
        tree is built without chat_id awareness, we use the passed *documents*
        list (same as vector_store.documents) to filter which sources are visible.

        If documents is None, all sources are returned (admin / no-auth mode).
        """
        visible_sources: set[str] | None = None
        if chat_id is not None and documents is not None:
            visible_sources = set()
            for doc in documents:
                doc_chat = doc.get("chat_id")
                if doc_chat and doc_chat != chat_id:
                    continue
                visible_sources.add(doc.get("source", ""))

        result = []
        for source, doc_node in self._docs.items():
            if visible_sources is not None and source not in visible_sources:
                continue
            for sec_node in doc_node.sections.values():
                result.append({
                    "source":  sec_node.source,
                    "section": sec_node.section,
                    "heading": sec_node.heading,
                    "summary": sec_node.summary[:200],  # keep prompt short
                })
        return result

    def get_chunk_ids_for_sections(
        self, sections: list[str], chat_id: str = None,
        documents: list[dict] | None = None,
    ) -> set[str]:
        """Return chunk_ids that belong to any of the given section types."""
        visible_sources: set[str] | None = None
        if chat_id is not None and documents is not None:
            visible_sources = {
                doc.get("source", "") for doc in documents
                if not doc.get("chat_id") or doc.get("chat_id") == chat_id
            }

        section_set = set(sections)
        ids: set[str] = set()
        for source, doc_node in self._docs.items():
            if visible_sources is not None and source not in visible_sources:
                continue
            for sec_node in doc_node.sections.values():
                if sec_node.section in section_set:
                    ids.update(sec_node.chunk_ids)
        return ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize to JSON file."""
        try:
            data = {src: node.to_dict() for src, node in self._docs.items()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("PageIndex saved to %s (%d docs)", path, len(self._docs))
        except Exception as e:
            logger.error("PageIndex save failed: %s", e)

    def load(self, path: str) -> bool:
        """Deserialize from JSON file. Returns True on success."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._docs = {src: DocumentNode.from_dict(v) for src, v in data.items()}
            logger.info(
                "PageIndex loaded from %s: %d docs, %d sections",
                path, len(self._docs),
                sum(len(d.sections) for d in self._docs.values()),
            )
            return True
        except FileNotFoundError:
            logger.info("No page_index.json found — will build on first ingest")
            return False
        except Exception as e:
            logger.warning("PageIndex load failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of section nodes across all documents."""
        return sum(len(d.sections) for d in self._docs.values())

    def __repr__(self) -> str:
        return f"PageIndex(docs={len(self._docs)}, sections={len(self)})"
