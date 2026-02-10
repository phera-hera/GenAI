"""
Heuristic chunk quality filter (MVP).

Removes chunks that are obviously garbage:
- Bibliography/reference sections
- Image/figure placeholders
- Dots-only or whitespace-heavy noise
- Headers/footers/page numbers
- Very short chunks (< 15 words)
- Very long chunks (> 300 words — likely rambling)

Industry finding: Can remove up to 60.5% of noisy chunks.
"""

import re
import logging
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

# Patterns for structural garbage
BIBLIOGRAPHY_PATTERNS = [
    r"^\s*\[\d+\]\s+[A-Z][a-z]+",           # [1] Author Name...
    r"^\s*\d+\.\s+[A-Z][a-z]+.*et al",       # 1. Author et al.
    r"^\s*references?\s*$",                     # "References" heading
    r"^\s*bibliography\s*$",                    # "Bibliography" heading
    r"doi\.org|DOI:\s*10\.",                    # DOI links
    r"PMID:\s*\d+",                             # PubMed IDs
]

NOISE_PATTERNS = [
    r"^\s*\[?fig(ure)?\s*\d+\]?",             # Figure references
    r"^\s*\[?table\s*\d+\]?",                  # Table references (caption only)
    r"^[\s\.·•\-_=]{10,}$",                    # Dots, bullets, lines
    r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",      # Page numbers
    r"^[^\w]*$",                                 # No word characters at all
]

HEADER_FOOTER_PATTERNS = [
    r"journal\s+of\s+\w+",                     # Journal headers
    r"copyright\s+©?\s*\d{4}",                 # Copyright notices
    r"all\s+rights\s+reserved",                # Rights notices
    r"downloaded\s+from",                       # Download notices
]

# Compiled patterns
_bib_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in BIBLIOGRAPHY_PATTERNS]
_noise_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in NOISE_PATTERNS]
_header_re = [re.compile(p, re.IGNORECASE) for p in HEADER_FOOTER_PATTERNS]


def is_low_quality_chunk(node: BaseNode, min_words: int = 15, max_words: int = 300) -> bool:
    """
    Check if a chunk is structural garbage and should be filtered out.

    Args:
        node: The chunk node to evaluate
        min_words: Minimum word count threshold
        max_words: Maximum word count threshold

    Returns:
        True if chunk should be REMOVED
    """
    text = node.get_content().strip()

    # Length check: too short or too long
    word_count = len(text.split())
    if word_count < min_words or word_count > max_words:
        return True

    # Check section type in metadata
    chunk_type = (node.metadata.get("chunk_type", "") or "").lower()
    if chunk_type in ("references", "bibliography", "acknowledgements"):
        return True

    # Bibliography patterns
    bib_matches = sum(1 for p in _bib_re if p.search(text))
    if bib_matches >= 2:
        return True

    # Pure noise
    lines = text.strip().split("\n")
    noise_lines = sum(1 for line in lines if any(p.match(line.strip()) for p in _noise_re))
    if noise_lines / max(len(lines), 1) > 0.5:
        return True

    # Header/footer content
    if word_count < 30 and any(p.search(text) for p in _header_re):
        return True

    # High ratio of non-alphabetic characters (equations, symbols)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return True

    return False


def filter_chunks(nodes: list[BaseNode]) -> list[BaseNode]:
    """
    Filter out structural garbage chunks.

    Args:
        nodes: List of chunk nodes from parser

    Returns:
        Filtered list with obvious garbage removed
    """
    original_count = len(nodes)
    filtered = [n for n in nodes if not is_low_quality_chunk(n)]
    removed = original_count - len(filtered)

    if removed > 0:
        logger.info(f"Chunk filter: {original_count} → {len(filtered)} (removed {removed} chunks, {removed/original_count*100:.1f}%)")

    return filtered
