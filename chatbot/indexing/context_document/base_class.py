from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing text.
    
    Args:
        clean_whitespace (bool, optional): Clean whitespace. Defaults to True.
        clean_empty_lines (bool, optional): Clean empty lines. Defaults to True.
        clean_header_footer (bool, optional): Clean header and footer. Defaults to True.
        remove_urls (bool, optional): Remove URLs. Defaults to True.
        remove_html_tags (bool, optional): Remove HTML tags. Defaults to True.
        normalize_unicode (bool, optional): Normalize unicode. Defaults to False.
        custom_patterns (List[str], optional): List of custom patterns to remove. Defaults to [].
    """
    clean_whitespace: bool = True
    clean_empty_lines: bool = True
    clean_header_footer: bool = True
    remove_urls: bool = True
    remove_html_tags: bool = True
    normalize_unicode: bool = False
    custom_patterns: List[str] = field(default_factory=list)


@dataclass
class ExtractedContext:
    """
    Extracted context from a document.
    
    Args:
        document (str): Document to extract context from.
        context (str): Context extracted from the document.
    """
    document: str
    context: str


@dataclass
class ReconstructedChunk:
    """
    Reconstructed chunk from a document.
    
    Args:
        id (str): Unique identifier for the chunk (UUID).
        chunk (str): Chunk reconstructed from the document.
        document (Optional[str]): Document to construct chunk from.
    """
    id: str
    chunk: str
    document: Optional[str] = None
