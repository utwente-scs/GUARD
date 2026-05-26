"""
Parser module for GUARD.

Each language-specific parser converts raw file lines into a standardised
list of ParsedLine objects. The detection logic in guard.py operates on
these objects regardless of the source language.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedLine:
    """Standardised representation of a single parsed configuration line.

    Attributes:
        line_number:      1-based line number in the original file.
        raw:              Original line text (for output/reporting).
        comment:          Extracted comment text (after #), or None.
        key:              Attribute/variable name, or None if not a key-value line.
        value:            Attribute value, or None.
        is_parameterized: True if the value is a variable reference that should
                          NOT be flagged as a hardcoded secret.
    """
    line_number: int
    raw: str
    comment: Optional[str] = None
    key: Optional[str] = None
    value: Optional[str] = None
    is_parameterized: bool = False


def get_parser(extension):
    """Return the appropriate parser module for a file extension.

    Args:
        extension: File extension including dot (e.g. '.yml', '.pp', '.tf').

    Returns:
        A module with a parse_file(filepath) function that returns List[ParsedLine].

    Raises:
        ValueError: If no parser exists for the given extension.
    """
    if extension in ('.yml', '.yaml'):
        from . import ansible_parser
        return ansible_parser
    elif extension == '.pp':
        from . import puppet_parser
        return puppet_parser
    elif extension == '.tf':
        from . import terraform_parser
        return terraform_parser
    elif extension == '.rb':
        from . import chef_parser
        return chef_parser
    else:
        raise ValueError(f"No parser available for extension: {extension}")