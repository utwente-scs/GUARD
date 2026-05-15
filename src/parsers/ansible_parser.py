"""
Ansible (YAML) parser for GUARD.

Extracts the line-parsing logic from the original guard_file() function.
Converts each line into a ParsedLine with key, value, comment, and
parameterization info so the detection logic can operate uniformly.
"""

from . import ParsedLine


def parse_file(filepath):
    """Parse an Ansible YAML file into a list of ParsedLine objects.

    Preserves the original GUARD parsing behaviour:
    - Skips YAML document markers (---) and empty lines
    - Splits comments on '#'
    - Strips YAML list markers (-) from non-comment lines
    - Splits key:value on first ':'
    - Treats {{ }} in value as parameterized (variable reference)

    Args:
        filepath: Path to the .yml/.yaml file.

    Returns:
        List of ParsedLine objects for lines that contain detectable content.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    parsed = []
    for line_number, raw in enumerate(lines, start=1):
        raw = raw.rstrip()

        # Skip YAML document markers and empty lines
        if '---' in raw or not raw.strip():
            continue

        comment = None
        code_part = raw

        # Extract comment
        if '#' in raw:
            parts = raw.split('#', 1)
            code_part = parts[0]
            comment = parts[1].lstrip() if len(parts) > 1 else ""
        else:
            # Strip YAML list markers (only when no comment — matches original behaviour)
            code_part = raw.lstrip('-').lstrip()

        # Try to parse key: value
        key = None
        value = None
        is_param = False

        if ':' in code_part:
            key, value = code_part.split(':', 1)
            key = key.strip()
            value = value.lstrip()
            is_param = '{{' in value

        # Only emit a ParsedLine if there's something to detect
        if comment is not None or key is not None:
            parsed.append(ParsedLine(
                line_number=line_number,
                raw=raw,
                comment=comment,
                key=key,
                value=value,
                is_parameterized=is_param,
            ))

    return parsed