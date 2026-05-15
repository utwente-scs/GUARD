"""
Puppet (.pp) parser for GUARD.

Handles Puppet-specific syntax:
- Attribute assignment via '=>' (e.g. ensure => present)
- Variable assignment via '=' (e.g. $db_pass = 'admin123')
- Variable references: $var, ${var}, $module::var
- Comments with '#'
- Resource declarations, class definitions, and structural lines

Design note: Puppet and Ansible differ in two key ways relevant to parsing:
1. Puppet uses '=>' for attribute assignment, not ':'
2. Puppet uses '$variable' for parameterization, not '{{ variable }}'
"""

import re
from . import ParsedLine


# Patterns for detecting parameterized (variable/dynamic) values
_PARAM_FUNCTIONS = (
    'hiera(', 'hiera_hash(', 'hiera_array(',
    'lookup(',
    'template(', 'epp(',
    'fqdn_rand(',
    'generate(',
    'inline_template(',
)

# Lines that are purely structural and carry no security-relevant content
_STRUCTURAL_TOKENS = frozenset(('', '{', '}', ')', '(', ');', '},', ')]'))

# Variable assignment pattern: $var = value
_VAR_ASSIGN_RE = re.compile(r'^\s*(\$[\w:]+)\s*=\s*(.+?)\s*$')


def parse_file(filepath):
    """Parse a Puppet .pp file into a list of ParsedLine objects.

    Args:
        filepath: Path to the .pp file.

    Returns:
        List of ParsedLine objects for lines containing detectable content.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    parsed = []
    for line_number, raw in enumerate(lines, start=1):
        result = _parse_line(raw.rstrip(), line_number)
        if result is not None:
            parsed.append(result)
    return parsed


def _parse_line(text, line_number):
    """Parse a single Puppet line into a ParsedLine.

    Returns None for lines that have no detectable content (empty, structural).
    """
    stripped = text.strip()

    # Skip empty lines
    if not stripped:
        return None

    comment = None
    code_part = text

    # --- Extract comments ---
    # Split on first '#' (simplified; doesn't handle # inside strings,
    # but this matches GUARD's existing approach for Ansible)
    if '#' in text:
        parts = text.split('#', 1)
        code_part = parts[0]
        comment = parts[1].strip()

    clean = code_part.strip()

    # Skip structural-only lines (braces, parens, etc.)
    # Strip trailing commas/semicolons before checking
    clean_stripped = clean.rstrip(',').rstrip(';').strip()
    if clean_stripped in _STRUCTURAL_TOKENS:
        if comment:
            # Still return comment-only lines for suspicious comment detection
            return ParsedLine(line_number, text, comment=comment)
        return None

    # --- Try '=>' attribute assignment (most common in resource bodies) ---
    # e.g.  ensure  => present,
    #        owner   => 'root',
    #        mode    => $file_mode,
    if '=>' in code_part:
        key, value = code_part.split('=>', 1)
        key = key.strip()
        value = value.strip().rstrip(',').rstrip(';').strip()
        is_param = _is_parameterized(value)
        return ParsedLine(line_number, text, comment, key, value, is_param)

    # --- Try '=' variable assignment ---
    # e.g.  $db_password = 'admin123'
    var_match = _VAR_ASSIGN_RE.match(code_part)
    if var_match:
        key = var_match.group(1)     # e.g. '$db_password'
        value = var_match.group(2).rstrip(',').rstrip(';').strip()
        is_param = _is_parameterized(value)
        return ParsedLine(line_number, text, comment, key, value, is_param)

    # --- Non-assignment lines ---
    # Resource declarations (file { '/path':), class/define headers,
    # include statements, etc. Return for comment detection and full-line
    # RegEx scanning (e.g. a line might contain http:// or 0.0.0.0).
    if comment:
        return ParsedLine(line_number, text, comment=comment, value=clean)

    # Return with value=clean so RegEx patterns can scan the full line
    return ParsedLine(line_number, text, value=clean)


def _is_parameterized(value):
    """Check if a Puppet value is a variable reference or dynamic expression.

    Parameterized values should not be flagged as hardcoded secrets because
    their actual content is determined at runtime.

    Covers:
    - $variable, ${variable}, $module::variable
    - String interpolation: "text ${var} text"
    - Function calls: hiera(), lookup(), template(), etc.
    - undef (Puppet's equivalent of null/undefined)
    """
    stripped = value.strip().strip("'\"")

    # Direct variable reference
    if '$' in value:
        return True

    # Puppet undef
    if stripped == 'undef':
        return True

    # Dynamic function calls
    lower = stripped.lower()
    if any(lower.startswith(func) for func in _PARAM_FUNCTIONS):
        return True

    return False