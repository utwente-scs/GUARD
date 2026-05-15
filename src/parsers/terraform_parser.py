"""
Terraform (HCL) parser for GUARD.

Handles Terraform-specific syntax:
- Attribute assignment via '=' (e.g. name = "example")
- Variable references: var.name, local.name, module.name.output, data.type.name.attr
- String interpolation: "${var.name}"
- Comments with '#' and '//'
- Block headers (resource, data, variable, etc.) — skipped as non-assignment
- List values: ["0.0.0.0/0"] (including multi-line lists)
- Heredoc strings: <<EOF ... EOF

Design note: Terraform's HCL uses '=' for all assignments, unlike Ansible's ':'
and Puppet's '=>'. Block headers (resource "type" "name" {) have no '=' and are
naturally excluded from key-value parsing.
"""

import re
from . import ParsedLine


# HCL block keywords — lines starting with these followed by strings/braces
# are block headers, not key-value assignments
_BLOCK_KEYWORDS = frozenset((
    'resource', 'data', 'variable', 'output', 'locals', 'local',
    'module', 'provider', 'terraform', 'provisioner', 'connection',
    'backend', 'required_providers', 'moved', 'import', 'check',
))

# Structural-only tokens
_STRUCTURAL_TOKENS = frozenset(('', '{', '}', ')', '(', '];', ']', '},', '})', '['))

# Patterns for parameterized (dynamic) values
_PARAM_PREFIXES = ('var.', 'local.', 'module.', 'data.', 'each.', 'self.',
                   'count.', 'path.', 'terraform.')

_PARAM_FUNCTIONS = (
    'file(', 'templatefile(', 'lookup(', 'coalesce(', 'try(',
    'element(', 'merge(', 'concat(', 'join(', 'format(',
    'base64encode(', 'base64decode(', 'jsonencode(', 'jsondecode(',
    'cidrsubnet(', 'cidrhost(', 'tolist(', 'toset(', 'tomap(',
)


def parse_file(filepath):
    """Parse a Terraform .tf file into a list of ParsedLine objects.

    Args:
        filepath: Path to the .tf file.

    Returns:
        List of ParsedLine objects for lines containing detectable content.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    parsed = []
    in_heredoc = False
    heredoc_marker = None
    skip_until = -1  # line index to skip to (for multi-line list consumption)

    for line_number, raw in enumerate(lines, start=1):
        # Skip lines consumed by multi-line list joining
        if line_number <= skip_until:
            continue

        raw_stripped = raw.rstrip()

        # Handle heredoc blocks (<<EOF ... EOF) — skip interior lines
        if in_heredoc:
            if raw_stripped.strip() == heredoc_marker:
                in_heredoc = False
            continue

        heredoc_match = re.search(r'<<-?\s*(\w+)', raw_stripped)
        if heredoc_match:
            heredoc_marker = heredoc_match.group(1)
            # Still parse the assignment line itself (key = <<EOF)
            result = _parse_line(raw_stripped, line_number)
            if result is not None:
                parsed.append(result)
            in_heredoc = True
            continue

        result = _parse_line(raw_stripped, line_number)
        if result is not None:
            # --- Multi-line list joining ---
            # If value is just '[', the actual list content is on subsequent lines.
            # Join them so RegEx patterns can match values inside the list.
            if (result.key is not None
                    and result.value is not None
                    and result.value.strip() == '['):
                joined_value, consumed_up_to = _join_multiline_list(
                    lines, line_number, max_lookahead=20
                )
                if joined_value is not None:
                    result = ParsedLine(
                        result.line_number,
                        result.raw,
                        result.comment,
                        result.key,
                        joined_value,
                        _is_parameterized(joined_value),
                    )
                    skip_until = consumed_up_to

            parsed.append(result)

    return parsed


def _join_multiline_list(lines, current_line_number, max_lookahead=20):
    """Join a multi-line list into a single value string.

    When the parser encounters `key = [`, this function reads ahead
    to collect all list elements until the closing `]`.

    Args:
        lines:               All file lines (0-indexed).
        current_line_number: 1-based line number of the `key = [` line.
        max_lookahead:       Maximum number of lines to read ahead.

    Returns:
        Tuple of (joined_value, last_line_consumed) where joined_value
        is the full list as a string (e.g. '["0.0.0.0/0"]'), and
        last_line_consumed is the 1-based line number of the closing bracket.
        Returns (None, -1) if closing bracket is not found.
    """
    parts = ['[']
    start_idx = current_line_number  # 0-indexed: this is the line AFTER current

    for i in range(start_idx, min(start_idx + max_lookahead, len(lines))):
        line_content = lines[i].strip()

        # Strip comments from continuation lines
        for marker in ('#', '//'):
            if marker in line_content:
                idx = line_content.index(marker)
                before = line_content[:idx]
                if before.count('"') % 2 == 0:
                    line_content = before.strip()
                    break

        if line_content:
            parts.append(line_content)

        if ']' in line_content:
            joined = ' '.join(parts)
            return joined, i + 1  # i is 0-indexed, return 1-based

    return None, -1


def _parse_line(text, line_number):
    """Parse a single Terraform/HCL line into a ParsedLine.

    Returns None for lines with no detectable content.
    """
    stripped = text.strip()

    # Skip empty lines
    if not stripped:
        return None

    comment = None
    code_part = text

    # --- Extract comments ---
    # HCL supports '#' and '//' for single-line comments
    for marker in ('#', '//'):
        if marker in code_part:
            idx = code_part.index(marker)
            # Don't split if marker is inside a quoted string
            before = code_part[:idx]
            if before.count('"') % 2 == 0:
                comment = code_part[idx + len(marker):].strip()
                code_part = before
                break

    clean = code_part.strip()

    # Skip structural-only lines
    clean_check = clean.rstrip(',').rstrip(';').strip()
    if clean_check in _STRUCTURAL_TOKENS:
        if comment:
            return ParsedLine(line_number, text, comment=comment)
        return None

    # --- Check for block headers (not assignments) ---
    # e.g. resource "aws_instance" "example" {
    #      dynamic "ingress" {
    first_word = clean.split()[0] if clean.split() else ''
    if first_word in _BLOCK_KEYWORDS:
        if comment:
            return ParsedLine(line_number, text, comment=comment, value=clean)
        return ParsedLine(line_number, text, value=clean)

    # --- Try '=' assignment ---
    # HCL uses '=' for all attribute assignments
    # Exclude comparison operators: ==, !=, >=, <=
    if '=' in clean:
        eq_idx = _find_assignment_eq(clean)
        if eq_idx is not None:
            key = clean[:eq_idx].strip()
            value = clean[eq_idx + 1:].strip().rstrip(',').rstrip(';').strip()

            # Validate key looks like an HCL identifier (letters, digits, _, -)
            if re.match(r'^[\w-]+$', key):
                is_param = _is_parameterized(value)
                return ParsedLine(line_number, text, comment, key, value, is_param)

    # --- Non-assignment lines ---
    # Nested block headers, standalone expressions, etc.
    # Return for comment detection and full-line RegEx scanning.
    if comment:
        return ParsedLine(line_number, text, comment=comment, value=clean)

    if clean:
        return ParsedLine(line_number, text, value=clean)

    return None


def _find_assignment_eq(line):
    """Find the index of the assignment '=' in an HCL line.

    Skips comparison operators (==, !=, >=, <=) and '=' inside strings.
    Returns None if no assignment '=' is found.
    """
    i = 0
    in_string = False
    while i < len(line):
        ch = line[i]

        # Track string boundaries (simplified — doesn't handle escapes fully)
        if ch == '"' and (i == 0 or line[i - 1] != '\\'):
            in_string = not in_string
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if ch == '=':
            # Skip == (comparison)
            if i + 1 < len(line) and line[i + 1] == '=':
                i += 2
                continue
            # Skip !=, >=, <= (comparison)
            if i > 0 and line[i - 1] in ('!', '>', '<'):
                i += 1
                continue
            return i

        i += 1

    return None


def _is_parameterized(value):
    """Check if a Terraform value is a variable reference or dynamic expression.

    Parameterized values should not be flagged as hardcoded secrets because
    their actual content is determined at runtime or from variables.

    Covers:
    - var.name, local.name, module.name, data.type.name
    - String interpolation: "${var.name}"
    - Function calls: file(), templatefile(), lookup(), etc.
    - Boolean/null literals (not secrets)
    """
    stripped = value.strip().strip('"').strip("'")
    lower = stripped.lower()

    # Boolean and null literals
    if lower in ('true', 'false', 'null'):
        return True

    # Direct variable/resource references
    if any(stripped.startswith(prefix) for prefix in _PARAM_PREFIXES):
        return True

    # String interpolation containing variable references
    if '${' in value:
        return True

    # Function calls
    if any(lower.startswith(func) for func in _PARAM_FUNCTIONS):
        return True

    # List/map containing only references (e.g. [var.cidr_blocks])
    if stripped.startswith('[') and any(prefix in stripped for prefix in _PARAM_PREFIXES):
        return True

    return False