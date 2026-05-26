"""
Chef (.rb) parser for GUARD.

Chef uses a Ruby-based DSL where resources are defined inside do...end
blocks with key-value attributes on individual lines.

Parsed patterns (inside resource blocks):
    key 'value'          — quoted string attribute (most common)
    key "value"          — double-quoted string attribute
    key :symbol          — Ruby symbol value
    key value            — unquoted value (booleans, numbers)
    key: 'value'         — Ruby hash-style (inside variables() etc.)
    :key => 'value'      — hash rocket style
    node[:x][:y] = val   — attribute assignment with =

Skipped patterns:
    do / end / do |x|    — block boundaries
    include_recipe ...   — DSL method calls
    if/unless/case/when  — Ruby control flow
    .each / %w[...]      — iterators
    method(args)         — standalone method calls

Parameterization (value references that should NOT be flagged):
    node['key']          — Chef attribute access
    node[:key]           — Chef attribute access (symbol key)
    "#{...}"             — Ruby string interpolation
    data_bag_item(...)   — data bag lookups
    function(args)       — any function call as a value
"""

import re

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parsers import ParsedLine


# ---------------------------------------------------------------------------
# Skip logic — lines that should not produce key-value ParsedLines
# ---------------------------------------------------------------------------

# Ruby / Chef keywords that start a non-attribute line
SKIP_FIRST_WORDS = {
    'end', 'if', 'unless', 'case', 'when', 'else', 'elsif',
    'begin', 'rescue', 'ensure', 'def', 'class', 'module',
    'return', 'raise', 'yield', 'do', 'then', 'while', 'until',
    'for', 'break', 'next', 'retry', 'redo',
}

# Chef DSL method calls — these take arguments but are not attributes
SKIP_PREFIXES = (
    'include_recipe', 'depends', 'require', 'require_relative',
    'Chef::', 'chef_gem', 'notifies', 'subscribes',
    'not_if', 'only_if', 'include ', 'extend ',
    'puts ', 'print ', 'log ',
)


def _should_skip(code):
    """Check if a code line should be skipped (not parsed for key-value)."""
    if not code:
        return True

    # Lines ending with 'do' or 'do |var|' are block openers
    if re.search(r'\bdo(\s*\|.*\|)?\s*$', code):
        return True

    first_word = code.split()[0]

    if first_word in SKIP_FIRST_WORDS:
        return True

    for prefix in SKIP_PREFIXES:
        if code.startswith(prefix):
            return True

    # Ruby iterators: .each, %w[...], %w(...)
    if '.each' in code or code.startswith('%w'):
        return True

    # Standalone method calls: word(args) with no = assignment
    # e.g. force_link("/var/log", dir), daemon_user(:hadoop)
    if re.match(r'[a-zA-Z_]\w*\s*\(', code) and '=' not in code.split('(')[0]:
        return True

    return False


# ---------------------------------------------------------------------------
# Parameterization detection
# ---------------------------------------------------------------------------

def _is_parameterized(raw_value):
    """Check if a value contains Chef variable references or function calls.

    These values should not be flagged as hardcoded secrets because they
    are resolved at runtime.
    """
    # node['attr'] or node[:attr] — Chef attribute access
    if re.search(r'node\[', raw_value):
        return True
    # Ruby string interpolation: "...#{expr}..."
    if '#{' in raw_value:
        return True
    # Data bag lookups
    if 'data_bag' in raw_value:
        return True
    # Function calls as values: func_name(args)
    if re.search(r'[a-zA-Z_]\w*\(', raw_value):
        return True
    return False


# ---------------------------------------------------------------------------
# Key-value extraction
# ---------------------------------------------------------------------------

def _extract_key_value(code):
    """Try to extract a key-value pair from a single Chef code segment.

    Tries patterns in order of specificity:
        1. Hash rocket:    :key => value  or  key => value
        2. Symbol-key:     key: value     (Ruby 1.9+ hash syntax)
        3. Assignment:     node[...] = value  or  $var = value
        4. Space-separated: key 'value'   (Chef attribute style)

    Returns:
        (key, value, is_parameterized) or (None, None, False).
    """
    # --- Pattern 1: hash rocket (key => value) ---
    m = re.match(r':?(\w[\w-]*)\s*=>\s*(.+)', code)
    if m:
        key = m.group(1)
        raw_val = m.group(2).strip()
        value = raw_val.strip("'\"")
        return key, value, _is_parameterized(raw_val)

    # --- Pattern 2: symbol-key hash (key: value) ---
    # Must have space after colon to avoid matching URL-like patterns
    m = re.match(r'(\w[\w-]*):\s+(.+)', code)
    if m:
        key = m.group(1)
        raw_val = m.group(2).strip()
        value = raw_val.strip("'\"")
        return key, value, _is_parameterized(raw_val)

    # --- Pattern 3: assignment with = ---
    # Handles: node[:section][:key] = value, $var = value
    # Excludes comparisons: ==, !=, >=, <=
    if '=' in code and '=>' not in code:
        m = re.match(r'(.+?)\s*(?<![!=><])=(?!=)\s*(.+)', code)
        if m:
            lhs = m.group(1).strip()
            raw_val = m.group(2).strip()

            # Extract key name from LHS
            # node[:key], node['key'], default['section']['key'] → key
            key_m = re.search(r"""[\['":]([\w]+)['"\]]?\]?\s*$""", lhs)
            if key_m:
                key = key_m.group(1)
                value = raw_val.strip("'\"")
                return key, value, _is_parameterized(raw_val)

            # $variable = value → variable
            key_m = re.match(r'\$(\w+)', lhs)
            if key_m:
                key = key_m.group(1)
                value = raw_val.strip("'\"")
                return key, value, _is_parameterized(raw_val)

            # Dot-notation: self.obj.key = value → key
            key_m = re.search(r'\.(\w+)\s*$', lhs)
            if key_m:
                key = key_m.group(1)
                value = raw_val.strip("'\"")
                return key, value, _is_parameterized(raw_val)

    # --- Pattern 4: space-separated (key value) ---
    # The most common Chef attribute pattern: mode '0755', owner 'root'
    m = re.match(r'(\w[\w-]*)\s+(.*\S.*)', code)
    if m:
        key = m.group(1)
        raw_val = m.group(2).strip()

        # Don't match if this looks like a resource declaration (ends with do)
        if re.search(r'\bdo(\s*\|.*\|)?\s*$', raw_val):
            return None, None, False

        # Don't match if value starts with ( — that's a method call
        if raw_val.startswith('('):
            return None, None, False

        value = raw_val.strip("'\"")
        return key, value, _is_parameterized(raw_val)

    return None, None, False


# ---------------------------------------------------------------------------
# Comment extraction
# ---------------------------------------------------------------------------

def _find_inline_comment(line):
    """Find the position of an inline # comment, ignoring # inside strings.

    Returns:
        Index of the # character, or None if no inline comment.
    """
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == '#' and not in_single and not in_double:
            # Make sure it's not Ruby string interpolation #{
            if i + 1 < len(line) and line[i + 1] == '{':
                continue
            return i
    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_file(filepath):
    """Parse a Chef .rb file into a list of ParsedLine objects.

    Reads the file line by line, extracts comments and key-value pairs,
    handles multi-line values (arrays/hashes spanning lines), and detects
    parameterized values that should not be flagged as hardcoded secrets.

    Args:
        filepath: Path to a .rb file.

    Returns:
        List of ParsedLine objects.
    """
    parsed_lines = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line_number = i + 1
        raw = lines[i].rstrip('\n')
        stripped = raw.strip()

        # Skip blank lines
        if not stripped:
            i += 1
            continue

        # --- Full-line comment ---
        if stripped.startswith('#'):
            comment_text = stripped.lstrip('#').strip()
            parsed_lines.append(ParsedLine(
                line_number=line_number,
                raw=raw,
                comment=comment_text if comment_text else None
            ))
            i += 1
            continue

        # --- Separate code from inline comment ---
        comment = None
        code_part = stripped

        comment_pos = _find_inline_comment(stripped)
        if comment_pos is not None and comment_pos > 0:
            code_part = stripped[:comment_pos].strip()
            comment_text = stripped[comment_pos + 1:].strip()
            comment = comment_text if comment_text else None

        # --- Check if line should be skipped ---
        if _should_skip(code_part):
            if comment:
                parsed_lines.append(ParsedLine(
                    line_number=line_number,
                    raw=raw,
                    comment=comment
                ))
            i += 1
            continue

        # --- Handle semicolon-separated statements ---
        # Some Chef files use one-liners like:
        #   directory("...") { owner 'root' ; mode '0755' }
        # We already skip .each{} lines, but handle remaining ; cases
        segments = [code_part]
        if ';' in code_part:
            # Try to extract the inner block content
            inner_m = re.search(r'\{(.+)\}', code_part)
            if inner_m:
                inner = inner_m.group(1)
            else:
                inner = code_part
            segments = [s.strip() for s in inner.split(';') if s.strip()]

        # --- Parse each segment for key-value ---
        found_kv = False
        for segment in segments:
            key, value, is_param = _extract_key_value(segment)

            if key is not None and value is not None:
                # Multi-line joining: value opens [ or ( but doesn't close
                if ('[' in value and ']' not in value) or \
                   ('(' in value and ')' not in value):
                    while i + 1 < len(lines):
                        i += 1
                        next_stripped = lines[i].strip()
                        value += ' ' + next_stripped
                        raw += '\n' + lines[i].rstrip('\n')
                        if ']' in next_stripped or ')' in next_stripped:
                            break
                    value = value.strip("'\"")
                    # Re-check parameterization on the full joined value
                    is_param = is_param or _is_parameterized(value)

                parsed_lines.append(ParsedLine(
                    line_number=line_number,
                    raw=raw,
                    comment=comment if not found_kv else None,
                    key=key,
                    value=value,
                    is_parameterized=is_param
                ))
                found_kv = True

        # If no key-value found but there's a comment, still record it
        if not found_kv and comment:
            parsed_lines.append(ParsedLine(
                line_number=line_number,
                raw=raw,
                comment=comment
            ))

        i += 1

    return parsed_lines