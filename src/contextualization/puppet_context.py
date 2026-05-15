"""
Puppet contextualization phase.

Analogous to contexualizationPhase.py for Ansible, this module resolves
Puppet variable references ($var, ${var}) by finding their definitions
across all .pp files in a project and replacing references with resolved
values.

This is a preprocessing step that runs BEFORE parsing and detection.
By resolving variables, the detection phase can check whether the actual
values contain security smells (e.g. a hardcoded password assigned to a
variable that is then referenced elsewhere).

Puppet variable scoping is complex (class scope, node scope, top scope
with $:: prefix, Hiera lookups). This implementation handles the common
cases relevant to security smell detection:
    - Simple variable assignment: $var = 'value'
    - Class parameter defaults: class myclass($param = 'default')
    - Top-scope references: $::var

Limitations:
    - Does not resolve Hiera lookups (would require hiera.yaml + data files)
    - Does not resolve function calls (template(), inline_template(), etc.)
    - Does not handle conditional assignments or dynamic scope chains
    - Scoping is simplified: all definitions are treated as globally reachable
"""

import os
import re


def find_puppet_files(folder):
    """Find all .pp files in a directory tree."""
    file_list = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.pp'):
                file_list.append(os.path.join(root, f))
    return file_list


def find_variables(files):
    """Find all $variable references used in values across Puppet files.

    Looks for $var and ${var} patterns, excluding:
    - Variable definitions (left side of =)
    - Resource titles
    - Variables that are themselves code expressions
    """
    # Match $var and ${var} patterns
    pattern = re.compile(r'\$\{?([\w:]+)\}?')
    all_vars = set()

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        for match in pattern.finditer(content):
            var_name = match.group(1)
            # Skip Puppet built-in variables and facts
            if var_name.startswith(('facts', 'trusted', 'server_facts')):
                continue
            all_vars.add(var_name)

    return list(all_vars)


def find_definitions(files, var_names):
    """Find variable definitions across all Puppet files.

    Looks for patterns like:
        $var = 'value'
        $var = "value"
        $var = value

    Args:
        files: List of .pp file paths to search.
        var_names: List of variable names to find definitions for.

    Returns:
        Dict mapping variable name -> resolved value string.
    """
    var_values = {}

    # Pattern: $var_name = value (with optional quotes)
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            stripped = line.strip()

            # Skip comments
            if stripped.startswith('#'):
                continue

            # Match $variable = value
            match = re.match(r"\s*\$(\w[\w:]*)\s*=\s*['\"]?([^'\"#\n,;]+)['\"]?", stripped)
            if match:
                var_name = match.group(1)
                value = match.group(2).strip()

                if var_name in var_names:
                    # Don't resolve if the value is itself a variable or function call
                    if '$' in value or '(' in value:
                        continue
                    var_values[var_name] = value

            # Also check class parameter defaults:
            # class myclass ( Type $param = 'default', ... )
            param_match = re.match(
                r"\s*(?:\w+\s+)?\$(\w[\w:]*)\s*=\s*['\"]?([^'\"#\n,);]+)['\"]?",
                stripped
            )
            if param_match and param_match.group(1) not in var_values:
                var_name = param_match.group(1)
                value = param_match.group(2).strip()
                if var_name in var_names and '$' not in value and '(' not in value:
                    var_values[var_name] = value

    return var_values


def replace_variables(files, replacements):
    """Replace $variable references with their resolved values in all files.

    Handles both $var and ${var} syntax.

    Args:
        files: List of .pp file paths to modify in-place.
        replacements: Dict mapping variable name -> resolved value.
    """
    if not replacements:
        return

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        for var_name, value in replacements.items():
            # Replace ${var_name} and $var_name
            content = content.replace(f'${{{var_name}}}', value)
            # For $var_name, use word boundary to avoid partial matches
            # e.g. $port should not match $port_number
            content = re.sub(
                rf'\$({re.escape(var_name)})(?![\w:])',
                value,
                content
            )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


def main(folder_path):
    """Run the Puppet contextualization phase.

    Finds all .pp files, identifies variable references and their definitions,
    and replaces references with resolved values in-place.

    Args:
        folder_path: Root directory to process (typically ./temp).
    """
    files = find_puppet_files(folder_path)
    if not files:
        return

    var_names = find_variables(files)
    if not var_names:
        return

    definitions = find_definitions(files, var_names)

    # Only replace variables that have non-empty resolved values
    definitions = {k: v for k, v in definitions.items() if v}

    if definitions:
        replace_variables(files, definitions)