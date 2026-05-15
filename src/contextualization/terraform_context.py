"""
Terraform contextualization phase.

Resolves variable references (var.name, local.name) by finding their
definitions across all .tf files in a project and replacing references
with resolved values.

This is a preprocessing step that runs BEFORE parsing and detection.

Terraform variables are defined in two main ways:
1. variable "name" { default = "value" }  →  referenced as var.name
2. locals { name = "value" }              →  referenced as local.name

Limitations:
    - Only resolves variables with explicit default values (not tfvars or CLI input)
    - Does not resolve function calls, conditional expressions, or dynamic blocks
    - Does not handle variable validation or type constraints
    - Locals referencing other locals/variables are not recursively resolved
    - Does not parse terraform.tfvars or *.auto.tfvars files
"""

import os
import re


def find_terraform_files(folder):
    """Find all .tf files in a directory tree."""
    file_list = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.tf'):
                file_list.append(os.path.join(root, f))
    return file_list


def find_variable_defaults(files):
    """Extract default values from variable blocks.

    Parses patterns like:
        variable "db_password" {
          default = "admin123"
        }

    Args:
        files: List of .tf file paths.

    Returns:
        Dict mapping variable name -> default value string.
    """
    variables = {}

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find variable blocks with defaults
        # Pattern: variable "name" { ... default = "value" ... }
        var_blocks = re.finditer(
            r'variable\s+"(\w+)"\s*\{([^}]*)\}',
            content,
            re.DOTALL
        )

        for match in var_blocks:
            var_name = match.group(1)
            block_body = match.group(2)

            # Find default = "value" within the block
            default_match = re.search(
                r'default\s*=\s*"([^"]*)"',
                block_body
            )
            if default_match:
                value = default_match.group(1)
                # Skip if the default itself is a variable reference or function
                if '${' not in value and not any(
                    value.startswith(p) for p in ('var.', 'local.', 'module.')
                ):
                    variables[var_name] = value
                continue

            # Also check for unquoted defaults (numbers, booleans)
            default_match = re.search(
                r'default\s*=\s*([^\s#\n}]+)',
                block_body
            )
            if default_match:
                value = default_match.group(1).strip()
                if value not in ('true', 'false', 'null', '""', "''"):
                    variables[var_name] = value

    return variables


def find_locals(files):
    """Extract local value definitions.

    Parses patterns like:
        locals {
          db_password = "admin123"
          port        = 3306
        }

    Args:
        files: List of .tf file paths.

    Returns:
        Dict mapping local name -> value string.
    """
    locals_map = {}

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find locals blocks
        locals_blocks = re.finditer(
            r'locals\s*\{([^}]*)\}',
            content,
            re.DOTALL
        )

        for match in locals_blocks:
            block_body = match.group(1)

            # Find name = "value" assignments
            assignments = re.finditer(
                r'(\w+)\s*=\s*"([^"]*)"',
                block_body
            )
            for assign in assignments:
                name = assign.group(1)
                value = assign.group(2)
                if '${' not in value:
                    locals_map[name] = value

    return locals_map


def replace_references(files, variables, locals_map):
    """Replace var.name and local.name references with resolved values.

    Args:
        files: List of .tf file paths to modify in-place.
        variables: Dict of variable name -> value.
        locals_map: Dict of local name -> value.
    """
    if not variables and not locals_map:
        return

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        modified = False

        # Replace var.name references
        for var_name, value in variables.items():
            # Replace "${var.name}" interpolation
            old_interp = f'${{var.{var_name}}}'
            if old_interp in content:
                content = content.replace(old_interp, value)
                modified = True

            # Replace standalone var.name (with word boundary)
            pattern = rf'\bvar\.{re.escape(var_name)}\b'
            new_content = re.sub(pattern, f'"{value}"', content)
            if new_content != content:
                content = new_content
                modified = True

        # Replace local.name references
        for local_name, value in locals_map.items():
            old_interp = f'${{local.{local_name}}}'
            if old_interp in content:
                content = content.replace(old_interp, value)
                modified = True

            pattern = rf'\blocal\.{re.escape(local_name)}\b'
            new_content = re.sub(pattern, f'"{value}"', content)
            if new_content != content:
                content = new_content
                modified = True

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)


def main(folder_path):
    """Run the Terraform contextualization phase.

    Finds all .tf files, resolves variable defaults and locals,
    and replaces references in-place.

    Args:
        folder_path: Root directory to process (typically ./temp).
    """
    files = find_terraform_files(folder_path)
    if not files:
        return

    variables = find_variable_defaults(files)
    locals_map = find_locals(files)

    if variables or locals_map:
        replace_references(files, variables, locals_map)