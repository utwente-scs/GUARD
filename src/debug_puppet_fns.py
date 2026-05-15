"""
Debug script: trace why specific Puppet FN lines are missed.

Run from GUARD/src/:
    python debug_puppet_fns.py

This will parse each FN file with the Puppet parser and show:
- What ParsedLine is produced for the target line
- Whether the RegEx patterns would match the extracted value
- Whether the ML model would be invoked
"""

import re
import sys
import os

# Add src to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsers import puppet_parser

# --- FN lines to investigate ---
FN_LINES = [
    # (file_path_suffix, line_number, expected_smell, description)
    
    # Weak crypto FNs
    ("openstack@fuel-library-deployment-puppet-osnailyfacter-manifests-ssh.pp", 34,
     "weak crypto", "arcfour in cipher list"),
    ("openstack@fuel-library-deployment-puppet-osnailyfacter-manifests-ssh.pp", 35,
     "weak crypto", "sha1 in MAC list"),
    ("puphpet@puppet-puphpet-manifests-nginx-params.pp", 25,
     "weak crypto", "unknown — need to check"),

    # Unrestricted IP FNs
    ("SecGen@SecGen-modules-services-unix-smb-samba-manifests-params.pp", 111,
     "unrestricted IP", "0.0.0.0 in params"),
    ("hubspotdevops@puppet-nexus-manifests-params.pp", 36,
     "unrestricted IP", "0.0.0.0 in nexus host"),
    ("redhat-openstack@puppet-galera-manifests-server.pp", 106,
     "unrestricted IP", "0.0.0.0 in wsrep_bind_address"),

    # Empty password FNs
    ("alphagov@govuk-puppet-modules-govuk_mysql-manifests-server.pp", 48,
     "empty password", "root_password"),
    ("cargomedia@puppet-packages-modules-mysql-manifests-server-instance.pp", 1,
     "empty password", "root_password = ''"),
    ("redhat-openstack@puppet-galera-manifests-server.pp", 99,
     "empty password", "wsrep_sst_password = undef"),
]

# GUARD's RegEx patterns (same as guard.py)
ERRORS = {
    'invalidBinding': r'0\.0\.0\.0',
    'weakCrypto': [r'md5', r'sha1', r'arcfour'],
    'http': r'http://'
}


def find_file(base_dirs, filename):
    """Search for a file across multiple possible dataset directories."""
    for base in base_dirs:
        for root, dirs, files in os.walk(base):
            for f in files:
                full = os.path.join(root, f)
                if full.endswith(filename) or f == filename:
                    return full
                # Also check flattened oracle naming (@ replaces /)
                if filename.replace('/', '-') in full or filename in full:
                    return full
    return None


def check_detection(parsed_line):
    """Simulate GUARD's detection logic on a ParsedLine."""
    results = []
    
    if parsed_line.comment is not None:
        results.append(f"  → Comment extracted: '{parsed_line.comment[:60]}...' (SetFit would run)")
    
    if parsed_line.key is not None and parsed_line.value is not None:
        value = parsed_line.value
        
        # Check RegEx smells
        if re.search(ERRORS['invalidBinding'], value):
            results.append(f"  → MATCH: invalid IP binding in value")
        elif 'http' in value.lower():
            if re.search(ERRORS['http'], value):
                results.append(f"  → MATCH: http without TLS in value")
        elif any(re.search(weak, value) for weak in ERRORS['weakCrypto']):
            matched = [w for w in ERRORS['weakCrypto'] if re.search(w, value)]
            results.append(f"  → MATCH: weak crypto ({matched}) in value")
        else:
            results.append(f"  → No RegEx match → ML hardcoded-secret check would run")
            if parsed_line.is_parameterized:
                results.append(f"  → BUT is_parameterized=True → would be SKIPPED")
            elif len(value.strip()) == 0:
                results.append(f"  → BUT len(value)==0 → would be SKIPPED (empty password gap)")
            else:
                results.append(f"  → ML would classify key='{parsed_line.key}'")
        
        # Also check case sensitivity
        value_upper = value.upper()
        if any(re.search(weak, value_upper, re.IGNORECASE) for weak in ERRORS['weakCrypto']):
            if not any(re.search(weak, value) for weak in ERRORS['weakCrypto']):
                results.append(f"  ⚠ CASE ISSUE: weak crypto present in uppercase but RegEx is lowercase-only!")
    
    elif parsed_line.value is not None:
        results.append(f"  → key=None, value present → detection logic likely SKIPS this line")
        # Still check if RegEx would match the value
        if any(re.search(weak, parsed_line.value) for weak in ERRORS['weakCrypto']):
            results.append(f"  ⚠ Weak crypto IS in the value, but no key → detection doesn't run!")
        if re.search(ERRORS['invalidBinding'], parsed_line.value):
            results.append(f"  ⚠ 0.0.0.0 IS in the value, but no key → detection doesn't run!")
    else:
        results.append(f"  → No key, no value → nothing to detect")
    
    return results


def main():
    # Try multiple possible dataset locations
    base_dirs = [
        'datasets/puppet/oracle-dataset',
    ]
    
    print("=" * 70)
    print("PUPPET FN DEBUG — Tracing why specific lines are missed")
    print("=" * 70)
    
    for filename, target_line, expected, desc in FN_LINES:
        print(f"\n{'─' * 70}")
        print(f"FILE: {filename}")
        print(f"LINE: {target_line}  |  EXPECTED: {expected}  |  NOTE: {desc}")
        print(f"{'─' * 70}")
        
        filepath = find_file(base_dirs, filename)
        if filepath is None:
            print(f"  ✗ File not found! Searched in: {base_dirs}")
            print(f"    Try: find .. -name '{filename}' -type f")
            continue
        
        print(f"  Found: {filepath}")
        
        # Read raw line
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if target_line > len(lines):
            print(f"  ✗ File only has {len(lines)} lines, target is {target_line}")
            continue
        
        raw_line = lines[target_line - 1].rstrip()
        print(f"  Raw line: {repr(raw_line)}")
        
        # Parse with Puppet parser
        parsed_lines = puppet_parser.parse_file(filepath)
        
        # Find the ParsedLine for target line number
        target_parsed = None
        for pl in parsed_lines:
            if pl.line_number == target_line:
                target_parsed = pl
                break
        
        if target_parsed is None:
            print(f"  ✗ Parser returned NO ParsedLine for line {target_line}!")
            print(f"    This means the parser is SKIPPING this line.")
            print(f"    Possible causes:")
            print(f"      - Line matches a structural token ({puppet_parser._STRUCTURAL_TOKENS})")
            print(f"      - Line is empty after stripping")
            print(f"      - No assignment operator (=> or =) found and no comment")
            
            # Manual parse attempt
            stripped = raw_line.strip()
            clean_stripped = stripped.rstrip(',').rstrip(';').strip()
            print(f"\n    Manual analysis:")
            print(f"      stripped:       {repr(stripped)}")
            print(f"      clean_stripped: {repr(clean_stripped)}")
            print(f"      in STRUCTURAL:  {clean_stripped in puppet_parser._STRUCTURAL_TOKENS}")
            print(f"      has '=>':       {'=>' in raw_line}")
            print(f"      has '=':        {'=' in raw_line}")
            print(f"      VAR_RE match:   {puppet_parser._VAR_ASSIGN_RE.match(raw_line)}")
            continue
        
        print(f"\n  ParsedLine:")
        print(f"    key:              {repr(target_parsed.key)}")
        print(f"    value:            {repr(target_parsed.value)}")
        print(f"    comment:          {repr(target_parsed.comment)}")
        print(f"    is_parameterized: {target_parsed.is_parameterized}")
        
        # Check what detection would do
        print(f"\n  Detection analysis:")
        for line in check_detection(target_parsed):
            print(line)
    
    print(f"\n{'=' * 70}")
    print("Done. Check the output above to identify root causes.")
    print("=" * 70)


if __name__ == "__main__":
    main()