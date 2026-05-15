"""
Debug script: trace why specific Terraform FN lines are missed.

Run from GUARD/:
    python src/debug_terraform_fns.py

Shows what the parser produces for each FN line and whether
detection logic would catch it.
"""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from parsers import terraform_parser

# --- All 25 Terraform FN lines ---
FN_LINES = [
    # Unrestricted IP — sec_invalid_bind (11)
    ("ComplianceRepo1_TerraformsWebGoat-terraform-aws-ec2.tf", 87,
     "unrestricted IP", "cidr_blocks 0.0.0.0/0"),
    ("ComplianceRepo1_TerraformsWebGoat-terraform-aws-ec2.tf", 94,
     "unrestricted IP", "cidr_blocks 0.0.0.0/0"),
    ("ComplianceRepo1_TerraformsWebGoat-terraform-aws-ec2.tf", 101,
     "unrestricted IP", "cidr_blocks 0.0.0.0/0"),
    ("gedas987654_terraform-terraform-aws-ec2.tf", 60,
     "unrestricted IP", "cidr_blocks 0.0.0.0/0"),
    ("gedas987654_terraform-terraform-aws-ec2.tf", 67,
     "unrestricted IP", "cidr_blocks 0.0.0.0/0"),
    ("gedas987654_terraform-terraform-aws-ec2.tf", 74,
     "unrestricted IP", "cidr_blocks 0.0.0.0/0"),
    ("ministryofjustice_modernisation-platform-environments-terraform-environments-laa-oem-oem_ec2_db_sg.tf", 14,
     "unrestricted IP", "cidr_blocks"),
    ("mongodb_terraform-provider-mongodbatlas-examples-aws-atlas-privatelink-serverless-aws-vpc.tf", 49,
     "unrestricted IP", "cidr_blocks"),
    ("mongodb_terraform-provider-mongodbatlas-examples-aws-privatelink-endpoint-serverless-instance-aws-vpc.tf", 47,
     "unrestricted IP", "cidr_blocks"),
    ("waritta26_terraform-terraform-aws-ec2.tf", 87,
     "unrestricted IP", "cidr_blocks 0.0.0.0/0"),
    ("waritta26_terraform-terraform-aws-ec2.tf", 94,
     "unrestricted IP", "cidr_blocks 0.0.0.0/0"),
    ("waritta26_terraform-terraform-aws-ec2.tf", 101,
     "unrestricted IP", "cidr_blocks 0.0.0.0/0"),

    # HTTPS / SSL/TLS policy — sec_https + sec_ssl_tls_policy (14)
    ("jbhoraniya_terraform1-azurerm-azurerm_function_app-simple-main.tf", 59,
     "sec_https", "HTTPS config"),
    ("tablexi_terraform_modules-aws-application_load_balancer-main.tf", 53,
     "sec_https", "HTTPS config"),
    ("wjrmffldrhrl_Terraform-Study-1st-personal-kevin-chapter_2-tf-main.tf", 87,
     "sec_https", "HTTPS config"),
    ("wwce_terraform-aws-Jenkins_proj-master-WebInDeploy-loadbalancers.tf", 38,
     "sec_https", "HTTPS config"),
    ("wwce_terraform-aws-Jenkins_proj-master-WebInDeploy-loadbalancers.tf", 98,
     "sec_https", "HTTPS config"),
    ("wwce_terraform-aws-Jenkins_proj-master-WebInDeploy-loadbalancers.tf", 149,
     "sec_https", "HTTPS config"),
    ("ps-interactive_labs-shared-terraform-cloudfront-main.tf", 58,
     "sec_ssl_tls_policy", "SSL/TLS policy"),
    ("terraform-google-modules_terraform-docs-samples-cloud_sql-instance_ha-main.tf", 18,
     "sec_ssl_tls_policy", "SSL/TLS policy"),
    ("terraform-google-modules_terraform-docs-samples-cloud_sql-instance_ha-main.tf", 36,
     "sec_ssl_tls_policy", "SSL/TLS policy"),
    ("terraform-google-modules_terraform-docs-samples-cloud_sql-instance_ha-main.tf", 54,
     "sec_ssl_tls_policy", "SSL/TLS policy"),
    ("terraform-google-modules_terraform-docs-samples-cloud_sql-instance_pitr-main.tf", 18,
     "sec_ssl_tls_policy", "SSL/TLS policy"),
    ("terraform-google-modules_terraform-docs-samples-cloud_sql-instance_pitr-main.tf", 36,
     "sec_ssl_tls_policy", "SSL/TLS policy"),

    # Empty password — sec_empty_pass (1)
    ("ComputeCanada_magic_castle-examples-advanced-spot_instances-gcp-main.tf", 36,
     "empty password", "empty password"),
]

# GUARD's RegEx patterns
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
                if f == filename:
                    return os.path.join(root, f)
    return None


def check_detection(parsed_line):
    """Simulate GUARD's detection logic on a ParsedLine."""
    results = []

    if parsed_line.comment is not None:
        results.append(f"  -> Comment: '{parsed_line.comment[:80]}'")

    if parsed_line.key is not None and parsed_line.value is not None:
        value = parsed_line.value

        if re.search(ERRORS['invalidBinding'], value):
            results.append(f"  -> MATCH: invalid IP binding")
        elif 'http' in value.lower():
            if re.search(ERRORS['http'], value):
                results.append(f"  -> MATCH: http without TLS")
        elif any(re.search(weak, value) for weak in ERRORS['weakCrypto']):
            results.append(f"  -> MATCH: weak crypto")
        else:
            results.append(f"  -> No RegEx match -> ML check would run on key='{parsed_line.key}'")
            value_stripped = value.strip().strip("'\"")
            if value_stripped in ('', 'undef'):
                results.append(f"  -> Empty password rule would trigger (if ML predicts secret)")
            elif parsed_line.is_parameterized:
                results.append(f"  -> is_parameterized=True -> SKIPPED")
    elif parsed_line.value is not None:
        results.append(f"  -> key=None, value present -> detection SKIPS")
        if re.search(ERRORS['invalidBinding'], parsed_line.value):
            results.append(f"  !! 0.0.0.0 IS in value but no key -> missed!")
    else:
        results.append(f"  -> No key, no value -> nothing to detect")

    return results


def main():
    base_dirs = [
        'datasets/terraform/oracle-dataset/files',
    ]

    print("=" * 70)
    print("TERRAFORM FN DEBUG — Tracing why specific lines are missed")
    print("=" * 70)

    for filename, target_line, expected, desc in FN_LINES:
        print(f"\n{'─' * 70}")
        print(f"FILE: {filename}")
        print(f"LINE: {target_line}  |  EXPECTED: {expected}  |  NOTE: {desc}")
        print(f"{'─' * 70}")

        filepath = find_file(base_dirs, filename)
        if filepath is None:
            print(f"  ✗ File not found! Try: find . -name '{filename}' -type f")
            continue

        print(f"  Found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if target_line > len(lines):
            print(f"  ✗ File only has {len(lines)} lines")
            continue

        raw_line = lines[target_line - 1].rstrip()
        print(f"  Raw line: {repr(raw_line)}")

        # Show surrounding context (2 lines before/after)
        print(f"  Context:")
        for i in range(max(0, target_line - 3), min(len(lines), target_line + 2)):
            marker = ">>>" if i == target_line - 1 else "   "
            print(f"    {marker} {i+1}: {lines[i].rstrip()}")

        parsed_lines = terraform_parser.parse_file(filepath)

        target_parsed = None
        for pl in parsed_lines:
            if pl.line_number == target_line:
                target_parsed = pl
                break

        if target_parsed is None:
            print(f"\n  ✗ Parser returned NO ParsedLine for line {target_line}!")
            print(f"    Line is being SKIPPED by the parser.")
            continue

        print(f"\n  ParsedLine:")
        print(f"    key:              {repr(target_parsed.key)}")
        print(f"    value:            {repr(target_parsed.value)}")
        print(f"    comment:          {repr(target_parsed.comment)}")
        print(f"    is_parameterized: {target_parsed.is_parameterized}")

        print(f"\n  Detection analysis:")
        for line in check_detection(target_parsed):
            print(line)

    print(f"\n{'=' * 70}")
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()