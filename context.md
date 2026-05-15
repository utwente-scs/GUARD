# Session Context — GUARD Extension (Puppet + Terraform)

## Architecture

Refactored `guard.py` into a dispatcher pattern:
- `guard.py` — dispatcher + shared `detect_smells()` function (RegEx + ML on `ParsedLine` objects)
- `parsers/` — language-specific parsers: `ansible_parser.py`, `puppet_parser.py`, `terraform_parser.py`
- `contextualization/` — language-specific variable resolution: `puppet_context.py`, `terraform_context.py`
- `ParsedLine(line_number, raw, comment, key, value, is_parameterized)` — standardised output from all parsers
- `is_parameterized` replaces hardcoded `'{{' not in value` — now handles `$var` (Puppet), `var.name`/`${...}` (Terraform), `{{ }}` (Ansible)
- Detection logic is identical across languages; only parsing and parameterization detection differ
- Parsers share a lot of structure and could be refactored into a base class with overrides (not yet done)

---

## Puppet Extension Results

### Baseline → Extended
| Metric    | Baseline | Extended |
|-----------|----------|----------|
| TP        | 5        | 26       |
| FP        | 138      | 154      |
| FN        | 35       | 14       |
| Precision | 0.035    | 0.144    |
| Recall    | 0.125    | 0.650    |
| F1        | 0.055    | 0.236    |

Oracle: 65 total smells, 40 GUARD-compatible, 25 out-of-scope.

### FP breakdown
- 117 suspicious comment FPs (SetFit trained on Ansible, misclassifies Puppet docs patterns)
- 48 hardcoded-secret FPs (Label Propagation triggers on unfamiliar Puppet key names)
- Remaining: minor use-of-http and weak-crypto FPs

### Puppet FNs to investigate (14)

**Empty password (3):**
- `alphagov@govuk-puppet-modules-govuk_mysql-manifests-server.pp` line 48
- `cargomedia@puppet-packages-modules-mysql-manifests-server-instance.pp` line 1
- `redhat-openstack@puppet-galera-manifests-server.pp` line 99

**Unrestricted IP Address (3):**
- `SecGen@SecGen-modules-services-unix-smb-samba-manifests-params.pp` line 111
- `hubspotdevops@puppet-nexus-manifests-params.pp` line 36
- `redhat-openstack@puppet-galera-manifests-server.pp` line 106

**Use of weak cryptography algorithms (3):**
- `openstack@fuel-library-deployment-puppet-osnailyfacter-manifests-ssh.pp` line 34
- `openstack@fuel-library-deployment-puppet-osnailyfacter-manifests-ssh.pp` line 35
- `puphpet@puppet-puphpet-manifests-nginx-params.pp` line 25

**Suspicious comment (5):**
- `openstack@puppet-openstack-integration-manifests-designate.pp` line 37
- `openstack@puppet-openstack-integration-manifests-designate.pp` line 100
- `purpleidea@puppet-gluster-manifests-simple.pp` line 22
- `purpleidea@puppet-gluster-manifests-simple.pp` line 81
- `purpleidea@puppet-gluster-manifests-simple.pp` line 145

---

## Terraform Extension Results

### Baseline → Extended
| Metric    | Baseline | Extended |
|-----------|----------|----------|
| TP        | 0        | 36       |
| FP        | 38       | 58       |
| FN        | 66       | 25       |
| Precision | 0.000    | 0.383    |
| Recall    | 0.000    | 0.590    |
| F1        | 0.000    | 0.465    |

Oracle: 376 total smells, only 66 GUARD-compatible (~17.5%), 310 out-of-scope. This confirms the ~20% taxonomy overlap finding from RQ1.

Precision is notably higher than Puppet (0.383 vs 0.144) because fewer suspicious comment FPs (32 vs 117).

### TP breakdown
- 22 improper IP address binding (RegEx `0.0.0.0` through `=` parser)
- 10 hardcoded-secret (ML model on keys like `password`, `username`, `access_key`)
- 3 use of http (RegEx `http://`)
- 1 weak crypto

### FP breakdown
- 32 suspicious comment FPs
- 29 hardcoded-secret FPs
- 3 use of http FPs (likely internal/dev URLs)
- 2 improper ip address binding FPs

### Terraform FNs to investigate (25)

**Unrestricted IP Address — sec_invalid_bind (11):**
- `ComplianceRepo1_TerraformsWebGoat-terraform-aws-ec2.tf` lines 87, 94, 101
- `gedas987654_terraform-terraform-aws-ec2.tf` lines 60, 67, 74
- `ministryofjustice_modernisation-platform-environments-terraform-environments-laa-oem-oem_ec2_db_sg.tf` line 14
- `mongodb_terraform-provider-mongodbatlas-examples-aws-atlas-privatelink-serverless-aws-vpc.tf` line 49
- `mongodb_terraform-provider-mongodbatlas-examples-aws-privatelink-endpoint-serverless-instance-aws-vpc.tf` line 47
- `waritta26_terraform-terraform-aws-ec2.tf` lines 87, 94, 101

Likely cause: `0.0.0.0/0` appears inside lists (`cidr_blocks = ["0.0.0.0/0"]`) — RegEx should match but the value extraction or line structure may prevent it. Need to check actual file lines.

**HTTPS / SSL/TLS policy — sec_https + sec_ssl_tls_policy (14):**
- `jbhoraniya_terraform1-azurerm-azurerm_function_app-simple-main.tf` line 59 (sec_https)
- `tablexi_terraform_modules-aws-application_load_balancer-main.tf` line 53 (sec_https)
- `wjrmffldrhrl_Terraform-Study-1st-personal-kevin-chapter_2-tf-main.tf` line 87 (sec_https)
- `wwce_terraform-aws-Jenkins_proj-master-WebInDeploy-loadbalancers.tf` lines 38, 98, 149 (sec_https)
- `ps-interactive_labs-shared-terraform-cloudfront-main.tf` line 58 (sec_ssl_tls_policy)
- `terraform-google-modules_terraform-docs-samples-cloud_sql-instance_ha-main.tf` lines 18, 36, 54 (sec_ssl_tls_policy)
- `terraform-google-modules_terraform-docs-samples-cloud_sql-instance_pitr-main.tf` lines 18, 36 (sec_ssl_tls_policy)

Likely cause: these are NOT `http://` patterns — they are TLS policy configurations like `minimum_tls_version = "TLS1_0"` or `ssl_policy = "ELBSecurityPolicy-2016-08"`. GUARD's RegEx only catches `http://` URLs, not missing/weak TLS configurations. These would need Terraform-specific detection rules beyond the original 7 smells.

**Empty password — sec_empty_pass (1):**
- `ComputeCanada_magic_castle-examples-advanced-spot_instances-gcp-main.tf` line 36

---

## Key findings for thesis

1. **Parser extension works**: Both languages go from near-zero to meaningful detection purely from correct parsing. The underlying detection logic (RegEx + ML) is sound — failures were upstream preprocessing issues.

2. **Puppet vs Terraform precision gap**: Puppet precision (0.144) is much lower than Terraform (0.383), driven almost entirely by SetFit comment classifier FPs (117 vs 32). Retraining SetFit on language-specific data is the highest-impact next step.

3. **Taxonomy mismatch is real for Terraform**: Only 17.5% of Terraform oracle smells overlap with GUARD's 7-smell taxonomy. The majority are infrastructure-level concerns (logging, IAM, encryption, firewall) that don't exist in config management tools. This is a fundamental scope limitation, not an engineering failure.

4. **sec_https/sec_ssl_tls_policy are different smells**: 14 of 25 Terraform FNs are TLS policy issues, not HTTP URL issues. GUARD's `http://` RegEx can't detect `minimum_tls_version = "TLS1_0"`. These would need new Terraform-specific rules if brought into scope.

5. **RQ1 section correction**: The thesis draft says "Puppet fails at the semantic classification level" — this is inaccurate. Puppet also fails at parsing (`:` vs `=>`), which is why the extension jumped recall from 12.5% to 65%. Both languages fail at parsing, but Puppet's failure is less severe.

---

## Next steps

1. **Investigate FN lines** — open the actual files for the FNs listed above, check what the lines contain, and determine root causes (parser issue vs detection gap vs out-of-scope)
2. **Retrain SetFit** on Puppet and Terraform comment data — highest-impact improvement for precision
3. **Retrain Label Propagation** on Puppet/Terraform key-value data — reduce hardcoded-secret FPs
4. **Consider Terraform-specific rules** — decide whether TLS policy / SSL config smells should be added as new detection patterns (extends beyond original 7 smells — discuss with Matteo)
5. **Parser refactoring** — the three parsers share ~70% of their code and could be collapsed into a base class with language-specific overrides (lower priority, cosmetic)
6. **Write RQ2 findings** — document what adaptations were required (parser, contextualization, parameterization detection) and what couldn't be solved by adaptation alone (taxonomy mismatch, TLS policy smells)