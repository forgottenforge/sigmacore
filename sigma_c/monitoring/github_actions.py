"""
Sigma-C GitHub Actions
=======================
Copyright (c) 2025 ForgottenForge.xyz

GitHub Actions integration for CI/CD criticality checks.
"""

GITHUB_ACTION_YAML = """
name: Criticality Check
description: 'Analyze code criticality with Sigma-C'
author: 'ForgottenForge'

inputs:
  threshold:
    description: 'Maximum allowed sigma_c value'
    required: false
    default: '0.7'
  fail-on-critical:
    description: 'Fail build if threshold exceeded'
    required: false
    default: 'true'
  path:
    description: 'Path to analyze'
    required: false
    default: '.'

outputs:
  sigma_c:
    description: 'Computed criticality value'
  status:
    description: 'Pass or fail status'

runs:
  using: 'composite'
  steps:
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install Sigma-C
      shell: bash
      run: pip install sigma-c-framework
    
    - name: Run Analysis
      shell: bash
      run: |
        python -c "
        from sigma_c.monitoring.ci import analyze_repository
        result = analyze_repository('${{ inputs.path }}')
        print(f'sigma_c={result[\"sigma_c\"]}')
        print(f'status={result[\"status\"]}')
        
        if result['sigma_c'] > float('${{ inputs.threshold }}') and '${{ inputs.fail-on-critical }}' == 'true':
            exit(1)
        "

branding:
  icon: 'activity'
  color: 'blue'
"""

EXAMPLE_WORKFLOW = """
# .github/workflows/criticality.yml
name: Criticality Check
on: [push, pull_request]

jobs:
  sigma-c:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Analyze Criticality
        uses: sigma-c/action@v1
        with:
          threshold: 0.7
          fail-on-critical: true
          path: ./src
      
      - name: Upload Report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: criticality-report
          path: sigma_c_report.json
"""
