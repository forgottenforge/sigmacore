#!/bin/bash
# Sigma-C Framework v2.0.0 Release Script
# ========================================

set -e  # Exit on error

echo "🚀 Sigma-C Framework v2.0.0 Release"
echo "===================================="
echo ""

# 1. Clean previous builds
echo "📦 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/
echo "✓ Clean complete"
echo ""

# 2. Build package
echo "🔨 Building package..."
python -m build
echo "✓ Build complete"
echo ""

# 3. Check package
echo "🔍 Checking package..."
twine check dist/*
echo "✓ Package check passed"
echo ""

# 4. Git operations
echo "📝 Git operations..."
git add .
git commit -m "Release v2.0.0 - Rigorous Control

- 22 production-ready integrations
- 9 domain adapters
- Observable discovery & multi-scale analysis
- Active control with PID
- Streaming O(1) calculation
- IonQ hardware support added
"
git tag -a v2.0.0 -m "Release v2.0.0 - Rigorous Control"
echo "✓ Git commit and tag created"
echo ""

# 5. Push to GitHub
echo "🌐 Pushing to GitHub..."
git push origin main
git push origin v2.0.0
echo "✓ Pushed to GitHub"
echo ""

# 6. Upload to PyPI
echo "📤 Uploading to PyPI..."
echo "⚠️  This will upload to PRODUCTION PyPI!"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    twine upload dist/*
    echo "✓ Uploaded to PyPI"
else
    echo "⚠️  PyPI upload skipped"
fi
echo ""

echo "✅ Release v2.0.0 complete!"
echo ""
echo "Next steps:"
echo "1. Create GitHub release: https://github.com/forgottenforge/sigmacore/releases/new"
echo "2. Verify PyPI: https://pypi.org/project/sigma-c-framework/"
echo "3. Update documentation site"
