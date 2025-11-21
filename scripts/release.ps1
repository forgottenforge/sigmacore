# Sigma-C Framework v2.0.0 Release Script (Windows)
# ===================================================

Write-Host "🚀 Sigma-C Framework v2.0.0 Release" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

# 1. Clean previous builds
Write-Host "📦 Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Path dist, build, *.egg-info -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "✓ Clean complete" -ForegroundColor Green
Write-Host ""

# 2. Build package
Write-Host "🔨 Building package..." -ForegroundColor Yellow
python -m build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "✓ Build complete" -ForegroundColor Green
Write-Host ""

# 3. Check package
Write-Host "🔍 Checking package..." -ForegroundColor Yellow
twine check dist/*
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "✓ Package check passed" -ForegroundColor Green
Write-Host ""

# 4. Git operations
Write-Host "📝 Git operations..." -ForegroundColor Yellow
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
Write-Host "✓ Git commit and tag created" -ForegroundColor Green
Write-Host ""

# 5. Push to GitHub
Write-Host "🌐 Pushing to GitHub..." -ForegroundColor Yellow
git push origin main
git push origin v2.0.0
Write-Host "✓ Pushed to GitHub" -ForegroundColor Green
Write-Host ""

# 6. Upload to PyPI
Write-Host "📤 Uploading to PyPI..." -ForegroundColor Yellow
Write-Host "⚠️  This will upload to PRODUCTION PyPI!" -ForegroundColor Red
$response = Read-Host "Continue? (y/n)"
if ($response -eq 'y') {
    twine upload dist/*
    Write-Host "✓ Uploaded to PyPI" -ForegroundColor Green
} else {
    Write-Host "⚠️  PyPI upload skipped" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "✅ Release v2.0.0 complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Create GitHub release: https://github.com/forgottenforge/sigmacore/releases/new"
Write-Host "2. Verify PyPI: https://pypi.org/project/sigma-c-framework/"
Write-Host "3. Update documentation site"
