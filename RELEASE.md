# Sigma-C Framework - Release Guide

**Version 1.0.0** | **Copyright (c) 2025 ForgottenForge.xyz**

## 🚀 Schritt-für-Schritt: GitHub + PyPI Release

### Vorbereitung (Einmalig)

#### 1. GitHub Repository erstellen
```bash
# Auf GitHub.com:
# - Gehe zu https://github.com/new
# - Repository Name: "sigmacore" (oder dein Wunschname)
# - Description: "Critical Susceptibility Framework"
# - Public oder Private (deine Wahl)
# - NICHT "Initialize with README" anklicken (haben wir schon)
# - Erstellen
```

#### 2. PyPI Account vorbereiten
```bash
# 1. Account erstellen auf https://pypi.org/account/register/
# 2. Email verifizieren
# 3. API Token erstellen:
#    - Gehe zu https://pypi.org/manage/account/token/
#    - "Add API token"
#    - Token Name: "sigma-c-upload"
#    - Scope: "Entire account" (beim ersten Upload)
#    - Token SICHER SPEICHERN (wird nur einmal angezeigt!)
```

### GitHub Upload

#### 1. Git Repository initialisieren
```bash
cd d:/code/sigmacore/sigma_c_framework

# Git initialisieren
git init

# Remote hinzufügen (ersetze USERNAME mit deinem GitHub-Namen)
git remote add origin https://github.com/forgottenforge/sigmacore.git

# Dateien hinzufügen (nur die wichtigen, .gitignore filtert automatisch)
git add .

# Ersten Commit
git commit -m "Initial release v1.0.0 - Sigma-C Framework"

# Branch umbenennen (falls nötig)
git branch -M main

# Hochladen
git push -u origin main
```

#### 2. GitHub Release erstellen
```bash
# Tag erstellen
git tag -a v1.0.0 -m "Release v1.0.0 - Production-ready Sigma-C Framework"
git push origin v1.0.0

# Dann auf GitHub:
# - Gehe zu deinem Repository
# - Klicke "Releases" → "Create a new release"
# - Tag: v1.0.0
# - Title: "Sigma-C Framework v1.0.0"
# - Description: (siehe unten)
# - "Publish release"
```

**Release Description Template:**
```markdown
# Sigma-C Framework v1.0.0 🚀

First production release of the critical susceptibility framework.

## Features
- 6 Domain Adapters: Quantum, GPU, Financial, Climate, Seismic, Magnetic
- High-performance C++ core with Python bindings
- Comprehensive documentation (English + German)
- Production-ready examples

## Installation
```bash
pip install sigma-c-framework
```

## Quick Start
See [QUICKSTART.md](QUICKSTART.md) for a 5-minute introduction.

## License
Dual-licensed: AGPL-3.0 or Commercial
Contact: nfo@forgottenforge.xyz
```

### PyPI Upload

#### 1. Build-Tools installieren
```bash
pip install --upgrade build twine
```

#### 2. Package bauen
```bash
cd d:/code/sigmacore/sigma_c_framework

# Clean build (wichtig!)
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Package erstellen
python -m build
```

Dies erstellt:
- `dist/sigma_c_framework-1.0.0.tar.gz` (Source)
- `dist/sigma_c_framework-1.0.0-*.whl` (Wheel)

#### 3. Auf PyPI hochladen

**WICHTIG:** Teste zuerst auf TestPyPI!

```bash
# Test-Upload auf TestPyPI (empfohlen für ersten Versuch)
python -m twine upload --repository testpypi dist/*
# Username: __token__
# Password: <dein TestPyPI API Token>

# Testen der Installation
pip install --index-url https://test.pypi.org/simple/ sigma-c-framework

# Wenn alles funktioniert: Echter Upload
python -m twine upload dist/*
# Username: __token__
# Password: <dein PyPI API Token>
```

#### 4. Verifizieren
```bash
# Nach 1-2 Minuten sollte es verfügbar sein:
pip install sigma-c-framework

# Testen
python -c "from sigma_c import Universe; print(Universe.quantum())"
```

### Was wird NICHT hochgeladen?

Dank `.gitignore` und `MANIFEST.in` werden automatisch ausgeschlossen:
- ❌ `examples/code/` (alte Beispiele)
- ❌ `cache*/` (Cache-Verzeichnisse)
- ❌ `*.png`, `*.pdf` (Output-Dateien)
- ❌ `build/`, `__pycache__/` (Build-Artefakte)
- ❌ `.venv/` (Virtual Environment)

Was WIRD hochgeladen:
- ✅ `sigma_c/` (Python Package)
- ✅ `sigma_c_core/` (C++ Source)
- ✅ `examples_v4/` (Demo Scripts)
- ✅ `README.md`, `DOCUMENTATION.md`, `QUICKSTART.md`
- ✅ `license_*.txt`

### Troubleshooting

#### Problem: "Package already exists"
```bash
# Version erhöhen in setup.py und pyproject.toml
# z.B. 1.0.0 → 1.0.1
# Dann neu bauen und uploaden
```

#### Problem: C++ Compilation Error beim Install
```bash
# Stelle sicher, dass pybind11 in pyproject.toml steht
# User brauchen einen C++ Compiler (MSVC, GCC, Clang)
# Dokumentiere das in README.md unter "Prerequisites"
```

#### Problem: Import Error nach Installation
```bash
# Prüfe, ob alle Packages in setup.py gelistet sind
# Teste in frischer venv:
python -m venv test_env
test_env\Scripts\activate
pip install sigma-c-framework
python -c "from sigma_c import Universe"
```

### Automatisierung (Optional)

Erstelle `.github/workflows/release.yml` für automatische Releases:
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Build
        run: |
          pip install build
          python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### Nächste Schritte nach Release

1. **Ankündigung**: Poste auf Reddit, Twitter, LinkedIn
2. **Badge hinzufügen**: PyPI Badge in README.md
3. **Monitoring**: Prüfe Download-Statistiken auf PyPI
4. **Issues**: Beantworte GitHub Issues zeitnah

### Checkliste vor Release

- [ ] Alle Tests laufen (alle 6 Demos funktionieren)
- [ ] Version in `setup.py` und `pyproject.toml` ist korrekt
- [ ] `CHANGELOG.md` erstellt (optional, aber empfohlen)
- [ ] Dokumentation ist aktuell
- [ ] `.gitignore` ist vollständig
- [ ] Lizenzen sind korrekt
- [ ] GitHub Repository ist erstellt
- [ ] PyPI Account + API Token bereit

---

**Du bist bereit! 🚀 Folge den Schritten oben und dein Framework ist live.**
