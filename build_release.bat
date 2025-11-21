@echo off
echo ==========================================
echo Sigma-C v1.2.0 Release Builder
echo ==========================================

echo 1. Cleaning old builds...
rmdir /s /q build
rmdir /s /q dist
rmdir /s /q sigma_c_framework.egg-info

echo 2. Building Source Distribution and Wheel...
python setup.py sdist bdist_wheel

echo 3. Verifying Build...
python -m twine check dist/*

echo ==========================================
echo Build Complete!
echo To upload to PyPI, run:
echo python -m twine upload dist/*
echo ==========================================
echo ==========================================
