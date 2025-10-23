#!/bin/bash

# Upload to PyPI Script
# This script uploads the package to PyPI

set -e  # Exit on error

echo "========================================="
echo "PyPI Upload Script"
echo "========================================="
echo ""

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "Error: dist/ directory not found!"
    echo "Please run 'python3 -m build' first."
    exit 1
fi

# Check if package files exist
if [ ! -f "dist/lownocompute_ai_baseline-1.0.0-py3-none-any.whl" ]; then
    echo "Error: Wheel file not found!"
    exit 1
fi

if [ ! -f "dist/lownocompute_ai_baseline-1.0.0.tar.gz" ]; then
    echo "Error: Source distribution not found!"
    exit 1
fi

echo "Step 1: Checking package with twine..."
twine check dist/*

if [ $? -ne 0 ]; then
    echo "Error: Package check failed!"
    exit 1
fi

echo "✓ Package check passed!"
echo ""

echo "Step 2: Choose upload destination"
echo "1) TestPyPI (recommended for first upload)"
echo "2) PyPI (production)"
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "Uploading to TestPyPI..."
        echo "You will need your TestPyPI API token"
        echo "Username: __token__"
        echo "Password: Your TestPyPI token (starts with pypi-)"
        echo ""
        twine upload --repository testpypi dist/*
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "========================================="
            echo "✓ Upload to TestPyPI successful!"
            echo "========================================="
            echo ""
            echo "Test installation:"
            echo "pip install --index-url https://test.pypi.org/simple/ lownocompute-ai-baseline"
            echo ""
            echo "View package:"
            echo "https://test.pypi.org/project/lownocompute-ai-baseline/"
        fi
        ;;
    2)
        echo ""
        echo "WARNING: You are about to upload to PRODUCTION PyPI!"
        read -p "Are you sure? (yes/no): " confirm
        
        if [ "$confirm" != "yes" ]; then
            echo "Upload cancelled."
            exit 0
        fi
        
        echo ""
        echo "Uploading to PyPI..."
        echo "You will need your PyPI API token"
        echo "Username: __token__"
        echo "Password: Your PyPI token (starts with pypi-)"
        echo ""
        twine upload dist/*
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "========================================="
            echo "✓ Upload to PyPI successful!"
            echo "========================================="
            echo ""
            echo "Installation:"
            echo "pip install lownocompute-ai-baseline"
            echo ""
            echo "View package:"
            echo "https://pypi.org/project/lownocompute-ai-baseline/"
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Done!"

