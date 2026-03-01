#!/bin/bash
# test_pypi_packaging.sh - Sanity check for PyPI releases

# Exit immediately if a command exits with a non-zero status
set -e 

## 0. Ensure build is installed
if ! command -v python -m build &> /dev/null; then
    echo "❌ ERROR: build is not installed. Run 'pip install build -y' first."
    exit 1
fi

## 1. Remove old build files
echo "🧹 Cleaning old builds and environments..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

## 2. Build the pypi package
echo "📦 Building the wheel..."
python -m build
WHEEL_FILE=$(ls dist/*.whl)

## 3. Create a local test environment
echo "🧪 Creating an isolated virtual environment..."
conda create -n test_env_pypi python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate test_env_pypi

## 4. Install the built PtyRAD package
echo "⚙️ Installing the fresh wheel..."
pip install "$WHEEL_FILE"

## 5. Test CLI entry point and module execution
echo "🚀 Testing the entry point..."

# Test CLI execution
if ptyrad --help > /dev/null; then
    echo "✅ SUCCESS: ptyrad --help executed correctly."
else
    echo "❌ ERROR: Entry point failed."
    exit 1
fi

# Test module execution (important for accelerate launch)
if python -m ptyrad --help > /dev/null; then
    echo "✅ SUCCESS: python -m ptyrad --help executed correctly."
else
    echo "❌ ERROR: 'python -m ptyrad' entry point failed."
    exit 1
fi

## 6. Test and Verify the 'ptyrad init' behavior
TEST_OUT_DIR="test_init"
echo "🔍 Running integration test script..."

if python tools/verify_ptyrad_init.py "$TEST_OUT_DIR"; then
    echo "✅ Integration test complete."
else
    echo "❌ Integration test failed."
    exit 1
fi

## 7. Clean up the local test environment
echo "🧹 Cleaning up..."
conda deactivate
conda remove -n test_env_pypi --all -y

echo "🎉 All tests passed! You are ready to push to PyPI."