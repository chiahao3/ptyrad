#!/bin/bash
# test_conda_packaging.sh - Sanity check for Conda-Forge local builds

# Exit immediately if a command exits with a non-zero status
set -e 

## 0. Ensure conda-build is installed
if ! command -v conda-build &> /dev/null; then
    echo "❌ ERROR: conda-build is not installed. Run 'conda install conda-build -y' first."
    exit 1
fi

## 1. Remove old conda build files
echo "🧹 Cleaning old conda build artifacts..."
conda build purge

## 2. Build the conda package
# This will build the package and run the `test` section defined in meta.yaml
echo "📦 Building the Conda package..."
conda build recipe/

## 3. Get the exact path to the generated package (.tar.bz2 or .conda)
CONDA_PKG=$(conda build recipe/ --output)
echo "✅ Package built at: $CONDA_PKG"

## 4. Create a local test environment
echo "🧪 Creating an isolated virtual environment..."
conda create -n test_env_conda python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate test_env_conda

## 5. Install the built PtyRAD package
# --use-local tells conda to look in your local build cache first
echo "⚙️ Installing the fresh Conda package..."
conda install --use-local ptyrad -y

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
conda remove -n test_env_conda --all -y

echo "🎉 All tests passed! You are ready to push to conda-forge."