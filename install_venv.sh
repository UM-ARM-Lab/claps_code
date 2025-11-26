#!/bin/bash
set -e  # Exit on any error

echo "🚀 Installing CLAPS with Python venv (no conda needed)..."

CLAPS_DIR=$(pwd)

if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 is required but not found. Please install Python 3.10."
    exit 1
fi

PYTHON_VERSION=$(python3.10 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python version check passed: $(python3.10 --version)"

echo "📦 Initializing git submodules..."
git submodule update --init --recursive

echo "🐍 Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi
python3.10 -m venv venv

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip

# Check if CUDA is available and install PyTorch accordingly
echo "🔍 Checking for CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
else
    echo "ℹ️  No CUDA detected. Installing CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "📦 Installing core dependencies..."
pip install numpy scipy matplotlib plotly tqdm pyyaml
pip install pyvista rerun-sdk h5py cgal alphashape opencv-python
pip install lcm  # For processing real MBot data from .lcm log files

echo "📦 Installing luis_utils..."
cd external/luis_utils
pip install -e .
cd $CLAPS_DIR

echo "📦 Installing pymatlie..."
cd external/pymatlie
pip install -e .
cd $CLAPS_DIR

echo "📦 Installing CLAPS package..."
pip install -e .

echo "✅ Installation complete!"
echo ""
echo "📝 Virtual environment created: venv/"
echo ""
echo "🚀 To use CLAPS:"
echo "   source venv/bin/activate"
echo ""