#!/bin/bash
set -e

echo "=== Floor Heights Project Setup ==="

mkdir -p data output config

if ! command -v mamba &> /dev/null && ! command -v conda &> /dev/null; then
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    curl -L $MINIFORGE_URL -o miniforge.sh
    bash miniforge.sh -b -p $HOME/miniforge3
    rm miniforge.sh
    
    export PATH="$HOME/miniforge3/bin:$PATH"
    
    if [[ "$SHELL" == */zsh ]]; then
        PROFILE="$HOME/.zshrc"
        CONDA_HOOK='eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"'
    else
        PROFILE="$HOME/.bashrc"
        CONDA_HOOK='eval "$($HOME/miniforge3/bin/conda shell.bash hook)"'
    fi
    
    if ! grep -q "miniforge3/bin/conda" "$PROFILE"; then
        echo "$CONDA_HOOK" >> "$PROFILE"
    fi
    
    eval "$($HOME/miniforge3/bin/conda shell.$(basename $SHELL) hook)"
fi

if ! conda env list | grep -q "floor-heights"; then
    mamba create -y -n floor-heights -c conda-forge \
        python=3.12 \
        pdal python-pdal \
        gdal proj proj-data \
        fiona shapely geopandas \
        scikit-image scikit-learn \
        uv
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate floor-heights

uv pip install -e .

if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
fi

if command -v pre-commit &> /dev/null; then
    pre-commit install
fi

echo "=== Setup Complete ==="
echo "To activate the environment:"
echo "  conda activate floor-heights"
echo "  source .env"