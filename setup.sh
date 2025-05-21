#!/usr/bin/env bash
set -euo pipefail

echo "=== Floor-Heights local bootstrap ==="

# --------------------------------------------------------------------
# 1. Install micromamba in ~/bin if it isn't already there
# --------------------------------------------------------------------
if ! command -v micromamba &>/dev/null; then
  mkdir -p "$HOME/bin"
  echo "Installing micromamba to $HOME/bin..."
  curl -Ls "https://micro.mamba.pm/api/micromamba/$(uname -s | tr A-Z a-z)-$(uname -m)/latest" \
    | tar -xvj -C "$HOME/bin" --strip-components=1 bin/micromamba
  
  # Add to PATH if not already there
  if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
    echo "Adding $HOME/bin to PATH for current session"
    export PATH="$HOME/bin:$PATH"
    
    # Suggest adding to shell startup
    if [[ "$SHELL" == */zsh ]]; then
      PROFILE="$HOME/.zshrc"
    else
      PROFILE="$HOME/.bashrc"
    fi
    
    echo "Consider adding the following line to your $PROFILE:"
    echo 'export PATH="$HOME/bin:$PATH"'
  fi
fi

# --------------------------------------------------------------------
# 2. Create directories for project data
# --------------------------------------------------------------------
mkdir -p data output config

# --------------------------------------------------------------------
# 3. Create/update the conda environment with system dependencies
# --------------------------------------------------------------------
echo "Creating/updating conda environment 'floor-heights'..."
micromamba create -y -n floor-heights -c conda-forge \
  python=3.12 \
  pdal python-pdal \
  gdal proj proj-data \
  fiona shapely geopandas \
  scikit-image scikit-learn \
  uv \
  pip

# --------------------------------------------------------------------
# 4. Activate env + install Python deps via UV
# --------------------------------------------------------------------
echo "Activating environment and installing Python dependencies..."
eval "$(micromamba shell hook bash)"
micromamba activate floor-heights

# Install Python package in development mode
uv pip install --system -e .

# --------------------------------------------------------------------
# 5. Setup environment file and hooks
# --------------------------------------------------------------------
if [ ! -f .env ] && [ -f .env.example ]; then
  echo "Creating .env file from template..."
  cp .env.example .env
fi

if command -v pre-commit &>/dev/null; then
  echo "Setting up pre-commit hooks..."
  pre-commit install
fi

echo "✅ Setup complete — activate any time with: micromamba activate floor-heights"
echo "  To use in a new terminal: eval \"\$(micromamba shell hook bash)\" && micromamba activate floor-heights"