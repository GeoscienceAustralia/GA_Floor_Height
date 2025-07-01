#!/bin/bash
if [ ! -d ~/miniconda3 ]; then
    echo "Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
else
    echo "Miniconda already installed, skipping..."
fi

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv already installed, skipping..."
fi

export PATH="$HOME/.local/bin:$HOME/miniconda3/bin:$PATH"

if ! grep -q "conda initialize" ~/.bashrc; then
    echo "Initializing conda for bash..."
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda config --set auto_activate_base false
else
    echo "Conda already initialized, skipping..."
fi

eval "$(~/miniconda3/bin/conda shell.bash hook)"

if ! conda env list | grep -q "^floor-heights "; then
    echo "Creating conda environment..."
    conda env create -f environment.yml
else
    echo "Conda environment 'floor-heights' already exists, updating..."
    conda env update -f environment.yml --prune
fi

source ~/miniconda3/bin/activate floor-heights

echo "Installing project, pre-commit, and detect-secrets..."
uv pip install --python "$(which python)" -e . "pre-commit" "detect-secrets" "commitizen"

if ! command -v git-lfs &> /dev/null; then
    echo "Installing Git LFS..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs install
else
    echo "Git LFS already installed, skipping..."
fi

git lfs pull

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
. "$HOME/.nvm/nvm.sh"
nvm install 24
node -v
nvm current
npm -v

echo ""
echo "Configuring pre-commit..."
pre-commit install
pre-commit install --hook-type commit-msg

if [ ! -f .secrets.baseline ]; then
    echo "Creating detect-secrets baseline..."
    detect-secrets scan . > .secrets.baseline
    echo "✓ Created .secrets.baseline. Please review and commit it."
else
    echo "✓ .secrets.baseline already exists."
fi

if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✓ Created .env file from .env.example"
        echo ""
        echo "⚠️  IMPORTANT: You need to edit .env and add your credentials:"
        echo "   1. AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for S3 access"
        echo "   2. AWS_DEFAULT_REGION (typically ap-southeast-2)"
        echo "   3. Optional: HF_TOKEN for Hugging Face model access"
        echo ""
        echo "Edit the file with: nano .env"
    else
        echo "⚠️  Warning: .env.example not found, cannot create .env file"
        echo "   Please create a .env file with your AWS credentials"
    fi
else
    echo ""
    echo "✓ .env file already exists, keeping existing configuration"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. If you haven't already, edit .env to add your credentials:"
echo "   nano .env"
echo ""
echo "2. To use conda in your current terminal, run:"
echo "   source ~/.bashrc"
echo ""
echo "3. Then activate the environment with:"
echo "   conda activate floor-heights"
echo ""
echo "4. Test the installation with:"
echo "   fh --help"
