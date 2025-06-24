FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    git-lfs \
    build-essential \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "floor-heights", "/bin/bash", "-c"]

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY . .

RUN uv pip install --python $(which python) -e . "pre-commit" "detect-secrets"

RUN git lfs install

RUN if [ -f .env.example ] && [ ! -f .env ]; then cp .env.example .env; fi

ENTRYPOINT ["conda", "run", "-n", "floor-heights"]

CMD ["fh", "--help"]