# ------------------------------------------------------------------
# Floor‑Heights runtime image ─ Conda system libs + UV Python layer
# ------------------------------------------------------------------
FROM mambaorg/micromamba:2.1.1-debian12-slim

WORKDIR /app

# ---------- 1. system / compiled dependencies ---------------------
COPY environment.yml .
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes

# ---------- 2. install uv (stand‑alone binary) --------------------
RUN micromamba run -n base pip install uv
ENV PATH="/opt/conda/bin:${PATH}"
ENV UV_SYSTEM_PYTHON=1
ENV CONDA_PREFIX=/opt/conda

# ---------- 3. cacheable layer: Python deps (no project code) -----
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --compile-bytecode

# ---------- 4. copy source & install project itself --------------
COPY src/ ./src
COPY config/ ./config  
RUN mkdir -p /tmp/build && cp -r /app/* /tmp/build/ && \
    cd /tmp/build && uv sync --locked --no-dev --no-editable --compile-bytecode && \
    cp -r .venv /app/

# ---------- 6. runtime environment vars --------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH=/opt/conda/lib \
    PROJ_LIB=/opt/conda/share/proj \
    GDAL_DATA=/opt/conda/share/gdal \
    UV_COMPILE_BYTECODE=1

# ---------- 7. entry‑point ---------------------------------------
ENTRYPOINT ["/app/.venv/bin/python", "-m"]
CMD ["floor_heights.main"]