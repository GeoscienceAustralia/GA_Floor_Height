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
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ---------- 3. cacheable layer: Python deps (no project code) -----
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --all-extras --no-install-project --compile-bytecode

# ---------- 4. copy source & install project itself --------------
COPY src/    ./src
COPY config/ ./config
RUN uv sync --locked --no-dev --compile-bytecode

# ---------- 5. non‑root user -------------------------------------
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN chown -R ${HOST_UID}:${HOST_GID} /app && \
    set -eux; \
    if ! getent group "${HOST_GID}" >/dev/null; then \
        groupadd -g "${HOST_GID}" appgroup; \
    fi; \
    if ! getent passwd "${HOST_UID}" >/dev/null; then \
        useradd -m -u "${HOST_UID}" -g "${HOST_GID}" appuser; \
    fi
USER ${HOST_UID}:${HOST_GID}

# ---------- 6. runtime environment vars --------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH=/opt/conda/lib \
    PROJ_LIB=/opt/conda/share/proj \
    GDAL_DATA=/opt/conda/share/gdal \
    UV_COMPILE_BYTECODE=1

# ---------- 7. entry‑point ---------------------------------------
ENTRYPOINT ["uv", "run", "--"]
CMD ["python", "-m", "floor_heights.main"]