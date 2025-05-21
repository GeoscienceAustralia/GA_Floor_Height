FROM condaforge/miniforge3:latest

WORKDIR /app

RUN mamba install -y -c conda-forge \
        python=3.12 \
        pdal python-pdal \
        gdal proj proj-data \            
        fiona shapely geopandas \
        scikit-image scikit-learn \
        uv \
    && mamba clean --all --yes

COPY pyproject.toml uv.lock README.md ./

COPY src/    ./src
COPY config/ ./config

RUN uv pip install --system -e .

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

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH=/opt/conda/lib \
    PROJ_LIB=/opt/conda/share/proj \
    GDAL_DATA=/opt/conda/share/gdal

CMD ["python", "-m", "floor_heights.main"]