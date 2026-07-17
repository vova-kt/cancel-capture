FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.6.13

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 999 cancel-capture \
    && useradd --system --uid 999 --gid cancel-capture --home-dir /tmp cancel-capture \
    && mkdir -p /data /app \
    && chown cancel-capture:cancel-capture /data /app \
    && chmod 0700 /data

WORKDIR /app
COPY --from=builder --chown=cancel-capture:cancel-capture /app/.venv ./.venv
COPY --chown=cancel-capture:cancel-capture src ./src
COPY --chown=cancel-capture:cancel-capture .streamlit/config.toml ./.streamlit/config.toml

USER cancel-capture
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["cancel-capture", "doctor", "--quiet"]

CMD ["cancel-capture", "bot"]
