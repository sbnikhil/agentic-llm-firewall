# LiteLLM's official proxy image (database variant) with the firewall added.
# Build context: the repository root (see docker-compose.yml).
#
# LiteLLM's own environment (/app/.venv) is left exactly as LiteLLM ships it.
# The firewall and the packages it needs are installed from uv.lock into a
# separate venv (/opt/llmfw), which a .pth file appends to the END of
# LiteLLM's sys.path: for any package both have, LiteLLM's version wins, and
# /opt/llmfw only supplies what LiteLLM lacks (Presidio, spaCy, torch,
# sentence-transformers, llmfirewall). See docs/decisions/0003.
FROM ghcr.io/berriai/litellm-database:v1.108.3@sha256:aecfcc459b5af208f04b8e916c9912cc3001798e013b4874be1831c20e4a2c26

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/llmfw \
    UV_PYTHON=/usr/bin/python3.13 \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/hf

WORKDIR /opt/llmfirewall-src
# Dependencies first, so this layer is cached when only code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
    && echo /opt/llmfw/lib/python3.13/site-packages \
       > /app/.venv/lib/python3.13/site-packages/zzz-llmfirewall.pth

# Data the engine reads at runtime (paths set in docker-compose.yml).
COPY policies /opt/llmfirewall/policies
COPY secrets_catalog.json /opt/llmfirewall/secrets_catalog.json
# The compose stack's LiteLLM config and dev identity map.
COPY deploy/compose/litellm/config.yaml /etc/litellm/config.yaml
COPY deploy/compose/policies/identity.yaml /etc/llmfirewall/identity.yaml

# Download the sentence model at build time, so the proxy starts without
# network access (HF_HUB_OFFLINE=1 at runtime).
RUN /opt/llmfw/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

WORKDIR /app
