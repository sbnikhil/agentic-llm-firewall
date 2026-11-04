FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
# Install dependencies first so this layer is cached when only code changes.
# uv.lock includes the spaCy model, so no separate `spacy download` step.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/opt/venv/bin:$PATH"
# Nothing is served yet: the v0 app is frozen in legacy/v0 (see
# docs/decisions/0001-retire-v0-app.md). The image checks that the engine
# imports; importing it loads no model and opens no connection.
CMD ["python", "-c", "import llmfirewall.engine.detectors.input_rules, llmfirewall.engine.detectors.output_filter, llmfirewall.engine.detectors.pii; print('llmfirewall engine imports OK')"]
