"""Settings for every test suite, applied before any test module imports.

LiteLLM downloads its model cost map from GitHub when it is imported, unless
LITELLM_LOCAL_MODEL_COST_MAP is set. The bundled map is used instead, so
importing the adapter opens no connection (tests/unit must run offline).
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
