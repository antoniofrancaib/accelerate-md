from .realnvp import create_realnvp_flow
from .tarflow import create_tarflow_flow

# Unified registry for flow factories – downstream code can remain agnostic
# to the concrete architecture.
MODEL_REGISTRY = {
    "realnvp": create_realnvp_flow,
    "tarflow": create_tarflow_flow,
}
