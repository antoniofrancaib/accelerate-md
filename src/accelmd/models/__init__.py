from .realnvp import create_realnvp_flow

# Unified registry for flow factories – downstream code can remain agnostic
# to the concrete architecture.
MODEL_REGISTRY = {
    "realnvp": create_realnvp_flow,
}
