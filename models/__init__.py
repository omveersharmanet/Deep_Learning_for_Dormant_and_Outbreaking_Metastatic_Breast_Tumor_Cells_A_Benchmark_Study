# __init__.py
def get_model_registry(framesize):

    if framesize == 96:
        from .all_models_96 import MODEL_REGISTRY

    elif framesize == 64:
        from .all_models_64 import MODEL_REGISTRY

    else:
        raise ValueError(f"Unsupported framesize: {framesize}")

    return MODEL_REGISTRY
