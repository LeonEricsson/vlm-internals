from .qwen_wrapper import Qwen2_5_VLWrapper

_WRAPPERS = {
    "qwen2.5": Qwen2_5_VLWrapper,
}


def get_wrapper(model_name: str, device: str, **kwargs):
    key = model_name.lower().split("/")[0]
    try:
        cls = _WRAPPERS[key]
    except KeyError:
        raise ValueError(f"No wrapper for model '{model_name}'")
    return cls(model_name=model_name, device=device, **kwargs)
