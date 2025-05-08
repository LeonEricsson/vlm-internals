from .qwen import Qwen2_5_VLWrapper, Qwen2_5_DataCollator

_WRAPPERS = {"qwen2.5": (Qwen2_5_VLWrapper, Qwen2_5_DataCollator)}


def get_wrapper_collator(model_name: str, device: str, **kwargs):
    name = model_name.lower()
    for family, (WrapperCls, CollatorCls) in _WRAPPERS.items():
        if family in name:
            wrapper = WrapperCls(model_name=model_name, device=device, **kwargs)
            collator = None
            if CollatorCls:
                collator = CollatorCls(wrapper)
            return wrapper, collator
    raise ValueError(f"No wrapper found for model '{model_name}'")
