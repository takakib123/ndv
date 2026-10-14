import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import VideoMAEForVideoClassification

from src import config


def _restore_qv_bias(model: VideoMAEForVideoClassification, model_name: str) -> int:
    """Copy the hub checkpoint's q_bias/v_bias into query.bias/value.bias.

    The hub VideoMAE weights store attention biases as `q_bias`/`v_bias` (no key bias). From
    transformers 5.x the module uses plain `query.bias`/`key.bias`/`value.bias`, and
    from_pretrained does not remap them -- the pretrained biases (|b| up to ~2.9) are silently
    left at zero init. key.bias stays zero, matching the original VideoMAE (a key bias is
    softmax-invariant anyway). Returns the number of tensors restored (0 on transformers 4.x,
    where q_bias/v_bias load natively).
    """
    path = hf_hub_download(model_name, "model.safetensors")
    restored = 0
    params = dict(model.named_parameters())
    with safe_open(path, framework="pt") as f, torch.no_grad():
        for key in f.keys():
            if not key.startswith("videomae.encoder.") or not key.endswith(("q_bias", "v_bias")):
                continue
            target = key.replace("q_bias", "query.bias").replace("v_bias", "value.bias")
            if target in params:
                params[target].copy_(f.get_tensor(key))
                restored += 1
    return restored


def build_model(num_classes: int = config.NUM_CLASSES, model_name: str = config.MODEL_NAME):
    """Pretrained VideoMAE with a fresh `num_classes` head and correctly loaded attention biases."""
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    _restore_qv_bias(model, model_name)
    return model
