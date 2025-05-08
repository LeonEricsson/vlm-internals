import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class Qwen2_5_VLWrapper:
    """
    Wrapper for Qwen2.5 VL that extracts self-attention, activations, and logits.
    """

    def __init__(self, model_name: str, device: str = "cuda", layers=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        # Layers to hook (defaults to all)
        self.layers = layers

    def _prepare_inputs(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        return self.processor(
            text=[text],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    def forward_and_capture(self, batch: dict) -> dict:
        """
        Runs a forward pass on a batch and returns:
          - attns: [num_layers, batch, heads, seq, seq]
          - acts:  [num_layers, batch, seq, hidden_dim]
          - logits: [batch, seq, vocab]
        """
        inputs = self._prepare_inputs(batch["messages"])
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
        )

        attns = outputs.attentions
        logits = outputs.logits
        acts = outputs.hidden_states

        return {"attns": attns, "acts": acts, "logits": logits}
