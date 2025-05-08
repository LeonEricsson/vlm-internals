import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from typing import List, Dict


class Qwen2_5_VLWrapper:
    """
    Wrapper for Qwen2.5 VL that extracts self-attention, activations, and logits.
    """

    def __init__(self, model_name: str, device: str = "cuda", eval: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        min_pixels = 128 * 28 * 28  # 128 tokens min
        max_pixels = 512 * 28 * 28  # 256 tokens max
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            padding_side="left",
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if eval else "eager",
        ).to(self.device)
        self.model.eval()

    def forward_and_capture(self, batch: List[dict]):
        # inputs = self._prepare_inputs(batch)
        outputs = self.model(
            **batch,
            output_hidden_states=True,
            output_attentions=True,
        )

        attns = outputs.attentions
        logits = outputs.logits
        acts = outputs.hidden_states

        return {"attns": attns, "acts": acts, "logits": logits}


class Qwen2_5_DataCollator:
    def __init__(
        self,
        wrapper: Qwen2_5_VLWrapper,
    ):
        self.processor = wrapper.processor
        self.image_token_id = wrapper.model.config.image_token_id
        self.system_prompt = (
            "You are a helpful AI assistant. "
            "You will be shown an image and a statement about that image. "
            "Respond with exactly one word: True if the statement is correct, "
            "or False if it is not."
        )

    def __call__(self, samples: List[Dict]) -> Dict:
        """
        samples: list of dicts, each with 'image_options', 'caption_options', etc.
        Returns: a batched dict ready for model(**batch)
        """
        batch_messages = []

        # 1) Build messages per sample
        for sample in samples:
            content = [
                {"type": "image", "image": sample["image_options"][0]},
                {"type": "text", "text": sample["caption_options"][0]},
            ]
            msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ]
            batch_messages.append(msgs)

        # 2) Apply chat template in batch
        text_inputs = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in batch_messages
        ]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        batch = self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        img_pos = batch["input_ids"] == self.img_token_id
        txt_pos = ~img_pos

        batch["image_positions"] = img_pos
        batch["text_positions"] = txt_pos

        batch["labels"] = torch.tensor(
            [s["labels"][0] for s in samples], dtype=torch.long
        )
        return batch
