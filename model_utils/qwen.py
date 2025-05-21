import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from typing import List, Dict

MIN_PIXELS = 250 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
MAX_SEQ_LEN = 300


class Qwen2_5_VLWrapper:
    """
    Wrapper around the Qwen2.5 vision-language model for inference and feature extraction.

    Args:
        model_name (str): Identifier of the pretrained model to load.
        device (str): Device specifier for torch (e.g., 'cuda' or 'cpu').
        eval (bool): If True, use flash attention. Can not be True if we want to extract
                    attention weights + hidden states.

    Attributes:
        processor: AutoProcessor handling tokenization and vision preprocessing.
        model: Qwen2.5 VL model instance loaded onto the specified device.
    """

    def __init__(self, model_name: str, device: str = "cuda", eval: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            padding_side="left",
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if eval else "eager",
        ).to(self.device)
        self.model.eval()

    def forward_and_capture(self, batch: List[dict]):
        """
        Run a forward pass to capture attention maps, hidden states, and output logits.

        Args:
            batch (Dict[str, torch.Tensor]):
                Contains model inputs such as 'input_ids', 'pixel_values',
                'attention_mask', etc., all as PyTorch tensors on correct device.

        Returns:
            Dict[str, tuple]:
                'attns': tuple of attention tensors per layer (each [B, heads, seq, seq]).
                'acts': tuple of hidden-state tensors per layer (each [B, seq, hidden_dim]).
                'logits': output logits tensor of shape [B, seq, vocab_size].
        """
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
    """
    Data collator that converts raw visual reasoning samples into model-ready batches.
    """

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
        Build batched inputs from a list of raw samples.

        Args:
            samples (List[Dict]):
                Each dict must have 'image_options', 'caption_options', and 'labels'.

        Returns:
            Dict[str, torch.Tensor]:
                - Model input tensors ('input_ids', 'pixel_values', etc.)
                - 'image_positions': BoolTensors [B, seq] masks
                - 'labels': LongTensor [B]
        """
        batch_messages = []

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
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )

        img_pos = batch["input_ids"] == self.image_token_id

        batch["image_positions"] = img_pos

        batch["labels"] = torch.tensor(
            [s["labels"][0] for s in samples], dtype=torch.long
        )

        batch["attention_mask"] = batch["attention_mask"].to(torch.int8)
        return batch
