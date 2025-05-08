import os
import argparse
import time

import torch
import numpy as np
from tqdm import tqdm

from data import get_dataset
from model_utils import get_wrapper_collator


def debug_batch(loader, wrapper, device):
    print("[DEBUG] Inspecting first batch via DataLoader + collator…\n")
    batch = next(iter(loader))

    batch = {k: v.to(device) for k, v in batch.items()}
    wrapper.model.to(device)
    gen_ids = wrapper.model.generate(
        **batch,
        max_new_tokens=20,
    )
    generated = wrapper.processor.tokenizer.batch_decode(
        gen_ids, skip_special_tokens=True
    )[0]
    print(">>> [DEBUG] Generation:")
    print(generated, "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract attention maps, activations, and logits from VLMs."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--dataset",
        type=str,
        default="VSR",
        help="Dataset name as understood by get_dataset().",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./extractions",
        help="Directory to save the extracted tensors (.npz).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for extraction."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to process (for quick tests).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print model output shapes and first‐sample predictions for sanity checks",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = get_dataset(args.dataset)

    wrapper, collator = get_wrapper_collator(
        model_name=args.model, device=args.device, eval=False
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"Wrapper: {wrapper.__class__}, Collator: {collator.__class__}")

    if args.debug:
        debug_batch(loader, wrapper, args.device)
        return

    # 2) storage for everything
    storage = {
        "attns": [],
        "acts": [],
        "logits": [],
        "correct": [],
        "image_positions": [],
        "text_positions": [],
    }

    start_time = time.time()
    for idx, batch in enumerate(tqdm(loader, desc="Extracting", unit="batch")):
        labels = batch.pop("labels").to(args.device)  # [B]
        batch = {k: v.to(args.device) for k, v in batch.items()}

        out = wrapper.forward_and_capture(batch)

        # ——— compute True/False prediction per sample ———
        # take logits for the next-token (last position)
        last_logits = out["logits"][:, -1, :]  # [B, vocab]
        pred_ids = last_logits.argmax(dim=-1)  # [B]
        preds, trues = [], labels.cpu().tolist()
        for pid in pred_ids.cpu().tolist():
            tok = (
                wrapper.processor.tokenizer.decode([pid], skip_special_tokens=True)
                .strip()
                .lower()
            )
            if tok.startswith("true"):
                preds.append(1)
            elif tok.startswith("false"):
                preds.append(0)
            else:
                preds.append(None)
        # record correctness (None ⇒ treated as wrong)
        for p, t in zip(preds, trues):
            storage["correct"].append(bool(p == t))

        # ——— detach & accumulate arrays ———
        attn_np = np.stack([a.detach().cpu().numpy() for a in out["attns"]], axis=0)
        act_np = np.stack([h.detach().cpu().numpy() for h in out["acts"]], axis=0)
        logit_np = out["logits"].detach().cpu().numpy()
        img_pos = batch["image_positions"].cpu().numpy()
        txt_pos = batch["text_positions"].cpu().numpy()

        storage["attns"].append(attn_np)
        storage["acts"].append(act_np)
        storage["logits"].append(logit_np)
        storage["image_positions"].append(img_pos)
        storage["text_positions"].append(txt_pos)

        if args.limit and idx + 1 >= args.limit:
            break

    attns = np.concatenate(
        storage["attns"], axis=1
    )  # [layers, samples, heads, seq, seq]
    acts = np.concatenate(
        storage["acts"], axis=1
    )  # [layers+1, samples, seq, hidden_dim]
    logits = np.concatenate(storage["logits"], axis=0)  # [samples, seq, vocab]
    correct = np.array(storage["correct"], dtype=bool)  # [samples]

    out_path = os.path.join(args.output_dir, "all_tensors.npz")
    np.savez_compressed(
        out_path,
        attns=attns,
        acts=acts,
        logits=logits,
        correct=correct,
        image_positions=image_positions,
        text_positions=text_positions,
    )

    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.2f}s.")
    print(f"Processed samples: {logits.shape[0]}")
    print("Saved:")
    print(f"  • attns, acts, logits")
    print(f"  • correct flags (shape={correct.shape})")
    print(
        f"  • image_positions ({len(image_positions)} indices), text_positions ({len(text_positions)} indices)"
    )
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
