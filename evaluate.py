import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from data import get_dataset
from model_utils import get_wrapper_collator


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VSR benchmark with Qwen VL True/False"
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument(
        "--dataset",
        type=str,
        default="VSR",
        help="Dataset name as understood by get_dataset().",
    )
    p.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    p.add_argument(
        "--device", type=str, default="cuda", help="Device for model inference"
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1) load
    dataset = get_dataset(args.dataset)
    wrapper, collator = get_wrapper_collator(
        model_name=args.model, device=device, eval=True
    )
    processor = wrapper.processor

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    # 2) move model to device
    model = wrapper.model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    unrecog = 0

    # 3) loop
    for batch in tqdm(loader, desc="Evaluating"):
        labels = batch.pop("labels").to(device)  # [B]
        batch = {k: v.to(device) for k, v in batch.items()}

        # generate one token per sample
        with torch.no_grad():
            gen_ids = model.generate(
                **batch,
                max_new_tokens=1,
                use_cache=False,
            )

        final_gen_ids = gen_ids[:, -1]

        answers = processor.tokenizer.batch_decode(
            final_gen_ids, skip_special_tokens=True
        )

        for ans, true_lbl in zip(answers, labels.tolist()):
            a = ans.strip().lower()
            if a.startswith("true"):
                pred = 1
            elif a.startswith("false"):
                pred = 0
            else:
                pred = None

            if pred is None:
                unrecog += 1
            else:
                y_true.append(true_lbl)
                y_pred.append(pred)

    # 4) report
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = 100 * correct / total if total > 0 else 0.0

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nEvaluated (excluding {unrecog} unrecognized): {total}")
    print(f"Correct:   {correct}")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    if unrecog:
        print(f"Warning: {unrecog} responses were not ‘True’/‘False’ and were skipped.")


if __name__ == "__main__":
    main()
