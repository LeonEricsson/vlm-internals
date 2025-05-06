import os
import argparse
import time

import torch
import numpy as np
from tqdm import tqdm

from data.datasets import get_dataset
from model_utils.wrapper import ModelWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract attention maps, activations, and logits from VLMs."
    )
    parser.add_argument(
        "--model", type=str, required=True
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name as understood by get_dataset()."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the extracted tensors (.npz)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for extraction."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional limit on number of samples to process (for quick tests)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = get_dataset(args.dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    wrapper = ModelWrapper(model_name=args.model, device=args.device)

    storage = {
        'attns': [], 
        'acts': [],  
        'logits': [] 
    }

    start_time = time.time()

    for idx, batch in enumerate(tqdm(loader, desc="Extracting", unit="batch")):
        out = wrapper.forward_and_capture(batch)

        storage['attns'].append(out['attns'].detach().cpu().numpy())
        storage['acts'].append(out['acts'].detach().cpu().numpy())
        storage['logits'].append(out['logits'].detach().cpu().numpy())

        if args.limit and idx + 1 >= args.limit:
            break

    elapsed = time.time() - start_time

    attns = np.stack(storage['attns'], axis=0)
    acts = np.stack(storage['acts'], axis=0)
    logits = np.stack(storage['logits'], axis=0)

    out_path = os.path.join(args.output_dir, 'all_tensors.npz')
    np.savez_compressed(out_path, attns=attns, acts=acts, logits=logits)

    num_samples = attns.shape[0]
    print(f"Extraction complete in {elapsed:.2f}s.")
    print(f"Processed samples: {num_samples}")
    print(f"Saved tensors to: {out_path}")


if __name__ == "__main__":
    main()
