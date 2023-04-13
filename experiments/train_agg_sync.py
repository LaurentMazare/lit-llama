"""
This script is a placeholder for training LLaMA from scratch.
Currently, it just trains on the Shakespeare dataset.
"""

import os
from pathlib import Path
import sys
import time
from typing import Tuple

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import numpy as np

import lightning as L
from lightning.fabric.loggers import CSVLogger

from lit_llama.model import Block, LLaMA, LLaMAConfig

out_dir = "out"

eval_iters = 100
log_interval = 10

# Hyperparameters
learning_rate = 6e-4
micro_batch_size = 5
max_iters = 20000 // 4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0


def copy_params(src, dst, weight=1.0, accumulate=False):
    src_state = src.state_dict()
    dst_state = dst.state_dict()

    for k in src_state:
        if accumulate:
            dst_state[k] += weight * src_state[k]
        else:
            dst_state[k].copy_(weight * src_state[k])


def avg_params(srcs, dst):
    n = len(srcs)
    weight = 1 / n

    copy_params(srcs[0], dst, weight=weight, accumulate=False)

    for src in srcs[1:]:
        copy_params(src, dst, weight=weight, accumulate=True)


def main(logs_dir: str = "logs", agg_interval: int=100) -> None:
    logger = CSVLogger(logs_dir, name=f"lit-llama_agg", flush_logs_every_n_steps=1)

    fabric = L.Fabric(accelerator="auto", devices=1, loggers=logger)
    fabric.launch()
    fabric.seed_everything(1335 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    val_data = load_val_dataset()

    config = LLaMAConfig.from_name("30M")
    config.vocab_size = 100  # from prepare_shakespeare.py
    
    with fabric.device:
        agg_model = LLaMA(config)

    n_params = sum(p.numel() for p in agg_model.parameters())
    print(f"N parameters: {n_params * 1e-9:.3f}B")

    agg_model = fabric.setup_module(agg_model)

    agg_model.apply(agg_model._init_weights)

    agg_iter_num = 0

    agg_checkpoint = f"checkpoint_agg_iter{agg_iter_num:04d}.ckpt"
    agg_checkpoint_tmp = f"_{agg_checkpoint}"
    torch.save(agg_model.state_dict(), os.path.join(out_dir, agg_checkpoint_tmp))
    os.rename(os.path.join(out_dir, agg_checkpoint_tmp), os.path.join(out_dir, agg_checkpoint))

    time.sleep(10.0)

    with fabric.device:
        peer_model = LLaMA(config)

    while True:
        peers = [int(el[5:]) for el in os.listdir(out_dir) if el.startswith("peer_")]
        peers.sort()
        n_peers = len(peers)

        weight = 1.0 / n_peers

        for n, peer_id in enumerate(peers):
            expected_checkpoint = os.path.join(out_dir, f"checkpoint_peer{peer_id:02d}_iter{agg_iter_num:04d}.ckpt")
            print(f"Expecting {expected_checkpoint} for aggregation")
            while True:
                if os.path.exists(expected_checkpoint):
                    break
                time.sleep(0.5)

            with torch.no_grad():
                peer_model.load_state_dict(torch.load(expected_checkpoint))

                if n == 0:
                    copy_params(peer_model, agg_model, weight=weight, accumulate=False)
                else:
                    copy_params(peer_model, agg_model, weight=weight, accumulate=True)

                os.remove(expected_checkpoint)

        iter_num = agg_iter_num * agg_interval

        agg_iter_num += 1

        # TODO: Save with Fabric
        agg_checkpoint = f"checkpoint_agg_iter{agg_iter_num:04d}.ckpt"
        agg_checkpoint_tmp = f"_{agg_checkpoint}"
        print(f"Saving checkpoint to {agg_checkpoint}")
        torch.save(agg_model.state_dict(), os.path.join(out_dir, agg_checkpoint_tmp))
        os.rename(os.path.join(out_dir, agg_checkpoint_tmp), os.path.join(out_dir, agg_checkpoint))

        prev_agg_checkpoint = f"checkpoint_agg_iter{agg_iter_num-1:04d}.ckpt"
        os.remove(os.path.join(out_dir, prev_agg_checkpoint))

        val_loss = validate(fabric, agg_model, val_data)
        fabric.logger.log_metrics({"val_loss": val_loss.item()}, iter_num)
        fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")

        if agg_iter_num * agg_interval > max_iters:
            break

    logger.finalize("success")


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(
            fabric,
            val_data,
            micro_batch_size,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def get_batch(fabric: L.Fabric, data: np.ndarray, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if fabric.device.type == "cuda":
        x.pin_memory(), y.pin_memory()
    x, y = fabric.to_device((x, y))
    return x, y


def load_val_dataset(data_dir: str = "data/shakespeare") -> np.ndarray:
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return val_data


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
