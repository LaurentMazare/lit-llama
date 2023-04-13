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

eval_interval = 100
eval_iters = 100
log_interval = 10

# Hyperparameters
learning_rate = 6e-4
micro_batch_size = 5
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0


def main(logs_dir: str = "logs", batch_size: int = 25) -> None:
    logger = CSVLogger(logs_dir, name=f"lit-llama_single", flush_logs_every_n_steps=1)

    fabric = L.Fabric(accelerator="auto", devices=1, loggers=logger)
    fabric.launch()
    fabric.seed_everything(1335 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("30M")
    config.vocab_size = 100  # from prepare_shakespeare.py
    
    with fabric.device:
        model = LLaMA(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"N parameters: {n_params * 1e-9:.3f}B")

    model = fabric.setup_module(model)

    model.apply(model._init_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    optimizer = fabric.setup_optimizers(optimizer)

    train(fabric, model, optimizer, batch_size, train_data, val_data)

    logger.finalize("success")


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> None:
    assert batch_size % micro_batch_size == 0, f"batch_size ({batch_size}) is not a multiple of micro_batch_size ({micro_batch_size})"

    grad_accumulation_steps = batch_size // micro_batch_size

    iter_num = 0

    while True:
        # TODO: add learning rate scheduling

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0 and fabric.global_rank == 0:
            val_loss = validate(fabric, model, val_data)
            fabric.logger.log_metrics({"val_loss": val_loss.item()}, iter_num)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            # TODO: Save with Fabric
            print(f"Saving checkpoint to {out_dir}")
            torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoint_single.pth'))

        t0 = time.time()

        input_ids, targets = get_batch(
            fabric,
            train_data,
            micro_batch_size,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )

        is_accumulating = iter_num % grad_accumulation_steps != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # .backward() accumulates when .zero_grad() wasn't called
            fabric.backward(loss)

        if not is_accumulating:
            # TODO: Gradient clipping
            if grad_clip != 0.0:
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            print("Stepping optimizer")

            optimizer.step()
            optimizer.zero_grad()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.logger.log_metrics({"train_loss": loss.item()}, iter_num)
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")

        print(f"Samples seen: {iter_num * micro_batch_size}")

        iter_num += 1

        if iter_num > max_iters:
            break


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


def load_datasets(data_dir: str = "data/shakespeare") -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
