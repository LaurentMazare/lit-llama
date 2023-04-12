"""
This script is a placeholder for training LLaMA from scratch.
Currently, it just trains on the Shakespeare dataset.
"""

import os
import time
from typing import Tuple

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
max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0


def main(peer_id: int, agg_interval: int, batch_size: int=25) -> None:
    logger = CSVLogger("logs", name=f"lit-llama_{peer_id}")

    peer_file_path = os.path.join(out_dir, f"peer_{peer_id:04d}")
    with open(peer_file_path, "w") as f:
        f.write("")

    fabric = L.Fabric(accelerator="auto", devices=1, loggers=logger)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank + 100 * peer_id)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("30M")
    config.vocab_size = 100  # from prepare_shakespeare.py

    with fabric.device:
        model = LLaMA(config)

    model = fabric.setup_module(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"N parameters: {n_params * 1e-9:.3f}B")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    optimizer = fabric.setup_optimizers(optimizer)

    agg_iter_num = 0

    while True:
        expected_checkpoint = os.path.join(out_dir, f"checkpoint_agg_iter{agg_iter_num:04d}.ckpt")
        while True:
            if os.path.exists(expected_checkpoint):
                break
            time.sleep(1.0)

        with torch.no_grad():
            model.load_state_dict(torch.load(expected_checkpoint))

        iter_num = agg_iter_num * agg_interval

        train_peer(fabric, model, optimizer, peer_id, iter_num, agg_interval, batch_size, train_data)

        # TODO: Save with Fabric
        print(f"Saving checkpoint to {out_dir}")
        peer_checkpoint = f"checkpoint_peer{peer_id:02d}_iter{agg_iter_num:04d}.ckpt"
        torch.save(model.state_dict(), os.path.join(out_dir, peer_checkpoint))

        val_loss = validate(fabric, model, val_data)
        fabric.logger.log_metrics({"val_loss": val_loss.item()}, iter_num)
        fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")

        agg_iter_num += 1

        if agg_iter_num * agg_interval > max_iters:
            break

    logger.finalize("success")


def train_peer(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    peer_id: int,
    start_iter: int,
    n_iter: int,
    batch_size: int,
    train_data: np.ndarray,
) -> None:
    assert batch_size % micro_batch_size == 0, f"batch_size ({batch_size}) is not a multiple of micro_batch_size ({micro_batch_size})"

    grad_accumulation_steps = batch_size // micro_batch_size

    iter_num = start_iter

    # in the async version, if there's a new checkpoint after accumulation, we just incorporate it
    # here we don't do this, we proceed sync

    while True:
        t0 = time.time()

        input_ids, targets = get_batch(
            fabric,
            train_data,
            micro_batch_size,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )

        is_accumulating = iter_num % grad_accumulation_steps != 0 or iter_num == start_iter

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            fabric.backward(loss)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.logger.log_metrics({f"train_loss_{peer_id}": loss.item()}, iter_num)
            fabric.print(f"peer: {peer_id}, iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")

        if not is_accumulating:
            # TODO: Gradient clipping
            if grad_clip != 0.0:
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            print(f"Stepping optimizer at iteration {iter_num}")

            optimizer.step()
            optimizer.zero_grad()

        iter_num += 1

        if iter_num - start_iter > n_iter:
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
    # we don't care about cross-peer contamination, we just draw randomly with a different seed
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
