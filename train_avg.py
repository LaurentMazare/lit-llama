"""
This script is a placeholder for training LLaMA from scratch.
Currently, it just trains on the Shakespeare dataset.
"""

import os
import time
from typing import Tuple

import lightning as L

import torch

import numpy as np

from lightning.fabric.loggers import TensorBoardLogger

from lit_llama.model import Block, LLaMA, LLaMAConfig

out_dir = "out"

n_peers = 2
reduce_interval = 100

eval_interval = 200
eval_iters = 100
log_interval = 1
# compilation fails as it does not support torch.complex64 for RoPE
# compile = False

# Hyperparameters
learning_rate = 6e-4
batch_size = 25
micro_batch_size = 5
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0


def main_single() -> None:

    logger = TensorBoardLogger("logs", name="lit-llama")

    fabric = L.Fabric(accelerator="auto", loggers=logger)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("30M")
    config.vocab_size = 100  # from prepare_shakespeare.py
    
    with fabric.device:
        model = LLaMA(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"N parameters: {n_params * 1e-9:.3f}B")

    # if compile:
    #     model = torch.compile(model)

    model = fabric.setup_module(model)

    model.apply(model._init_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    optimizer = fabric.setup_optimizers(optimizer)

    train(fabric, model, optimizer, train_data, val_data)

    logger.finalize("success")


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


def main_peer() -> None:
    logger = TensorBoardLogger("logs", name="lit-llama")

    fabric = L.Fabric(accelerator="auto", devices=1, loggers=logger)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("30M")
    config.vocab_size = 100  # from prepare_shakespeare.py
    
    with fabric.device:
        avg_model = LLaMA(config)
        target_model = LLaMA(config)

    n_params = sum(p.numel() for p in avg_model.parameters())
    print(f"N parameters: {n_params * 1e-9:.3f}B")

    avg_model = fabric.setup_module(avg_model)
    target_model = fabric.setup_module(target_model)

    avg_model.apply(avg_model._init_weights)

    weight = 1.0 / n_peers

    iter_num = 0

    models = []
    optimizers = []
    for n in range(n_peers):
        with fabric.device:
            model = LLaMA(config)
        model = fabric.setup_module(model)
        models.append(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
        optimizer = fabric.setup_optimizers(optimizer)
        optimizers.append(optimizer)

    while True:
        for n in range(n_peers):
            # TODO: partition train_data
 
            with torch.no_grad():
                copy_params(avg_model, models[n], accumulate=False)

            train_peer(fabric, models[n], optimizers[n], n, iter_num, reduce_interval, train_data)
 
            with torch.no_grad():
                if n == 0:
                    copy_params(models[n], target_model, weight=weight, accumulate=False)
                else:
                    copy_params(models[n], target_model, weight=weight, accumulate=True)

        copy_params(target_model, avg_model, accumulate=False)

        iter_num += reduce_interval

        # val_loss = validate(fabric, avg_model, val_data)
        # fabric.logger.log_metrics({"val_loss": val_loss.item()}, iter_num)
        # fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
        # # TODO: Save with Fabric
        # print(f"Saving checkpoint to {out_dir}")
        # torch.save(avg_model.state_dict(), os.path.join(out_dir, 'ckpt.pt'))

    logger.finalize("success")



def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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
            torch.save(model.state_dict(), os.path.join(out_dir, 'ckpt.pt'))

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

        iter_num += 1


def train_peer(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    peer_id: int,
    start_iter: int,
    n_iter: int,
    train_data: np.ndarray,
) -> None:
    assert batch_size % micro_batch_size == 0, f"batch_size ({batch_size}) is not a multiple of micro_batch_size ({micro_batch_size})"

    grad_accumulation_steps = batch_size // micro_batch_size // n_peers

    iter_num = start_iter

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
    torch.set_float32_matmul_precision("high")
    # main_single()
    main_peer()
