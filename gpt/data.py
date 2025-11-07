import os
from collections import defaultdict, deque
from dataclasses import asdict
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from gpt.config import DataConfig


def build_dataset(config: DataConfig, tokenizer: Tokenizer, eos_token_id: int):
    """Tokenize, pack, and convert to tensors. Will load from local cache if available."""
    # load -> tokenize -> fast bfd packing -> pad and build labels
    ds = load_dataset(
        **asdict(config.dataset),
        num_proc=os.cpu_count(),
        download_mode="force_redownload" if config.skip_cache else None,
    )
    ds = ds.map(
        partial(encode, tokenizer=tokenizer, column=config.column, eos_token_id=eos_token_id),
        desc="Tokenizing data...",
        batched=True,
        num_proc=os.cpu_count(),
        batch_size=config.process_batch_size,
        remove_columns=ds.column_names,
    )
    ds = ds.map(
        partial(pack, seq_len=config.seq_len),
        desc="Packing data...",
        batched=True,
        batch_size=config.process_batch_size,
        num_proc=(len(ds) // config.process_batch_size) + 1,
        remove_columns=ds.column_names,
    ).with_format("torch")
    return ds.map(
        partial(build_inputs, pad_id=eos_token_id, pad_to=config.pad_to),
        desc="Building inputs...",
        num_proc=os.cpu_count(),
        remove_columns=ds.column_names,
    )


def encode(examples, tokenizer, column, eos_token_id) -> dict:
    """Batch tokenize input text."""
    encodings = tokenizer.encode_batch_fast(examples[column])
    return {"input_ids": [encoding.ids + [eos_token_id] for encoding in encodings]}


def pack(examples: pa.Table, seq_len: int) -> pa.Table:
    """Pack examples into sequences up to seq_len."""
    ids = pc.list_slice(examples["input_ids"], 0, seq_len)
    lens = pc.list_value_length(ids).to_numpy()
    order = np.argsort(-lens)

    succ = IntSucc(seq_len)
    succ.add(seq_len)  # sentinel enables new bins
    by_space = defaultdict(deque)  # space -> deque[bins]
    bins = []  # each: {"ids": [...], "len": int}

    for i in order:
        L = int(lens[i])
        if not L:
            continue
        s = succ.next_geq(L)
        b = by_space[s].popleft() if s < seq_len else {"ids": [], "len": 0}
        if s < seq_len and not by_space[s]:
            succ.discard(s)
        b["ids"].append(int(i))
        b["len"] += L
        if s == seq_len:
            bins.append(b)
        ns = s - L
        by_space[ns].append(b)
        if ns:
            succ.add(ns)

    reorder = [j for b in bins for j in b["ids"]]
    ids_taken = take(ids, reorder)

    # offsets (match ListArray vs LargeListArray via dtype)
    tok_counts = [b["len"] for b in bins]
    odtype = ids_taken.offsets.type.to_pandas_dtype()
    offs = np.cumsum([0] + tok_counts, dtype=odtype)

    LA = type(ids_taken)
    packed_ids = LA.from_arrays(offs, ids_taken.values)

    # position_ids: reset to 0 at each original example boundary
    dl = lens[reorder]
    T = int(offs[-1])
    pos = np.ones(T, dtype=np.int32)
    pos[0] = 0
    if dl.size > 1:
        cut = dl[:-1].cumsum()
        pos[cut] = -(dl[:-1] - 1)
    pos = pos.cumsum()
    position_ids = LA.from_arrays(offs, pa.array(pos, type=pa.int32()))

    return pa.Table.from_arrays([packed_ids, position_ids], names=["input_ids", "position_ids"])


def pad(t: torch.Tensor, pad_id: int, pad_to: int) -> torch.Tensor:
    """Pad a 1D tensor to the next multiple of pad_to."""
    remainder = t.size(0) % pad_to
    if remainder == 0:
        return t
    pad_len = pad_to - remainder
    return torch.cat((t, torch.full((pad_len,), pad_id, dtype=t.dtype, device=t.device)))


def build_inputs(batch: dict[str, torch.Tensor], pad_id: int, pad_to: int) -> dict[str, torch.Tensor]:
    """Pad tensors to multiple of pad_to and build labels column w/ appropriate masking."""
    labels = batch["input_ids"].clone()
    labels[batch["position_ids"] == 0] = -100  # mask boundary tokens
    batch["input_ids"] = pad(batch["input_ids"], pad_id=pad_id, pad_to=pad_to)
    batch["position_ids"] = pad(batch["position_ids"], pad_id=0, pad_to=pad_to)
    batch["labels"] = pad(labels, pad_id=-100, pad_to=pad_to)

    return batch


class IntSucc:
    """Find next greater integer in a set of integers."""

    __slots__ = ("N", "bits")

    def __init__(self, maxval: int):
        assert maxval >= 1
        self.N, self.bits = maxval, 0

    def add(self, i: int):
        self.bits |= 1 << (i - 1)

    def discard(self, i: int):
        self.bits &= ~(1 << (i - 1))

    def next_geq(self, x: int) -> int:
        y = self.bits >> (x - 1)
        assert y, "no successor present (missing sentinel?)"
        return x + ((y & -y).bit_length() - 1)


def take(arr, idx):
    """Take elements from a pyarrow array based on indices."""
    idx = np.asarray(idx, dtype=np.int32)
    out = pc.take(arr, pa.array(idx, type=pa.int32()))
    return out.combine_chunks() if isinstance(out, pa.ChunkedArray) else out
