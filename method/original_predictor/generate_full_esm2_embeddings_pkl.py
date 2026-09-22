#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate reusable, variable-length FULL residue-level ESM2 embeddings from
filtered_sequences.xlsx WITHOUT global padding.

Why this format
---------------
A dense array shaped [N, Lmax, D] wastes substantial disk space because every
protein is padded to the longest sequence. This script instead stores each
protein as its own [Li, D] NumPy array in an indexed pickle stream.

Outputs
-------
full_esm2_cache_pkl/
    protein_full_embs.pkl        # sequential pickle records; each is [Li, 1280]
    protein_full_embs_index.pkl  # byte offsets + lengths + row mapping
    embedding_metadata.json

The data file is written one sequence at a time, so generation does not retain
all embeddings in RAM. The companion index enables random access without
loading the whole data pickle into memory.

Default storage dtype is float16 to halve disk usage. During training, cast the
loaded tensor to float32 if required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


ESM_ALLOWED = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
FORMAT_VERSION = "indexed-ragged-esm2-pickle-v1"
DATA_FILENAME = "protein_full_embs.pkl"
INDEX_FILENAME = "protein_full_embs_index.pkl"
METADATA_FILENAME = "embedding_metadata.json"


def clean_sequence(value: Any) -> str:
    """Normalize a sequence while retaining ESM-supported ambiguity tokens."""
    seq = str(value).strip().upper()
    seq = re.sub(r"\s+", "", seq)
    seq = seq.replace("*", "")
    return "".join(aa if aa in ESM_ALLOWED else "X" for aa in seq)


def parse_sheet(value: str) -> int | str:
    """Allow --sheet 0 or --sheet Sheet1."""
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def sequence_hash(sequences: List[str]) -> str:
    """Stable fingerprint used to validate automatic resume."""
    digest = hashlib.sha256()
    for seq in sequences:
        digest.update(seq.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    """Write an index/checkpoint atomically."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def atomic_json_dump(obj: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def human_bytes(n_bytes: float) -> str:
    value = float(n_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def build_index(
    *,
    source_file: Path,
    seq_col: str,
    sequences: List[str],
    original_rows: List[int],
    lengths: List[int],
    offsets: List[int],
    data_end_offset: int,
    storage_dtype: str,
    completed: int,
) -> Dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "data_file": DATA_FILENAME,
        "source_file": str(source_file.resolve()),
        "sequence_column": seq_col,
        "sequence_sha256": sequence_hash(sequences),
        "n_sequences": len(sequences),
        "completed": completed,
        "offsets": offsets,
        "lengths": lengths,
        "original_rows": original_rows,
        "sequences": sequences,
        "data_end_offset": int(data_end_offset),
        "embedding_model": "esm2_t33_650M_UR50D",
        "representation_layer": 33,
        "embedding_dim": 1280,
        "storage_dtype": storage_dtype,
        "bos_eos_removed": True,
        "global_padding": False,
    }


def validate_resume_index(
    index: Dict[str, Any],
    sequences: List[str],
    storage_dtype: str,
) -> None:
    checks = {
        "format": FORMAT_VERSION,
        "sequence_sha256": sequence_hash(sequences),
        "n_sequences": len(sequences),
        "storage_dtype": storage_dtype,
        "embedding_model": "esm2_t33_650M_UR50D",
        "representation_layer": 33,
        "embedding_dim": 1280,
    }
    mismatches = {
        key: (index.get(key), expected)
        for key, expected in checks.items()
        if index.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Existing checkpoint does not match this run. "
            f"Mismatches: {mismatches}. Use --restart to regenerate."
        )


def read_embedding(data_path: Path, offset: int) -> np.ndarray:
    """Read one embedding record at a known byte offset."""
    with open(data_path, "rb") as handle:
        handle.seek(int(offset))
        array = pickle.load(handle)
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise TypeError(f"Invalid embedding record at offset {offset}")
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", default="filtered_sequences.xlsx")
    parser.add_argument("--sheet", default="0")
    parser.add_argument("--seq_col", default="enzyme")
    parser.add_argument("--out_dir", default="full_esm2_cache_pkl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--storage_dtype",
        choices=["float16", "float32"],
        default="float16",
        help="float16 is recommended for storage; cast to float32 during training.",
    )
    parser.add_argument(
        "--max_residues",
        type=int,
        default=1022,
        help="No silent truncation. The script stops if a sequence exceeds this length.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Save the random-access index every N completed sequences.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete an existing partial cache and regenerate from sequence 0.",
    )
    args = parser.parse_args()

    xlsx = Path(args.xlsx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / DATA_FILENAME
    index_path = out_dir / INDEX_FILENAME
    metadata_path = out_dir / METADATA_FILENAME

    if not xlsx.exists():
        raise FileNotFoundError(f"Cannot find: {xlsx.resolve()}")

    df = pd.read_excel(xlsx, sheet_name=parse_sheet(args.sheet))
    if args.seq_col not in df.columns:
        raise ValueError(
            f"Sequence column '{args.seq_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    work = df[[args.seq_col]].copy()
    work["original_row"] = np.arange(len(work), dtype=np.int64)
    work = work.dropna(subset=[args.seq_col]).copy()
    work[args.seq_col] = work[args.seq_col].map(clean_sequence)
    work = work[work[args.seq_col].str.len() > 0].reset_index(drop=True)

    sequences = work[args.seq_col].astype(str).tolist()
    original_rows = work["original_row"].astype(int).tolist()
    lengths = [len(seq) for seq in sequences]

    if not sequences:
        raise ValueError("No valid protein sequences found.")

    n = len(sequences)
    lmax = max(lengths)
    total_residues = int(sum(lengths))
    itemsize = np.dtype(args.storage_dtype).itemsize
    ragged_estimate = total_residues * 1280 * itemsize
    old_padded_float32 = n * lmax * 1280 * np.dtype(np.float32).itemsize

    print(f"Rows in Excel        : {len(df)}")
    print(f"Valid sequences      : {n}")
    print(f"Min sequence length  : {min(lengths)}")
    print(f"Max sequence length  : {lmax}")
    print(f"Mean sequence length : {np.mean(lengths):.2f}")
    print(f"Total residues       : {total_residues:,}")
    print(f"Estimated ragged {args.storage_dtype}: {human_bytes(ragged_estimate)}")
    print(f"Old padded float32 estimate    : {human_bytes(old_padded_float32)}")

    if lmax > args.max_residues:
        bad = [i for i, length in enumerate(lengths) if length > args.max_residues]
        examples = [
            {
                "cache_row": i,
                "original_row": original_rows[i],
                "length": lengths[i],
            }
            for i in bad[:10]
        ]
        raise ValueError(
            f"{len(bad)} sequences exceed --max_residues={args.max_residues}. "
            f"No truncation was performed. Examples: {examples}"
        )

    if args.restart:
        for path in (data_path, index_path, metadata_path):
            if path.exists():
                path.unlink()

    offsets: List[int] = []
    completed = 0
    data_end_offset = 0

    if data_path.exists() or index_path.exists():
        if not (data_path.exists() and index_path.exists()):
            raise RuntimeError(
                "Found an incomplete cache without both data and index files. "
                "Use --restart to regenerate safely."
            )
        with open(index_path, "rb") as handle:
            index = pickle.load(handle)
        validate_resume_index(index, sequences, args.storage_dtype)
        offsets = [int(x) for x in index["offsets"]]
        completed = int(index["completed"])
        data_end_offset = int(index["data_end_offset"])
        if completed != len(offsets):
            raise RuntimeError("Checkpoint is inconsistent: completed != len(offsets).")
        if completed > n:
            raise RuntimeError("Checkpoint contains more sequences than the current input.")
        print(f"Resuming from sequence {completed}/{n}.")
        data_mode = "r+b"
    else:
        data_mode = "w+b"

    import esm

    print("\nLoading ESM2: esm2_t33_650M_UR50D")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    repr_layer = 33
    emb_dim = 1280

    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    target_dtype = np.float16 if args.storage_dtype == "float16" else np.float32

    print("Device:", device)
    print("Storage: variable-length indexed pickle; global padding = False")

    with open(data_path, data_mode) as data_handle:
        if completed > 0:
            # Remove any trailing partially-written record left by interruption.
            data_handle.truncate(data_end_offset)
            data_handle.seek(data_end_offset)
        else:
            data_handle.seek(0)

        with torch.inference_mode():
            iterator = tqdm(
                range(completed, n),
                initial=completed,
                total=n,
                desc="ESM2 ragged full embeddings",
            )
            for i in iterator:
                seq = sequences[i]
                _, _, tokens = batch_converter([("protein", seq)])
                tokens = tokens.to(device)

                output = model(
                    tokens,
                    repr_layers=[repr_layer],
                    return_contacts=False,
                )

                embedding = (
                    output["representations"][repr_layer][0, 1 : len(seq) + 1]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(target_dtype, copy=False)
                )

                if embedding.shape != (len(seq), emb_dim):
                    raise RuntimeError(
                        f"Embedding shape mismatch at cache row {i}: "
                        f"expected {(len(seq), emb_dim)}, got {embedding.shape}"
                    )

                offset = int(data_handle.tell())
                pickle.dump(embedding, data_handle, protocol=pickle.HIGHEST_PROTOCOL)
                data_handle.flush()

                offsets.append(offset)
                completed = i + 1
                data_end_offset = int(data_handle.tell())

                if (
                    args.checkpoint_every > 0
                    and completed % args.checkpoint_every == 0
                ):
                    os.fsync(data_handle.fileno())
                    checkpoint = build_index(
                        source_file=xlsx,
                        seq_col=args.seq_col,
                        sequences=sequences,
                        original_rows=original_rows,
                        lengths=lengths,
                        offsets=offsets,
                        data_end_offset=data_end_offset,
                        storage_dtype=args.storage_dtype,
                        completed=completed,
                    )
                    atomic_pickle_dump(checkpoint, index_path)

                del output, embedding, tokens

        data_handle.flush()
        os.fsync(data_handle.fileno())

    final_index = build_index(
        source_file=xlsx,
        seq_col=args.seq_col,
        sequences=sequences,
        original_rows=original_rows,
        lengths=lengths,
        offsets=offsets,
        data_end_offset=data_end_offset,
        storage_dtype=args.storage_dtype,
        completed=completed,
    )
    atomic_pickle_dump(final_index, index_path)

    metadata = {
        key: final_index[key]
        for key in [
            "format",
            "data_file",
            "source_file",
            "sequence_column",
            "sequence_sha256",
            "n_sequences",
            "completed",
            "embedding_model",
            "representation_layer",
            "embedding_dim",
            "storage_dtype",
            "bos_eos_removed",
            "global_padding",
        ]
    }
    metadata.update(
        {
            "min_sequence_length": min(lengths),
            "max_sequence_length": max(lengths),
            "total_residues": total_residues,
            "data_file_size_bytes": data_path.stat().st_size,
            "data_file_size": human_bytes(data_path.stat().st_size),
        }
    )
    atomic_json_dump(metadata, metadata_path)

    # Lightweight verification: first and last record.
    first = read_embedding(data_path, offsets[0])
    last = read_embedding(data_path, offsets[-1])
    assert first.shape == (lengths[0], emb_dim)
    assert last.shape == (lengths[-1], emb_dim)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nDONE")
    print("Data  :", data_path.resolve(), human_bytes(data_path.stat().st_size))
    print("Index :", index_path.resolve())
    print("Rows  :", completed)
    print("Padding stored: none")
    print("Record shapes: [Li, 1280], where Li is each sequence's true length")


if __name__ == "__main__":
    main()
