#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def load_unikp_modules(unikp_dir: Path):
    unikp_dir = unikp_dir.resolve()
    required = [
        unikp_dir / "build_vocab.py",
        unikp_dir / "pretrain_trfm.py",
        unikp_dir / "utils.py",
        unikp_dir / "enumerator.py",
        unikp_dir / "dataset.py",
        unikp_dir / "vocab.pkl",
        unikp_dir / "trfm_12_23000.pkl",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(
            "UniKP repository is incomplete. Missing:\n  " + "\n  ".join(missing)
        )

    sys.path.insert(0, str(unikp_dir))
    import build_vocab
    from pretrain_trfm import TrfmSeq2seq
    from utils import split
    return build_vocab, TrfmSeq2seq, split


class VocabCompatUnpickler(pickle.Unpickler):
    """Map old UniKP vocab.pkl references such as __main__.WordVocab
    onto the classes imported from build_vocab.py.
    """
    def __init__(self, file_obj, build_vocab_module):
        super().__init__(file_obj)
        self.build_vocab_module = build_vocab_module

    def find_class(self, module, name):
        if module == "__main__" and name in {"WordVocab", "Vocab", "TorchVocab"}:
            return getattr(self.build_vocab_module, name)
        return super().find_class(module, name)


def load_vocab_compat(path: Path, build_vocab_module):
    with open(path, "rb") as f:
        return VocabCompatUnpickler(f, build_vocab_module).load()


def get_ids(smiles: str, split_fn, vocab, seq_len: int = 220):
    PAD, UNK, EOS, SOS = 0, 1, 2, 3
    tokens = split_fn(smiles).split()

    # Match UniKP truncation rule: 218 content tokens max.
    if len(tokens) > seq_len - 2:
        keep = (seq_len - 2) // 2
        tokens = tokens[:keep] + tokens[-keep:]

    ids = [vocab.stoi.get(tok, UNK) for tok in tokens]
    ids = [SOS] + ids + [EOS]
    ids += [PAD] * (seq_len - len(ids))
    return ids


@torch.inference_mode()
def encode_batch_unikp(model, src: torch.Tensor) -> torch.Tensor:
    """GPU-safe equivalent of UniKP TrfmSeq2seq._encode().

    Returns:
      [mean(last), max(last), first(last), first(penultimate)]
      => 4 * 256 = 1024 dimensions.
    """
    embedded = model.embed(src)
    embedded = model.pe(embedded)

    output = embedded
    for i in range(model.trfm.encoder.num_layers - 1):
        output = model.trfm.encoder.layers[i](output, None)

    penul = output
    output = model.trfm.encoder.layers[-1](output, None)

    if model.trfm.encoder.norm is not None:
        output = model.trfm.encoder.norm(output)

    feat = torch.cat(
        [
            output.mean(dim=0),
            output.max(dim=0).values,
            output[0, :, :],
            penul[0, :, :],
        ],
        dim=1,
    )
    return feat.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="filtered_sequences.xlsx")
    ap.add_argument("--sheet", default="0")
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--cache_dir", default="full_esm2_cache_pkl")
    ap.add_argument("--unikp_dir", default="UniKP")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--seq_len", type=int, default=220)
    ap.add_argument("--out", default="all_smiles_embs_unikp1024.npy")
    args = ap.parse_args()

    xlsx = Path(args.xlsx)
    cache_dir = Path(args.cache_dir)
    unikp_dir = Path(args.unikp_dir)
    out = Path(args.out)
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet

    df = pd.read_excel(xlsx, sheet_name=sheet)
    if args.smiles_col not in df.columns:
        raise KeyError(
            f"Missing SMILES column '{args.smiles_col}'. "
            f"Columns: {list(df.columns)}"
        )

    index_path = cache_dir / "protein_full_embs_index.pkl"
    if index_path.exists():
        with open(index_path, "rb") as f:
            idx = pickle.load(f)
        original_rows = np.asarray(idx["original_rows"], dtype=np.int64)
        smiles = (
            df.iloc[original_rows][args.smiles_col]
            .astype(str)
            .str.strip()
            .tolist()
        )
        if len(smiles) != int(idx["n_sequences"]):
            raise RuntimeError(
                f"SMILES rows={len(smiles)} != protein cache n={idx['n_sequences']}"
            )
        alignment = "protein_full_embs_index.pkl/original_rows"
    else:
        smiles = df[args.smiles_col].astype(str).str.strip().tolist()
        alignment = "xlsx row order"

    bad = [i for i, sm in enumerate(smiles) if not sm or sm.lower() == "nan"]
    if bad:
        raise ValueError(f"Empty/NaN SMILES rows; examples: {bad[:10]}")

    build_vocab, TrfmSeq2seq, split_fn = load_unikp_modules(unikp_dir)

    vocab_path = unikp_dir / "vocab.pkl"
    weights_path = unikp_dir / "trfm_12_23000.pkl"

    # Important compatibility fix for old UniKP vocab.pkl.
    vocab = load_vocab_compat(vocab_path, build_vocab)

    model = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)

    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state, strict=True)

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )
    model = model.to(device).eval()

    print("Rows       :", len(smiles))
    print("Alignment  :", alignment)
    print("Vocab size :", len(vocab))
    print("Device     :", device)
    print("Output     :", out.resolve())

    features = np.empty((len(smiles), 1024), dtype=np.float32)

    for start in tqdm(
        range(0, len(smiles), args.batch_size),
        desc="UniKP SMILES-1024",
    ):
        end = min(start + args.batch_size, len(smiles))
        ids = [
            get_ids(sm, split_fn, vocab, args.seq_len)
            for sm in smiles[start:end]
        ]
        src = (
            torch.tensor(ids, dtype=torch.long, device=device)
            .t()
            .contiguous()
        )

        feat = encode_batch_unikp(model, src)
        expected = (end - start, 1024)
        if tuple(feat.shape) != expected:
            raise RuntimeError(
                f"Feature shape mismatch {tuple(feat.shape)} != {expected}"
            )
        features[start:end] = feat.cpu().numpy()

    if not np.isfinite(features).all():
        raise RuntimeError("Generated SMILES embeddings contain NaN/Inf.")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, features)

    metadata = {
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "hidden_dim": 256,
        "final_dim": 1024,
        "components": [
            "mean(last_encoder_layer)",
            "max(last_encoder_layer)",
            "first_token(last_encoder_layer)",
            "first_token(penultimate_encoder_layer)",
        ],
        "alignment": alignment,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "vocab_size": len(vocab),
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nDONE")
    print("shape :", features.shape)
    print("dtype :", features.dtype)
    print("finite:", bool(np.isfinite(features).all()))
    print("saved :", out.resolve())


if __name__ == "__main__":
    main()
