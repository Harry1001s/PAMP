"""UniKP substrate encoding used by the global-head trainer."""
import pickle
import sys
from pathlib import Path
import torch

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
