"""Read CataPro residue features by original dataset row or unique sequence index."""
from common import *


class CataproResidueCache:
    def __init__(self, directory=None):
        self.directory=Path(directory) if directory else OUT/'cache/esm2_residue'
        self.index=json.loads((self.directory/'index.json').read_text())
        self.row_mapping=np.asarray(self.index['row_mapping'],dtype=np.int64)

    def __len__(self):
        return len(self.row_mapping)

    def by_sequence(self, unique_sequence_index):
        i=int(unique_sequence_index)
        if not 0<=i<len(self.index['lengths']): raise IndexError(i)
        value=np.load(self.directory/f'{i:05d}.npy',mmap_mode='r',allow_pickle=False)
        if value.shape!=(self.index['lengths'][i],1280): raise ValueError(f'Invalid residue shape for sequence {i}')
        return value

    def __getitem__(self, dataset_row):
        i=int(dataset_row)
        if not 0<=i<len(self.row_mapping): raise IndexError(i)
        return self.by_sequence(self.row_mapping[i])
