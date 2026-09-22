import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models import build_model
from src.preprocessing import TargetScaler, FeatureScaler
from src.metrics import regression_metrics
from src.data import load_splits
from src.data import load_vector_data
from src.training import epoch_batches


class Correctness(unittest.TestCase):
    def test_training_api_hides_test_targets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.save(root / 'p.npy', np.ones((5, 3), dtype=np.float32))
            np.save(root / 's.npy', np.ones((5, 2), dtype=np.float32))
            np.save(root / 'y.npy', [1., 2., 3., 9999., 8888.])
            np.savez(root / 'split.npz', train_idx=[0, 1], val_idx=[2], test_idx=[3, 4])
            manifest = {'derived_protein_features': {'mean': {'path': str(root / 'p.npy')}},
                        'smiles_embeddings': str(root / 's.npy'), 'labels': str(root / 'y.npy'),
                        'split_indices': str(root / 'split.npz')}
            _, _, labels, _ = load_vector_data(manifest)
            self.assertTrue(np.isnan(labels[3:]).all())
            np.testing.assert_array_equal(labels[:3], [1., 2., 3.])
            with self.assertRaises(PermissionError):
                load_vector_data(manifest, include_test_labels=True)

    def test_no_batch_duplicates_or_missing_rows(self):
        for n in (2, 3, 17, 33, 129):
            order = torch.arange(n)
            batches = epoch_batches(order, 16)
            self.assertTrue(torch.equal(torch.cat(batches), order))
            self.assertTrue(all(len(b) >= 2 for b in batches))

    def test_split_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'split.npz'
            np.savez(path, train_idx=[0, 0], val_idx=[1], test_idx=[2])
            with self.assertRaises(ValueError): load_splits(path)

    def test_scalers_and_metrics(self):
        y = np.array([-2., 1., 4.])
        scaler = TargetScaler.fit(y)
        np.testing.assert_allclose(scaler.inverse(scaler.transform(y)), y, atol=1e-6)
        feature = FeatureScaler.fit(y[:, None], 'standard')
        self.assertGreater(feature.transform(np.array([[100.]]))[0, 0], 10)
        m = regression_metrics(y, y + 100)
        self.assertLess(m['r2'], 0)
        self.assertAlmostEqual(m['pearson'], 1)
        self.assertIsNone(regression_metrics([1, 1], [1, 2])['r2'])

    def test_fusion_gradients_bn_reload_and_ensemble(self):
        torch.set_num_threads(2)
        p, s = torch.randn(8, 7), torch.randn(8, 5)
        for fusion in ('concat', 'sum', 'gated', 'bilinear'):
            c = dict(protein_dim=7, smiles_dim=5, hidden_dims=[16, 16],
                     projection_mode='both', projection_dim=8, fusion=fusion,
                     bilinear_rank=3, norm='batchnorm', residual=True)
            m = build_model(c)
            out = m(p, s)
            self.assertEqual(out.shape, (8,))
            out.square().mean().backward()
            self.assertTrue(all(v.grad is not None for v in m.parameters()))
            m.eval()
            before = {k: v.clone() for k, v in m.named_buffers()}
            with torch.no_grad(): a = m(p, s)
            for k, v in m.named_buffers(): self.assertTrue(torch.equal(before[k], v))
            clone = build_model(c).eval(); clone.load_state_dict(m.state_dict())
            with torch.no_grad(): b = clone(p, s)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(torch.stack([a, b]).mean(0), (a+b)/2)


if __name__ == '__main__': unittest.main()
