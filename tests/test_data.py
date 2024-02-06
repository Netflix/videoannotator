import unittest

from videoannotator import data


class TestLabeledDataset(unittest.TestCase):
    def test_valid(self):
        lds = data.LabeledDataset(
            label="test", pos=frozenset({"x", "y"}), neg=frozenset({"z"})
        )
        assert len(lds) == 3
        assert lds.pos_cnt == 2
        assert lds.neg_cnt == 1

    def test_invalid_common_keys(self):
        with self.assertRaises(ValueError):
            data.LabeledDataset(
                label="test", pos=frozenset({"x"}), neg=frozenset({"x"})
            )

    def test_invalid_no_data(self):
        with self.assertRaises(ValueError):
            data.LabeledDataset(label="test", pos=frozenset(), neg=frozenset())
