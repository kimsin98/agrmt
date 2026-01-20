"""
Pytest tests for agreement() function.

Test cases are loaded from CSV exported by the reference R implementation.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from agrmt import agreement


# Load test cases from CSV at module level (runs once)
def _load_test_cases():
    csv_path = Path(__file__).parent / "agreement_test_cases.csv"
    test_cases = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            vector = np.array([int(x) for x in row["vector"].split(",")])
            expected = float(row["agreement"])
            test_cases.append(pytest.param(vector, expected, id=name))
    return test_cases


TEST_CASES = _load_test_cases()


@pytest.mark.parametrize("vector,expected", TEST_CASES)
def test_agreement_matches_r(vector, expected):
    """Test that Python implementation matches R function's output."""
    result = agreement(vector)
    assert result == pytest.approx(expected, abs=1e-9)


class TestAgreementEdgeCases:
    """Test error handling and edge cases."""

    def test_length_less_than_3_raises(self):
        with pytest.raises(ValueError, match="Length of vector < 3"):
            agreement([1, 2])

    def test_length_2_raises(self):
        with pytest.raises(ValueError, match="Length of vector < 3"):
            agreement(np.array([10, 20]))

    def test_length_1_raises(self):
        with pytest.raises(ValueError, match="Length of vector < 3"):
            agreement([5])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Length of vector < 3"):
            agreement([])

    def test_negative_values_raises(self):
        with pytest.raises(ValueError, match="Negative values"):
            agreement([10, -5, 20])

    def test_float_values_raises(self):
        with pytest.raises(ValueError, match="must contain integers"):
            agreement([10.5, 20.5, 30.0])

    def test_all_zeros(self):
        """All zeros should return 0 (no agreement pattern)."""
        result = agreement([0, 0, 0, 0, 0])
        assert result == pytest.approx(0.0)
