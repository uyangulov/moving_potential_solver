import pytest
import numpy as np

class TestApplyCircuit:

    @pytest.fixture
    def a(self):
        return np.array([1])

    def test_single_qubit_gate(self, a):
        assert a.shape[0] == 1

   