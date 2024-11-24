import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from montecarlo_parallel import run_parallel
from montecarlo_sequential import run_sequential


def test_sequential():
    assert run_sequential(1000) is not None

def test_parallel():
    assert run_parallel(1000, 4) is not None