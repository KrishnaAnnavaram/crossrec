import pytest

from crossrec.overlap import prepare
from crossrec.synthetic import SyntheticSpec, generate


@pytest.fixture(scope="session")
def synth():
    return generate(SyntheticSpec(seed=7))


@pytest.fixture(scope="session")
def core(synth):
    data, _ = prepare(synth, min_user=3, min_item=3, max_users=0, seed=7)
    return data
