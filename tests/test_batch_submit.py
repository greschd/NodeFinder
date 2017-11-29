# pylint: disable=redefined-outer-name

import asyncio

import pytest

from nodefinder._batch_submit import BatchSubmitter


@pytest.fixture
def echo_submitter():
    echo = lambda x: x
    with BatchSubmitter(echo) as func:
        yield func


@pytest.mark.parametrize('num_inputs', [10, 150, 300, 600])
def test_simple_submit(echo_submitter, num_inputs):
    loop = asyncio.get_event_loop()
    input_ = list(range(num_inputs))
    fut = asyncio.gather(*[echo_submitter(i) for i in input_])
    loop.run_until_complete(fut)
    assert fut.result() == input_


def test_failing_run():
    def func(x):  # pylint: disable=unused-argument
        raise ValueError

    loop = asyncio.get_event_loop()
    with BatchSubmitter(func) as f:
        with pytest.raises(ValueError):
            loop.run_until_complete(f(1))
