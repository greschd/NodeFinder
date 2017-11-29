# pylint: disable=redefined-outer-name

import asyncio

import pytest

from nodefinder._calculation_batcher import CalculationBatcher


@pytest.fixture
def default_echo_batcher():
    echo = lambda x: x
    loop = asyncio.get_event_loop()
    batcher = CalculationBatcher(echo, loop=loop)
    batcher.start()
    yield batcher
    batcher.stop()


@pytest.mark.parametrize('num_inputs', [10, 150, 300, 600])
def test_simple_submit(default_echo_batcher, num_inputs):
    loop = asyncio.get_event_loop()
    input_ = list(range(num_inputs))
    fut = asyncio.gather(*[default_echo_batcher.submit(i) for i in input_])
    loop.run_until_complete(fut)
    assert fut.result() == input_