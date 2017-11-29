import time
import asyncio


class CalculationBatcher:
    def __init__(
        self,
        func,
        *,
        loop=None,
        timeout=0.,
        sleep_time=0.1,
        min_batch_size=100,
        max_batch_size=200
    ):
        self._func = func
        self._loop = loop or asyncio.get_event_loop()
        self._timeout = timeout
        self._sleep_time = sleep_time
        self._min_batch_size = min_batch_size
        self._max_batch_size = max_batch_size
        self._tasks = asyncio.Queue()
        self._step_task = None

    def start(self):
        self._step_task = asyncio.Task(self.step(), loop=self._loop)

    def stop(self):
        self._step_task.cancel()
        self._loop.run_until_complete(self._wait_for_cancel())
        self._step_task = None

    async def _wait_for_cancel(self):
        while not self._step_task.cancelled():
            asyncio.sleep(self._sleep_time)

    async def step(self):
        while True:
            # make sure that the event loop is always released
            await asyncio.sleep(0)
            start_time = time.time()
            while ((self._tasks.qsize() < self._min_batch_size) and
                   (time.time() - start_time < self._timeout)
                   ) or (self._tasks.qsize() == 0):
                await asyncio.sleep(self._sleep_time)
            inputs = []
            futures = []
            for _ in range(self._max_batch_size):
                try:
                    key, fut = self._tasks.get_nowait()
                    inputs.append(key)
                    futures.append(fut)
                except asyncio.QueueEmpty:
                    break

            results = self._func(inputs)
            for fut, res in zip(futures, results):
                fut.set_result(res)

    async def submit(self, x):
        fut = self._loop.create_future()
        self._tasks.put_nowait((x, fut))
        return await fut
