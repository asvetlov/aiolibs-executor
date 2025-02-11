import asyncio
import unittest
from collections.abc import AsyncIterator
from typing import Any

from aiolibs_executor import Executor


class TestExecutor(unittest.IsolatedAsyncioTestCase):
    def make_executor(
        self,
        *,
        max_workers: int | None = None,
        max_pending: int = 0,
        task_name_prefix: str = "",
    ) -> Executor:
        executor = Executor(
            max_workers=max_workers,
            max_pending=max_pending,
            task_name_prefix=task_name_prefix,
        )
        self.addAsyncCleanup(executor.shutdown)
        return executor

    async def test_submit_nowait(self) -> None:
        executor = self.make_executor()

        async def f(
            *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            await asyncio.sleep(0)
            return args, kwargs

        self.assertEqual(
            await executor.submit_nowait(f(1, a=2)), ((1,), {"a": 2})
        )

    async def test_submit(self) -> None:
        executor = self.make_executor()

        async def f(
            *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            await asyncio.sleep(0)
            return args, kwargs

        fut = await executor.submit(f(1, a=2))
        self.assertEqual(await fut, ((1,), {"a": 2}))

    async def test_map(self) -> None:
        executor = self.make_executor()

        async def f(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        arg = list(range(3))
        ret = [i async for i in executor.map(f, arg, arg)]
        self.assertEqual(ret, [0, 2, 4])

    async def test_amap(self) -> None:
        executor = self.make_executor()

        async def f(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        async def inp() -> AsyncIterator[int]:
            for i in range(1, 4):
                await asyncio.sleep(0)
                yield i

        ret = [i async for i in executor.amap(f, inp(), inp())]
        self.assertEqual(ret, [2, 4, 6])


if __name__ == "__main__":
    unittest.main()
