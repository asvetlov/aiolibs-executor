import asyncio
import unittest
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


if __name__ == "__main__":
    unittest.main()
