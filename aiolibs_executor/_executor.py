import contextvars
import dataclasses
import itertools
from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Future,
    Queue,
    QueueShutDown,
    Task,
    create_task,
    gather,
    get_running_loop,
)
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from typing import Any
from warnings import catch_warnings


class Executor:
    _counter = itertools.count().__next__

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        max_pending: int = 0,
        task_name_prefix: str = "",
    ) -> None:
        if max_workers is None:
            max_workers = 100
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._max_workers = max_workers
        self._task_name_prefix = (
            task_name_prefix or f"Executor-{Executor._counter()}"
        )
        self._init_context = contextvars.copy_context()
        self._init = False
        self._shutdown = False
        self._jobs: Queue[_Job[Any]] = Queue(max_pending)
        # tasks are much cheaper than threads or processes,
        # there is no need for adjusting tasks count on the fly like
        # ThreadPoolExecutor or ProcessPoolExecutor do.
        self._tasks: list[Task[None]] = []

    def submit_nowait[R](
        self,
        coro: Coroutine[Any, Any, R],
        *,
        context: contextvars.Context | None = None,
    ) -> Future[R]:
        self._lazy_init()
        job = _Job(
            get_running_loop(),
            context if context is not None else self._init_context,
            coro,
        )
        self._jobs.put_nowait(job)
        return job.future

    async def submit[R](
        self,
        coro: Coroutine[Any, Any, R],
        *,
        context: contextvars.Context | None = None,
    ) -> Future[R]:
        self._lazy_init()
        job = _Job(
            get_running_loop(),
            context if context is not None else self._init_context,
            coro,
        )
        await self._jobs.put(job)
        return job.future

    async def map[R, *IT](
        self,
        fn: Callable[..., Coroutine[Any, Any, R]],
        /,
        *iterables: Iterable[Any],
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]:
        jobs = [
            self.submit_nowait(fn(*args), context=context)
            for args in zip(*iterables, strict=False)
        ]

        try:
            # reverse to keep finishing order
            jobs.reverse()
            while jobs:
                # Careful not to keep a reference to the popped future
                yield await jobs.pop()
        finally:
            # The current task was cancelled, e.g. by timeout
            for job in jobs:
                job.cancel()

    async def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        self._shutdown = True
        if not self._init:
            return
        if cancel_futures:
            # Drain all work items from the queue, and then cancel their
            # associated futures.
            while not self._jobs.empty():
                self._jobs.get_nowait().cancel()

        self._jobs.shutdown()
        if not wait:
            for task in self._tasks:
                if not task.done():
                    task.cancel()

        rets = await gather(*self._tasks, return_exceptions=True)
        excs = [
            exc
            for exc in rets
            if isinstance(exc, BaseException)
            and type(exc) is not CancelledError
        ]
        if excs:
            try:
                raise BaseExceptionGroup(
                    "unhandled errors during Executor.shutdown()",
                    excs,
                ) from None
            finally:
                del excs

    def _lazy_init(self) -> None:
        # Lazy init exists for allowing Executor instantiation then there is no
        # active event loop yet.
        # .submit(), .map(), and .shutdown() all require running loop though.
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")
        if self._init:
            return
        self._init = True
        for i in range(self._max_workers):
            task_name = self._task_name_prefix + f"_{i}"
            self._tasks.append(
                create_task(
                    self._work(task_name),
                    name=task_name,
                    context=self._init_context,
                )
            )

    async def _work(self, prefix: str) -> None:
        try:
            while True:
                await (await self._jobs.get()).execute(prefix)
        except QueueShutDown:
            pass


@dataclasses.dataclass
class _Job[R]:
    loop: AbstractEventLoop
    context: contextvars.Context
    coro: Coroutine[Any, Any, R]

    def __post_init__(self) -> None:
        self.future: Future[R] = self.loop.create_future()

    async def execute(self, prefix: str) -> None:
        fut = self.future
        if fut.done():
            return
        try:
            name = prefix
            try:
                name += f" [{self.coro.__qualname__}]"
            except AttributeError:
                pass
            task = self.loop.create_task(
                self.coro,
                context=self.context,
                name=name,
            )
            fut.add_done_callback(_sync(task))
            ret = await task
        except BaseException as ex:
            if not fut.done():
                fut.set_exception(ex)
        else:
            if not fut.done():
                fut.set_result(ret)

    def cancel(self) -> None:
        fut = self.future
        if not fut.done():
            fut.cancel()
        with catch_warnings(action="ignore", category=RuntimeWarning):
            # Suppress RuntimeWarning: coroutine 'coro' was never awaited.
            # The warning is possible if .shutdown() was called
            # with cancel_futures=True and there are non-started coroutines
            # in pedning jobs list.
            del self.coro


def _sync[R](task: Task[R]) -> Callable[[Future[R]], None]:
    def f(fut: Future[R]) -> None:
        if fut.cancelled():
            if not task.done():
                task.cancel()

    return f
