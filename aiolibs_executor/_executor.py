import asyncio
import contextvars
from collections.abc import Callable, Awaitable, Iterable, AsyncIterator
from itertools import counter
from typing import Any


class Executor:
    _counter = itertools.count().__next__

    def __init__(
            self,
            max_workers: int | None = None,
            task_name_prefix: str = "",
            initializer: Callable[..., Awaitable[None]] | None = None,
            initargs: tuple[Any, ...] = (),
    ) -> None:
        if max_workers is None:
            max_workers = 100
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._max_workers = max_workers
        self._task_name_prefix = task_name_prefix or f'Executor-{self._counter()}'
        if initializer is not None:
            if not callable(initializer):
                raise TypeError("initializer must be a callable")
        self._initializer = initializer
        self._initargs = initargs
        self._init = False
        self._shutdown = False
        self._jobs: asyncio.Queue[_Job] = asyncio.Queue()
        # tasks are much cheaper than threads or processes,
        # there is no need for adjusting tasks count on the fly like
        # ThreadPoolExecutor or ProcessPoolExecutor do.
        self._tasks = []

    def submit[R, **P](
        self,
        fn: Callable[P, Awaitable[R]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> asyncio.Future[R]:
        return self.submit_with_context(None, fn, *args, **kwargs)

    def submit_with_context[R, **P](
        self,
        context: contextvars.Context | None,
        fn: Callable[P, Awaitable[R]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> asyncio.Future[R]:
        if self._shutdown:
            raise RuntimeError('cannot schedule new futures after shutdown')
        self._lazy_init()
        loop = asyncio.get_running_loop()
        job = _Job(loop, context, None, fn, *args, **kwargs)
        self._jobs.put_nowait(job)
        return job

    def map[R, *IT](
        self,
        fn: Callable[..., Awaitable[R]],
        /,
        *iterables: Iterable[Any],
    ) -> AsyncIterator[R]:
        jobs = [self.submit(fn, *args) for args in zip(*iterables, strict=False)]

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        async def result_iterator():
            try:
                # reverse to keep finishing order
                jobs.reverse()
                while jobs:
                    # Careful not to keep a reference to the popped future
                    yield await jobs.pop()
            finally:
                for job in jobs:
                    job.cancel()
        return result_iterator()

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
                job = self._jobs.get_nowait()
                if not job.done():
                    job.cancel()
                del job

        self._jobs.shutdown()
        if wait:
            rets = await asyncio.gather(self._tasks, return_exceptions=True)
            excs = [exc for exc in rets if isinstance(exc, BaseException)]
            if excs:
                try:
                    raise BaseExceptionGroup(
                        "unhandled errors during Executor.shutdown()",
                        excs,
                    ) from None
                finally:
                    excs = None

    def _lazy_init(self) -> None:
        # Lazy init exists for allowing Executor instantiation then there is no
        # active event loop yet.
        # .submit(), .map(), and .shutdown() all require running loop though.
        if self._init:
            return
        self._init = True
        for i in range(self._max_workers):
            task_name = self._task_name_prefix + f"_{i}"
            self._tasks.append(asyncio.create_task(self._work(), name=task_name))

    async def _work(self) -> None:
        try:
            if self._initializer is not None:
                await self._initializer(*self._initargs)
            while True:
                job = self._jobs.get()
                await job._execute()
                await self._outcome.put(job)
                del job
        except asyncio.QueueShutDown:
            pass


class _Job[R, **P](asyncio.Future[R]):
    def __init__(
            self,
            loop: asyncio.AbstractEventLoop,
            outcome: asyncio.Queue['_Job'] | None,
            fn: Callable[P, Awaitable[R]],
            /,
            *args: P.args,
            **kwargs: P.kwargs,
    ) -> None:
        super().__init__(loop=loop)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._outcome = outcome

    async def _execute(self) -> R:
        try:
            ret = await self._fn(*self._args, **self._kwargs)
        except BaseException as ex:
            if not self.done():
                self.set_exception(ex)
        else:
            if not self.done():
                self.set_result(ret)
        if self._outcome is not None:
            # executor.map() mode
            self._outcome.put(self)
            # break cycle reference
            self._outcome = None
