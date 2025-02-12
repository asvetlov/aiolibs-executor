# aiolibs-executor

Asyncio version of `concurrent.futures` executor.

## Installaion

`pip install aiolibs-executor`

## Usage

```
from aiolibs_executor import Executor

async def worker(arg):
   return await do_stuff(arg)

async with Executor(num_workers=3) as executor:
	futs = [await executor.submit(worker(i)) for i in range(10)]
	for fut in futs:
		print(await fut)
```

All submitted `worker(i)` coroutines are distributed to three concurrent streams,
awaiting returned functions provides a value returned by a coroutine.

TODO: Make comprehensive description of all public `Executor`'s methods.


## License

The library is published under Apache 2.0 license.
