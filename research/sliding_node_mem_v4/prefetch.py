"""Bounded CPU batch preparation; the consumer alone advances model state."""
from concurrent.futures import ThreadPoolExecutor


def prefetch_batches(iterable):
    """Overlap one next() call with training, preserving order and exceptions.

    Use only with a CPU data producer that does not touch model state or RNG.
    Early closure waits for its one pending read and then closes the producer.
    """
    iterator = iter(iterable)
    done = object()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='v4-batch-reader')
    try:
        pending = pool.submit(next, iterator, done)
        while True:
            batch = pending.result()
            if batch is done:
                break
            pending = pool.submit(next, iterator, done)
            yield batch
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
