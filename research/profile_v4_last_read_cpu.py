"""Local CPU-only microbenchmark; does not launch or modify Modal jobs."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from sliding_node_mem_v4.temporal import TemporalAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/sliding_v4/last_read_cpu.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    module = TemporalAttention().train()
    x = torch.randn(256, 9, 128, requires_grad=True)
    valid = torch.ones(256, 9, dtype=torch.bool)
    valid[:128, :4] = False
    times = torch.arange(9, dtype=torch.float64).repeat(256, 1) * .001 + 1_700_000_000
    upstream = torch.randn(256, 128)
    result = dict(scope='CPU temporal module forward+backward only; not GPU or full training',
                  torch=torch.__version__, threads=1, batch=256, window=9, seconds={})
    def measure(last):
        module.zero_grad(set_to_none=True)
        x.grad = None
        start = time.perf_counter()
        y = module.forward_last(x, valid, times) if last else module(x, valid, times)[:, -1]
        (y * upstream).sum().backward()
        return time.perf_counter() - start
    for _ in range(3):
        measure(False)
        measure(True)
    full, last = [], []
    for i in range(12):
        # Alternate order to reduce systematic warmup/thermal bias.
        if i % 2:
            last.append(measure(True)); full.append(measure(False))
        else:
            full.append(measure(False)); last.append(measure(True))
    result['seconds'] = dict(full=full, last=last)
    result['median_seconds'] = dict(full=statistics.median(full), last=statistics.median(last))
    result['module_speedup'] = statistics.median(full) / statistics.median(last)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'seconds'}, indent=2))


if __name__ == '__main__':
    main()
