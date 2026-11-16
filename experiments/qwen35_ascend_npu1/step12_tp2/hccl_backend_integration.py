import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu
from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend

rank = int(os.environ['RANK'])
assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'
assert os.environ['ASCEND_VISIBLE_DEVICES'] == '2,3'
assert int(os.environ['WORLD_SIZE']) == 2 and torch.npu.device_count() == 2
torch.npu.set_device(rank)
dist.init_process_group('hccl', timeout=timedelta(seconds=90))
try:
    backend = HcclBackend()
    backend._process_group = lambda group: dist.group.WORLD
    group = (0, 1)
    local = torch.tensor([rank + 1], dtype=torch.float32, device=f'npu:{rank}')
    gathered = torch.empty(2, dtype=local.dtype, device=local.device)
    backend.all_gather_single(gathered, local, group)
    source = torch.tensor([rank + 1, rank + 2], dtype=local.dtype, device=local.device)
    reduced = backend.reduce_scatter(source, group)
    torch.npu.synchronize()
    got_gather = gathered.cpu().tolist()
    got_reduce = reduced.cpu().tolist()
    assert got_gather == [1.0, 2.0], got_gather
    assert got_reduce == ([3.0] if rank == 0 else [5.0]), got_reduce
    print(json.dumps({'rank': rank, 'device': str(local.device), 'all_gather': got_gather, 'reduce_scatter': got_reduce}), flush=True)
finally:
    dist.destroy_process_group()
