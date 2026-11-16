import json
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '2,3'
assert os.environ.get('ASCEND_VISIBLE_DEVICES') == '2,3'
assert int(os.environ['WORLD_SIZE']) == 2
assert torch.npu.device_count() == 2
assert local_rank in (0, 1)
torch.npu.set_device(local_rank)
dist.init_process_group(backend='hccl', timeout=timedelta(seconds=90))
try:
    value = torch.tensor([rank + 1], dtype=torch.float32, device=f'npu:{local_rank}')
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    result = value.cpu().item()
    assert result == 3.0, f'Unexpected all_reduce result: {result}'
    print(json.dumps({'rank': rank, 'local_rank': local_rank, 'world_size': dist.get_world_size(), 'visible_physical_devices': os.environ['ASCEND_RT_VISIBLE_DEVICES'], 'expected_physical_id': (2, 3)[local_rank], 'device': str(value.device), 'allreduce_sum': result, 'hostname': socket.gethostname(), 'pid': os.getpid()}), flush=True)
finally:
    dist.destroy_process_group()
