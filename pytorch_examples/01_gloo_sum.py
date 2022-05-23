import json
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(my_rank, world_size):
    group = dist.new_group(list(range(world_size)))
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f'Rank {my_rank} has data {tensor} on host {socket.gethostname()}')


def init(master_url, my_rank, world_size, fn):
    dist.init_process_group(init_method=master_url, rank=my_rank, world_size=world_size, backend='gloo')
    fn(my_rank, world_size)


if __name__ == '__main__':

    with open('/valohai/config/distributed.json') as fp:
        distributed_config = json.load(fp)

    master = None
    for member in distributed_config['members']:
        if member['member_id'] == '0':
            master = member
    assert master, 'Master not found in distributed configuration.'

    master_port = 1234
    master_ip = master['network']['local_ips'][0]
    url = f"tcp://{master_ip}:{master_port}"
    size = distributed_config['config']['required_count']
    rank = int(distributed_config['self']['member_id'])

    mp.set_start_method('spawn')
    p = mp.Process(target=init, args=(url, rank, size, run))
    p.start()
    p.join()
