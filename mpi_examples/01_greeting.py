import valohai

if not valohai.distributed.is_distributed_task():
    print('not running as a Valohai distributed task, aborting')
    exit(1)

print(f'hello from {valohai.distributed.me().identity}, I am rank {valohai.distributed.rank}!')
