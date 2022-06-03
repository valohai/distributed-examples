import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from subprocess import Popen

import valohai
from Crypto.PublicKey import RSA


def generate_key_pair(
    seed,
    private_key_format='PEM',
    public_key_format='OpenSSH',
):
    """
    Generate an SSH keypair (private and public key) from the given seed.

    This intentionally generates the same SSH key pair given the same seed.
    Effectively this makes the SSH authentication as secure as the seed used to generate these keys
    for an entity that has gained access to the network.

    The inter-worker network is secure nevertheless but this restricts unintended communication
    from workers outside the specific worker group.
    """
    previous_state = random.getstate()
    random.seed(a=seed, version=2)
    # random.randbytes was added in Python 3.9 which does the same thing as the following lambda
    key = RSA.generate(2048, randfunc=lambda n: random.getrandbits(n * 8).to_bytes(n, 'little'))
    private = key.export_key(private_key_format).decode()
    public = key.publickey().export_key(format=public_key_format).decode()
    random.setstate(previous_state)
    return private, public


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', '-v',
        required=False,
        default=False,
        action='store_true',
        help="show debugging information while running",
    )
    parser.add_argument(
        '--dry-run', '--dr',
        required=False,
        default=False,
        action='store_true',
        help="show the resulting command but don't run it",
    )
    parser.add_argument(
        '--master-wait', '--mw',
        required=False,
        default=5,
        type=int,
        metavar='SECONDS',
    )
    parser.add_argument(
        '--processes-per-host', '--pph',
        required=False,
        default=1,
        type=int,
        metavar='COUNT',
        help='the number of worker processes per host; usually the number of GPUs per the host, default: 1',
    )
    parser.add_argument(
        '--ssh-port', '--sp',
        required=False,
        default=1234,
        type=int,
        metavar='PORT',
        help='the port to use for SSH connection, default: 1234'
    )
    _, *args = sys.argv
    settings, target_command = parser.parse_known_args(args=args)

    print('Settings:', settings)

    print('Me:', valohai.distributed.me())

    home_dir = os.environ.get('HOME')
    if home_dir != '/root':
        ssh_dir = os.path.abspath(f'{__file__}/../.ssh')
    else:
        ssh_dir = f'{home_dir}/.ssh'

    private_key, public_key = generate_key_pair(seed=valohai.distributed.group_name)

    ssh_file = 'id_rsa'
    ssh_private_target = os.path.join(ssh_dir, ssh_file)
    ssh_public_target = os.path.join(ssh_dir, f'{ssh_file}.pub')
    authorized_keys_target = os.path.join(ssh_dir, 'authorized_keys')
    ssh_config_target = os.path.join(ssh_dir, 'config')
    os.makedirs(ssh_dir, mode=0o700, exist_ok=True)

    with open(ssh_private_target, mode='w') as fp:
        fp.write(private_key)
        fp.write(os.linesep)

    with open(ssh_public_target, mode='w') as fp:
        fp.write(f'{public_key}{os.linesep}')

    with open(authorized_keys_target, mode='a') as fp:
        fp.write(f'{public_key}{os.linesep}')

    with open(ssh_config_target, mode='a') as fp:
        fp.write(f'Host *{os.linesep}')
        fp.write(f'    User root{os.linesep}')
        fp.write(f'    IdentityFile {ssh_private_target}{os.linesep}')
        fp.write(f'    StrictHostKeyChecking no{os.linesep}')
        fp.write(f'    UserKnownHostsFile /dev/null{os.linesep}')

    for target in [ssh_private_target, ssh_public_target, authorized_keys_target, ssh_config_target]:
        os.chmod(target, mode=0o700)

    if not valohai.distributed.me().is_master:
        # fixes "Missing privilege separation directory: /run/sshd" if `sshd` was freshly installed
        os.makedirs('/run/sshd', mode=0o700, exist_ok=True)
        sshd_cmd = shutil.which('sshd')
        sshd_process = Popen([
            sshd_cmd,
            '-d',  # debug, but also accepts just a single connection before exit
            '-D',  # don't detach
            '-o', 'PermitUserEnvironment=yes',
            '-o', 'PermitRootLogin=yes',
            '-p', str(settings.ssh_port),
            '-h', ssh_private_target,
        ])
        output, error = sshd_process.communicate()
        print(output, error)
        print(f'Exit Status: {sshd_process.returncode}')
        exit()  # don't exit with the return code as all return codes from debugging `sshd` are >0

    time.sleep(settings.master_wait)

    primary_local_ips = [m.primary_local_ip for m in valohai.distributed.members()]
    host_configurations = [f'{ip}:{settings.processes_per_host}' for ip in primary_local_ips]
    host_value = ','.join(host_configurations)

    mpi_executable = [shutil.which('mpirun')]

    options = []
    options = [*options, '--allow-run-as-root']

    options = [*options, *['--mca', 'plm_rsh_agent', 'ssh']]
    ssh_args = [f'-p {settings.ssh_port}']
    if not settings.verbose:
        ssh_args.append('-o LogLevel=ERROR')
    options = [*options, *['--mca', 'plm_rsh_args', f'"{" ".join(ssh_args)}"']]

    if settings.verbose:
        options = [*options, *['--mca', 'orte_base_help_aggregate', '0']]

    options = [*options, *['-bind-to', 'none']]  # don't bind a training process to a single CPU core
    options = [*options, *['-map-by', 'slot']]  # allows you to have a mixture of different NUMA configurations

    # specify/copy environment variables to all the workers
    options = [*options, *['-x', 'VH_CONFIG_DIR=/valohai/config']]
    options = [*options, *['-x', 'NCCL_DEBUG=INFO']]
    options = [*options, *['-x', 'PATH']]

    options = [*options, *['-np', str(valohai.distributed.required_count * settings.processes_per_host)]]
    options = [*options, *['--host', host_value]]

    if settings.verbose:
        options = [*options, *['-v']]  # mpi debug

    command = [*mpi_executable, *options, *target_command]
    print('Command:', command)

    if settings.dry_run:
        exit()

    # TODO: retry logic?
    process = Popen(command, stdin=subprocess.DEVNULL)
    output, error = process.communicate()
    print(output, error)
    exit(process.returncode)
