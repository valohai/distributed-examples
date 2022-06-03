import json

# Valohai prepared distributed task configuration in various formats under /valohai/config/
#   * /valohai/config/distributed.json
#   * /valohai/config/distributed.yaml
import socket

with open('/valohai/config/distributed.json') as fp:
    distributed_config = json.load(fp)

# 'config' contains information about the distributed group and how this specific execution relates to that group
membership = distributed_config['config']
assert membership['group_name'].startswith('task-')  # task id e.g. "task-0180acd6-c7ad-ca5b-bb1a-00f278d4183c"
assert 'member_id' in membership  # running number starting from 0 as a string i.e. '0', '1', '2' etc.
assert 'required_count' in membership  # total number of members

# 'members' contains a list of _all_ running executions in the group, including this execution
members = distributed_config['members']
for member in members:
    assert 'member_id' in member  # member id of this member i.e. '0', '1', '2' etc.
    assert 'identity' in member  # identity string of the cloud instance; format depends on the infrastructure used
    assert 'announce_time' in member  # when the member joined the group in ISO 8061

    member_network = member['network']  # network configuration of this specific member
    assert 'exposed_ports' in member_network  # if any ports were exposed through `VH_EXPOSE_PORTS` env var

    # lists of registered local and public IPs
    assert isinstance(member_network['local_ips'], list)
    assert isinstance(member_network['public_ips'], list)

    # usually, the first local ip is the most important
    assert member_network['local_ips'][0]  # primary local ip reported e.g. 10.0.16.21

# if you want to only get details about this particular member,
# instead of finding it in 'members', there is a helper key 'self'
# which is a copy of the 'members' item
me = distributed_config['self']
assert me['member_id'] == membership['member_id']

me_in_members = list(filter(lambda m: m['member_id'] == membership['member_id'], members))[0]
assert me_in_members['member_id'] == membership['member_id']

print(f'Distributed configuration is valid on {socket.gethostname()}!')
