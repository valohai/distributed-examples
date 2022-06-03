import valohai

# Valohai prepares distributed task configuration in various formats under /valohai/config/
#   * /valohai/config/distributed.json
#   * /valohai/config/distributed.yaml
#
# 'valohai.distributed' will automatically search and parse this configuration

# 'valohai.distributed' contains information about the distributed group itself
assert valohai.distributed.group_name.startswith('task-')  # task id e.g. "task-0180acd6-c7ad-ca5b-bb1a-00f278d4183c"
assert valohai.distributed.required_count  # total number of members

# 'members()' contains a list of _all_ running executions in the group, including this execution
for member in valohai.distributed.members():
    assert member.rank >= 0  # running number integer from zero, each member has a different assigned rank
    assert member.member_id  # member id of this member i.e. '0', '1', 'worker-123' etc.
    assert member.identity  # identity string of the cloud instance; format depends on the infrastructure used
    assert member.announce_time  # when the member joined the group in ISO 8061

    # registered local IPs, public IPs and exposed port pairings
    assert isinstance(member.local_ips, list)
    assert isinstance(member.public_ips, list)
    assert isinstance(member.exposed_ports, dict)

    assert member.primary_local_ip  # the first local ip reported e.g. 10.0.16.21
    assert member.primary_public_ip  # the first public ip reported

# if you want to only get details about this particular member,
# instead of finding it in 'members()', there is the 'me()' helper
me = valohai.distributed.me()
print(f'Distributed configuration is valid on {me.identity}!')
