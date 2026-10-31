"""Human-readable entity details from recorded attributes and real events."""
import json
from pathlib import PurePosixPath

from web_app.backend.cadets import RELATIONS


def first(attributes, *names):
    for name in names:
        value = attributes.get(name)
        if value is not None and str(value).strip().lower() not in ('', 'none', 'null', '<unknown>'):
            return str(value)
    return None


def describe(node):
    a = node['attributes']
    kind = first(a, 'type') or 'entity'
    command = first(a, 'cmd_line', 'command_line', 'commandLine', 'command', 'cmd')
    path = first(a, 'path', 'file_path', 'filePath', 'executable', 'image', 'Image')
    return dict(kind=kind, command=command,
                executable_path=path if kind in ('subject', 'process') else first(a, 'executable', 'image', 'Image'),
                file_path=path if kind == 'file' else None,
                file_name=(PurePosixPath(path.replace('\\', '/')).name if path else first(a, 'name')),
                pid=first(a, 'pid', 'process_id', 'processId', 'ProcessId'),
                parent_pid=first(a, 'ppid', 'parent_pid', 'parentProcessId'),
                user=first(a, 'user', 'username', 'user_id', 'uid'),
                host=first(a, 'host', 'hostname', 'host_name', 'computer'),
                uuid=first(a, 'uuid', 'node_uuid', 'process_guid'),
                hash=first(a, 'hash', 'hash_id', 'sha256', 'md5'),
                local_ip=first(a, 'local_ip', 'localAddress', 'src_addr'),
                local_port=first(a, 'local_port', 'localPort', 'src_port'),
                remote_ip=first(a, 'remote_ip', 'remoteAddress', 'dst_addr'),
                remote_port=first(a, 'remote_port', 'remotePort', 'dst_port'),
                command_has_arguments=bool(command and any(c.isspace() for c in command)))


def details(db, node, split):
    node = dict(node)
    node['attributes'] = json.loads(node.pop('attrs'))
    recorded = describe(node)
    selection = ('SELECT * FROM events WHERE split=? AND src=? '
                 'UNION SELECT * FROM events WHERE split=? AND dst=?')
    args = (split, node['id'], split, node['id'])
    activity = db.execute('SELECT count(*) AS events,min(time) AS first_seen,max(time) AS last_seen FROM (' + selection + ')', args).fetchone()
    operations = db.execute('SELECT relation,count(*) AS events FROM (' + selection + ') GROUP BY relation ORDER BY events DESC', args).fetchall()
    # Exact node IDs join the metadata index; never reverse an embedding/hash or
    # infer executable paths from the process name alone.
    query = ('SELECT n.id,n.attrs,e.relation,count(*) AS events FROM (' + selection + ') e '
             'JOIN nodes n ON n.id=CASE WHEN e.src=? THEN e.dst ELSE e.src END '
             'WHERE n.id!=? GROUP BY n.id,e.relation ORDER BY events DESC,n.id LIMIT 30')
    related = []
    for row in db.execute(query, args + (node['id'], node['id'])):
        attrs = json.loads(row['attrs'])
        info = describe(dict(attributes=attrs))
        related.append(dict(id=row['id'], relation=RELATIONS[row['relation']], events=row['events'],
                            **info))
    missing = []
    if recorded['kind'] in ('subject', 'process'):
        if not recorded['command_has_arguments']:
            missing.append('Full command-line arguments were not recorded; the available command is a program name only.')
        if not recorded['executable_path']:
            missing.append('The process executable path was not recorded. Executed files below come from observed EXECUTE events.')
    if recorded['kind'] == 'file' and not recorded['file_path']:
        missing.append('The source data contains no filename or path for this file. Its numeric entity ID cannot be decoded into a filename.')
    return dict(id=node['id'], recorded=recorded, activity=dict(activity),
                operations=[dict(relation=RELATIONS[r['relation']], events=r['events']) for r in operations],
                related_entities=related, related_limit=30, limitations=missing,
                attributes=node['attributes'], provenance='Recorded entity attributes and observed event endpoints; no inferred filenames or command arguments.')
