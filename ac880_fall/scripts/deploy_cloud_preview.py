"""Provision a loopback-only authenticated preview, never a public website."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shlex
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ml'))
import cache_env  # E: caches before third-party imports.
from cloud_ssh import CloudSSH


def deploy(a):
    source=Path(__file__).resolve().parents[1]/'cloud'
    a.output.mkdir(parents=True,exist_ok=True)
    # This directory was created with current-user-only permissions, outside Git.
    credentials=a.key.parent/'home_web_preview.json'
    if credentials.exists():
        account=json.loads(credentials.read_text(encoding='utf8'))
    else:
        account=dict(username='ac880',password=secrets.token_urlsafe(24),
                     preview_url='http://127.0.0.1:18089',public=False)
        with credentials.open('x',encoding='utf8') as f:
            json.dump(account,f,indent=2);f.write('\n')
    if account['username']!='ac880' or len(account['password'])<24:
        raise ValueError('Unexpected preview account; do not weaken its password')
    result=subprocess.run([str(a.openssl),'passwd','-6','-stdin'],
        input=account['password']+'\n',text=True,capture_output=True,check=True)
    digest=result.stdout.strip()
    if not digest.startswith('$6$'):raise ValueError('Expected SHA-512 crypt password hash')
    log=a.output/'provision.log'
    client=CloudSSH(a.cloud_host,a.cloud_user,a.key,a.known_hosts,log,
                    jump_host=a.jump_host,jump_user=a.jump_user,jump_key=a.jump_key)
    def run(command,timeout=180):
        return client.run(command,timeout=timeout)
    stage=None
    try:
        run('sudo -n true')
        # Runtime-mask the distribution's default service before installation so
        # a default unprotected port-80 site can never start as a package hook.
        script='''set -eu
if ! command -v nginx >/dev/null 2>&1; then
    sudo systemctl mask --runtime nginx.service
    trap 'sudo systemctl unmask --runtime nginx.service >/dev/null 2>&1 || true' EXIT
    sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nginx
    sudo systemctl disable nginx.service
    sudo systemctl unmask --runtime nginx.service
    trap - EXIT
fi
sudo install -d -m 0755 /opt/ac880-home-preview /opt/ac880-home-preview/logs
'''
        print('Preparing private Nginx runtime; no public listener',flush=True)
        run('bash -s <<\'AC880_PREPARE\'\n'+script+'AC880_PREPARE',timeout=300)
        stage=run('mktemp -d "$HOME/.ac880-preview.XXXXXXXX"').strip()
        if not stage.startswith('/') or '\n' in stage:
            raise RuntimeError('Unexpected remote staging directory')
        for name in ('nginx-preview.conf','ac880-home-preview.service'):
            client.put_file(source/name,stage+'/'+name)
        client.put_bytes(stage+'/users.htpasswd',
                         (account['username']+':'+digest+'\n').encode('utf8'))
        install='''set -eu
sudo install -o root -g root -m 0644 STAGE/nginx-preview.conf /opt/ac880-home-preview/nginx.conf
sudo install -o root -g www-data -m 0640 STAGE/users.htpasswd /opt/ac880-home-preview/users.htpasswd
sudo install -o root -g root -m 0644 STAGE/ac880-home-preview.service /etc/systemd/system/ac880-home-preview.service
sudo /usr/sbin/nginx -t -c /opt/ac880-home-preview/nginx.conf
sudo systemctl daemon-reload
sudo systemctl restart ac880-home-preview.service
sudo systemctl is-active ac880-home-preview.service
ss -lnt
'''.replace('STAGE',shlex.quote(stage))
        output=run('bash -s <<\'AC880_INSTALL\'\n'+install+'AC880_INSTALL',timeout=60)
        print(output,flush=True)
        check="""import socket,urllib.request,urllib.error,json
assert socket.create_connection(('127.0.0.1',18089),timeout=3)
result={}
for path in ('/','/api/state','/snapshot.jpg','/stream.mjpg','/api/incidents'):
 try:urllib.request.urlopen('http://127.0.0.1:18089'+path,timeout=3)
 except urllib.error.HTTPError as e:result[path]=e.code
assert set(result.values())=={401} and len(result)==5,result
print(json.dumps(result))
"""
        checks=json.loads(run('python3 -c '+shlex.quote(check)))
        receipt=dict(cloud_host=a.cloud_host,cloud_user=a.cloud_user,
            transport='OpenSSH',jump_host=a.jump_host,
            public=False,listen='127.0.0.1:18089',
            upstream='127.0.0.1:18088',unauthenticated_routes=checks,
            credential_file=str(credentials),manual_start=True,
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source.iterdir() if p.is_file()})
        (a.output/'provision_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('Private preview ready. Credentials are in '+str(credentials),flush=True)
    finally:
        if stage:
            files=[shlex.quote(stage+'/'+name) for name in
                   ('nginx-preview.conf','ac880-home-preview.service','users.htpasswd')]
            try:run('rm -f -- '+' '.join(files)+'\nrmdir -- '+shlex.quote(stage))
            except (RuntimeError,subprocess.TimeoutExpired):
                print('Remote staging cleanup failed; inspect the provision log.',file=sys.stderr)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cloud-host',default='182.92.239.147')
    p.add_argument('--cloud-user',default='ecs-user')
    p.add_argument('--jump-host',default=None)
    p.add_argument('--jump-user',default='ecs-user')
    p.add_argument('--jump-key',type=Path,default=Path('E:/Zynq_keys/aliyun/ac880_home.pem'))
    p.add_argument('--key',type=Path,default=Path('E:/Zynq_keys/aliyun/ac880_home.pem'))
    p.add_argument('--known-hosts',type=Path,default=Path('E:/Zynq_keys/aliyun/known_hosts'))
    p.add_argument('--openssl',type=Path,default=Path('E:/Anaconda/Library/bin/openssl.exe'))
    p.add_argument('--output',type=Path,required=True)
    deploy(p.parse_args())
