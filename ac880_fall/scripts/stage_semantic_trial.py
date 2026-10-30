"""Stage a new bounded trial beside the stopped production tree, never replace it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko


def main():
    p=argparse.ArgumentParser();p.add_argument('runtime',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    root=a.runtime.resolve();manifest=json.loads((root/'manifest.json').read_text())
    for name,item in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Local runtime changed')
    config=json.loads((root/'config.json').read_text());remote=str(Path(config['overlay_path']).parent.parent).replace('\\','/')
    if remote not in ('/home/xilinx/fall_semantic_trial','/home/xilinx/fall_gesture_trial'):raise ValueError('Unexpected bounded trial root')
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('192.168.2.99',username='xilinx',password=os.getenv('AC880_PASSWORD','xilinx'),timeout=8)
    try:
        _,o,e=c.exec_command('/home/xilinx/fall/run.sh status',timeout=15)
        if json.loads(o.read().decode())['running']:raise RuntimeError('Stop production gracefully before a hardware trial')
        with c.open_sftp() as s:
            s.mkdir(remote)  # never overwrite an existing trial or a user directory
            for folder in sorted((v for v in root.rglob('*') if v.is_dir()),key=lambda p:len(p.parts)):s.mkdir(remote+'/'+folder.relative_to(root).as_posix())
            for path in (v for v in root.rglob('*') if v.is_file()):
                dest=remote+'/'+path.relative_to(root).as_posix();s.put(str(path),dest)
                if path.suffix=='.sh':s.chmod(dest,0o755)
        code='''import fcntl,hashlib,json,pathlib,subprocess,os
p=pathlib.Path(REMOTE_ROOT);assert p.resolve()==p
lockpath=pathlib.Path('/run/lock/ac880_fall.lock')
with lockpath.open('r' if lockpath.exists() else 'x') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m=json.loads((p/'manifest.json').read_text())
 for name,item in m['files'].items():
  path=p/name;assert p in path.resolve().parents
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],name
  if path.suffix=='.sh':subprocess.run(['bash','-n',str(path)],check=True)
  if path.suffix=='.py':compile(path.read_bytes(),str(path),'exec')
  with path.open('rb') as f:os.fsync(f.fileno())
 os.sync()
 print(json.dumps(dict(root=str(p),verified=True,files=len(m['files']),production_modified=False)))
'''
        code=code.replace('REMOTE_ROOT',repr(remote))
        _,o,e=c.exec_command('sudo -n /usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(code),timeout=90)
        result=o.read().decode();error=e.read().decode()
        if o.channel.recv_exit_status():raise RuntimeError(error)
        a.receipt.write_text(result);print(result)
    finally:c.close()


if __name__=='__main__':main()
