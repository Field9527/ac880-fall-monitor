"""Run the guarded gesture probe on an idle board and retrieve its results."""
import argparse
import json
import os
from pathlib import Path
import shlex
import paramiko


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--case',type=Path,action='append',default=[]);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);source=Path(__file__).resolve().parents[1];remote=None
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.2.99',username='xilinx',password=os.getenv('AC880_PASSWORD','xilinx'),timeout=10)
    try:
        _,o,e=c.exec_command("python3 -c 'import tempfile;print(tempfile.mkdtemp(prefix=\"ac880_gesture_probe_\",dir=\"/tmp\"))'")
        remote=o.read().decode().strip()
        if not remote.startswith('/tmp/ac880_gesture_probe_'):raise RuntimeError('Invalid probe directory')
        (a.out/'remote_state.json').write_text(json.dumps(dict(remote=remote)))
        uploads=[(source/'sw'/n,n) for n in ('ncnn_backend.py','gesture_control.py','hand_models.py','hand_geometry.py')]
        meta=json.loads((a.model/'gesture.json').read_text());names=['gesture.json','export_reference.npz']
        for cfg in meta['models'].values():names.extend((cfg['param'],cfg['weights']))
        uploads.extend((a.model/n,n) for n in names)
        uploads.extend(((a.library,'libncnn.so.1'),(source/'scripts/gesture_probe_worker.py','worker.py')))
        uploads.extend((v,'case_%02d.jpg'%i) for i,v in enumerate(a.case))
        with c.open_sftp() as s:
            for path,name in uploads:s.put(str(path),remote+'/'+name)
        _,o,e=c.exec_command('env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/local/share/pynq-venv/bin/python3 '+shlex.quote(remote+'/worker.py')+' '+shlex.quote(remote),timeout=150)
        stdout=o.read().decode();stderr=e.read().decode();code=o.channel.recv_exit_status()
        (a.out/'stdout.txt').write_text(stdout);(a.out/'stderr.txt').write_text(stderr)
        if code:raise RuntimeError('Gesture CPU probe failed (%d)'%code)
        with c.open_sftp() as s:s.get(remote+'/result.json',str(a.out/'result.json'))
        print(stdout)
    finally:
        if remote:
            code='import pathlib,shutil;p=pathlib.Path('+repr(remote)+').resolve();assert p.parent==pathlib.Path("/tmp") and p.name.startswith("ac880_gesture_probe_");shutil.rmtree(str(p));print("removed")'
            _,o,e=c.exec_command('python3 -c '+shlex.quote(code),timeout=20);(a.out/'cleanup.txt').write_text(o.read().decode())
        c.close()


if __name__=='__main__':main()
