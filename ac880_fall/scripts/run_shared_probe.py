"""Upload only CPU probe assets to an isolated, guarded temporary directory."""
import argparse
import json
import os
from pathlib import Path
import shlex
import paramiko


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path);p.add_argument('--cases',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--limit',type=int)
    p.add_argument('--runtime-library',type=Path);p.add_argument('--scene',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    cases=json.loads((a.cases/'cases.json').read_text())
    if a.limit:cases['items']=cases['items'][:a.limit]
    (a.out/'cases.json').write_text(json.dumps(cases))
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('192.168.2.99',username='xilinx',password=os.getenv('AC880_PASSWORD','xilinx'),timeout=8);remote=None
    try:
        _,out,err=client.exec_command('/home/xilinx/fall/run.sh status',timeout=15)
        status=json.loads(out.read().decode())
        if status['running']:raise RuntimeError('Stop the production service gracefully before this idle-CPU benchmark')
        _,out,err=client.exec_command("python3 -c 'import tempfile;print(tempfile.mkdtemp(prefix=\"ac880_shared_20261028_\",dir=\"/tmp\"))'",timeout=10)
        remote=out.read().decode().strip()
        if out.channel.recv_exit_status() or not remote.startswith('/tmp/ac880_shared_20261028_'):raise RuntimeError('Cannot allocate bounded probe directory')
        (a.out/'remote_state.json').write_text(json.dumps(dict(remote=remote))+'\n')
        uploads=[(root/'sw'/name,name) for name in ('fall_lite.py','ncnn_backend.py','semantic_features.py','semantic_decode.py','shared_detector.py','semantic_probe.py')]
        uploads.append((root/'scripts/shared_probe_worker.py','worker.py'))
        uploads += [(a.model/name,name) for name in ('shared_semantic.ncnn.param','shared_semantic.ncnn.bin','shared_semantic.json','region_verifier.npz')]
        model_meta=json.loads((a.model/'shared_semantic.json').read_text(encoding='utf8'))
        if model_meta.get('reference_verifier'):
            name=model_meta['reference_verifier']['file']
            if Path(name).name!=name:raise ValueError('Invalid reference verifier path')
            uploads.append((a.model/name,name))
        uploads += [(a.cases/r['file'],r['file']) for r in cases['items']]
        uploads.append((a.out/'cases.json','cases.json'))
        if (a.model/'export_reference.npz').exists():uploads.append((a.model/'export_reference.npz','export_reference.npz'))
        if a.runtime_library:uploads.append((a.runtime_library,'libncnn_semantic.so'))
        if a.scene:uploads += [(a.scene/name,'scene/'+name) for name in ('nanodet-m.param','nanodet-m.bin','source_manifest.json')]
        with client.open_sftp() as sftp:
            if a.scene:sftp.mkdir(remote+'/scene')
            for local,name in uploads:sftp.put(str(local),remote+'/'+name)
        _,out,err=client.exec_command('env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/local/share/pynq-venv/bin/python3 '+shlex.quote(remote+'/worker.py')+' '+shlex.quote(remote),timeout=180)
        stdout,stderr=out.read().decode(),err.read().decode();code=out.channel.recv_exit_status()
        (a.out/'stdout.txt').write_text(stdout);(a.out/'stderr.txt').write_text(stderr)
        if code:raise RuntimeError('Shared probe failed (%d); inspect saved stderr'%code)
        with client.open_sftp() as sftp:sftp.get(remote+'/result.json',str(a.out/'result.json'))
        print(stdout)
    finally:
        cleanup=dict(remote=remote,result='not_allocated')
        if remote:
            try:
                script='import pathlib,shutil;p=pathlib.Path('+repr(remote)+').resolve();assert p.parent==pathlib.Path("/tmp") and p.name.startswith("ac880_shared_20261028_");shutil.rmtree(str(p));print("removed")'
                _,out,err=client.exec_command('python3 -c '+shlex.quote(script),timeout=15)
                cleanup.update(result=out.read().decode().strip(),exit_code=out.channel.recv_exit_status())
            except Exception as error:cleanup.update(result='pending',error_type=type(error).__name__)
        (a.out/'cleanup.json').write_text(json.dumps(cleanup)+'\n');client.close()


if __name__=='__main__':main()
