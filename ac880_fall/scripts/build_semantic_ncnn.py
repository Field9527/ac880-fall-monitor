"""Build an isolated ARMv7 probe library; never replaces the production runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import warnings
warnings.filterwarnings('ignore')
import paramiko


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--probe',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--extra-param',action='append',default=[],type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    accepted=a.workspace/'out/false_alarm_speed_20261016'
    source=accepted/'ncnn_source.tar.gz'
    metadata=json.loads((accepted/'ncnn_build_source.json').read_text())
    ops={line.split()[0] for line in (a.probe/'nanodet-m.param').read_text().splitlines()[2:] if line.strip()}
    for path in a.extra_param:
        ops.update(line.split()[0] for line in path.read_text().splitlines()[2:] if line.strip())
    # ModelBin calls the Cast layer internally while expanding FP16-stored
    # weights to FP32, even though Cast is absent from the .param graph.
    enabled=set(metadata['enabled_layers'])|ops|{'Cast'}
    cache=Path('E:/Zynq_cache/ncnn-20260526')
    names=re.findall(r'ncnn_add_layer\((\w+)',(cache/'src/CMakeLists.txt').read_text())
    if enabled-set(names):raise ValueError('Unknown source operators: '+str(enabled-set(names)))
    flags=['-DWITH_LAYER_%s=%s'%(name.lower(),'ON' if name in enabled else 'OFF') for name in names]
    cmake=['cmake','-S','ncnn','-B','build','-DCMAKE_BUILD_TYPE=Release','-DNCNN_VERSION=20260526',
           '-DNCNN_SHARED_LIB=ON','-DNCNN_C_API=ON','-DNCNN_OPENMP=ON','-DNCNN_RUNTIME_CPU=OFF',
           '-DNCNN_VFPV4=OFF','-DNCNN_VULKAN=OFF','-DNCNN_PIXEL=OFF','-DNCNN_PIXEL_ROTATE=OFF',
           '-DNCNN_PIXEL_AFFINE=OFF','-DNCNN_PIXEL_DRAWING=OFF','-DNCNN_INT8=OFF','-DNCNN_BF16=OFF',
           '-DNCNN_BUILD_TOOLS=OFF','-DNCNN_BUILD_EXAMPLES=OFF','-DNCNN_BUILD_BENCHMARK=OFF',
           '-DCMAKE_C_FLAGS=-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard',
           '-DCMAKE_CXX_FLAGS=-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard']+flags
    script='''import subprocess,tarfile,pathlib
root=pathlib.Path(__file__).parent
with tarfile.open(str(root/'ncnn_source.tar.gz')) as archive:archive.extractall(str(root))
subprocess.run(%r,cwd=str(root),check=True)
subprocess.run(['nice','-n','10','cmake','--build','build','-j','2'],cwd=str(root),check=True)
print('SEMANTIC_NCNN_BUILD_PASS',flush=True)
'''%cmake
    (a.out/'build_worker.py').write_text(script)
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('192.168.2.99',username='xilinx',password=os.getenv('AC880_PASSWORD','xilinx'),timeout=10)
    remote=None
    try:
        _,out,err=client.exec_command("python3 -c 'import tempfile;print(tempfile.mkdtemp(prefix=\"ac880_ncnn_20261028_\",dir=\"/tmp\"))'",timeout=10)
        remote=out.read().decode().strip()
        if out.channel.recv_exit_status() or not remote.startswith('/tmp/ac880_ncnn_20261028_'):
            raise RuntimeError('Cannot allocate isolated build')
        with client.open_sftp() as s:
            s.put(str(source),remote+'/ncnn_source.tar.gz')
            s.put(str(a.out/'build_worker.py'),remote+'/build_worker.py')
        command='env PYTHONDONTWRITEBYTECODE=1 python3 '+shlex.quote(remote+'/build_worker.py')+' > '+shlex.quote(remote+'/build.log')+' 2>&1'
        _,out,err=client.exec_command(command,timeout=1800)
        code=out.channel.recv_exit_status()
        with client.open_sftp() as s:
            s.get(remote+'/build.log',str(a.out/'build.log'))
            if code==0:s.get(remote+'/build/src/libncnn.so.1.0.20260526',str(a.out/'libncnn_semantic.so'))
        if code:raise RuntimeError('Isolated NCNN build failed; see local build.log')
        result=dict(source_commit=metadata['commit'],source_archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    enabled_layers=sorted(enabled),added_layers=sorted(enabled-set(metadata['enabled_layers'])),
                    library_sha256=hashlib.sha256((a.out/'libncnn_semantic.so').read_bytes()).hexdigest(),
                    production_modified=False,cpu='Cortex-A9 ARMv7 NEON hard-float, no VFPv4')
        (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    finally:
        if remote:
            code='import pathlib,shutil;p=pathlib.Path('+repr(remote)+').resolve();assert p.parent==pathlib.Path("/tmp") and p.name.startswith("ac880_ncnn_20261028_");shutil.rmtree(str(p));print("build_removed")'
            _,out,err=client.exec_command('python3 -c '+shlex.quote(code),timeout=30)
            result=out.read().decode().strip()
            (a.out/'cleanup.json').write_text(json.dumps(dict(result=result,exit_code=out.channel.recv_exit_status()))+'\n')
        client.close()


if __name__=='__main__':main()
