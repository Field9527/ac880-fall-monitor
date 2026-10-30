"""CPU-only ARM numerical and timing probe: no PYNQ, camera, GPIO or DMA."""
import json
import fcntl
from pathlib import Path
import resource
import sys
import time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU,(90,90))
lock_path=Path('/run/lock/ac880_fall.lock')
hardware_lock=lock_path.open('r' if lock_path.exists() else 'x')
fcntl.flock(hardware_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
import cv2
import numpy as np
from ncnn_backend import NCNNNet
from hand_models import HandModels

root=Path(sys.argv[1]);meta=json.loads((root/'gesture.json').read_text());refs=np.load(str(root/'export_reference.npz'))
cv2.setNumThreads(1);result=dict(cpu_only=True,outputs=[],timing=[])
for name in ('palm','hand'):
    c=meta['models'][name];net=NCNNNet(root/'libncnn.so.1',root/c['param'],root/c['weights'],1,output_names=c['output_names'],output_layouts=c['output_layouts'])
    for repeat in range(3):
        start=time.perf_counter();actual=net.forward(refs[name+'_input']);elapsed=time.perf_counter()-start
        result['timing'].append(dict(model=name,repeat=repeat,seconds=elapsed))
    for i,v in enumerate(actual):
        ref=refs[name+'_out'+str(i)]
        if v.size!=ref.size or not np.allclose(v.reshape(ref.shape),ref,rtol=.003,atol=.005):raise ValueError('ARM output parity failed: %s out%d'%(name,i))
        result['outputs'].append(dict(model=name,output=i,shape=list(v.shape),max_error=float(np.max(abs(v.reshape(ref.shape)-ref)))))
    net.close()
hands=HandModels(root,root/'libncnn.so.1');result['images']=[]
for path in sorted(root.glob('case_*.jpg')):
    hands.reset()  # unrelated fixed images are not a video track
    image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);start=time.perf_counter();hand=hands.detect(image)
    result['images'].append(dict(file=path.name,hand=hand,seconds=time.perf_counter()-start))
hands.close();result['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result['pass_']=True
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
