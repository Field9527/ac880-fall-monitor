"""CPU-only bounded shared-model regression; never opens PYNQ or GPIO."""
import argparse
import fcntl
import json
from pathlib import Path
import resource
import time
import cv2
import numpy as np
from fall_lite import FallLiteDetector
from shared_detector import SharedFallDetector


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
    resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
    resource.setrlimit(resource.RLIMIT_CPU,(90,90));resource.setrlimit(resource.RLIMIT_CORE,(0,0));cv2.setNumThreads(1)
    lock_path=Path('/run/lock/ac880_fall.lock')
    hardware_lock=lock_path.open('r' if lock_path.exists() else 'x')
    fcntl.flock(hardware_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    production=Path('/home/xilinx/fall');library=production/'assets/libncnn.so.1'
    old=FallLiteDetector(dict(model=str(production/'assets/models/fall_lite.onnx'),metadata=str(production/'assets/models/fall_lite.json'),engine='ncnn',runtime_library=str(library),threads=1,score_threshold=.7))
    sample=np.random.RandomState(17).normal(size=(1,3,192,320)).astype(np.float32)
    expected=old.net.forward(sample)
    cases=json.loads((root/'cases.json').read_text())['items'];rows=[]
    for index,item in enumerate(cases):
        rgb=np.load(str(root/item['file']),allow_pickle=False)[:,:,::-1].copy()
        if index==0:old.predict_rgb(rgb)
        start=time.perf_counter();found=old.predict_rgb(rgb)
        rows.append(dict(id=item['id'],kind=item['kind'],expected_fall_boxes=item['boxes'],baseline_boxes=found,baseline_seconds=time.perf_counter()-start))
    old.close();print('BASELINE_DONE',flush=True)
    cfg=dict(metadata=str(root/'shared_semantic.json'),runtime_library=str(root/'libncnn_semantic.so') if (root/'libncnn_semantic.so').exists() else str(library),engine='semantic',threads=1,allow_experimental=True)
    if (root/'scene').exists():cfg['scene_model']=str(root/'scene')
    new=SharedFallDetector(cfg)
    outputs=new.net.forward(sample);errors=[float(np.abs(a-b).max()) for a,b in zip(expected,outputs[:3])]
    if not new.nano and not all(np.allclose(a,b,atol=.001,rtol=.003) for a,b in zip(expected,outputs[:3])):raise RuntimeError('Shared ARM fall path parity failed')
    export_errors=[]
    if (root/'export_reference.npz').exists():
        refs=np.load(str(root/'export_reference.npz'),allow_pickle=False)
        for i,actual in enumerate(outputs):
            ref=refs['out'+str(i)]
            if actual.ndim==2:ref=ref[0]
            export_errors.append(float(abs(actual-ref).max()))
            if actual.shape!=ref.shape or not np.allclose(actual,ref,atol=.001,rtol=.003):raise RuntimeError('ARM export parity failed: '+str(i))
    for index,item in enumerate(cases):
        rgb=np.load(str(root/item['file']),allow_pickle=False)[:,:,::-1].copy()
        if index==0:new.predict_rgb(rgb)
        start=time.perf_counter();found=new.predict_rgb(rgb);seconds=time.perf_counter()-start
        rows[index].update(shared_boxes=found,shared_seconds=seconds,semantic=new.last_semantic)
        print('CASE',item['id'],round(seconds,3),len(found),len(new.last_semantic['reviewed_candidates']),flush=True)
    new.close()
    summary={name:dict(mean=float(np.mean([r[name] for r in rows])),p95=float(np.percentile([r[name] for r in rows],95))) for name in ('baseline_seconds','shared_seconds')}
    result=dict(rows=rows,summary=summary,fall_output_max_abs_errors=errors if not new.nano else None,export_max_abs_errors=export_errors,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,hardware_programmed=False,production_modified=False,scope='CPU-only fixed regression; no live video concurrency')
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
