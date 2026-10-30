"""Check NHWC source, NCHW adaptation and NCNN on identical owned tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ml'))
import cache_env
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('--ncnn-python',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.ncnn_python));import ncnn
    cv2.setNumThreads(1);rows=[];references={}
    for name,size,count in (('palm',192,2),('hand',224,4)):
        source=cv2.dnn.readNetFromONNX(str(a.model/(name+'.onnx')))
        adapted=cv2.dnn.readNetFromONNX(str(a.model/(name+'_nchw.onnx')))
        net=ncnn.Net();net.opt.num_threads=1
        net.opt.use_fp16_storage=False;net.opt.use_fp16_packed=False;net.opt.use_fp16_arithmetic=False;net.opt.use_vulkan_compute=False
        if net.load_param(str(a.model/(name+'_nchw.ncnn.param'))) or net.load_model(str(a.model/(name+'_nchw.ncnn.bin'))):raise RuntimeError('NCNN load failed')
        for seed in (0,17):
            nhwc=np.random.RandomState(seed).uniform(0,1,(1,size,size,3)).astype(np.float32);nchw=np.ascontiguousarray(nhwc.transpose(0,3,1,2))
            source.setInput(nhwc);expected=source.forward(['Identity'+('_'+str(i) if i else '') for i in range(count)])
            adapted.setInput(nchw);converted=adapted.forward(['Identity'+('_'+str(i) if i else '') for i in range(count)])
            actual=[]
            with net.create_extractor() as ex:
                owned=ncnn.Mat(nchw[0]);ex.input('in0',owned)
                for i in range(count):
                    code,v=ex.extract('out'+str(i))
                    if code:raise RuntimeError('NCNN extraction failed')
                    actual.append(np.asarray(v).copy().reshape(expected[i].shape))
            for i,(ref,cv,nc) in enumerate(zip(expected,converted,actual)):
                if not np.allclose(ref,cv,atol=.003,rtol=.003) or not np.allclose(ref,nc,atol=.005,rtol=.003):raise ValueError('%s out%d parity failed: cv=%g ncnn=%g'%(name,i,np.max(abs(ref-cv)),np.max(abs(ref-nc))))
                rows.append(dict(model=name,seed=seed,output=i,shape=list(ref.shape),adapted_max_error=float(np.max(abs(ref-cv))),ncnn_max_error=float(np.max(abs(ref-nc)))))
                if seed==17:references[name+'_out'+str(i)]=ref
            if seed==17:references[name+'_input']=nchw
        net.clear()
    np.savez(a.model/'export_reference.npz',**references)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    models={name:dict(param=name+'_nchw.ncnn.param',weights=name+'_nchw.ncnn.bin',param_sha256=sha(a.model/(name+'_nchw.ncnn.param')),weights_sha256=sha(a.model/(name+'_nchw.ncnn.bin')),output_names=['out'+str(i) for i in range(count)],output_layouts=[2,2] if name=='palm' else [1]*4) for name,count in (('palm',2),('hand',4))}
    (a.model/'gesture.json').write_text(json.dumps(dict(architecture='mediapipe_hands',models=models,source_commit='48b10fa505aeb51fbd127fc60a5c660040e823b7',license='Apache-2.0',pc_parity=True,field_qualified=False),indent=2)+'\n')
    report=dict(pass_=True,outputs=rows,scope='PC numerical conversion only; not board performance or recognition acceptance')
    (a.model/'parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
