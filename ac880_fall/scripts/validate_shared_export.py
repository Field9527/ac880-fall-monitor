"""PC Torch/ONNX/NCNN parity before a bounded ARM trial; no hardware access."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ml'))
import cache_env
import argparse
import hashlib
import json
import cv2
import numpy as np
import torch
from shared_semantic_model import SharedSemantic
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sw'))
from semantic_features import OUTPUT_NAMES,RegionVerifier,region_descriptor
from fall_lite import letterbox


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--ncnn-python',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.ncnn_python));import ncnn
    torch.set_num_threads(2);cv2.setNumThreads(1)
    checkpoint=torch.load(a.model/'object_head.pt',map_location='cpu')
    model=SharedSemantic(a.checkpoint,object_neck=checkpoint.get('object_neck',False)).eval();model.objects.load_state_dict(checkpoint['objects'])
    sample=np.random.RandomState(17).normal(size=(1,3,192,320)).astype(np.float32)
    with torch.no_grad():expected=[v.numpy() for v in model(torch.from_numpy(sample))]
    onnx=cv2.dnn.readNetFromONNX(str(a.model/'shared_semantic.onnx'));onnx.setInput(sample)
    converted=onnx.forward(list(OUTPUT_NAMES));net=ncnn.Net();net.opt.num_threads=1
    net.opt.use_vulkan_compute=False;net.opt.use_fp16_storage=False;net.opt.use_fp16_packed=False;net.opt.use_fp16_arithmetic=False
    if net.load_param(str(a.model/'shared_semantic.ncnn.param')) or net.load_model(str(a.model/'shared_semantic.ncnn.bin')):raise RuntimeError('NCNN shared model load failed')
    actual=[]
    with net.create_extractor() as ex:
        if ex.input('in0',ncnn.Mat(sample[0])):raise RuntimeError('NCNN input failed')
        for index in range(9):
            code,value=ex.extract('out'+str(index))
            if code:raise RuntimeError('NCNN named output failed')
            actual.append(np.asarray(value).copy()[None])
    rows=[]
    for name,original,cv,nc in zip(OUTPUT_NAMES,expected,converted,actual):
        if cv.shape!=original.shape or nc.shape!=original.shape:raise ValueError('Output shape changed: '+name)
        if not np.allclose(original,cv,atol=.001,rtol=.003) or not np.allclose(original,nc,atol=.001,rtol=.003):raise ValueError('Shared output parity failed: '+name)
        rows.append(dict(name=name,shape=list(original.shape),onnx_max_abs_error=float(np.max(np.abs(original-cv))),ncnn_max_abs_error=float(np.max(np.abs(original-nc)))))
    verifier=RegionVerifier(a.model/'region_verifier.npz');box=[30,40,290,170]
    p1=verifier.predict(expected[6:],box);p2=verifier.predict(actual[6:],box)
    delta=max(abs(p1[k]-p2[k]) for k in p1)
    if delta>.001:raise ValueError('Region probability changed after conversion')
    sha=lambda name:hashlib.sha256((a.model/name).read_bytes()).hexdigest()
    meta=json.loads((a.model/'shared_semantic.json').read_text());meta['ncnn']=dict(param='shared_semantic.ncnn.param',weights='shared_semantic.ncnn.bin',param_sha256=sha('shared_semantic.ncnn.param'),weights_sha256=sha('shared_semantic.ncnn.bin'),input_name='in0',output_names=['out'+str(i) for i in range(9)],weight_precision='fp32')
    (a.model/'shared_semantic.json').write_text(json.dumps(meta,indent=2)+'\n')
    report=dict(pass_=True,outputs=rows,verifier_max_abs_error=delta,scope='PC fixed tensor parity; not ARM timing or field accuracy')
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
