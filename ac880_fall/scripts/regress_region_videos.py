"""Repeated URFall RGB-half regression; no frame-aligned truth or new field claim."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ml'))
import cache_env
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sw'))
from fall_lite import FallLiteDetector
from shared_detector import SharedFallDetector
from tracked_fall import TrackedFallDetector
from fall_detect import FallDetector,FallState


def main():
    p=argparse.ArgumentParser();p.add_argument('--videos',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
    old=FallLiteDetector(dict(model=str(a.baseline/'fall_lite.onnx'),metadata=str(a.baseline/'fall_lite.json'),engine='opencv',score_threshold=.7))
    new=SharedFallDetector(dict(metadata=str(a.model/'shared_semantic.json'),engine='opencv_shared',allow_experimental=True))
    cfg=json.loads(a.config.read_text())['decision'];reports=[]
    for path in sorted(a.videos.glob('*-cam0.mp4')):
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);index=0;rows=[];previous=FallDetector(cfg);tracker=TrackedFallDetector(cfg)
        while True:
            ok,bgr=cap.read()
            if not ok:break
            t=index/fps;index+=1
            if (index-1)%15:continue
            if bgr.shape[:2]!=(240,640):raise ValueError('Expected author depth-left, RGB-right video')
            rgb=bgr[:,320:,::-1].copy();baseline=old.predict_rgb(rgb);current=new.predict_rgb(rgb);best=max(baseline,key=lambda v:v['score'],default=None)
            with patch('fall_detect.time.monotonic',return_value=1000+t):
                before=previous.step(bool(best),best['score'] if best else 0.,best['bbox'] if best else None)
                after=tracker.step_many(current,index,1000+t,1000+t)
            rows.append(dict(video_s=t,baseline_boxes=len(baseline),kept_boxes=len(current),reviewed=len(new.last_semantic['reviewed_candidates']),baseline_state=before['state_name'],new_state=after['state_name']))
        cap.release();report=dict(video=path.name,sampled_frames=len(rows),sample_fps=2.,
            baseline_candidate_frames=sum(r['baseline_boxes']>0 for r in rows),new_candidate_frames=sum(r['kept_boxes']>0 for r in rows),
            baseline_alarm_frames=sum(r['baseline_state']=='ALARM' for r in rows),new_alarm_frames=sum(r['new_state']=='ALARM' for r in rows),rows=rows)
        reports.append(report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    new.close();a.out.write_text(json.dumps(dict(reports=reports,scope='Repeated six-clip RGB-half regression; 2Hz software replay; no aligned frame truth, native latency or new field acceptance'),indent=2)+'\n')


if __name__=='__main__':main()
