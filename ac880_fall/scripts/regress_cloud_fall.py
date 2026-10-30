"""Compare ROI head versions on fixed and historical video regressions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ml'))
import cache_env
import argparse
import json
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sw'))
from shared_detector import SharedFallDetector
from tracked_fall import TrackedFallDetector
from semantic_decode import matched_count


def main():
    p=argparse.ArgumentParser()
    for name in ('before','after','cases','previews','videos','config','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1)
    cfg=json.loads(a.config.read_text(encoding='utf8'))['decision']
    models={key:SharedFallDetector(dict(metadata=str(folder/'shared_semantic.json'),engine='opencv_shared',allow_experimental=True,max_verify=8)) for key,folder in (('before',a.before),('after',a.after))}
    cases=json.loads((a.cases/'cases.json').read_text(encoding='utf8'))['items']
    fixed={};detail={}
    for key,det in models.items():
        matched=0;negative={};rows=[]
        for item in cases:
            # The existing probe writer saves BGR canvases.
            rgb=np.load(a.cases/item['file'],allow_pickle=False)[:,:,::-1].copy()
            detections=det.predict_rgb(rgb)
            if item['kind']=='fallen_person_regression':
                boxes=[dict(box=[d['bbox'][0],d['bbox'][1],d['bbox'][0]+d['bbox'][2],d['bbox'][1]+d['bbox'][3]],score=d['score']) for d in detections]
                matched+=matched_count(item['boxes'],boxes)
            elif item['kind']!='synthetic_rotated_normal_source':negative[str(item['id'])]=len(detections)
            rows.append(dict(id=item['id'],kind=item['kind'],detections=detections))
        for path in sorted(a.previews.glob('preview_*.jpg')):
            rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            detections=det.predict_rgb(rgb);negative[path.name]=len(detections)
            rows.append(dict(id=path.name,kind='visually_reviewed_no_fall_historical_preview',detections=detections))
        fixed[key]=dict(matched_fall_boxes=matched,negative_images=len(negative),false_positive_images=sum(v>0 for v in negative.values()),false_positive_boxes=sum(negative.values()),fp_cases={k:v for k,v in negative.items() if v})
        detail[key]=rows
    videos=[]
    for path in sorted(a.videos.glob('*-cam0.mp4')):
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);index=0;observations=[]
        while True:
            valid,bgr=cap.read()
            if not valid:break
            current=index/fps;index+=1
            if (index-1)%15:continue
            assert bgr.shape[:2]==(240,640),'Expected author depth-left RGB-right layout'
            rgb=bgr[:,320:,::-1].copy()
            observations.append(dict(video_s=current,detections={key:det.predict_rgb(rgb) for key,det in models.items()}))
        cap.release();metrics={}
        for key in models:
            metrics[key]={}
            for stride in (1,4):
                tracker=TrackedFallDetector(cfg);events=[];tracker.on_event=events.append;states=[]
                for seq,row in enumerate(observations[::stride]):
                    stamp=1000+row['video_s']
                    with patch('fall_detect.time.monotonic',return_value=stamp):
                        state=tracker.step_many([dict(d) for d in row['detections'][key]],seq,stamp,stamp)
                    states.append(state['state_name'])
                metrics[key][str(2/stride)]=dict(alarm=any(v['event']=='alarm' for v in events),candidate=any(v!='IDLE' for v in states),alarm_events=sum(v['event']=='alarm' for v in events),detected_frames=sum(bool(r['detections'][key]) for r in observations[::stride]))
        row=dict(video=path.name,sampled_frames=len(observations),metrics=metrics)
        videos.append(row);print('VIDEO',json.dumps(row),flush=True)
    checks=dict(fixed_positive_preserved=fixed['after']['matched_fall_boxes']>=fixed['before']['matched_fall_boxes'],fixed_negative_not_worse=fixed['after']['false_positive_images']<=fixed['before']['false_positive_images'])
    for rate in ('2.0','0.5'):
        for category in ('fall','adl'):
            counts={key:sum(v['metrics'][key][rate]['alarm'] for v in videos if v['video'].startswith(category)) for key in models}
            checks[category+'_'+rate+'_not_worse']=(counts['after']>=counts['before'] if category=='fall' else counts['after']<=counts['before'])
    report=dict(fixed=fixed,videos=videos,checks=checks,qualified=all(checks.values()),fixed_details=detail,scope='Historical/regression only; repeated URFall videos are not frame-aligned truth or fresh field acceptance; synthetic rotated normals excluded from negatives')
    a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('REGRESSION',json.dumps(dict(fixed=fixed,checks=checks,qualified=report['qualified'])),flush=True)


if __name__=='__main__':main()
