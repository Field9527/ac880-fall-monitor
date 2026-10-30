"""Measure three actual HTTP video readers plus detector freshness, not UI FPS alone."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import time
import requests
from PIL import Image


def session():
    s=requests.Session();s.trust_env=False;return s


def read_stream(base,seconds):
    times=[];size=None;deadline=time.monotonic()+seconds;data=b''
    with session() as s,s.get(base+'/stream.mjpg',stream=True,timeout=6) as r:
        r.raise_for_status()
        for block in r.iter_content(65536):
            data+=block
            while True:
                start=data.find(b'\xff\xd8');end=data.find(b'\xff\xd9',start+2)
                if start<0 or end<0:break
                image=data[start:end+2];data=data[end+2:];times.append(time.monotonic())
                if size is None:size=list(Image.open(BytesIO(image)).size)
            if time.monotonic()>=deadline:break
            if len(data)>4*1024*1024:raise ValueError('Malformed or excessive video frame')
    intervals=[b-a for a,b in zip(times,times[1:])]
    return dict(frames=len(times),resolution=size,fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0,
                largest_gap_s=max(intervals) if intervals else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8089');p.add_argument('--seconds',type=int,default=30)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();states=[]
    with ThreadPoolExecutor(max_workers=3) as pool,session() as s:
        futures=[pool.submit(read_stream,a.base,a.seconds) for _ in range(3)]
        deadline=time.monotonic()+a.seconds
        while time.monotonic()<deadline:
            r=s.get(a.base+'/api/state',timeout=5);r.raise_for_status();d=r.json()
            states.append(dict(detector=d['detector'],video=d['video'],environment=d.get('environment'),viewers=d['video']['viewers']))
            time.sleep(1)
        readers=[f.result() for f in futures]
    report=dict(readers=readers,states=states,scope='Three PC HTTP clients through the LAN gateway; not three independent WiFi devices')
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(readers,indent=2))


if __name__=='__main__':main()
