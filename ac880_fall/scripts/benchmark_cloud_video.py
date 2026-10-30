"""Measure received MJPEG frames and fresh detector status; input JSON on stdin."""
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import time
import urllib.error
import urllib.request


def benchmark(parameters):
    base=parameters['url'].rstrip('/')
    duration=float(parameters.get('seconds',20))
    readers=int(parameters.get('readers',1))
    headers={}
    if parameters.get('username'):
        encoded=base64.b64encode((parameters['username']+':'+parameters['password']).encode()).decode()
        headers['Authorization']='Basic '+encoded
    def opener():return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def stream():
        arrivals=[];sequences=[];sizes=[];ages=[];client=opener()
        request=urllib.request.Request(base+'/stream.mjpg',headers=headers)
        with client.open(request,timeout=12) as response:
            deadline=time.monotonic()+duration
            while time.monotonic()<deadline:
                while True:
                    line=response.readline(1024)
                    if not line:raise RuntimeError('Video stream ended')
                    if line.strip()==b'--homeframe':break
                fields={}
                while True:
                    line=response.readline(1024)
                    if line in (b'\r\n',b'\n'):break
                    if not line:raise RuntimeError('Truncated video header')
                    key,_,value=line.partition(b':');fields[key.lower()]=value.strip()
                length=int(fields[b'content-length'])
                if not 0<length<5000000:raise RuntimeError('Invalid video frame length')
                image=response.read(length)
                if len(image)!=length or image[:2]!=b'\xff\xd8':raise RuntimeError('Invalid JPEG frame')
                arrivals.append(time.monotonic());sequences.append(int(fields[b'x-sequence']));sizes.append(length)
                if b'x-capture-unix-ms' in fields:ages.append(time.time()-float(fields[b'x-capture-unix-ms'])/1000)
        intervals=[b-a for a,b in zip(arrivals,arrivals[1:])]
        elapsed=arrivals[-1]-arrivals[0] if len(arrivals)>1 else duration
        return dict(frames=len(arrivals),fps=(len(arrivals)-1)/elapsed if elapsed else 0.,
                    max_gap_s=max(intervals) if intervals else None,
                    mean_jpeg_bytes=sum(sizes)/max(1,len(sizes)),payload_mbps=sum(sizes)*8/max(.001,elapsed)/1e6,
                    first_sequence=sequences[0] if sequences else None,last_sequence=sequences[-1] if sequences else None,
                    frame_age_s=dict(mean=sum(ages)/len(ages),p95=sorted(ages)[int((len(ages)-1)*.95)],maximum=max(ages)) if ages else None)
    samples=[];client=opener()
    with ThreadPoolExecutor(max_workers=readers) as pool:
        futures=[pool.submit(stream) for _ in range(readers)]
        deadline=time.monotonic()+duration
        while time.monotonic()<deadline:
            request=urllib.request.Request(base+'/api/state',headers=headers)
            started=time.monotonic()
            try:
                with client.open(request,timeout=12) as response:value=json.load(response)
            except (OSError,ValueError,urllib.error.URLError) as exc:
                samples.append(dict(status_error=type(exc).__name__,status_code=getattr(exc,'code',None),
                    status_latency_s=round(time.monotonic()-started,3)))
                time.sleep(1)
                continue
            detector=value['detector']
            samples.append(dict(video=value['video'],cloud_video=value.get('cloud_video'),status_age_s=value.get('status_age_s'),
                status_latency_s=round(time.monotonic()-started,3),
                alarm_count=value.get('controls',{}).get('alarm_count'),
                detector={k:detector.get(k) for k in ('running','ready','state_name','display_fps','inference_fps','inference_ms','gesture_ms','gesture_fps','hand_status','evidence_age_s','coverage_limited','hand_age_s')}))
            time.sleep(1)
        results=[future.result() for future in futures]
    return dict(url=base,seconds=duration,readers=results,samples=samples,
                scope='HTTP readers measured here; not independent mobile-network acceptance')


if __name__=='__main__':
    import sys
    print(json.dumps(benchmark(json.load(sys.stdin))))
