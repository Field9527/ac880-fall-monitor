"""Persistent incident HTTP contract, auth, isolation and freshness."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sw'))
from alarm_store import AlarmStore
from home_web import HomeWeb


class Outputs:
    def snapshot(self):return dict(alarm_count=1)


class IncidentHTTPTests(unittest.TestCase):
    def test_http_persistence_and_review_auth(self):
        with tempfile.TemporaryDirectory(dir='E:/Zynq_cache/temp') as folder:
            archive=AlarmStore(folder)
            archive.observe(dict(run_id='test',detector=dict(running=True,state_name='ALARM',input_source='synthetic_output_test'),controls=dict(alarm_count=1)))
            web=HomeWeb(Outputs(),dict(host='127.0.0.1',port=0),archive=archive)
            client=http.client.HTTPConnection('127.0.0.1',web.server.server_port,timeout=3)
            def request(method,path,body=None,headers=None):
                client.request(method,path,body,headers or {});response=client.getresponse();data=response.read()
                return response.status,json.loads(data)
            try:
                status,items=request('GET','/api/incidents');self.assertEqual(status,200)
                item=items['items'][0];self.assertEqual(item['context'],'synthetic_output_test')
                url='/api/incidents/'+item['id']+'/review'
                status,_=request('POST',url,'{}',{'Content-Type':'application/json'})
                self.assertEqual(status,403);client.close()
                client=http.client.HTTPConnection('127.0.0.1',web.server.server_port,timeout=3)
                status,_=request('POST',url,json.dumps(dict(review='false_alarm',note='<script>')),{'Content-Type':'application/json','X-Home-Token':web.token})
                self.assertEqual(status,200)
                self.assertEqual(AlarmStore(folder).list()[0]['note'],'<script>')
                status,_=request('GET','/api/incidents/bad/snapshot.jpg');self.assertEqual(status,400)
                snapshot=web.snapshot();self.assertGreaterEqual(snapshot['status_age_s'],0)
            finally:client.close();web.close()


if __name__=='__main__':unittest.main()
