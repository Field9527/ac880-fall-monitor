import tempfile
import time
import unittest
from pathlib import Path
from alarm_store import AlarmStore


class AlarmTests(unittest.TestCase):
    def test_dedup_restart_review_snapshot_and_episode_end(self):
        with tempfile.TemporaryDirectory(dir='E:/Zynq_cache/temp') as root:
            archive=AlarmStore(root)
            state=dict(run_id='run1',detector=dict(running=True,state_name='ALARM',pe=.9,test_context='phone_screen_replay'),controls=dict(alarm_count=1))
            self.assertTrue(archive.observe(state));archive.observe(state)
            self.assertEqual(len(archive.list()),1)
            archive.first_seen-=1;archive.observe(state,b'jpeg',time.monotonic())
            item=archive.list()[0];self.assertEqual(archive.snapshot(item['id']),b'jpeg')
            archive.review(item['id'],dict(review='false_alarm',note='black clothing'))
            archive=AlarmStore(root);archive.observe(state)
            self.assertEqual(len(archive.list()),1);self.assertEqual(archive.list()[0]['review'],'false_alarm')
            state['detector']['state_name']='IDLE';archive.observe(state)
            self.assertIsNotNone(archive.list()[0]['ended'])
            with self.assertRaises(ValueError):archive.snapshot('../secrets')
            with self.assertRaises(ValueError):archive.review(item['id'],dict(review='not_an_alarm',note='x'))


if __name__=='__main__':unittest.main()
