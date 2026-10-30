"""PC-side persistent incident records; never runs in the FPGA camera loop."""
from datetime import datetime,timezone
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid


class AlarmStore:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.database=self.folder/'events.sqlite3';self.lock=threading.Lock()
        self.active=None;self.first_seen=0.
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS incidents (id TEXT PRIMARY KEY, event_key TEXT UNIQUE, started TEXT, ended TEXT, context TEXT, score REAL, review TEXT, note TEXT, snapshot TEXT, model TEXT)')

    @contextmanager
    def _db(self):
        db=sqlite3.connect(str(self.database),timeout=3.)
        try:
            with db:yield db
        finally:db.close()

    def observe(self,state,jpeg=None,published_at=0.):
        detector=state['detector'];controls=state['controls']
        run=state.get('run_id')
        # Missing run identity or a disconnected board must never create a new alarm.
        if not run or not detector.get('running'):return False
        alarm=detector.get('state_name')=='ALARM'
        key='%s:%s'%(run,controls['alarm_count'])
        stamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self.lock,self._db() as db:
            if self.active and (not alarm or self.active[1]!=key):
                db.execute('UPDATE incidents SET ended=COALESCE(ended,?) WHERE id=?',(stamp,self.active[0]))
                self.active=None
            if alarm:
                if not self.active:
                    row=db.execute('SELECT id,snapshot FROM incidents WHERE event_key=?',(key,)).fetchone()
                    identity=row[0] if row else uuid.uuid4().hex
                    db.execute('INSERT OR IGNORE INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (identity,key,stamp,None,detector.get('input_source') if detector.get('input_source','').startswith('synthetic') else detector.get('test_context','unknown'),float(detector.get('pe',0)),
                         'unreviewed','',None,detector.get('model_sha256','')))
                    self.active=(identity,key);self.first_seen=time.monotonic()
                identity=self.active[0]
                row=db.execute('SELECT snapshot FROM incidents WHERE id=?',(identity,)).fetchone()
                if row[0] is None and jpeg and published_at>self.first_seen+.25:
                    name=identity+'.jpg';target=self.folder/name
                    tmp=target.with_suffix('.tmp');tmp.write_bytes(jpeg);tmp.replace(target)
                    db.execute('UPDATE incidents SET snapshot=? WHERE id=?',(name,identity))
                return db.execute('SELECT snapshot FROM incidents WHERE id=?',(identity,)).fetchone()[0] is None
        return False

    def list(self):
        with self.lock,self._db() as db:
            db.row_factory=sqlite3.Row
            return [dict(row) for row in db.execute('SELECT id,started,ended,context,score,review,note,snapshot,model FROM incidents ORDER BY started DESC,rowid DESC LIMIT 100')]

    def review(self,identity,value):
        self._identity(identity)
        if not isinstance(value,dict) or set(value)-{'review','note'}:raise ValueError('反馈格式无效')
        if value.get('review') not in ('unreviewed','confirmed','false_alarm'):raise ValueError('反馈状态无效')
        note=value.get('note','')
        if not isinstance(note,str) or len(note)>300:raise ValueError('备注最多300字')
        with self.lock,self._db() as db:
            changed=db.execute('UPDATE incidents SET review=?,note=? WHERE id=?',(value['review'],note,identity)).rowcount
            if not changed:raise ValueError('报警记录不存在')
        # Human feedback is an annotation for review, never an automatic training label.
        return dict(saved=True)

    def snapshot(self,identity):
        self._identity(identity)
        with self.lock,self._db() as db:
            row=db.execute('SELECT snapshot FROM incidents WHERE id=?',(identity,)).fetchone()
        return (self.folder/row[0]).read_bytes() if row and row[0] else None

    @staticmethod
    def _identity(identity):
        if not re.fullmatch('[0-9a-f]{32}',identity):raise ValueError('报警编号无效')
