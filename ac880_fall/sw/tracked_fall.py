"""各目标独立执行报警判决，仅新鲜证据推进状态。"""

from collections import deque
import math
import time
from fall_detect import FallDetector, FallState


def similarity(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    area = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return area / max(1, aw * ah + bw * bh - area)


class TrackedFallDetector:
    def __init__(self, cfg, logger=None, max_tracks=16, max_age_s=3.0, ttl_s=4.0):
        self.cfg, self.logger = cfg, logger
        self.max_tracks = max_tracks
        self.max_age_s = max_age_s
        self.ttl_s = ttl_s
        self.tracks = {}
        self.next_track = 1
        self.event_id = 0
        self.last_evidence = -1
        self.events = deque(maxlen=1000)
        self.last_roi = self.last_bbox = None
        self.pe = self.peak_p = 0.0
        self.on_event = None

    def _event(self, track_id, event):
        track = self.tracks[track_id]
        key = event["event_id"]
        if event["event"] == "candidate_start" or key not in track["events"]:
            self.event_id += 1
            track["events"][key] = self.event_id
        value = dict(
            event,
            event_id=track["events"][key],
            track_id=track_id,
            test_context=track["evidence"].get("test_context", "live_scene"),
            region_evidence=track["evidence"].get("region_probabilities"),
        )
        self.last_roi = track["logic"].last_roi
        self.events.append(value)
        if self.on_event:
            self.on_event(value)

    def _result(self, status="fresh", limited=False):
        tracks = list(self.tracks.values())
        best = max(
            tracks, key=lambda t: (int(t["result"]["state"]), t["result"]["pe"]), default=None
        )
        result = (
            dict(best["result"])
            if best
            else dict(
                state=FallState.IDLE,
                state_name="IDLE",
                pe=0.0,
                peak_p=0.0,
                motion=False,
                score=0.0,
                bbox=None,
            )
        )
        self.last_bbox = result["bbox"]
        self.pe = result["pe"]
        self.peak_p = result["peak_p"]
        result.update(
            evidence_status=status,
            tracking_limited=limited,
            tracks=[dict(track_id=k, **v["result"]) for k, v in self.tracks.items()],
        )
        return result

    def step_many(self, detections, evidence_id, sampled_at, now=None, roi=None, roi_bbox=None):
        now = time.monotonic() if now is None else now
        if evidence_id <= self.last_evidence:
            return self._result("duplicate")
        if not all(math.isfinite(v) for v in (sampled_at, now)):
            raise ValueError("Invalid evidence timestamp")
        self.last_evidence = evidence_id
        for key in list(self.tracks):
            if now - self.tracks[key]["seen"] > self.ttl_s:
                self.tracks[key]["logic"].reset()
                del self.tracks[key]
        if now < sampled_at or now - sampled_at > self.max_age_s:
            return self._result("stale")
        for d in detections:
            b = d["bbox"]
            p = d["score"]
            if (
                len(b) != 4
                or not all(math.isfinite(v) for v in (*b, p))
                or min(b[2:]) <= 0
                or not 0 <= p <= 1
            ):
                raise ValueError("Invalid target evidence")
        available = set(self.tracks)
        limited = False
        updated = set()
        for d in sorted(detections, key=lambda v: v["score"], reverse=True):
            chosen = max(
                available, key=lambda k: similarity(self.tracks[k]["box"], d["bbox"]), default=None
            )
            if chosen is None or similarity(self.tracks[chosen]["box"], d["bbox"]) < 0.2:
                if len(self.tracks) >= self.max_tracks:
                    limited = True
                    continue
                chosen = self.next_track
                self.next_track += 1
                logic = FallDetector(self.cfg, self.logger)
                self.tracks[chosen] = dict(
                    logic=logic, box=d["bbox"], seen=now, evidence=dict(d), events={}, result=None
                )
                logic.on_event = lambda event, key=chosen: self._event(key, event)
            else:
                available.remove(chosen)
            track = self.tracks[chosen]
            track.update(box=d["bbox"], seen=now, evidence=dict(d))
            updated.add(chosen)
            track["result"] = track["logic"].step(
                d["score"] >= self.cfg["score_th"],
                d["score"],
                d["bbox"],
                roi if d["bbox"] == roi_bbox else None,
            )
            d["track_id"] = chosen
        for key, track in self.tracks.items():
            if key not in updated:
                track["result"] = track["logic"].step(False, 0.0)
        return self._result(limited=limited)

    def reset(self):
        for track in self.tracks.values():
            track["logic"].reset()
        self.tracks.clear()
        self.last_roi = self.last_bbox = None
        self.pe = self.peak_p = 0.0
