"""摔倒与手势独立处理最新帧，各最多一个在途任务。"""

from concurrent.futures import ThreadPoolExecutor
import os
from fall_lite import LatestFrameInference
from hand_region_review import review_hand_region


class ConcurrentGestureInference(LatestFrameInference):
    def __init__(self, detector, timeout_s=3.0, partition_cpus=False):
        self.owner_affinity = None
        cpus = []
        if partition_cpus and hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
            cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) >= 2:
            self.owner_affinity = set(cpus)
            owner_cpu, hand_cpu = cpus[:2]
            # Linux 的 pid=0 仅设置当前线程。
            os.sched_setaffinity(0, {owner_cpu})
            super().__init__(
                detector, timeout_s, worker_initializer=lambda: os.sched_setaffinity(0, {owner_cpu})
            )
            self.hand_pool = ThreadPoolExecutor(
                max_workers=1, initializer=lambda: os.sched_setaffinity(0, {hand_cpu})
            )
        else:
            super().__init__(detector, timeout_s)
            self.hand_pool = ThreadPoolExecutor(max_workers=1)
        self.hand_future = None
        self.hand_submitted_at = None
        self.hand_completed = 0
        self.hand_observation = {}

    def poll(self, frame, now):
        # 两个只读任务共享自有快照，避免持有 DMA 视图。
        # 任务忙碌时不复制新帧。
        owned = []

        def source():
            if not owned:
                owned.append(frame() if callable(frame) else frame)
            return owned[0]

        observation, fresh = super().poll(source, now)
        if self.hand_future is not None:
            if self.hand_future.done():
                self.hand_observation = self.hand_future.result()
                self.hand_future = None
                self.hand_completed += 1
            elif now - self.hand_submitted_at > self.timeout_s:
                raise TimeoutError("Hand inference exceeded %.1f seconds" % self.timeout_s)
        enabled = self.detector.gestures_enabled()
        if not enabled:
            self.hand_observation = {}
        interval = self.detector.hand_interval_s
        if (
            enabled
            and self.hand_future is None
            and (self.hand_submitted_at is None or now - self.hand_submitted_at >= interval)
        ):
            self.hand_submitted_at = now
            self.hand_future = self.hand_pool.submit(self.detector.process_hand, source(), now)
        result = dict(observation)
        hand = self.hand_observation
        options = self.detector.semantic_options.configuration
        if (
            enabled
            and hand.get("hand_semantic_generation") == options["generation"]
            and hand.get("gesture_generation") == options["gesture_generation"]
        ):
            result.update(hand)
        else:
            result.update(
                hand=None,
                hand_sampled_at=None,
                hand_sequence=0,
                gesture_generation=options["gesture_generation"],
            )
        if fresh:
            result = review_hand_region(result, now)
            self.observation = dict(result)
        # 手势任务完成不能刷新旧摔倒证据的时间。
        return result, fresh

    def close(self):
        self.hand_pool.shutdown(wait=True)
        super().close()
        if self.owner_affinity is not None:
            os.sched_setaffinity(0, self.owner_affinity)
