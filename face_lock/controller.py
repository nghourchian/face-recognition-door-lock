import math


class LockController:
    """Lock timing uses a monotonic clock."""

    def __init__(self, threshold=0.45):
        self.threshold = threshold
        self.state = "LOCKED"
        self.deadline = None
        self.match_since = None
        self.hits = 0
        self.empty_since = None
        self.last_observation = None
        self.last_capture = None
        self.faulted = False
        self.unlocks = 0

    def reset_match(self):
        self.match_since, self.hits = None, 0

    def fault(self):
        self.state = "WAIT_CLEAR"
        self.deadline = None
        self.empty_since = None
        self.reset_match()
        if not self.faulted:
            self.faulted = True
            return [("CAMERA_FAULT_LOCKED", None)]
        return []

    def tick(self, now):
        events = []
        if self.state == "UNLOCKED" and now >= self.deadline:
            self.state = "WAIT_CLEAR"
            self.deadline = None
            self.empty_since = None
            events.append(("AUTO_RELOCK", None))
        if self.last_observation is not None and now - self.last_observation > 1.0:
            events += self.fault()
        return events

    def observe(self, now, captured, count, score, usable):
        events = self.tick(now)
        # Never confirm identity from stale or duplicate observations.
        if now - captured > 1.0 or captured > now:
            return events + self.fault()
        if self.last_capture is not None and captured <= self.last_capture:
            return events
        if self.last_observation is not None and now - self.last_observation > 0.75:
            self.reset_match()
            self.empty_since = None
        self.last_observation, self.last_capture = now, captured
        if self.faulted:
            self.faulted = False
            events.append(("CAMERA_RECOVERED_WAIT_CLEAR", None))
        if self.state == "UNLOCKED":
            return events
        if self.state == "WAIT_CLEAR":
            if count == 0:
                if self.empty_since is None:
                    self.empty_since = now
                if now - self.empty_since >= 1.0:
                    self.state = "LOCKED"
                    self.empty_since = None
                    events.append(("READY", None))
            else:
                self.empty_since = None
            return events
        matched = (
            count == 1
            and usable
            and score is not None
            and math.isfinite(score)
            and score >= self.threshold
        )
        if not matched:
            self.reset_match()
            return events
        if self.match_since is None:
            self.match_since = now
        self.hits += 1
        elapsed = now - self.match_since
        if self.hits >= 3 and elapsed >= 0.5:
            self.state = "UNLOCKED"
            self.deadline = now + 3.0
            self.unlocks += 1
            self.reset_match()
            events.append(("UNLOCK", elapsed * 1000))
        return events


class PassiveGate:
    def __init__(self, cutoff=0.8):
        self.cutoff = cutoff
        self.start, self.last, self.count = None, None, 0

    def update(self, now, identity_ok, scores):
        good = (
            identity_ok
            and scores is not None
            and len(scores) == 2
            and all(math.isfinite(v) and self.cutoff <= v <= 1 for v in scores)
        )
        if not good or (
            self.last is not None and (now <= self.last or now - self.last > 0.75)
        ):
            self.start, self.last, self.count = None, None, 0
            return False
        if self.start is None:
            self.start = now
        self.last = now
        self.count += 1
        return self.count >= 5 and now - self.start >= 1.0
