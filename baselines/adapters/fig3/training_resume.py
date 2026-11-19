"""Atomic epoch-boundary recovery; no change to model, loss or selection rules."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch


class TrainingResume:
    def __init__(self, out, interval=5):
        self.out = Path(out)
        self.path = self.out / 'training_resume.pt'
        self.interval = interval
        self.stop = False
        self.key = hashlib.sha256((self.out / 'RUN_LOCK.json').read_bytes()).hexdigest()
        self.state = None
        if self.path.exists():
            self.state = torch.load(self.path, map_location='cpu', weights_only=False)
            if self.state['key'] != self.key:
                raise RuntimeError('Resume configuration/source mismatch; preserve checkpoint for audit')
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

    def request_stop(self, *_):
        self.stop = True

    def restore(self, candidate, model, optimizer, rng, scheduler=None):
        s = self.state
        if s is None or s['candidate'] != candidate:
            return None
        model.load_state_dict(s['model'])
        optimizer.load_state_dict(s['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(s['scheduler'])
        rng.set_state(s['order_rng'])
        random.setstate(s['python_rng'])
        np.random.set_state(s['numpy_rng'])
        torch.set_rng_state(s['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s['cuda_rng'])
        return s

    def skip(self, candidate):
        return self.state is not None and candidate < self.state['candidate']

    def selections(self):
        return list(self.state['selections']) if self.state is not None else []

    def save(self, candidate, epoch, model, optimizer, rng, values, selections,
             scheduler=None, force=False):
        if not (force or self.stop or epoch % self.interval == 0):
            return
        s = dict(key=self.key, candidate=candidate, epoch=epoch,
                 model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict() if scheduler else None,
                 order_rng=rng.get_state(), python_rng=random.getstate(),
                 numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 values=values, selections=list(selections), saved_unix=time.time())
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            torch.save(s, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.path)
        info = dict(candidate=candidate, epoch=epoch, saved_unix=s['saved_unix'],
                    complete_training_state=True, stop_requested=self.stop)
        (self.out / 'training_resume.json').write_text(json.dumps(info, indent=2))
        if self.stop:
            raise SystemExit(75)
