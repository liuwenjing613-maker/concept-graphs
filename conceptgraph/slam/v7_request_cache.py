"""Run-local, durable request/evidence cache with single-flight inference.

A claimed but incomplete request is DEFER on restart, never silently reissued.
Cache hits are rebound to the current frozen snapshot; original timing stays linked.
"""
import copy
import hashlib
import json
import threading
import time
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class EvidenceRequestCache:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.guard=threading.Lock();self.locks={}

    def execute(self,key,invoke):
        with self.guard:
            lock=self.locks.setdefault(key,threading.Lock())
        with lock:
            path=self.root/(key+'.json')
            try:
                # Cross-process claim also prevents a restart retrying an unknown request.
                with path.open('x') as f:
                    json.dump(dict(state='IN_FLIGHT',claimed_at=time.time()),f)
            except FileExistsError:
                try:record=json.loads(path.read_text())
                except (ValueError,OSError):record={}
                if record.get('state')=='COMPLETE':
                    return copy.deepcopy(record['result']),True
                return dict(value=None,error='CACHE_INCOMPLETE_REQUEST_DEFER',
                            format_mode=None,attempts=[],seconds=0),True
            result=invoke()
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(state='COMPLETE',result=result),ensure_ascii=False))
            temporary.replace(path)
            return result,False
