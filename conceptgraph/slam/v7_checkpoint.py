"""Atomic frame-boundary checkpoints for the original, fresh online run."""
import atexit, copy, fcntl, hashlib, io, json, os, pickle, random, time
from pathlib import Path
import numpy as np
import torch


def geometry(kind, data):
    import open3d as o
    if kind == 'pcd':
        p=o.geometry.PointCloud()
        for name, value in data.items():setattr(p,name,o.utility.Vector3dVector(value))
        return p
    if kind == 'obb':
        p=o.geometry.OrientedBoundingBox(*data[:3]);p.color=data[3];return p
    p=o.geometry.AxisAlignedBoundingBox(*data[:2]);p.color=data[2];return p


class StatePickler(pickle.Pickler):
    def reducer_override(self, obj):
        import open3d as o
        if isinstance(obj,o.geometry.PointCloud):
            return geometry,('pcd',{k:np.asarray(getattr(obj,k)).copy() for k in ('points','colors','normals')})
        if isinstance(obj,o.geometry.OrientedBoundingBox):
            return geometry,('obb',(obj.center,obj.R,obj.extent,obj.color))
        if isinstance(obj,o.geometry.AxisAlignedBoundingBox):
            return geometry,('aabb',(obj.min_bound,obj.max_bound,obj.color))
        return NotImplemented


def atomic(path, data):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def attrs(obj, excluded=()):
    return {k:v for k,v in vars(obj).items() if k not in excluded}


def code_signature():
    root=Path(__file__).resolve().parents[1]
    files=list((root/'slam').glob('v7*.*'))+[root/'slam/rerun_realtime_mapping.py',root/'slam/association_gate.py',root/'utils/evidence.py']
    files+=list((root/'slam/prompts/v7').glob('*'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


class Checkpoints:
    def __init__(self, root, cfg):
        from omegaconf import OmegaConf
        self.root=Path(root).resolve();self.directory=self.root/'checkpoints';self.directory.mkdir(exist_ok=True)
        self.lock=(self.directory/'run.lock').open('a+')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close();raise RuntimeError('This run already has an active checkpoint writer')
        self.cfg=OmegaConf.to_container(cfg,resolve=True) if OmegaConf.is_config(cfg) else copy.deepcopy(cfg)
        for k in ('device','v7_resume','v7_checkpoint'):self.cfg.pop(k,None)
        self.code=code_signature();self.loaded=None

    def files(self):
        return {str(p.relative_to(self.root)):p for p in self.root.rglob('*')
                if p.is_file() and not p.is_relative_to(self.directory)}

    def save(self, next_frame, objects, map_edges, evidence, gate, tracker, local):
        started=time.perf_counter()
        for f in evidence._files.values():f.flush();os.fsync(f.fileno())
        runtime=gate.vlm_runtime
        if runtime.forced_groups:raise RuntimeError('Cannot checkpoint pending map mutations')
        disk={}
        for name,p in self.files().items():
            # Immutable per-event evidence stays in place; mutable indexes are restored exactly.
            mutable=p.suffix in {'.json','.html'} and '/events/' not in '/'+name and '/cases/' not in '/'+name
            disk[name]={'size':p.stat().st_size,'content':p.read_bytes() if mutable else None}
        state=dict(schema=1,root=str(self.root),cfg=self.cfg,code=self.code,next_frame=next_frame,
            fallback=runtime.fallback,objects=objects,map_edges=map_edges,
            evidence=attrs(evidence,('_files',)),file_keys=list(evidence._files),
            gate=attrs(gate,('rerun','_human_input','vlm_runtime','_instance_merge_gate')),
            runtime=attrs(runtime,('owner','evidence','staged_bindings')),
            live_evidence=attrs(runtime.evidence,('runtime',)),
            merge=attrs(gate._instance_merge_gate,('owner','runtime')) if gate._instance_merge_gate else None,
            tracker=attrs(tracker),local=local,disk=disk,dirs=[str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_dir() and not p.is_relative_to(self.directory)],
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        buf=io.BytesIO();StatePickler(buf,protocol=5).dump(state);payload=buf.getvalue()
        filename=f'frame_{next_frame:06d}_{time.time_ns()}.pkl';atomic(self.directory/filename,payload)
        pointer=dict(schema=1,file=filename,sha256=hashlib.sha256(payload).hexdigest(),next_frame=next_frame,completed_frame=next_frame-1,status='ready')
        atomic(self.directory/'latest.json',json.dumps(pointer,indent=2).encode())
        # Keep latest and previous complete generations, never replace a valid pointer with a partial file.
        old=sorted(self.directory.glob('frame_*.pkl'))
        for p in old[:-2]:p.unlink()
        print(f'[checkpoint] latest next_frame={next_frame} saved in {time.perf_counter()-started:.2f}s',flush=True)

    def load(self):
        pointer=json.loads((self.directory/'latest.json').read_text())
        if pointer.get('status')=='completed':raise ValueError('This run is already completed')
        p=self.directory/pointer['file']
        if p.parent!=self.directory or p.is_symlink():raise ValueError('Invalid checkpoint path')
        blob=p.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=pointer['sha256']:raise ValueError('Checkpoint checksum mismatch')
        state=pickle.loads(blob) # Only local checkpoints produced by this trusted run.
        if state['root']!=str(self.root) or state['cfg']!=self.cfg:raise ValueError('Resume configuration differs from original run')
        if state['code']!=self.code:raise ValueError('Resume code/prompts differ from checkpoint; use the original version')
        if state['fallback']!=os.environ.get('V7_FALLBACK','auto'):raise ValueError('Resume human/auto mode differs')
        # Validate everything before rolling back any files.
        for name,item in state['disk'].items():
            p=self.root/name
            if not p.resolve().is_relative_to(self.root) or p.is_symlink():raise ValueError('Unsafe evidence path')
            if not p.is_file() or p.stat().st_size<item['size'] and item['content'] is None:raise ValueError('Checkpoint evidence missing/truncated: '+name)
        self.loaded=state;return state

    def rollback(self):
        state=self.loaded;archive=self.directory/'interrupted'/str(time.time_ns())
        for name,p in self.files().items():
            item=state['disk'].get(name)
            if item is None:
                dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);os.replace(p,dest)
            elif item['content'] is not None:atomic(p,item['content'])
            elif p.suffix=='.jsonl' and p.stat().st_size>item['size']:
                dest=archive/(name+'.tail');dest.parent.mkdir(parents=True,exist_ok=True)
                with p.open('r+b') as f:f.seek(item['size']);dest.write_bytes(f.read());f.truncate(item['size'])
        for p in sorted(self.root.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if p.is_dir() and not p.is_relative_to(self.directory) and str(p.relative_to(self.root)) not in state['dirs']:
                try:p.rmdir()
                except OSError:pass
        for name in state['dirs']:(self.root/name).mkdir(parents=True,exist_ok=True)
        print('[checkpoint] rolled back unfinished frame; preserved partial files under',archive,flush=True)

    def restore(self, rerun, tracker):
        from conceptgraph.utils.evidence import EvidenceRecorder
        from conceptgraph.slam.association_gate import BlockingAssociationGate
        from conceptgraph.slam.v7_runtime import V7Runtime
        from conceptgraph.slam.v7_evidence import LiveEvidence
        from conceptgraph.slam.v7_merge import V7MergeGate
        state=self.loaded
        evidence=EvidenceRecorder.__new__(EvidenceRecorder);evidence.__dict__.update(state['evidence'])
        evidence._files={k:(evidence.evidence_dir/k).open('a',encoding='utf-8') for k in state['file_keys']}
        atexit.register(evidence._finalize_abandoned_run)
        gate=BlockingAssociationGate.__new__(BlockingAssociationGate);gate.__dict__.update(state['gate']);gate.rerun=rerun;gate._human_input=input
        runtime=V7Runtime.__new__(V7Runtime);runtime.__dict__.update(state['runtime']);runtime.owner=gate;runtime.staged_bindings={};gate.vlm_runtime=runtime
        live=LiveEvidence.__new__(LiveEvidence);live.__dict__.update(state['live_evidence']);live.runtime=runtime;runtime.evidence=live
        gate._instance_merge_gate=None
        if state['merge'] is not None:
            merge=V7MergeGate.__new__(V7MergeGate);merge.__dict__.update(state['merge']);merge.owner=gate;merge.runtime=runtime;gate._instance_merge_gate=merge
        tracker.__dict__.update(state['tracker'])
        py,npstate,cpu,gpu=state['rng'];random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if gpu:torch.cuda.set_rng_state_all(gpu)
        runtime.publish()
        return state['objects'],state['map_edges'],evidence,gate,state['local'],state['next_frame']

    def completed(self):
        p=self.directory/'latest.json';d=json.loads(p.read_text());d['status']='completed';atomic(p,json.dumps(d,indent=2).encode())
