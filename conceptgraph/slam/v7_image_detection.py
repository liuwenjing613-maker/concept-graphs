"""Versioned full-resolution YOLO cache; SAM and downstream coordinates stay unchanged."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def contract(imgsz,classes,weights_root,image_hw):
    return dict(schema='v7-image-detection-v1',yolo_imgsz=int(imgsz),yolo_rect=True,
                yolo_conf=0.1,sam_imgsz=1024,image_hw=list(image_hw),
                classes=list(classes),ultralytics=version('ultralytics'),
                weights={name:digest(Path(weights_root)/name) for name in ['yolov8l-world.pt','sam_l.pt']},
                clip='ViT-H-14/laion2b_s32b_b79k')


class DetectionCache:
    def __init__(self,root,expected,sources,create):
        self.root=Path(root);self.path=self.root/'image_detection_manifest.json'
        self.expected=expected
        self.sources={Path(p).stem:str(Path(p).resolve()) for p in sources}
        if len(self.sources)!=len(sources):raise ValueError('duplicate detection source names')
        if create:
            if self.path.exists():raise FileExistsError('refusing to overwrite detector cache')
            self.data=dict(contract=expected,status='building',frames={})
            save(self.path,self.data)
        else:
            if not self.path.is_file():raise ValueError('unversioned/640 detection cache refused; use a new cache name')
            self.data=json.loads(self.path.read_text())
            if self.data.get('contract')!=expected:raise ValueError('detection cache configuration/weights/classes mismatch')
            missing=[stem for stem in self.sources if stem not in self.data.get('frames',{})]
            if missing:raise ValueError(f'incomplete detection cache: {len(missing)} requested frames missing')
            for stem in self.sources:
                folder=self.root/'detections'/stem
                for filename in ['xyxy.npz','mask.npz','class_id.npz','confidence.npz','image_feats.npz','v7_frame.json']:
                    if not (folder/filename).is_file():raise ValueError(f'incomplete detection cache file: {folder/filename}')

    def check_source(self,path):
        path=Path(path);row=self.data['frames'][path.stem]
        if str(path.resolve())!=row['source'] or digest(path)!=row['rgb_sha256']:
            raise ValueError('cached detection RGB source changed')
        return row

    def record(self,path,input_hw,original_hw):
        path=Path(path)
        if not input_hw:raise ValueError('YOLO input tensor shape was not recorded')
        row=dict(source=str(path.resolve()),rgb_sha256=digest(path),
                 requested_imgsz=self.expected['yolo_imgsz'],actual_yolo_input_hw=list(input_hw),
                 original_hw=list(original_hw))
        save(self.root/'detections'/path.stem/'v7_frame.json',row)
        self.data['frames'][path.stem]=row
        if set(self.sources)<=set(self.data['frames']):self.data['status']='complete'
        save(self.path,self.data)
        return row


def validate_coordinates(gobs,image_hw,classes):
    import numpy as np
    boxes=np.asarray(gobs['xyxy']);masks=np.asarray(gobs['mask']);ids=np.asarray(gobs['class_id'])
    h,w=map(int,image_hw)
    if boxes.shape!=(len(ids),4) or masks.shape!=(len(ids),h,w):
        raise ValueError('box/mask/RGB coordinate shape mismatch')
    if not np.isfinite(boxes).all() or (len(boxes) and
        ((boxes[:,[0,2]] < -0.1).any() or (boxes[:,[0,2]] > w+0.1).any() or
         (boxes[:,[1,3]] < -0.1).any() or (boxes[:,[1,3]] > h+0.1).any())):
        raise ValueError('detection boxes outside original image coordinates')
    if list(gobs['classes'])!=list(classes) or (len(ids) and ((ids<0).any() or (ids>=len(classes)).any())):
        raise ValueError('cached detection class ordering mismatch')
