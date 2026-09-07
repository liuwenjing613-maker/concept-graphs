"""Evidence cards for staged observation audit and pairwise identity. No inference."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from conceptgraph.slam.history_projection import project_points,sha
BG=(19,27,38); PANEL=(27,38,52); TEXT=(237,242,248); MUTED=(172,190,208)
PURPLE=(219,92,226); CYAN=(42,212,237)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)
def label(im,xy,s,size=23,color=TEXT):
    ImageDraw.Draw(im).text(xy,s,font=font(size),fill=color)
def crop_box(mask,pad=.18,minimum=72):
    yy,xx=np.where(mask);h,w=mask.shape
    if not len(xx):return (0,0,w,h)
    x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1
    cx=(x0+x1)/2;cy=(y0+y1)/2
    bw=max(minimum,(x1-x0)*(1+2*pad));bh=max(minimum,(y1-y0)*(1+2*pad))
    return (max(0,int(cx-bw/2)),max(0,int(cy-bh/2)),min(w,int(cx+bw/2+1)),min(h,int(cy+bh/2+1)))
def union_box(mask,projections):
    union=mask.copy()
    for p in projections:
        uv=p['reliable_uv'];union[uv[:,1],uv[:,0]]=True
    return crop_box(union,.12)
def paint(rgb,mask,color,fill=0.,hatch=False,bbox=None):
    a=rgb.copy()
    if fill:
        a[mask]=np.rint(a[mask]*(1-fill)+np.array(color)*fill).astype(np.uint8)
    if hatch:
        y,x=np.indices(mask.shape)
        hit=mask & ((x-y)%14==0)
        a[hit]=np.rint(a[hit]*.3+np.array(color)*.7).astype(np.uint8)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(a,contours,-1,color,2,cv2.LINE_AA)
    if bbox is not None:
        x0,y0,x1,y1=np.rint(bbox).astype(int)
        cv2.rectangle(a,(x0,y0),(x1,y1),color,1,cv2.LINE_AA)
    return a
def tile(rgb,box,size,mask=None,color=PURPLE,fill=0,hatch=False,target_only=False,bbox=None):
    x0,y0,x1,y1=box;w,h=size
    raw=rgb[y0:y1,x0:x1]
    k=min(w/raw.shape[1],h/raw.shape[0])
    rw,rh=max(1,round(raw.shape[1]*k)),max(1,round(raw.shape[0]*k))
    out=cv2.resize(raw,(rw,rh),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC)
    if mask is not None:
        m=cv2.resize(mask[y0:y1,x0:x1].astype(np.uint8),(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
        if target_only:out[~m]=(45,49,57)
        else:
            bb=None if bbox is None else [(bbox[0]-x0)*k,(bbox[1]-y0)*k,(bbox[2]-x0)*k,(bbox[3]-y0)*k]
            out=paint(out,m,color,fill,hatch,bb)
    canvas=Image.new('RGB',(w,h),BG);canvas.paste(Image.fromarray(out),((w-rw)//2,(h-rh)//2))
    return canvas
def unproject(mask,depth,pose,K):
    yy,xx=np.where(mask & np.isfinite(depth) & (depth>0))
    pix=np.column_stack([xx,yy,np.ones(len(xx))])
    camera=(pix @ np.linalg.inv(np.asarray(K)[:3,:3]).T)*depth[yy,xx,None]
    world=np.column_stack([camera,np.ones(len(xx))])@np.asarray(pose).T
    return world[:,:3],np.column_stack([xx,yy])
def audit_card(current,out):
    rgb,mask=current['rgb'],current['mask'];h,w=mask.shape;box=crop_box(mask)
    im=Image.new('RGB',(1280,1160),BG)
    label(im,(24,16),'OBSERVATION AUDIT | CURRENT',30)
    label(im,(24,59),'Purple outline = CURRENT mask. Gray background = removed pixels.',21,MUTED)
    label(im,(24,101),'CONTEXT | full frame + detection box + mask outline',22)
    im.paste(tile(rgb,(0,0,w,h),(780,366),mask,bbox=current['bbox']),(20,141))
    label(im,(830,173),'ONE OBSERVATION',23)
    label(im,(830,219),'Locate it in the scene.',21,MUTED)
    label(im,(830,257),'Inspect its local structure.',21,MUTED)
    label(im,(830,295),'Check pixels inside the mask.',21,MUTED)
    label(im,(830,361),'frame '+str(current['frame_idx']),22)
    label(im,(830,401),'mask pixels: '+str(int(mask.sum())),20,MUTED)
    label(im,(24,534),'LOCAL RGB | mask outline',24)
    label(im,(660,534),'TARGET ONLY | same crop / scale',23)
    im.paste(tile(rgb,box,(608,544),mask),(24,578))
    im.paste(tile(rgb,box,(608,544),mask,target_only=True),(648,578))
    im.save(out,quality=97,subsampling=0)
    return dict(width=1280,height=1160,current_crop=list(box))
def geometry_tile(current,projection,box,size):
    rgb,mask=current['rgb'],current['mask'];x0,y0,x1,y1=box
    w,h=size;k=min(w/(x1-x0),h/(y1-y0));rw,rh=max(1,round((x1-x0)*k)),max(1,round((y1-y0)*k))
    raw=cv2.resize(rgb[y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC)
    cm=cv2.resize(mask[y0:y1,x0:x1].astype(np.uint8),(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
    pm=np.zeros(mask.shape,np.uint8);uv=projection['reliable_uv'];pm[uv[:,1],uv[:,0]]=1
    pm=cv2.resize(pm[y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
    raw=paint(raw,cm,PURPLE,fill=.22)
    raw=paint(raw,pm,CYAN,hatch=True)
    canvas=Image.new('RGB',(w,h),BG);canvas.paste(Image.fromarray(raw),((w-rw)//2,(h-rh)//2))
    return canvas
def pair_card(alias,current,histories,out):
    """Minimal labels; independently cropped CURRENT target; frozen projections."""
    n=len(histories);assert 1<=n<=3
    cb=crop_box(current['mask'])
    tb=crop_box(current['mask'],pad=.15,minimum=1)
    assert int(current['mask'][tb[1]:tb[3],tb[0]:tb[2]].sum())==int(current['mask'].sum())
    gb=union_box(current['mask'],[h['projection'] for h in histories])
    cr=(cb[2]-cb[0])/(cb[3]-cb[1]);gr=(gb[2]-gb[0])/(gb[3]-gb[1])
    portrait=cr<.72 and gr<.9 and n<=2
    labels=[]
    def txt(x,y,value,size=23,width=1232,color=TEXT):
        draw=ImageDraw.Draw(im)
        while draw.textlength(value,font=font(size))>width and size>12:size-=1
        bounds=draw.textbbox((x,y),value,font=font(size))
        assert bounds[0]>=0 and bounds[2]<=1280 and bounds[3]<=height
        draw.text((x,y),value,font=font(size),fill=color)
        labels.append(dict(text=value,bounds=list(bounds),font_size=size))
    if portrait:
        height=842;im=Image.new('RGB',(1280,height),BG)
        txt(24,101,'CURRENT · RGB + MASK',20,310)
        txt(350,101,'CURRENT · TARGET ONLY',18,224)
        im.paste(tile(current['rgb'],cb,(310,674),current['mask'],fill=.16),(24,144))
        im.paste(tile(current['rgb'],tb,(224,674),current['mask'],target_only=True),(350,144))
        txt(616,101,'CANDIDATE '+alias+' · VISIBLE PROJECTION',22,640)
        pw=(640-16*(n-1))//n
        for i,h in enumerate(histories):
            x=616+i*(pw+16)
            txt(x,145,alias+'-'+h['role']+' → CURRENT',20,pw)
            im.paste(geometry_tile(current,h['projection'],gb,(pw,640)),(x,178))
        layout='current_projection_columns'
        ts=(224,674)
    else:
        ch=min(350,max(220,round(824/cr)));pw=(1232-16*(n-1))//n
        ph=min(530,max(250,round(pw/gr)));gy=146+ch+64
        height=gy+ph+58;im=Image.new('RGB',(1280,height),BG)
        txt(24,101,'CURRENT · RGB + MASK',23,824)
        txt(868,101,'CURRENT · TARGET ONLY',23,388)
        im.paste(tile(current['rgb'],cb,(824,ch),current['mask'],fill=.16),(24,140))
        im.paste(tile(current['rgb'],tb,(388,ch),current['mask'],target_only=True),(868,140))
        txt(24,151+ch,'CANDIDATE '+alias+' · VISIBLE PROJECTION',24)
        for i,h in enumerate(histories):
            x=24+i*(pw+16)
            txt(x,gy-1,alias+'-'+h['role']+' → CURRENT',22,pw)
            im.paste(geometry_tile(current,h['projection'],gb,(pw,ph)),(x,gy+34))
        layout='current_projection_rows';ts=(388,ch)
    txt(24,14,'CURRENT vs CANDIDATE '+alias,29)
    txt(24,57,'Purple: CURRENT | Cyan + hatch: CANDIDATE '+alias,22)
    for i,a in enumerate(labels):
        ax0,ay0,ax1,ay1=a['bounds']
        for b in labels[i+1:]:
            bx0,by0,bx1,by1=b['bounds']
            assert min(ax1,bx1)<=max(ax0,bx0) or min(ay1,by1)<=max(ay0,by0),'overlapping labels'
    im.save(out,quality=97,subsampling=0)
    oldscale=min(ts[0]/(cb[2]-cb[0]),ts[1]/(cb[3]-cb[1]))
    newscale=min(ts[0]/(tb[2]-tb[0]),ts[1]/(tb[3]-tb[1]))
    return dict(width=1280,height=height,geometry_crop=list(gb),selected_histories=n,layout=layout,
        historical_rgb_panels=0,historical_target_only_panels=0,current_rgb_crop=list(cb),
        target_only_crop=list(tb),target_only_margin=.15,target_mask_pixels_preserved=int(current['mask'].sum()),
        target_linear_scale_gain=newscale/oldscale,image_labels=labels)
