"""User-confirmed observation-quality A/B renderer; no map mutation."""
import math
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(F,15)
title=ImageFont.truetype(F,22)
small=ImageFont.truetype(F,13)
CYAN=np.array([0,225,225]);W=640;H=480;HEADER=64;GAP=18
def box(mask,scale):
 yy,xx=np.where(mask);a,b=int(xx.min()),int(xx.max())+1;c,d=int(yy.min()),int(yy.max())+1
 cx=(a+b)/2;cy=(c+d)/2
 return [max(0,math.floor(cx-(b-a)*scale/2)),max(0,math.floor(cy-(d-c)*scale/2)),min(mask.shape[1],math.ceil(cx+(b-a)*scale/2)),min(mask.shape[0],math.ceil(cy+(d-c)*scale/2))]


def panel(rgb,mask,main_component,scale,focused):
 crop=box(mask,scale);l,t,r,b=crop;raw=rgb[t:b,l:r];m0=mask[t:b,l:r]
 f=min(W/raw.shape[1],H/raw.shape[0]);ww=round(raw.shape[1]*f);hh=round(raw.shape[0]*f)
 natural=np.array(Image.fromarray(raw).resize((ww,hh),Image.Resampling.LANCZOS))
 m=cv2.resize(m0.astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST).astype(bool)
 a=natural.copy()
 if focused:
  a[~m]=np.rint(a[~m].astype(float)*.65).astype(np.uint8)
  a[m]=np.rint(.9*a[m].astype(float)+.1*CYAN).astype(np.uint8)
 expected=a.copy()
 k=np.ones((3,3),np.uint8)
 d2=cv2.dilate(m.astype(np.uint8),k,iterations=2).astype(bool)
 d3=cv2.dilate(m.astype(np.uint8),k,iterations=3).astype(bool)
 a[d3&~d2]=[10,15,18];a[d2&~m]=CYAN
 im=Image.fromarray(a)
 # Caption and line live outside the target, never in a hole enclosed by its bounding rectangle.
 yy,xx=np.where(m);x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1
 tb=font.getbbox('TARGET');tw=tb[2]+8;th=tb[3]-tb[1]+8
 candidates=[]
 main_crop=main_component[t:b,l:r]
 main_display=cv2.resize(main_crop.astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST).astype(bool)
 ay,ax=np.where(main_display);assert len(ax)>0
 points=np.column_stack((ax,ay))
 from scipy.spatial import cKDTree
 point_tree=cKDTree(points)
 for y in range(5,max(6,hh-th-4),12):
  for x in range(5,max(6,ww-tw-4),12):
   if x+tw>ww or y+th>hh:continue
   if not (x+tw<x0 or x>x1 or y+th<y0 or y>y1):continue
   if d3[y:y+th,x:x+tw].any():continue
   center=np.array([x+tw/2,y+th/2]);distance,j=point_tree.query(center);j=int(j)
   # Prefer a short but visible line with a clear gap between text and contour.
   score=abs(distance-55)
   candidates.append((score,x,y,points[j].tolist()))
 draw=ImageDraw.Draw(im);label=None
 if candidates:
  _,x,y,end=min(candidates)
  center=np.array([x+tw/2,y+th/2]);v=np.array(end)-center
  length=max(abs(v[0])/(tw/2+2),abs(v[1])/(th/2+2),1.)
  start=np.rint(center+v/length).astype(int).tolist()
  line=np.zeros(m.shape,np.uint8);cv2.line(line,tuple(start),tuple(end),255,3,cv2.LINE_8)
  aa=np.array(im);aa[(line>0)&~m]=[5,10,15]
  line[:]=0;cv2.line(line,tuple(start),tuple(end),255,1,cv2.LINE_8)
  aa[(line>0)&~m]=CYAN
  im=Image.fromarray(aa);draw=ImageDraw.Draw(im)
  draw.text((x+4,y+4-tb[1]),'TARGET',font=font,fill=tuple(CYAN),stroke_width=1,stroke_fill=(0,0,0))
  label={'location':'outside_target_bbox','rect':[x,y,x+tw,y+th],'line_start':start,'line_endpoint_target_pixel':end}
 else:
  label={'location':'header','line_endpoint_target_pixel':[int(xx[len(xx)//2]),int(yy[len(yy)//2])]}
 assert np.array_equal(np.array(im)[m],expected[m]),'annotations must not overwrite target'
 canvas=Image.new('RGB',(W,H+HEADER),(249,250,252));draw=ImageDraw.Draw(canvas)
 draw.text((18,10),'B. Focused RGB' if focused else 'A. Context RGB',font=title,fill=(20,35,50))
 if label['location']!='header':draw.text((18,39),'1.35x crop | background 65% | tint 10%' if focused else '2.0x crop | natural RGB',font=small,fill=(66,80,96))
 ox=(W-ww)//2;oy=HEADER+(H-hh)//2;canvas.paste(im,(ox,oy))
 if label['location']=='header':
  fronts=[]
  for qx in range(ww):
   qys=np.flatnonzero(m[:,qx])
   if len(qys) and main_display[qys[0],qx]:fronts.append((int(qys[0]),abs(qx-ww/2),qx))
  assert fronts,'No unobstructed principal component from header'
  py,_,px=min(fronts);tx=max(8,min(W-tw-8,ox+px-tw//2))
  draw=ImageDraw.Draw(canvas);draw.text((tx,38),'TARGET',font=font,fill=(0,150,150))
  sx=ox+px
  draw.line([(sx,59),(sx,oy+py-1)],fill=(5,10,15),width=3)
  draw.line([(sx,59),(sx,oy+py-1)],fill=(0,225,225),width=1)
  label.update(line_endpoint_target_pixel=[px,py],location='header_external_vertical_leader')
 return canvas,{'crop_xyxy':crop,'crop_scale':scale,'source_crop_size':[raw.shape[1],raw.shape[0]],'display_rgb_size':[ww,hh],'display_scale':f,'mask_resampling':'nearest','rgb_resampling':'Lanczos','target_overlay_alpha':.1 if focused else 0.,'background_factor':.65 if focused else 1.,'target_annotation_overwrite':False,'target_label':label,'source_target_pixels':int(m0.sum())}


def audit_quality_card(current,destination):
    rgb=np.asarray(current['rgb'],dtype=np.uint8)
    mask=np.asarray(current['mask'],dtype=bool)
    if rgb.shape[:2]!=mask.shape or not mask.any():
        raise ValueError('empty or mismatched observation mask')
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    main=labels==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))
    a,am=panel(rgb,mask,main,2.0,False)
    b,bm=panel(rgb,mask,main,1.35,True)
    card=Image.new('RGB',(W*2+GAP,H+HEADER),(225,230,236))
    card.paste(a,(0,0));card.paste(b,(W+GAP,0));card.save(destination,format='PNG')
    return dict(A=am,B=bm,source_mask_pixels=int(mask.sum()),components=n-1)
