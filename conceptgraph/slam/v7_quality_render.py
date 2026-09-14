"""Exact accepted RGB/mask rendering; consumes only frozen online evidence."""
import math
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
RED=np.array([235,55,65]);FILL=.12;DIM=.25
def box(mask):
 y,x=np.where(mask);h,w=mask.shape;dx=max(16,math.ceil((x.max()-x.min()+1)*.18));dy=max(16,math.ceil((y.max()-y.min()+1)*.18))
 return max(0,int(x.min())-dx),max(0,int(y.min())-dy),min(w,int(x.max())+1+dx),min(h,int(y.max())+1+dy)
def panel(rgb,mask,size,crop,focus=False):
 x0,y0,x1,y1=crop;assert mask[y0:y1,x0:x1].sum()==mask.sum()
 w,h=size;k=min(w/(x1-x0),h/(y1-y0));rw,rh=round(k*(x1-x0)),round(k*(y1-y0))
 a=cv2.resize(rgb[y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC)
 m=cv2.resize(mask[y0:y1,x0:x1].astype('uint8'),(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
 if focus:a[~m]=np.rint(a[~m]*DIM).astype('uint8')
 alpha=.25 if focus else FILL
 a[m]=np.rint(a[m]*(1-alpha)+RED*alpha).astype('uint8')
 contours,hier=cv2.findContours(m.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(a,contours,-1,tuple(map(int,RED)),2,cv2.LINE_AA)
 canvas=Image.new('RGB',size,(244,246,248));canvas.paste(Image.fromarray(a),((w-rw)//2,(h-rh)//2))
 return canvas,dict(crop=list(crop),scale=k,rendered_mask_pixels=int(m.sum()),contours=len(contours),holes=sum(int(x[3]>=0) for x in hier[0]) if hier is not None else 0)

def quality_card(views,dest,node=False):
 if not views or len(views)>3 or (not node and len(views)!=1):raise ValueError('invalid quality views')
 W,H,label=1944,704,34 if node else 0
 im=Image.new('RGB',(W,(H+label)*len(views)),(244,246,248));metas=[]
 for i,v in enumerate(views):
  rgb=np.asarray(v['rgb']);m=np.asarray(v['mask'],dtype=bool)
  if rgb.shape[:2]!=m.shape or not m.any():raise ValueError('invalid quality mask')
  y=i*(H+label)
  if node:ImageDraw.Draw(im).text((12,y+2),'H'+str(i+1),font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25),fill=(25,35,45))
  y+=label
  left,lm=panel(rgb,m,(1200,680),(0,0,m.shape[1],m.shape[0]))
  right,rm=panel(rgb,m,(708,680),box(m),True)
  im.paste(left,(12,y+12));im.paste(right,(1224,y+12))
  metas.append(dict(uid=v['uid'],context=lm,focus=rm))
 im.save(dest,format='PNG')
 return dict(width=W,height=im.height,views=metas,fill=.12,focus_fill=.25,background=.25)

def select_quality(rows):
 h1=max(rows,key=lambda r:(r['quality'],r['uid']))
 rem=[r for r in rows if r['uid']!=h1['uid'] and r['mask_pixels']>0 and r['quality']>0 and r['quality']>=.25*h1['quality']]
 out=[dict(h1,role='representative')]
 if rem:
  h2=max(rem,key=lambda r:(r['visual_distance'],r['quality'],r['uid']));out.append(dict(h2,role='visual_outlier'));rem=[r for r in rem if r['uid']!=h2['uid']]
 if rem:out.append(dict(max(rem,key=lambda r:(r['geometry_score'],r['visual_distance'],r['uid'])),role='geometric_outlier'))
 return out
