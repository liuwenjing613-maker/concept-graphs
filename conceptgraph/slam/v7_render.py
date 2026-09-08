"""Validated v6 renderers, adapted to explicit live inputs; no offline data access."""
import math
import numpy as np
import cv2
from PIL import Image,ImageDraw
from conceptgraph.slam import staged_identity_cards as cards
from conceptgraph.slam.staged_identity_cards import BG,TEXT,MUTED,PURPLE,CYAN,font,crop_box,union_box
from conceptgraph.slam.history_projection import project_points
COLORS={'A':PURPLE,'B':CYAN,'C':(255,207,62)}
YELLOW=(255,207,62)


def outline(rgb,m,c,width=2):
 out=rgb.copy();cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(out,cs,-1,c,width,cv2.LINE_AA);return out

def text(im,x,y,s,size=23,width=None,color=TEXT):
 d=ImageDraw.Draw(im)
 if width:
  while d.textlength(s,font=font(size))>width and size>14:size-=1
 bb=d.textbbox((x,y),s,font=font(size));assert bb[2]<=im.width and bb[3]<=im.height,(s,bb,im.size)
 d.text((x,y),s,font=font(size),fill=color)

def view_tile(v,size,color,target=False,projection=None):
 m=v['mask'];rgb=v['rgb'];box=crop_box(m,.15,1) if target else crop_box(m,.22,72)
 if projection is not None:box=union_box(m,[projection])
 x0,y0,x1,y1=box;w,h=size;k=min(w/(x1-x0),h/(y1-y0))
 rw,rh=max(1,round((x1-x0)*k)),max(1,round((y1-y0)*k))
 if False and not target and projection is None:
  rgb=rgb.copy();rgb[~m]=np.rint(rgb[~m]*.22+12).clip(0,255).astype(np.uint8)
 raw=cv2.resize(rgb[y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC)
 mm=cv2.resize(m[y0:y1,x0:x1].astype(np.uint8),(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
 if projection is not None:
  pm=np.zeros_like(m);uv=projection['reliable_uv'];pm[uv[:,1],uv[:,0]]=True
  a,b=(m,pm) if color==PURPLE else (pm,m)
  region=a.astype(np.uint8)+2*b.astype(np.uint8)
  r=cv2.resize(region[y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_NEAREST)
  for value,c in [(1,PURPLE),(2,CYAN),(3,YELLOW)]:
   hit=r==value;raw[hit]=np.rint(raw[hit]*.68+np.array(c)*.32).astype(np.uint8)
   raw=outline(raw,hit,c)
 elif target:raw[~mm]=(45,49,57)
 else:raw=outline(raw,mm,color)
 canvas=Image.new('RGB',size,BG);canvas.paste(Image.fromarray(raw),((w-rw)//2,(h-rh)//2))
 assert int(m[y0:y1,x0:x1].sum())==int(m.sum())
 return canvas,dict(crop=list(box),original_mask_pixels=int(m.sum()),render_scale=k)

def node_card(alias,chosen,dest,visual):
 n=len(chosen);assert 1<=n<=3
 if n==1:
  im=Image.new('RGB',(1440,650),BG);pw=684;ph=474
  placements=[(24,124,24,162,732,124,732,162)]
 else:
  pw=528;ph=416;im=Image.new('RGB',(24+(pw+24)*n,1080),BG)
  placements=[(24+i*(pw+24),118,24+i*(pw+24),155,24+i*(pw+24),603,24+i*(pw+24),640) for i in range(n)]
 text(im,24,16,'OBJECT '+alias+' | HISTORY AUDIT',30)
 text(im,24,61,({'A':'Purple','B':'Cyan','C':'Yellow'}[alias])+': target | RGB: context | Gray: removed',22)
 panels=[]
 for i,(r,pos) in enumerate(zip(chosen,placements)):
  v=visual(r['uid']);lx,ly,ix,iy,tx,ty,jx,jy=pos
  text(im,lx,ly,'H'+str(i+1)+' | RGB + MASK',23,pw)
  text(im,tx,ty,'H'+str(i+1)+' | TARGET ONLY',23,pw)
  for target,x,y in [(False,ix,iy),(True,jx,jy)]:
   tile,meta=view_tile(v,(pw,ph),COLORS[alias],target);im.paste(tile,(x,y))
   assert x+pw<=im.width and y+ph<=im.height
   panels.append(dict(history=r['uid'],target_only=target,box=[x,y,pw,ph],**meta))
 im.save(dest,quality=97,subsampling=0)
 return dict(width=im.width,height=im.height,history_count=n,panels=panels)

def history_scores(uids,obs,arr):
 rows=[];features=[]
 for uid in uids:
  o=obs[uid];idx=int(o['frame_uid'].rsplit('_f',1)[1]);a=int(o['processed_mask_area'])
  feature=arr(o['image_feat_ref']).astype(float).ravel();feature/=max(1e-12,np.linalg.norm(feature))
  features.append(feature)
  rows.append(dict(uid=uid,frame_idx=idx,mask_pixels=a,quality=a*float(o['valid_depth_ratio'])*(1-min(.9,float(o['boundary_touch_ratio']))),
    center=o['bbox_3d_center'],geometry_score=0.0))
 centers=np.array([r['center'] for r in rows]);med=np.median(centers,axis=0)
 for r in rows:r['geometry_score']=float(np.linalg.norm(np.array(r['center'])-med))
 h1=max(range(len(rows)),key=lambda i:(rows[i]['quality'],rows[i]['uid']))
 for i,r in enumerate(rows):r['visual_distance']=float(np.clip(1-features[i]@features[h1],0,2))
 selected=[dict(rows[h1],role='representative')]
 remaining=[r for i,r in enumerate(rows) if i!=h1 and r['mask_pixels']>0]
 if remaining:
  h2=max(remaining,key=lambda r:(r['visual_distance'],r['quality'],r['uid']));selected.append(dict(h2,role='visual_outlier'))
  remaining=[r for r in remaining if r['uid']!=h2['uid']]
 if remaining:
  h3=max(remaining,key=lambda r:(r['geometry_score'],r['visual_distance'],r['uid']));selected.append(dict(h3,role='geometric_outlier'))
 return rows,selected

def anchor_scores(alias,rows,points,obs,visual):
 out=[]
 for r in rows:
  v=visual(r['uid']);m=v['mask'];p=project_points(points,v['pose'],v['K'],v['depth'])
  uv=p['reliable_uv'];counts=p['counts'];q=int(m.sum());n=len(uv)
  reliability=n/max(1,len(p['uv']))
  border=0 if not n else float(((uv[:,0]<3)|(uv[:,0]>=m.shape[1]-3)|(uv[:,1]<3)|(uv[:,1]>=m.shape[0]-3)).mean())
  own=float(obs[r['uid']]['valid_depth_ratio'])*(1-min(.9,float(obs[r['uid']]['boundary_touch_ratio'])))
  score=float(np.sqrt(min(q,n))*np.sqrt(reliability)*own*(1-border))
  out.append(dict(uid=r['uid'],frame_idx=r['frame_idx'],alias=alias,score=score,target_pixels=q,visible_pixels=n,reliability=reliability,projected_border_fraction=border,counts=counts))
 return out

def geometry(current,p):
 cm=current['mask'];pm=np.zeros_like(cm);uv=p['reliable_uv'];pm[uv[:,1],uv[:,0]]=True
 intersect=int((cm&pm).sum());n=int(pm.sum());q=int(cm.sum())
 return cm,pm,dict(current_pixels=q,projection_pixels=n,shared_pixels=intersect,projection_in_current=round(intersect/max(1,n),6),current_covered=round(intersect/max(1,q),6),iou=round(intersect/max(1,int((cm|pm).sum())),6))

def paint_holes(rgb,mask,color,fill=0.,hatch=False,bbox=None):
 a=rgb.copy()
 if fill:a[mask]=np.rint(a[mask]*(1-fill)+np.array(color)*fill).astype(np.uint8)
 if hatch:
  y,x=np.indices(mask.shape);hit=mask&((x-y)%14==0);a[hit]=np.rint(a[hit]*.3+np.array(color)*.7).astype(np.uint8)
 contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(a,contours,-1,color,2,cv2.LINE_AA)
 return a

def region_card(item,h,p,dest):
 cur=item['current'];cm,pm,g=geometry(cur,p)
 W,H=1740,1150;im=Image.new('RGB',(W,H),cards.BG);draw=ImageDraw.Draw(im)
 def txt(x,y,t,size=22,color=cards.TEXT):draw.text((x,y),t,font=cards.font(size),fill=color)
 txt(24,18,'CURRENT vs CANDIDATE '+item['alias'],32)
 txt(24,65,'Source views: purple = CURRENT | cyan = CANDIDATE',24)
 old=cards.paint;cards.paint=paint_holes
 try:
  for y,who,v,color,fill,hatch in [(110,'CURRENT',cur,cards.PURPLE,.16,False),(624,'CANDIDATE '+item['alias']+' | H1',h,cards.CYAN,0,True)]:
   txt(24,y,who+' | RGB + MASK',23);txt(566,y,who+' | TARGET ONLY',21)
   cb=cards.crop_box(v['mask']);tb=cards.crop_box(v['mask'],pad=.15,minimum=1)
   im.paste(cards.tile(v['rgb'],cb,(518,452),v['mask'],color=color,fill=fill,hatch=hatch),(24,y+40))
   im.paste(cards.tile(v['rgb'],tb,(518,452),v['mask'],target_only=True),(566,y+40))
 finally:cards.paint=old
 txt(1108,110,'PROJECTION IN CURRENT CAMERA',22)
 colors=[cards.PURPLE,(255,207,62),cards.CYAN];names=['CURRENT ONLY','SHARED MASK AREA','CANDIDATE ONLY']
 for i,(color,label) in enumerate(zip(colors,names)):
  y=148+i*29;draw.rectangle((1108,y+4,1127,y+21),fill=color);txt(1136,y,label,20)
 box=cards.union_box(cm,[p]);x0,y0,x1,y1=box;k=min(608/(x1-x0),750/(y1-y0));rw=max(1,round((x1-x0)*k));rh=max(1,round((y1-y0)*k))
 raw=cv2.resize(cur['rgb'][y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC)
 # Classify before resizing, nearest-neighbour preserves mask membership and holes.
 region=cm.astype(np.uint8)+pm.astype(np.uint8)*2
 region=cv2.resize(region[y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_NEAREST)
 for code,color in [(1,colors[0]),(3,colors[1]),(2,colors[2])]:raw=paint_holes(raw,region==code,color,fill=.32)
 im.paste(Image.fromarray(raw),(1108+(608-rw)//2,250+(750-rh)//2))
 ptxt='n/a (no visible pixels)' if not g['projection_pixels'] else f"{100*g['projection_in_current']:.1f}%"
 txt(1108,1013,'Projection inside CURRENT: '+ptxt,20)
 txt(1108,1042,f"CURRENT covered: {100*g['current_covered']:.1f}%",20)
 txt(1108,1071,f"Visible projection: {g['projection_pixels']} pixels",20)
 txt(1108,1100,f"Source masks: C {int(cm.sum())} / H {int(h['mask'].sum())} px",19)
 im.save(dest,quality=97,subsampling=0)
 assert int((cm&pm).sum())==g['shared_pixels']
 return g

old_tile=cards.tile

def focus_tile(rgb,box,size,mask=None,color=cards.PURPLE,fill=0,hatch=False,target_only=False,bbox=None):
 if mask is None or target_only:return old_tile(rgb,box,size,mask=mask,color=color,fill=fill,hatch=hatch,target_only=target_only,bbox=bbox)
 out=rgb.copy();out[~mask]=np.rint(out[~mask]*.22+12).clip(0,255).astype(np.uint8)
 # No tint/hatching inside source target: preserve color and texture exactly.
 return old_tile(out,box,size,mask=mask,color=color,fill=0,hatch=False,target_only=False,bbox=bbox)

def focused_card(current,alias,h,p,path):
 old=cards.tile;cards.tile=focus_tile
 try:return region_card(dict(current=current,alias=alias),h,p,path)
 finally:cards.tile=old


def select(m,a,frames):
 first=m['selected_anchor'] if m['selected_anchor']['alias']==a else m['other_appearance_history'];hist=m['histories'][a]['all'];uids=set(m['objects'][a]['member_observation_uids']);h=m['source_timeline']['h_frame'];assert first['uid'] in uids and first['frame_idx']<=h
 valid=[x for x in hist if x['uid'] in uids and x['frame_idx']<=h and x['quality']>0];center=np.median([x['center'] for x in valid],axis=0);maxq=max(x['quality'] for x in valid)
 def direction(x):
  c=np.asarray(frames[x['frame_idx']]['pose'])[:3,3];v=c-center;return v/max(np.linalg.norm(v),1e-9)
 d0=direction(first);candidates=[]
 for x in valid:
  if x['frame_idx']==first['frame_idx'] or x['quality']<.25*maxq:continue
  angle=float(np.degrees(np.arccos(np.clip(np.dot(d0,direction(x)),-1,1))));score=(angle+1e-6)*math.sqrt(x['quality']/maxq);candidates.append(dict(**x,view_angle_degrees=angle,selection_score=score))
 if not candidates:
  for x in valid:
   if x['frame_idx']!=first['frame_idx']:candidates.append(dict(**x,view_angle_degrees=None,selection_score=x['quality']))
 if not candidates:return [dict(first,selection_role='only_available_view')]
 second=max(candidates,key=lambda x:(x['selection_score'],x['quality'],-x['frame_idx']));return [dict(first,selection_role='existing_comparison_view'),dict(second,selection_role='quality_constrained_view_direction_change')]

def history_tile(obs,a,mode,visual):
 v=visual(obs['uid']);mask=v['mask'];box=crop_box(mask,.45,160);x0,y0,x1,y1=box;w,h=684,420;k=min(w/(x1-x0),h/(y1-y0));rw,rh=round((x1-x0)*k),round((y1-y0)*k);rgb=cv2.resize(v['rgb'][y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC);mm=cv2.resize(mask[y0:y1,x0:x1].astype('uint8'),(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
 if mode=='target_rgb':rgb[~mm]=(45,48,55)
 else:
  color=COLORS[a];rgb=outline(rgb,mm,color,2);n,lab,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'));big=(lab==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype('uint8');dist=cv2.distanceTransform(np.pad(big,1),cv2.DIST_L2,5)[1:-1,1:-1];yy,xx=np.unravel_index(dist.argmax(),dist.shape);assert mask[yy,xx];tip=(min(rw-1,int((xx-x0)*k)),min(rh-1,int((yy-y0)*k)));start=(max(25,min(rw-26,tip[0]-70 if tip[0]>rw/2 else tip[0]+70)),max(25,min(rh-26,tip[1]-70)));cv2.arrowedLine(rgb,start,tip,color,2,cv2.LINE_AA,tipLength=.2);cv2.circle(rgb,start,16,color,-1);cv2.putText(rgb,a,(start[0]-9,start[1]+7),cv2.FONT_HERSHEY_SIMPLEX,.65,(15,24,32),2,cv2.LINE_AA)
 im=Image.new('RGB',(w,h),BG);im.paste(Image.fromarray(rgb),((w-rw)//2,(h-rh)//2));return im,dict(uid=obs['uid'],frame=obs['frame_idx'],crop=list(box),render_size=[rw,rh],scale=k,fill=0,mask_source='original_frozen_observation',source_mask_pixels=int(mask.sum()))

def merge_card(m,selected,visual,cm,pm):
 im=Image.new('RGB',(1440,1510),BG);text(im,24,15,'A vs B | SAME PHYSICAL OBJECT?',30);text(im,24,62,'Historical views | Target RGB only; gray = excluded',22)
 for a,x in [('A',24),('B',732)]:
  if len(selected[a])<2:text(im,x,700,a+'2 | No second historical view',24,650)
  for j,ob in enumerate(selected[a]):
   y=145+j*490;text(im,x,y-40,a+str(j+1)+' | frame '+str(ob['frame_idx']),24);t,_=history_tile(ob,a,'target_rgb',visual);im.paste(t,(x,y))
 a=m['selected_anchor']['alias'];other='B' if a=='A' else 'A';uid=m['selected_anchor']['uid'];v=visual(uid)
 assert np.array_equal(cm,v['mask']);support=cm|pm
 def overlay(box,w,h,hide):
  x0,y0,x1,y1=box;k=min(w/(x1-x0),h/(y1-y0));rw,rh=round((x1-x0)*k),round((y1-y0)*k);rgb=cv2.resize(v['rgb'][y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC);mask=cv2.resize(cm[y0:y1,x0:x1].astype('uint8'),(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
  if hide:
   keep=cv2.resize(support[y0:y1,x0:x1].astype('uint8'),(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool);rgb[~keep]=(45,48,55)
  rgb=outline(rgb,mask,COLORS[a],1);yy,xx=np.where(pm[y0:y1,x0:x1]);py=np.minimum(rh-1,np.floor(yy*k).astype(int));px=np.minimum(rw-1,np.floor(xx*k).astype(int));rgb[py,px]=COLORS[other];tile=Image.new('RGB',(w,h),BG);tile.paste(Image.fromarray(rgb),((w-rw)//2,(h-rh)//2));return tile,dict(crop=list(box),scale=k)
 context,ctx=overlay(crop_box(support,.45,120),684,310,False);zoom,zm=overlay(crop_box(support,.08,1),660,310,True);draw=ImageDraw.Draw(im);draw.rectangle((0,1080,1439,1509),fill=BG);text(im,24,1090,'RGB + PROJECTION | '+a+'1 CAMERA',25);draw.line((24,1144,72,1144),fill=COLORS[a],width=3);text(im,84,1129,a+' mask boundary',21);draw.ellipse((372,1140,377,1145),fill=COLORS[other]);draw.ellipse((389,1136,394,1141),fill=COLORS[other]);text(im,410,1129,other+' projected points',21);im.paste(context,(24,1175));text(im,752,1090,'TARGET RGB + POINTS | ZOOM',24,660);text(im,752,1130,'Same camera | Gray: no selected support',21,660);im.paste(zoom,(752,1175))
 return im,dict(source_h_snapshot_uid=m['source_h_snapshot_uid'],timeline=m['source_timeline'],selected_histories=selected,context=ctx,zoom=zm,projection_source=m['projection_source'],projection_cloud_sha256=m['projection_cloud_sha256'],point_expansion=False)
