#!/usr/bin/env python3
"""Select a completed result folder, then evaluate with the frozen protocol."""
import argparse,json,re,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def select(items,label):
 if len(items)==1:return items[0]
 if not sys.stdin.isatty():raise ValueError(label+'有多个候选，请使用 --map 或 --scene 明确指定')
 for i,v in enumerate(items,1):print(f'{i}. {v}')
 # Explicit bounds prevent zero/negative indices.
 while True:
  try:
   n=int(input(label+'编号：'))
   if 1<=n<=len(items):return items[n-1]
  except ValueError:pass
  print('请输入有效编号。')
def main():
 p=argparse.ArgumentParser(description='选择结果文件夹，使用统一完整网格协议评估。支持 ConceptGraphs pcd_*.pkl.gz 和已导出的 OVI npz。')
 p.add_argument('folder',nargs='?',help='结果文件夹；省略时交互输入')
 p.add_argument('--map',help='文件夹中的指定地图文件名')
 p.add_argument('--scene',choices=SCENES)
 p.add_argument('--method',default='selected_result')
 p.add_argument('--out',help='输出目录；默认 <结果文件夹>/unified_metrics')
 p.add_argument('--config',default=os.environ.get('UNIFIED_EVAL_CONFIG',str(ROOT/'config.local.json')),help='参考数据配置 JSON；也可设置 UNIFIED_EVAL_CONFIG')
 p.add_argument('--dry-run',action='store_true',help='检查并生成清单，不运行评估')
 args=p.parse_args();folder=args.folder
 if not folder:
  if not sys.stdin.isatty():p.error('请提供结果文件夹路径')
  folder=input('请输入结果文件夹完整路径：').strip().strip('\"\'')
 d=Path(folder).expanduser().resolve()
 if not d.is_dir():p.error('结果文件夹不存在：'+str(d))
 if args.map:
  mp=(d/args.map).resolve()
  if mp.parent!=d or not mp.is_file():p.error('--map 必须是所选文件夹中的文件')
 else:
  candidates=sorted(d.glob('pcd_*.pkl.gz'))
  if not candidates:candidates=sorted(d.glob('*.npz'))
  if not candidates:p.error('未找到最终地图。请选择直接包含 pcd_*.pkl.gz 或已导出 OVI npz 的文件夹。')
  mp=select(candidates,'选择地图')
 scene=args.scene or next((part for part in reversed(d.parts) if part in SCENES),None)
 if not scene:
  matches=[s for s in SCENES if re.search(r'(?<![a-z0-9])'+s+r'(?![a-z0-9])',mp.stem)]
  scene=matches[0] if len(matches)==1 else None
 if not scene:
  if not sys.stdin.isatty():p.error('无法识别场景，请传入 --scene')
  scene=select(SCENES,'选择场景')
 if not re.fullmatch(r'[A-Za-z0-9_-]+',args.method):p.error('--method 仅允许字母、数字、下划线和连字符')
 out=Path(args.out).expanduser().resolve() if args.out else d/'unified_metrics';out.mkdir(parents=True,exist_ok=True)
 config=Path(args.config).expanduser().resolve()
 if not config.is_file():p.error('找不到参考配置；复制 config.example.json 为 config.local.json 并填写数据路径，或使用 --config')
 template=json.load(open(config))
 for key in ['reference_root','clip_text']:
  value=Path(template[key]).expanduser()
  if not value.is_absolute():value=config.parent/value
  template[key]=str(value.resolve())
  if not value.exists():p.error('参考数据不存在：'+str(value))
 entry={'method':args.method,'scene':scene,'map':str(mp),'kind':'cg' if mp.name.endswith('.pkl.gz') else 'npz','cost':{'frames':None,'vlm_queries':None,'mapping_runtime_seconds':None,'end_to_end_runtime_seconds':None,'runtime_comparable':False}}
 cfg=d/'config_params.json'
 if cfg.exists():
  c=json.load(open(cfg));entry['input']={k:c.get(k) for k in ['start','end','stride','image_width','image_height','clip_masked_weight','detections_exp_suffix']}
 template['entries']=[entry];manifest=out/'selection_manifest.json';manifest.write_text(json.dumps(template,ensure_ascii=False,indent=2))
 print('地图：',mp,'\n场景：',scene,'\n结果：',out,flush=True)
 if args.dry_run:return
 subprocess.run([sys.executable,str(ROOT/'evaluate_unified.py'),'--manifest',str(manifest),'--out',str(out)],check=True)
 result=json.load(open(out/args.method/scene/'result.json'))
 print(json.dumps(result['metrics'],ensure_ascii=False,indent=2));print('评估完成：',out/args.method/scene/'result.json')
if __name__=='__main__':main()
