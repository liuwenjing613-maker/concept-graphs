#!/usr/bin/env python3
"""Check actual vision inference, not merely a listening port/model listing."""
import argparse, base64, io, json, time, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ports', nargs='+', type=int, default=[11435,11436,11437,11463,11464])
    p.add_argument('--model', default='qwen3.6:35b-a3b-mtp-q4_K_M')
    p.add_argument('--num-ctx', type=int, default=32768, help='Match v7 experiment context; avoid server default 262144')
    p.add_argument('--timeout', type=float, default=60)
    p.add_argument('--output', type=Path)
    a=p.parse_args()
    b=io.BytesIO();Image.new('RGB',(64,64),(255,0,0)).save(b,format='PNG')
    payload=dict(model=a.model,stream=False,think=False,messages=[dict(role='user',content='What is the main color of this image? Reply with one word.',images=[base64.b64encode(b.getvalue()).decode()])],options=dict(temperature=0,num_predict=32,num_ctx=a.num_ctx))
    report=dict(checked_at=datetime.now(timezone.utc).isoformat(),model=a.model,num_ctx=a.num_ctx,results=[])
    for port in a.ports:
        start=time.perf_counter();row=dict(port=port)
        try:
            req=urllib.request.Request(f'http://127.0.0.1:{port}/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=a.timeout) as response:
                d=json.load(response);row['http_status']=response.status
            content=d.get('message',{}).get('content','').strip()
            row.update(ok=bool(d.get('done') and content and not d.get('error')),answer=content,done_reason=d.get('done_reason'))
        except urllib.error.HTTPError as e:
            row.update(ok=False,http_status=e.code,error=e.read().decode("utf-8",errors="replace")[:2000])
        except Exception as e:
            row.update(ok=False,error=str(e))
        row['seconds']=round(time.perf_counter()-start,3)
        report['results'].append(row);print(json.dumps(row),flush=True)
        if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
