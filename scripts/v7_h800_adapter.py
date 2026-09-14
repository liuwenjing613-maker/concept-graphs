"""Lossless nonstreaming Ollama-to-vLLM transport; no visual decision logic."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse,json,time,uuid,hashlib,base64,threading,httpx

def translate(data,model):
 if data['model']!=model:raise ValueError('Model mismatch')
 if data.get('stream',False):raise ValueError('Streaming unsupported')
 opts=data.get('options',{})
 if set(opts)-{'num_ctx','num_predict','temperature'}:raise ValueError('Unmapped options')
 if opts.get('num_ctx',32768)!=32768:raise ValueError('Context limit mismatch')
 messages=[]
 for m in data['messages']:
  content=m['content']
  if m.get('images'):
   content=[{'type':'text','text':content}]
   for b in m['images']:
    raw=base64.b64decode(b,validate=True)
    mime='png' if raw.startswith(b'\x89PNG\r\n\x1a\n') else 'jpeg' if raw.startswith(b'\xff\xd8\xff') else None
    if mime is None:raise ValueError('Invalid image format')
    content.append({'type':'image_url','image_url':{'url':'data:image/'+mime+';base64,'+b}})
  messages.append({'role':m['role'],'content':content})
 req=dict(model=model,messages=messages,stream=False,temperature=opts.get('temperature',0),max_tokens=opts.get('num_predict',1024),chat_template_kwargs={'enable_thinking':data.get('think',False)})
 fmt=data.get('format')
 if isinstance(fmt,dict):req['response_format']={'type':'json_schema','json_schema':{'name':'stage_response','strict':True,'schema':fmt}}
 elif fmt=='json':req['response_format']={'type':'json_object'}
 elif fmt:raise ValueError('Unmapped format')
 return req

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--identity',required=True);ap.add_argument('--upstream',required=True);ap.add_argument('--ports',type=int,nargs='+',required=True);ap.add_argument('--log-root',required=True);a=ap.parse_args()
 ident=json.loads(Path(a.identity).read_text());root=Path(a.log_root);root.mkdir(parents=True,exist_ok=True)
 assert hashlib.sha256(json.dumps(ident['files'],sort_keys=True).encode()).hexdigest()==ident['digest']
 class Handler(BaseHTTPRequestHandler):
  def respond(self,code,data):
   body=json.dumps(data,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
  def do_GET(self):
   try:
    with httpx.Client(timeout=10,trust_env=False) as c:r=c.get(a.upstream+'/v1/models');r.raise_for_status()
    matches=[m for m in r.json()['data'] if m['id']==ident['model']]
    if len(matches)!=1:raise ValueError('Upstream model missing')
    if self.path=='/api/tags':return self.respond(200,{'models':[dict(name=ident['model'],digest=ident['digest'],backend=ident['backend'])]})
    self.respond(200,r.json())
   except Exception as e:self.respond(503,{'error':str(e)})
  def do_POST(self):
   rid=uuid.uuid4().hex;start=time.perf_counter()
   try:
    if self.path!='/api/chat':return self.respond(404,{'error':'Unsupported endpoint'})
    data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));req=translate(data,ident['model'])
    with httpx.Client(timeout=310,trust_env=False) as c:r=c.post(a.upstream+'/v1/chat/completions',json=req)
    raw=r.json();elapsed=time.perf_counter()-start
    (root/(rid+'.json')).write_text(json.dumps(dict(request_id=rid,request_sha256=hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest(),model_identity={k:ident[k] for k in ('model','digest','backend')},options=data.get('options'),format=data.get('format'),enable_thinking=data.get('think',False),http_status=r.status_code,seconds=elapsed,upstream_response=raw),ensure_ascii=False,indent=2))
    if r.status_code!=200:return self.respond(r.status_code,raw)
    if raw['model']!=ident['model']:raise ValueError('Response model mismatch')
    choice=raw['choices'][0];usage=raw.get('usage',{})
    self.respond(200,dict(model=raw['model'],message=dict(role='assistant',content=choice['message'].get('content') or ''),done=True,done_reason=choice['finish_reason'],total_duration=int(elapsed*1e9),prompt_eval_count=usage.get('prompt_tokens'),eval_count=usage.get('completion_tokens'),adapter_request_id=rid))
   except Exception as e:
    (root/(rid+'_error.json')).write_text(json.dumps(dict(request_id=rid,error=type(e).__name__+': '+str(e),seconds=time.perf_counter()-start)))
    self.respond(502,{'error':type(e).__name__+': '+str(e),'request_id':rid})
 servers=[ThreadingHTTPServer(('127.0.0.1',p),Handler) for p in a.ports]
 for s in servers[:-1]:threading.Thread(target=s.serve_forever,daemon=True).start()
 servers[-1].serve_forever()
if __name__=='__main__':main()
