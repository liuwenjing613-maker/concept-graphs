"""Deterministic parsing and event decisions; no model calls or map mutation."""
import json,re
QUALITY_SCHEMA={'status':('USABLE','CORRUPTED','INSUFFICIENT'),'reason':str}
PAIR_SCHEMA={'choice':('SAME','DIFFERENT','UNCERTAIN'),'confidence':int}
class DuplicateKey(ValueError):pass
def unique_pairs(items):
    d={}
    for k,v in items:
        if k in d:raise DuplicateKey(k)
        d[k]=v
    return d
def reject_constant(v):raise ValueError('nonfinite_json')
DECODER=json.JSONDecoder(object_pairs_hook=unique_pairs,parse_constant=reject_constant)
def schema_valid(o,task):
    schema=QUALITY_SCHEMA if task=='quality' else PAIR_SCHEMA
    if not isinstance(o,dict) or set(o)!=set(schema):return False
    if task=='quality':
        return o['status'] in schema['status'] and isinstance(o['reason'],str) and 1<=len(o['reason'])<=120
    return o['choice'] in schema['choice'] and type(o['confidence']) is int and 0<=o['confidence']<=5
def parse_response(text,task):
    bad=lambda why:dict(accepted=False,strict=False,mode='REJECTED',value=None,error=why)
    if not isinstance(text,str) or not text.strip():return bad('empty_output')
    def parse_all(t):
        obj,end=DECODER.raw_decode(t.lstrip())
        if t.lstrip()[end:].strip():raise ValueError('trailing_text')
        return obj
    try:
        obj=parse_all(text)
        if not schema_valid(obj,task):return bad('invalid_schema')
        return dict(accepted=True,strict=True,mode='STRICT_JSON',value=obj,error=None)
    except DuplicateKey:return bad('duplicate_key')
    except (ValueError,TypeError):pass
    fence=re.fullmatch(r'\s*'+chr(96)*3+r'(?:json)?\s*(.*?)\s*'+chr(96)*3+r'\s*',text,re.S)
    if fence:
        try:
            obj=parse_all(fence.group(1))
            if not schema_valid(obj,task):return bad('invalid_fenced_schema')
            return dict(accepted=True,strict=False,mode='FENCED_JSON',value=obj,error=None)
        except DuplicateKey:return bad('duplicate_key')
        except (ValueError,TypeError):return bad('invalid_fenced_json')
    objects=[];end=0
    for m in re.finditer(r'[\{\[]',text):
        start=m.start()
        if start<end:continue
        try:
            obj,n=DECODER.raw_decode(text[start:]);end=start+n;objects.append((obj,start,end))
        except DuplicateKey:return bad('duplicate_key')
        except (ValueError,TypeError):continue
    if len(objects)!=1:return bad('missing_or_multiple_json_values')
    obj,start,end=objects[0]
    if not schema_valid(obj,task):return bad('invalid_schema')
    return dict(accepted=True,strict=False,mode='EXTRACTED_UNIQUE_JSON',value=obj,error=None,span=[start,end])
def decide_event(quality,pairs,expected_aliases,pool_complete=True):
    def result(kind,reason,target=None,same=None):
        return dict(kind=kind,reason_code=reason,target_alias=target,same_aliases=same or [],applied_to_map=False)
    if quality is None or not schema_valid(quality,'quality'):return result('PENDING','QUALITY_PARSE_OR_API_FAILURE')
    if quality['status']!='USABLE':return result('PENDING','QUALITY_'+quality['status'])
    if not pool_complete or not expected_aliases:return result('PENDING','CANDIDATE_POOL_INCOMPLETE_OR_EMPTY')
    if len(expected_aliases)!=len(set(expected_aliases)):return result('PENDING','DUPLICATE_EXPECTED_ALIAS')
    aliases=[p['alias'] for p in pairs]
    if len(aliases)!=len(set(aliases)) or set(aliases)!=set(expected_aliases):return result('PENDING','PAIRWISE_MISSING_OR_DUPLICATE')
    if any(not schema_valid(p.get('value'),'pairwise') for p in pairs):return result('PENDING','PAIRWISE_PARSE_OR_API_FAILURE')
    same=[p['alias'] for p in pairs if p['value']['choice']=='SAME']
    if len(same)>1:return result('MERGE_REVIEW','MULTIPLE_SAME_REQUIRES_NODE_CHECK',same=same)
    # Referenced table: one SAME associates even if another received answer is UNCERTAIN.
    if len(same)==1:return result('ASSOCIATE','UNIQUE_SAME',target=same[0],same=same)
    if all(p['value']['choice']=='DIFFERENT' for p in pairs):return result('NEW','ALL_DISPLAYED_CANDIDATES_DIFFERENT')
    return result('PENDING','NO_SAME_WITH_UNCERTAIN')
