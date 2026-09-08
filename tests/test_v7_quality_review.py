import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from conceptgraph.slam.v7_runtime import V7Runtime
from conceptgraph.slam.human_instance_merge import object_state
from test_v7_split import obj

class QualityReview(unittest.TestCase):
    def run_case(self,answer):
        with tempfile.TemporaryDirectory() as tmp:
            r=V7Runtime.__new__(V7Runtime);r.root=Path(tmp);r.rows={};r.fallback='human';r.publish=lambda:None
            d=r.root/'f000001_d000_create';d.mkdir();(d/'quality.jpg').write_bytes(b'x')
            candidate=obj('a');r.staged_bindings={d.name:dict(images=[dict(label='QUALITY',path='quality.jpg')],objects=[object_state(candidate)])}
            calls=[];questions=[]
            def stage(e,d,task,images,snapshot):
                calls.append(task);return dict(value={'status':'UNUSABLE','reason':'unclear'} if task=='observation_quality' else {'choice':'SAME','confidence':5})
            r.stage=stage
            def human(e,d,allowed,images,reason,snapshot,**kw):questions.append((allowed,kw));return answer
            r.human_choice=human
            decision,_,_=r.adjudicate(d,[('A',0,candidate)],'snapshot',1,'f1')
            return decision,calls,questions
    def test_usable_continues_identity(self):
        d,c,q=self.run_case('USABLE');self.assertEqual(d['choice'],'A');self.assertEqual(c,['observation_quality','pairwise']);self.assertEqual(q[0][0],['USABLE','UNUSABLE']);self.assertEqual(q[0][1]['question_type'],'observation_quality')
    def test_unusable_drops_without_candidate_question(self):
        d,c,q=self.run_case('UNUSABLE');self.assertEqual(d['choice'],'DISCARD');self.assertEqual(c,['observation_quality']);self.assertEqual(len(q),1)

if __name__=='__main__':unittest.main()
