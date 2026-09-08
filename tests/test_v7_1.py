import copy,json,unittest
from unittest.mock import patch
import httpx
from test_v7_split import GateIntegration
from test_v7_image_parallel import Stages,response
from conceptgraph.slam.v7_1_review import unique_owner,validate_joint

class UnknownGate(GateIntegration):
    def test_unknown_does_not_lock_and_waits_for_new_history(self):
        self.quality='CONTAMINATED';self.review(1);n=len(self.calls)
        for frame in (2,3):self.review(frame)
        row=self.gate.votes.state(self.gate.votes.key('a','b'))
        self.assertFalse(row['locked']);self.assertEqual(row['reject_total'],0)
        self.assertEqual(len(self.calls),n)
        self.a['obs_uids'].append('a-h2');self.quality='CLEAN';self.answer='SAME'
        self.review(4);self.assertEqual(row['merge_streak'],1)
        self.review(5);self.assertTrue(row['awaiting_execution'])
    def test_containment_conflict_is_not_negative_vote(self):
        self.b['pcd'].points=self.a['pcd'].points.copy();self.review(1)
        row=self.gate.votes.state(self.gate.votes.key('a','b'))
        self.assertEqual(row['reject_total'],0);self.assertEqual(len(self.humans),0)
    def test_unknown_interrupts_positive_streak(self):
        self.answer='SAME';self.review(1)
        self.answer='UNCERTAIN';self.review(2)
        row=self.gate.votes.state(self.gate.votes.key('a','b'))
        self.assertEqual(row['merge_streak'],0);self.assertEqual(row['reject_total'],0)

class CompletionRecovery(Stages):
    def test_length_recovers_same_H_once(self):
        with patch('httpx.Client') as c:
            c.return_value.__enter__.return_value.post.side_effect=[response(reason='length'),response()]
            r=self.r.stage('e',self.root/'A','pairwise',[('A',self.image)],'H')
        self.assertEqual(r['value']['choice'],'SAME');self.assertEqual(len(r['attempts']),2)
        self.assertEqual([a['num_predict'] for a in r['attempts']],[1024,2048])
        self.assertEqual(len(self.r.rows['e']['stage_results']),1)
        for p in (self.root/'A/attempts').glob('*/attempt.json'):
            self.assertEqual(json.loads(p.read_text())['h_snapshot_uid'],'H')
    def test_timeouts_shared_across_length_recovery(self):
        with patch('httpx.Client') as c:
            c.return_value.__enter__.return_value.post.side_effect=[httpx.ReadTimeout('x'),response(reason='length')]+[httpx.ReadTimeout('x')]*3+[response()]
            r=self.r.stage('e',self.root/'A','pairwise',[('A',self.image)],'H')
            self.assertEqual(c.return_value.__enter__.return_value.post.call_count,5)
        self.assertIsNone(r['value']);self.assertEqual(r['timeout_count'],4)

class Joint(unittest.TestCase):
    def good(self):
        return dict(current_status='USABLE',choice='A',candidates=[dict(alias=a,relation='SAME' if a=='A' else 'DIFFERENT',node_quality='CLEAN',evidence='visible independent structure') for a in 'ABC'],reason='unique owner')
    def test_requires_unique_clean_original_same(self):
        x=self.good();self.assertEqual(unique_owner(x,list('ABC'),['A','B']),'A')
        self.assertIsNone(unique_owner(x,list('ABC'),['B','C']))
        for field,value in [('relation','UNCERTAIN'),('relation','SAME')]:
            y=copy.deepcopy(x);y['candidates'][1][field]=value
            with self.assertRaises(ValueError):validate_joint(y,list('ABC'))
        x['candidates'][0]['node_quality']='CONTAMINATED'
        with self.assertRaises(ValueError):validate_joint(x,list('ABC'))
    def test_alias_and_current_invariants(self):
        for mutate in [lambda x:x['candidates'].reverse(),lambda x:x.update(current_status='INSUFFICIENT'),lambda x:x['candidates'].pop()]:
            x=self.good();mutate(x)
            with self.assertRaises(ValueError):validate_joint(x,list('ABC'))
    def test_unresolved_never_selects_owner(self):
        x=self.good();x['choice']='UNRESOLVED';x['candidates'][1]['relation']='SAME'
        self.assertIsNone(unique_owner(x,list('ABC'),['A','B']))

class Granularity(unittest.TestCase):
    def test_positive_merge_pair_cannot_be_overridden(self):
        x=Joint().good()
        self.assertIsNone(unique_owner(x,list('ABC'),['A','B'],[['A','B']]))
    def test_part_whole_is_not_a_distinct_candidate(self):
        x=Joint().good();x['candidates'][1]['relation']='SAME_OBJECT_PART'
        with self.assertRaises(ValueError):validate_joint(x,list('ABC'))
        x['choice']='UNRESOLVED';validate_joint(x,list('ABC'))
