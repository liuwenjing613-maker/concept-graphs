import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('mod_launcher',ROOT/'scripts/run_blocking_association_gate.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)

class LaunchTests(unittest.TestCase):
    def test_selected_ports_reach_mapping_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene=root/'raw/room0';(scene/'results').mkdir(parents=True)
            for name in ['traj.txt','results/frame000000.jpg','results/depth000000.png']:(scene/name).touch()
            argv=['launch','--mode','vlm','--fallback','auto','--mask-weight','.75','--vlm-ports','11437','11464','--project-root',tmp,'--dataset-root',str(root/'raw'),'--output-root',str(root/'out'),'--no-web-link']
            with patch.object(sys,'argv',argv),patch.dict(os.environ,{'V7_VLM_URLS':'["http://wrong:1"]'}),patch.object(launcher.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
                self.assertEqual(launcher.main(),0)
            call=run.call_args
            self.assertEqual(json.loads(call.kwargs['env']['V7_VLM_URLS']),['http://127.0.0.1:11437','http://127.0.0.1:11464'])
            self.assertIn('clip_masked_weight=0.75',call.args[0])
            self.assertIn('start=0',call.args[0])
            self.assertEqual(call.kwargs['cwd'],ROOT)
            manifest=json.loads(next((root/'results/blocking_association_gate_v1/launches').glob('*.json')).read_text())
            self.assertIn('mask0p75',manifest['experiment_root'])
            self.assertEqual(manifest['clip_bbox_weight'],.25)
            self.assertEqual(manifest['vlm_urls'],json.loads(call.kwargs['env']['V7_VLM_URLS']))

if __name__=='__main__':unittest.main()
