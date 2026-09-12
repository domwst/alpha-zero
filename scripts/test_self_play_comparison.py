import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from contextlib import ExitStack

from experiment_io import read, write
from run_self_play_comparison import run, battle_command


class HandoffTests(unittest.TestCase):
    def exercise(self, fails):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp); selfplay=root/'selfplay';selfplay.mkdir()
            write(root/'jobs/job.json',{'stage':'running'})
            plan={'schema_version':1,'id':'match','title':'Test match','repository':str(root),
                  'self_play_root':str(root),'self_play_dir':str(selfplay),'self_play_id':'job',
                  'supervisor_script':'supervisor.py','supervisor_sha256':'sha',
                  'supervisor':{'pid':101,'argv':['python3','supervisor.py']}, 'worker':{'pid':102},
                  'binary':'trainer','first':{'path':'first'},'second':{'path':'second'},
                  'settings':{'games':1000,'simulations':4000,'temperature':.7,'parallelism':500,
                              'inference_batch_size':128,'seed':1,'batch_timeout_us':1000,'heartbeat_seconds':30}}
            write(root/'plan.json',plan)
            write(root/'result.partial.json',{'config':{'games_parallelism':500,'inference_batch_size':128,'seed':1}})
            battle=Mock(pid=200,returncode=1 if fails else 0);battle.poll.return_value=battle.returncode
            resumed=Mock(pid=300);resumed.poll.return_value=None
            popen=stack.enter_context(patch('run_self_play_comparison.subprocess.Popen',side_effect=[battle,resumed]))
            for name in ['validate_plan','wait_gone','validate_battle','stop']:
                stack.enter_context(patch('run_self_play_comparison.'+name))
            stack.enter_context(patch('run_self_play_comparison.digest',return_value='sha'))
            stack.enter_context(patch('run_self_play_comparison.os.kill'))
            stack.enter_context(patch('run_self_play_comparison.time.sleep'))
            stack.enter_context(patch('run_self_play_comparison.resource_sample',return_value={
                'working_bytes':0,'memory_limit':100,'gpu_memory_mib':0}))
            if fails:
                with self.assertRaisesRegex(RuntimeError,'Battle failed'):
                    run(root/'plan.json')
            else:
                run(root/'plan.json')
                self.assertTrue((root/'result.json').exists())
            self.assertEqual(popen.call_args_list[1].args[0],plan['supervisor']['argv'])
            status=read(root/'status.json')
            self.assertEqual(status['stage'],'failed' if fails else 'completed')
            self.assertEqual(status['resume']['pid'],300)
            argv=battle_command(plan,root/'result.json')
            self.assertEqual(argv[argv.index('--games-parallelism')+1],'500')

    def test_success_resumes_original_supervisor(self):
        self.exercise(False)

    def test_failed_comparison_still_resumes_original_supervisor(self):
        self.exercise(True)

class DashboardComparisonTests(unittest.TestCase):
    def test_new_identity_displays_without_modifying_historical_recipe_plan(self):
        from additional_comparisons import append_additional_comparisons
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);queue=root/'queue';job=root/'match'
            agent=Mock(queue=queue)
            plan={'id':'new-job','title':'New match','first':{'model':{'architecture':'one'}},
                  'second':{'model':{'architecture':'two'}},'settings':{'games':1000},
                  'metadata':{'participants':{'first':{'label':'Latest'},'second':{'label':'Best'}}}}
            values={queue/'additional-comparisons.json':[{'directory':str(job)}],job/'plan.json':plan,
                    job/'status.json':{'stage':'running'}}
            agent.json.side_effect=lambda path,default=None: values.get(path,default)
            agent.battle.return_value={'id':'new-job','title':'New match','kind':'battle','result':None}
            with patch('additional_comparisons.locked',return_value=True):
                result=append_additional_comparisons(agent,{'phases':[]})
            self.assertEqual(result['phases'][0]['participants']['first']['label'],'Latest')
            self.assertEqual(agent.battle.call_args.args[4],'running')
