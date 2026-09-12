from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment_agent import Experiments
from experiment_statistics import summarize_live_match


def event(game,first='first_checkpoint',winner='first_checkpoint',plies=9):
    second='second_checkpoint' if first=='first_checkpoint' else 'first_checkpoint'
    return f'battle game complete game={game} first_seat="{first}" second_seat="{second}" winner="{winner}" plies={plies}\n'


class LiveStatisticsTests(unittest.TestCase):
    def test_out_of_order_games_deduplicate_and_do_not_infer_unfinished_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'battle.log'
            log.write_text(event(40)+event(3,'second_checkpoint','first_checkpoint',20)+event(40)+event(6,winner='draw',plies=361)+event(7)[:-1])
            data=summarize_live_match(log,1000)
            self.assertEqual(data['games'],3)
            self.assertEqual(data['outcomes'],{'first_player_wins':1,'second_player_wins':1,'draws':1})
            self.assertEqual(data['seat_results']['first_checkpoint']['second']['wins'],1)
            self.assertEqual(data['lengths']['all']['frequencies'],[{'moves':9,'count':1},{'moves':20,'count':1},{'moves':361,'count':1}])
            self.assertTrue(data['partial'])
            self.assertIsNone(data['completion'])

    def test_restart_discards_previous_games_even_before_any_new_game_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'battle.log'
            log.write_text(event(20)+'using compute device device=Cuda(0)\n')
            self.assertIsNone(summarize_live_match(log,1000))
            with log.open('a') as f:f.write(event(2,winner='second_checkpoint',plies=17))
            self.assertEqual(summarize_live_match(log,1000)['games'],1)
            self.assertEqual(summarize_live_match(log,1000)['lengths']['all']['mean'],17)

    def test_invalid_events_and_missing_logs_are_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'battle.log'
            self.assertIsNone(summarize_live_match(log,1000))
            log.write_text(event(1000)+event(2,plies=362)+event(3,winner='unknown'))
            self.assertIsNone(summarize_live_match(log,1000))

    def test_agent_publishes_live_statistics_without_marking_result_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=root/'battle.log';log.write_text(event(2))
            agent=Experiments(root,root)
            with patch('experiment_agent.summarize_live_match',wraps=summarize_live_match) as summarize:
                for _ in range(2):
                    data=agent.battle('live','Live',root/'result.json',log,'running','a / b',{'games':1000})
                    self.assertIsNone(data['result'])
                    self.assertEqual(data['state'],'running')
                    self.assertEqual(data['live_statistics']['games'],1)
                self.assertEqual(summarize.call_count,1)
                with log.open('a') as f:f.write(event(5,plies=12))
                self.assertEqual(agent.battle('live','Live',root/'result.json',log,'running','a / b',{'games':1000})['live_statistics']['games'],2)
                self.assertEqual(summarize.call_count,2)


if __name__=='__main__':
    unittest.main()
