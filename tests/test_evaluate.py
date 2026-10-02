import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import evaluate


class EvaluationTests(unittest.TestCase):
    def test_avg4_and_invalid_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = {**evaluate.SETTINGS, 'num_problems': 30, 'average_at_n': 30,
                     'results': [{'generations': [{'correct': True}] +
                                 [{'correct': False}] * 3} for _ in range(30)]}
            for name in evaluate.DATASETS:
                (Path(tmp) / (name + '.json')).write_text(json.dumps(value))
            self.assertEqual(evaluate.summarize(tmp)['mean'], 25)
            for bad in ({'num_problems': 29}, {'average_at_n': 120}, {'val_n': 1}):
                (Path(tmp) / 'aime24.json').write_text(json.dumps({**value, **bad}))
                with self.assertRaises(ValueError):
                    evaluate.summarize(tmp)

    def test_adapter_is_required(self):
        path = Path('/example/adapter')
        evaluate.require_adapter(SimpleNamespace(lora_path=str(path)), path)
        for request in (None, SimpleNamespace(lora_path='/wrong/adapter')):
            with self.assertRaises(RuntimeError):
                evaluate.require_adapter(request, path)

    def test_original_answer_scoring(self):
        # Execute the actual scoring functions without importing the GPU runtime.
        from math_verify import parse, verify
        tree = ast.parse((evaluate.ROOT / 'evaluation/official_opsd.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ('extract_boxed_answer', 'grade_answer')]
        scope = {'parse': parse, 'verify': verify}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'scoring', 'exec'), scope)
        extract, grade = scope['extract_boxed_answer'], scope['grade_answer']
        self.assertEqual(extract(r'First \boxed{1}, finally \boxed{2}'), '2')
        self.assertTrue(grade(extract(r'\boxed{2}'), '2'))
        self.assertFalse(grade(extract('2'), '2'))
        self.assertFalse(grade('3', '2'))

    def test_download_all_sizes(self):
        for size in evaluate.SIZES:
            model = json.loads((evaluate.ROOT / 'evaluation/models.json').read_text())[size]
            with patch('huggingface_hub.snapshot_download', return_value='/cache/snapshot') as download, \
                 patch.object(evaluate, 'sha256', return_value=model['adapter_sha256']), \
                 patch('sys.argv', ['evaluate.py', '--size', size, '--download-only']):
                evaluate.main()
            self.assertEqual(download.call_args_list[0].args, ('starrylay/RC-OPD',))
            self.assertEqual(download.call_args_list[1].args, (model['base_model'],))
            self.assertEqual(download.call_args_list[0].kwargs['allow_patterns'], [
                f'RC-OPD-{size}/adapter_config.json', f'RC-OPD-{size}/adapter_model.safetensors'])

    def test_each_benchmark_runs_in_separate_process(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('sys.argv', ['evaluate.py', '--checkpoint', '/adapter', '--model-path', '/base', '--output', tmp]), \
             patch.object(evaluate.subprocess, 'run') as run, \
             patch.object(evaluate, 'summarize', return_value={}):
            evaluate.main()
        self.assertEqual(run.call_count, 3)
        for call, name in zip(run.call_args_list, evaluate.DATASETS):
            command = call.args[0]
            self.assertEqual(command[command.index('--dataset') + 1], name)
            self.assertEqual(command[command.index('--checkpoint') + 1], '/adapter')
            self.assertTrue(call.kwargs['check'])


if __name__ == '__main__':
    unittest.main()
