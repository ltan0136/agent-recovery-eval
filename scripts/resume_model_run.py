"""Continue a frozen local model run without discarding completed failures.

Use --one to checkpoint one new episode per process. This is for deterministic
local simulators: an unknown interrupted in-flight attempt is explicitly logged
and restarted, not presented as an observed model result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_eval.agents import LocalTransformersAdapter, run_agent
from recovery_eval.artifacts import finalize_run, read_jsonl, source_hashes, summarize, write_json
from recovery_eval.workflows import build_matrix, workflow_specs
from recovery_eval.engine import Environment
from scripts.reproduce_release import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--interruption-record', type=Path, required=True)
    parser.add_argument('--one', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    directory = args.out.resolve()
    if source == directory:
        parser.error('Preserve the interrupted source directory; choose a different output')
    original = json.loads((source/'manifest.json').read_text())
    if original.get('command') != 'model' or original.get('requested_backend') != 'local':
        parser.error('Only a saved local model CLI run is supported')
    if original.get('status') == 'completed':
        parser.error('Source is already complete')
    if original.get('code_sha256') != source_hashes():
        parser.error('Current package source differs from the frozen original')
    if (original.get('variants'),original.get('seeds'),original.get('conditions'),original.get('prompt_variants')) != (1,1,['clean','lost_ack'],['basic','guarded']):
        parser.error('This recovery command supports the declared release protocol only')
    if (original.get('max_turns'),original.get('max_tool_calls'),original.get('max_new_tokens'),original.get('format_retries')) != (8,12,160,1):
        parser.error('Original execution budgets differ')
    workflows = original.get('requested_workflows')
    workflows = list(workflow_specs()) if workflows is None else workflows
    if (not isinstance(workflows, list) or len(workflows) != 4
            or not all(isinstance(name, str) for name in workflows)
            or set(workflows) != set(workflow_specs())):
        parser.error('The release protocol requires the complete four-workflow set')
    locked_models = {model: revision for _, model, revision in MODELS}
    if locked_models.get(original.get('requested_model')) != original.get('requested_revision'):
        parser.error('Model and revision do not match the declared release locks')
    jobs = [(case, prompt) for case in build_matrix(workflows, 1, 1, ['clean', 'lost_ack'])
            for prompt in ['basic', 'guarded']]
    if len(jobs) != 16:
        parser.error('The release protocol must contain exactly sixteen episode cells')
    original_rows = read_jsonl(source/'progress.jsonl')
    if not original_rows:
        parser.error('No complete episodes were preserved')
    interruption = json.loads(args.interruption_record.read_text())
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    origin_digest = hashlib.sha256((source/'progress.jsonl').read_bytes()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    # OS releases this advisory lock if the worker process disappears.
    try:
        import fcntl
    except ImportError:
        parser.error('Continuation currently requires a Unix host with fcntl')
    with (directory/'continuation.lock').open('a+') as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another continuation worker owns this directory')
        if any(path.name != 'continuation.lock' for path in directory.iterdir()):
            metadata = json.loads((directory/'manifest.json').read_text())
            if (not metadata.get('managed_continuation') or metadata.get('resumed_from') != str(source)
                    or metadata.get('original_progress_sha256') != origin_digest
                    or metadata.get('resume_script_sha256') != script_digest):
                parser.error('Existing output is not this exact managed continuation')
            if metadata.get('status') == 'completed':
                print(json.dumps({'status':'already_completed','episodes':metadata['completed_episodes']})); return
            rows = read_jsonl(directory/'progress.jsonl')
        else:
            rows = original_rows.copy()
            (directory/'progress.jsonl').write_bytes((source/'progress.jsonl').read_bytes())
            metadata = {**original, 'status':'running', 'managed_continuation':True,
                'resumed_from':str(source), 'original_progress_sha256':origin_digest,
                'resume_script_sha256':script_digest, 'continuation_started_utc':now,
                'preserved_original_episodes':len(original_rows), 'continuation_sessions':[],
                'infrastructure_interruption':interruption,
                'unobserved_inflight_attempt':'unknown; preserved complete failures are never removed; next unfinished simulator case restarts',
                'timing_note':'Wall elapsed includes interruption and model reloads; recorded model calls are a lower bound including only complete episode records.'}
        if rows[:len(original_rows)] != original_rows:
            parser.error('The original completed denominator was changed')
        summarize(rows)  # Reject duplicate agent/case records and mixed agent kinds.
        keys = [(r['case_id'],r['prompt_variant']) for r in rows]
        expected = [(c['case_id'],p) for c,p in jobs]
        if keys != expected[:len(keys)]:
            parser.error('Saved complete rows are not the exact frozen matrix prefix')
        # Recompute the expected frozen execution contract without executing a policy.
        # This validates saved evidence; it does not repeat an observed model attempt.
        for row, (case, prompt) in zip(rows, jobs):
            expected_row = Environment(case, max_calls=12).finish(False)
            for field in ('case_config', 'execution_config', 'case_hash', 'condition_hash', 'config_hash', 'public_task'):
                if row.get(field) != expected_row[field]:
                    parser.error('Preserved frozen episode configuration differs: ' + row['case_id'] + '/' + field)
            for field in ('case_id', 'workflow', 'fault', 'seed', 'variant_id'):
                if row.get(field) != case[field]:
                    parser.error('Preserved episode metadata differs from the frozen matrix')
        model = original['requested_model']; revision = original['requested_revision']
        for row in rows:
            if (row['agent_kind'] != 'local_model' or row['model'] != model
                    or row['agent_id'] != model + ':' + row['prompt_variant']
                    or row['model_backend'].get('requested_revision') != revision
                    or row['model_backend']['resolved_revision_sha'] != revision
                    or (row['max_turns'],row['max_tool_calls'],row['format_retry_limit']) != (8,12,1)):
                parser.error('Preserved row identity or protocol differs')
        session = {'started_utc':now, 'completed_before':len(rows), 'new_episodes':0}
        metadata['continuation_sessions'].append(session)
        write_json(directory/'manifest.json',metadata)
        adapter = LocalTransformersAdapter(model, revision=revision, cache_dir=args.cache_dir,
            max_new_tokens=160,threads=4,local_files_only=True)
        started=time.perf_counter()
        try:
            remaining = jobs[len(rows):]
            for case,prompt in remaining[:1] if args.one else remaining:
                row = run_agent(case,adapter,prompt_variant=prompt,max_turns=8,max_tool_calls=12,format_retries=1)
                with (directory/'progress.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                rows.append(row);session['new_episodes'] += 1
                print(json.dumps({'case_id':row['case_id'],'prompt':prompt,'success':row['success'],'completed':len(rows)}),flush=True)
            session.update({'finished_utc':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.perf_counter()-started})
            metadata.update({'completed_episodes':len(rows),'model_backend':adapter.describe()})
            if len(rows)==len(jobs):
                metadata.update({'status':'completed','duration_seconds':(
                    datetime.now(timezone.utc)-datetime.fromisoformat(metadata['created_utc'])).total_seconds()})
                finalize_run(rows,directory,metadata)
            else:
                write_json(directory/'manifest.json',metadata)
                print(json.dumps({'status':'checkpointed','remaining':len(jobs)-len(rows)}),flush=True)
        except BaseException as error:
            session.update({'exception_type':type(error).__name__,'duration_seconds':time.perf_counter()-started})
            metadata.update({'status':'interrupted','completed_episodes':len(rows)})
            write_json(directory/'manifest.json',metadata)
            raise


if __name__=='__main__':
    main()
