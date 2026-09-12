# Historical experiment recipes

These scripts capture the activation, value-head, pooling, depth/width,
training-recipe, and board-mask experiments from September 2026. Their fixed
seeds, checkpoint/pass choices, queue identifiers, paths, build revisions, and
handoff assumptions are part of those particular experiments. They are retained
for auditing and reproducing the associated reports, rather than presented as
portable launchers for a new pod.

Python recipes can still be imported as `archive.<module>` with `scripts` on
`PYTHONPATH`, or invoked directly as `python3 scripts/archive/<name>.py`.
The regression tests remain under `scripts/test_*.py`. Shared record and resource
helpers now live outside this archive, so current tools do not import a historical
recipe just to read JSON, check a checkpoint, or monitor a process.

The shell scripts may reference retired hosts' directory layouts or an old
compatibility build. Inspect those requirements before reproducing an experiment.
Moving a script also changes its recorded source path/checksum: historical frozen
plans are not rewritten or made to appear compatible with different code. Keep
an old deployed workspace if an unfinished historical queue still needs it.

Current self-play supervisors, collectors, dashboard readers, archive tools,
and reusable diagnostics remain in the parent directory. See its README.
