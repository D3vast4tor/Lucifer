# Canonical Lucifer paths

The project and operational environment live below `/home/Lucifer`:

- Application code and documentation: `/home/Lucifer/Soul`.
- Serving/training Python environment: `/home/Lucifer/vLLM/Lucifer`.
- Conversations, adapters, datasets, jobs and logs: `/home/Lucifer/Soul/var/state/runtime`.
- Cached Qwen model, including its original blobs and snapshot layout: `/home/Lucifer/models/Qwen3.5-4B`.
- Primary Strata model and runtime: `/home/Lucifer/Soul/var/strata`.

`scripts/lucifer` and `scripts/vllm` locate their interpreters relative to the
project. The vLLM launcher deliberately uses `python -m` so an old console-script
shebang cannot prevent model startup.

## Offline migration

Run the read-only report first:

```sh
/home/Lucifer/Soul/scripts/migrate-paths
```

After the backend and its owned model/training processes have stopped, apply the
same plan:

```sh
/home/Lucifer/Soul/scripts/migrate-paths --apply
```

The command refuses to apply while the backend holds its data-directory lock.
It never stops a process. Nonempty destination directories are never merged or
overwritten. A genuinely empty destination created by startup defaults can be
removed atomically before the rename. Cross-filesystem moves are refused before removing the source;
they require a separately verified copy.

Before relocation, SQLite's backup API saves a consistent database under
`/home/Lucifer/Soul/var/state/migration-backups/TIMESTAMP`. Modified configuration
and console-entrypoint files are backed up there too. The JSON plan and result
record all moves and changed operational field names without exporting chat
content.

The application data directory and Qwen cache repository are renamed into their
canonical locations. Narrow compatibility symlinks remain at the former
application-data and Qwen-cache paths. All actual bytes are under `/home/Lucifer`;
these links keep immutable historical file references usable. No compatibility
link is created at `/home/d3vast4tor/Lucifer` or the former vLLM directory.

The command updates current model paths and nested serving specs, current adapter
locations, dataset location pointers, the model YAML's path fields, console-script
interpreter lines and virtual-environment activation paths. In particular, the
persisted `lucifer-train` executable uses the canonical dynamic
`/home/Lucifer/Soul/scripts/vllm` launcher.

The four exact dangling user-service links for `lucifer-local-search.service`
and `lucifer-post-training.service`, including their `default.target.wants`
counterparts, are recorded in the backup plan and removed only when they still
point to the removed repository. Other units and symlinks are preserved. The
command never reloads systemd or starts/stops units.

Historical transcripts, raw logs, agent checkpoints, evaluation/admission
bindings, adaptive receipts, artifact configuration, provenance and hashes remain
unchanged. Updating their embedded paths would invalidate recorded identities.
Their old application-data/model paths resolve through the narrow links instead.
Older agent checkpoints that explicitly pinned the removed repository workspace
remain historical; open a new mission in `/home/Lucifer/Soul` to execute there.

The operation is repeatable: a completed migration reports no further moves or
field rewrites. If an interrupted migration needs recovery, inspect its saved
plan and backup with the backend stopped before restoring files or renaming
directories. The command does not silently merge a partial destination.
