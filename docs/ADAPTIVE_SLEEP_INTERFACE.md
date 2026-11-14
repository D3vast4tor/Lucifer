# Lucifer adaptive Sleep interface

Ordinary `/sleep start` remains a one-shot job. Start an adaptive job explicitly:

```text
/sleep start --adaptive --thinking max --generation-model lucifer
/sleep start --adaptive --seed 19 --dry-run
/sleep resume SLEEP_RUN_ID
```

Adaptive jobs request 50 optimizer updates initially and refresh their data every
50 optimizer updates. A seed fixes randomized schedule choices. A resumed job
retains its checkpoint and source bindings; resuming does not silently opt a
conversation into training or activate an adapter.

The API accepts an optional `schedule` on `POST /sleep/start`:

```json
{
  "schedule": {
    "mode": "until_admitted",
    "refresh_optimizer_steps": 50,
    "initial_update_budget": 50,
    "seed": 19
  },
  "generation_model": "lucifer",
  "generation_thinking": "max"
}
```

The two update counts are fixed by this schedule. Omitting `schedule` retains the
existing pipeline. Teacher reasoning remains transient; training artifacts
contain verified final targets only. The terminal defaults an adaptive request
to maximum teacher reasoning; the API also normalizes an adaptive request's
omitted or `off` value to `max`. Other adaptive teacher levels are rejected.
The backward-compatible one-shot default remains `off`.

`POST /sleep/runs/{run_id}/resume` requires an `idempotency_key`. Repeating a lost
request uses the same key. The terminal retains an unacknowledged resume key;
`/sleep resume ID --idempotency-key KEY` also permits an explicit retry.

The Sleep view receives adaptive epoch, update budget, cumulative optimizer
updates, data passes, next refresh and admission gates through the existing
telemetry heartbeat. It displays the current adjustment as
`3 + RANDINT(randint_min, randint_max)`, using the persisted integer bounds;
a separate round index appears when available. Both bounds grow after each
completed round and reset with the budget at an adaptive-epoch boundary.
See [the complete growing-bounds schedule](ADAPTIVE_SLEEP.md#adaptive-rounds-and-growing-random-bounds).
Unknown counters remain unknown. Failed gates show their recorded reasons.

An `admitted` adapter has met its admission checks. Admission does not activate
it. Select it explicitly using the compatible loaded Lucifer profile; selection
does not switch models. The Cortex shows admitted artifacts and their measured
gates. Source revocation or invalidation continues to prevent reuse. Native
adapter compatibility remains a runtime requirement: a checkpoint for one
architecture cannot be attached to another by renaming it.

Lucifer is the public model and application identity. Checkpoint names, runtime
identifiers and licenses remain accurate in technical provenance.
