# Shared autonomous training and behavioral evaluation

User-authorized replacement scope, 14 November 2026. Applies to Sleep and
Meditation. The latest user instruction supersedes the previous requirement to
keep working until a statistically admitted new base is obtained.

## Completion evidence

Implemented and checked against the replacement stop condition. Evidence is in
`var/knowledge/agent-llm-curriculum/behavioral-pilot/completion-audit.json`.
The full regression run passed 1,156 checks with four skips; subsequent focused
controller checks passed 111 and page checks passed 54. The wheel includes the
master prompt and new controllers. Both pages were rendered against the actual
backend at wide and compact terminal sizes.

Actual supervised primary-head pilot `meditation_9d51150a67894687` completed one
retained optimizer update after 15 separate disposable calibration updates.
It changed 1,362,649 saved head elements, passed sampled numerical retention,
changed native logits and reproduced bit-identical native parent rollback. The
original parent head hash remained unchanged and the original serving profile
was restored.

The live test found that the bounded verifier rejected a correct `set.add`
implementation. That allowlist defect was fixed and both models' existing code
artifacts were rescored under the corrected checker, with no model or optimizer
replay and original journals preserved. The corrected diagnostic is **2/8 tasks
for each head**, zero gains and zero regressions. Six answer/tool-contract
failures per head limit what can be inferred about reasoning. No improvement,
base admission or activation is claimed. Sleep's new controller paths were
tested; this revision does not claim a new actual Sleep weight-training run.

## Requirements and completion evidence

1. Generate fresh, sealed evaluation episodes automatically before optimization.
   Cover novel rule inference, unfamiliar code repair with hidden tests,
   multi-step tool tasks with verified final state, and recovery/transfer under
   changed constraints. Evaluation episodes and oracle answers never become
   teacher prompts or training examples. Track family/seed/source exclusions and
   suite identity. Same generator author is not claimed to provide independent
   real-world repository evidence.
2. Score outcomes using trusted code, hidden tests and state invariants. The
   model chooses its implementation/action path within the task tools and fixed
   budget. No model self-grading or explanation-length reward. Execute proposed
   code only in the bounded verifier, and tool effects only in owned episode
   resources. Record fingerprints before effects and reuse completed effects.
3. Compare baseline and candidate with identical episodes, tool permissions,
   generation settings and budgets. Report solved tasks by category, gains,
   regressions, incomplete cases, loops/duplicate effects and elapsed time.
   Missing evidence cannot pass. A small pilot may prove the workflow while
   honestly rejecting promotion; it cannot establish universal intelligence.
4. Default new Sleep and Meditation jobs to this behavioral protocol. Keep
   numerical parity, retention, source consent, immutable artifacts and rollback
   controls. Preserve pre-existing legacy run histories rather than relabeling
   old likelihood scores as behavioral outcomes.
5. Automatically prepare needed teaching material, manifests and episode files
   from consented chats/documents and trusted generators. Loaded-model proposals
   may choose supported teaching topics/rules; an independent verifier controls
   target admission and evaluation truth. Unknown domains remain unsupported,
   never receive fabricated ground truth and cannot require Codex to approve
   ordinary verified material. Consent and unavailable hardware are real
   constraints, not reasons to silently opt in chats or invent results.
6. Ship and consume a shared master training prompt. Sleep produces compatible
   adapters/experts with replay; Meditation changes only the primary output head.
   Both exclude provider reasoning, respect source lineage and separate
   training/development/evaluation. The prompt cannot replace enforced checks.
7. Show meaningful job readings and a compact legend on both pages: completed
   work versus retained checkpoint, task success by category, baseline comparison,
   regressions, evaluation coverage/status, resource usage and restoration. Do
   not call a probability-loss decrease an intelligence score. Persist evaluation
   and rollback decisions even when large working files are pruned.
8. Test pure generators/oracles, forged or missing evidence, hidden-test
   isolation, resource limits, idempotency, both job integrations and UI rendering.
   Run a bounded actual-model pilot proving real training, behavioral evaluation
   and restoration. Inspect its receipts before marking the revised goal done.

## Implementation sequence

Complete the in-progress append-only resume and durable evaluation-history work;
implement the shared sealed task generators and bounded episode executor; wire
the master prompt and automatic resource preparation into both controllers;
make behavioral admission and reporting the default for new runs; update both
pages and legends; execute focused/combined tests and a small native pilot.

No automatic activation or mandatory human review is added. Behavioral tasks
are bounded to their owned workspace; downloaded instructions cannot grant
runtime permissions or change the verifiers. Public authored worked solutions
are distinguished from excluded provider reasoning.
