#!/usr/bin/env python3
"""Build authored public exercises using trusted local checks, not model traces.

Reference documents supply conceptual provenance. These are local teaching
fixtures, not upstream benchmark items or claims about an installed API version.
The builder does not run downloaded programs, contact the model, or start jobs.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, ValidationError

from lucifer.problem_solution import FORMAT, parse_record, source_family


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def serialized(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    count: int
    action: Literal["read", "inspect"]


def task(family, i):
    """Return public problem, checked answer and a bounded verifier receipt."""
    checks = {}
    if family == "bash":
        guard_passes, pipeline_check = bool(i % 2), bool(i // 2 % 2)
        guard = "true" if guard_passes else "false"
        if pipeline_check:
            guard += " | cat"
        # A deliberately failed guard must prevent the later effect. Exercise
        # both successful and failed paths; the only effect is our private file.
        code = "set -o pipefail\nif " + guard + "; then touch result; else exit 7; fi\n"
        with tempfile.TemporaryDirectory(prefix="lucifer-teaching-guard-") as folder:
            result = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-c", code], cwd=folder,
                                    capture_output=True, timeout=3)
            exists = (Path(folder) / "result").exists()
        assert exists == guard_passes and result.returncode == (0 if guard_passes else 7)
        question = (f"For local workflow case {i}, predict this Bash script's exit code and whether its effect file is created. "
                    "The directory starts empty. The commands true, false, cat and touch have their standard meanings.\n" + code)
        answer = {"exit_code": result.returncode, "effect_created": exists,
                  "explanation": "The if condition controls the effect explicitly. With pipefail, a failed stage makes the pipeline fail; the else branch exits before creating the file. A passing condition enters the effect branch."}
        checks = {"actual_exit_code": result.returncode, "actual_effect_created": exists}
    elif family in {"pydantic", "jsonschema"}:
        examples = [({"count": 3, "action": "read"}, True), ({"count": "3", "action": "read"}, False),
                    ({"count": True, "action": "inspect"}, False), ({"action": "read"}, False),
                    ({"count": 4, "action": "delete"}, False), ({"count": 0, "action": "inspect"}, True),
                    ({"count": 2, "action": "read", "secret": "extra"}, False),
                    ({"count": -2, "action": "inspect"}, True)]
        instance, expected = examples[i]
        schema = {"type": "object", "properties": {"count": {"type": "integer"},
                  "action": {"enum": ["read", "inspect"]}}, "required": ["count", "action"], "additionalProperties": False}
        if family == "pydantic":
            try:
                Command.model_validate(instance)
                valid = True
            except ValidationError:
                valid = False
            contract = "Pydantic model: strict=True, extra='forbid'; required count:int and action:Literal['read','inspect']."
        else:
            valid = jsonschema.Draft202012Validator(schema).is_valid(instance)
            contract = "JSON Schema (draft 2020-12): " + serialized(schema)
        assert valid is expected
        question = f"Validate local {family} request case {i}. {contract} Instance: {serialized(instance)}. Does it satisfy the stated contract?"
        answer = {"valid": valid, "explanation": "Check required fields, integer type, the action enumeration and forbidden extra properties separately. The contract permits negative integers; it does not permit a string or boolean in the integer field."}
        checks = {"actual_library_accepts": valid, "independent_table_expected": expected}
    elif family == "accelerate":
        counts = [3 + i, 7 + i, 2 + i]
        # Per-sample losses allow a direct oracle, separate from the weighted
        # microbatch-mean calculation being taught.
        groups = [[Fraction(j % 5 + 1) for j in range(n)] for n in counts]
        means = [sum(group) / len(group) for group in groups]
        direct = sum((group for group in groups), [])
        oracle = sum(direct) / len(direct)
        weighted = sum(mean * n for mean, n in zip(means, counts)) / sum(counts)
        assert weighted == oracle
        question = (f"A single-device training step accumulates three unequal microbatches in case {i}. "
                    f"Sample counts are {counts}; their mean losses are {[str(x) for x in means]}. "
                    "Use a per-sample mean objective, with no other loss scaling. Return total samples and the exact mean loss as numerator/denominator.")
        answer = {"samples": len(direct), "loss_fraction": str(oracle),
                  "explanation": "Multiply each microbatch mean by its sample count, sum those loss totals, and divide by the total number of samples. Equal weighting of unequal microbatch means would change the objective. Device count is one in this problem."}
        checks = {"direct_per_sample_mean": str(oracle), "weighted_microbatch_mean": str(weighted)}
    elif family == "transformers":
        layers, sequence, kvheads, dim = 4 + i, 48 + i * 8, 2 + i % 3, 16
        # Small enumerated array sizes validate the formula without allocating
        # a model or making assumptions about Lucifer's compressed MLA cache.
        one_layer = bytearray(sequence * kvheads * dim * 2)
        oracle = sum(len(one_layer) + len(one_layer) for _ in range(layers))
        expected = 2 * layers * sequence * kvheads * dim * 2
        assert oracle == expected
        question = (f"For a hypothetical ordinary uncompressed GQA model in case {i}, batch=1, layers={layers}, sequence={sequence}, "
                    f"KV heads={kvheads}, dimension per head={dim}, and BF16=2 bytes. Store both K and V, with no padding or metadata. "
                    "Compute cache bytes. State whether this formula alone determines the size of a compressed MLA cache.")
        answer = {"cache_bytes": oracle, "determines_compressed_mla_size": False,
                  "explanation": "Count both key and value arrays, then multiply layers, tokens, KV heads, dimensions and bytes per element. Compression, latent dimensions and additional architectural buffers need their own layout; the ordinary GQA result cannot simply be assigned to MLA."}
        checks = {"enumerated_array_bytes": oracle, "formula_bytes": expected}
    elif family == "mcp":
        schema = {"type": "object", "properties": {"query": {"type": "string"}},
                  "required": ["query"], "additionalProperties": False}
        arguments = {"query": "cache design"} if i % 2 == 0 else {"query": 123}
        allowed = bool(i % 3)
        validates = jsonschema.Draft202012Validator(schema).is_valid(arguments)
        expected = "call" if validates and allowed else "repair_arguments" if not validates else "request_access"
        question = (f"Local tools-client case {i}: a discovered tool has inputSchema {serialized(schema)}. "
                    f"Proposed arguments: {serialized(arguments)}. The user's runtime access policy permits this tool: {allowed}. "
                    "The tool has not run. Under the stated local policy, invalid arguments must be repaired first, then access checked. "
                    "Choose next_action from call, repair_arguments, request_access. Discovery alone is not permission.")
        answer = {"next_action": expected,
                  "explanation": "Validate the actual arguments against inputSchema before execution, then enforce the user's local access policy. Discovery and tool metadata describe capabilities; neither silently grants permission. No mutation or remote call is performed while deciding."}
        checks = {"schema_valid": validates, "runtime_allowed": allowed, "action": expected}
    elif family == "smolagents":
        completed = {0: ["inspect", "patch"], 1: ["inspect"], 2: ["inspect", "patch", "test"]}[i]
        plan = ["inspect", "patch", "test"]
        pending = [step for step in plan if step not in completed]
        next_step = pending[0] if pending else "finish"
        question = (f"A sandboxed code-agent teaching workflow has ordered steps {plan}. Durable completed results exist for {completed}. "
                    "Dependencies require that order; no failure or unknown side effect is recorded. What is the next step, and should completed actions be repeated?")
        answer = {"next_step": next_step, "repeat_completed": False,
                  "explanation": "Use the completed evidence to advance through the dependency order. Run only the first unfinished step. If every required result exists, finish; repeating completed effects adds risk without supplying missing evidence."}
        checks = {"pending_steps": pending, "next_step": next_step}
    elif family == "pytorch":
        # This constant program is authored here. No downloaded code is run.
        program = ("import json,torch; x=torch.tensor(2.,requires_grad=True); "
                   "loss=x*x; loss.backward(retain_graph=True); first=x.grad.item(); "
                   + ("x.grad=None; " if i == 1 else "")
                   + "loss.backward(); print(json.dumps({'first':first,'second':x.grad.item()}))")
        python = Path(__file__).resolve().parents[1] / ".train-venv/bin/python"
        result = subprocess.run([str(python), "-c", program], text=True, capture_output=True, check=True, timeout=30)
        values = json.loads(result.stdout)
        assert values == {"first": 4.0, "second": 4.0 if i == 1 else 8.0}
        cleared = i == 1
        question = (f"CPU PyTorch autograd case {i}: x=2.0 requires gradients; loss=x*x. "
                    f"Call loss.backward(retain_graph=True), then {'set x.grad=None' if cleared else 'leave x.grad unchanged'}, "
                    "then call loss.backward() again. No optimizer step runs. Return the first and second x.grad values and whether x itself changed.")
        answer = {"first_gradient": values["first"], "second_gradient": values["second"], "parameter_changed": False,
                  "explanation": "The derivative at x=2 is 4. Backward accumulates into an existing gradient, while clearing it starts a fresh accumulation. Backward alone does not perform an optimizer update, so the parameter stays at its original value."}
        checks = values
    elif family == "langgraph":
        state = ("completed", "started_unknown", "not_started")[i]
        choice = ("reuse_receipt", "reconcile_effect", "execute_once")[i]
        question = (f"A local durable agent resumes an external-effect node. Its checkpoint status is {state}. "
                    "For completed, a verified result receipt exists; for started_unknown, the effect may have occurred but no result exists; "
                    "for not_started, there was no attempt. The effect is non-idempotent. Choose reuse_receipt, reconcile_effect or execute_once. "
                    "Does restoring the checkpoint alone prove exactly-once execution of an external effect?")
        answer = {"next_action": choice, "checkpoint_alone_proves_exactly_once": False,
                  "explanation": "Use a verified completed receipt, inspect an uncertain effect before retrying, or execute a genuinely unstarted operation once. Restoring local state does not by itself establish what happened in an external system during a disconnect."}
        checks = {"state": state, "expected_transition": choice, "external_effects_executed_by_verifier": 0}
    elif family == "autogen":
        dependencies = ({"research": [], "write": ["research"], "test": ["write"]},
                        {"read_a": [], "read_b": [], "combine": ["read_a", "read_b"]},
                        {"planner": ["worker"], "worker": ["planner"]})[i]
        try:
            order = list(TopologicalSorter(dependencies).static_order())
            cyclic = False
            seen = set()
            for node in order:
                assert set(dependencies[node]) <= seen
                seen.add(node)
        except CycleError:
            cyclic, order = True, []
        question = ("A local multi-agent workflow has prerequisite lists " + serialized(dependencies)
                    + ". No task is completed. Return an order that respects all prerequisites, or an empty list if a dependency cycle prevents execution. "
                    "Return cyclic as well. Ties use the insertion order shown; do not start agents in a cycle.")
        answer = {"cyclic": cyclic, "order": order,
                  "explanation": "A task becomes runnable only after all prerequisites finish. Independent reads can precede their combination. A dependency cycle has no valid starting task and requires replanning rather than more delegation or endless waiting."}
        checks = {"stdlib_topological_order": order, "cycle_detected": cyclic}
    elif family == "peft":
        m, n, rank = 24 + i * 8, 16 + i * 4, 2 + i
        a, b = [[0] * n for _ in range(rank)], [[0] * rank for _ in range(m)]
        oracle = sum(map(len, a)) + sum(map(len, b))
        assert oracle == rank * (m + n)
        question = (f"A hypothetical LoRA update to one frozen dense matrix has output size {m}, input size {n} and rank {rank}. "
                    "Only A[rank,input] and B[output,rank] are trainable; ignore bias and all other layers. Return trainable parameter count, "
                    "original dense parameter count and whether the original dense entries receive optimizer updates.")
        answer = {"trainable_parameters": oracle, "dense_parameters": m * n, "original_dense_updated": False,
                  "explanation": "Count the two trainable low-rank factors separately and add them. A frozen original matrix does not receive optimizer updates. This exercise describes LoRA factors; it does not imply that Lucifer's native head-only Meditation trains or serves LoRA adapters."}
        checks = {"enumerated_factor_entries": oracle, "formula_count": rank * (m + n)}
    elif family == "vllm":
        tokens, block, available = (37, 16, 4) if i == 0 else (65, 16, 4) if i == 1 else (32, 16, 2)
        pages = [list(range(start, min(start + block, tokens))) for start in range(0, tokens, block)]
        needed = len(pages)
        assert sum(map(len, pages)) == tokens and needed == (tokens + block - 1) // block
        question = (f"An idealized paged KV-cache allocator has blocks of {block} tokens and {available} free blocks. "
                    f"A new sequence needs space for {tokens} tokens. No sharing, eviction, metadata or speculative allocation applies. "
                    "Return blocks_needed and fits. Count a partially filled last block as a full allocated block; this is a toy allocation contract, not an exact vLLM memory estimate.")
        answer = {"blocks_needed": needed, "fits": needed <= available,
                  "explanation": "Divide tokens into fixed-capacity blocks and round the number of blocks upward. Compare that allocation with the number of free blocks. Tokens fitting into a partial final block do not permit omitting its allocation."}
        checks = {"enumerated_pages": needed, "covered_tokens": sum(map(len, pages))}
    elif family == "sglang":
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}
        value = ({"answer": 9}, {"answer": "9"}, {"answer": 12})[i]
        supplied_evidence = 9
        valid = jsonschema.Draft202012Validator(schema).is_valid(value)
        factual = type(value["answer"]) is int and value["answer"] == supplied_evidence
        question = (f"A structured-generation response is {serialized(value)} under schema {serialized(schema)}. "
                    f"The independently supplied correct answer is {supplied_evidence}. Return schema_valid and matches_evidence. "
                    "Does satisfying a JSON schema alone prove a factual answer correct?")
        answer = {"schema_valid": valid, "matches_evidence": factual, "schema_alone_proves_truth": False,
                  "explanation": "Check the object's syntax and field types against the schema separately from comparing its content to evidence. A well-typed integer may still be the wrong integer; constrained output format cannot replace factual or semantic validation."}
        checks = {"actual_jsonschema_result": valid, "independent_evidence_match": factual}
    elif family == "llamacpp":
        elements, bits = 1024 + i * 256, 4 + i
        packed = (elements * bits + 7) // 8
        original = elements * 2
        assert packed == len(bytearray((elements * bits + 7) // 8))
        question = (f"A toy bit-packed tensor has {elements} elements at {bits} bits each. Its original representation is BF16, two bytes each. "
                    "Ignore scales, block metadata, headers and alignment. Return packed_bytes and original_bytes. "
                    "Would this toy formula alone give an exact real GGUF model file size?")
        answer = {"packed_bytes": packed, "original_bytes": original, "exact_gguf_size_proven": False,
                  "explanation": "Multiply elements by bits, then round upward to whole bytes. The BF16 reference uses two bytes per element. Real quantized formats add format-specific block data and file metadata; a theoretical payload estimate is not an exact model-file measurement."}
        checks = {"packed_allocation_bytes": packed, "bf16_payload_bytes": original}
    elif family == "llamaindex":
        evidence = ([{"source": "A", "value": 7}, {"source": "B", "value": 7}],
                    [{"source": "C", "value": 4}, {"source": "D", "value": 9}],
                    [])[i]
        values = {row["value"] for row in evidence}
        status = "unsupported" if not values else "conflicting" if len(values) > 1 else "supported"
        supported = next(iter(values)) if len(values) == 1 else None
        question = ("A retrieval task supplies these authoritative passages about the same versioned configuration value: "
                    + serialized(evidence) + ". No other evidence is available. Each value is a literal source statement. "
                    "Return status (supported/conflicting/unsupported), answer (integer or null), and source_ids. Do not resolve contradictions by inventing an authority order.")
        answer = {"status": status, "answer": supported, "source_ids": [r["source"] for r in evidence],
                  "explanation": "A single agreed value supports a sourced answer. Differing values require reporting the conflict, and no evidence requires an explicit unknown. Retrieval adds evidence to the context; it does not by itself perform an optimizer update."}
        checks = {"distinct_source_values": sorted(values), "status": status}
    elif family == "dspy":
        candidates = ({"A": [7, 10], "B": [8, 10]}, {"A": [9, 12], "B": [7, 8]}, {"A": [2, 3], "B": [3, 5]})[i]
        scores = {key: Fraction(correct, total) for key, (correct, total) in candidates.items()}
        winner = max(scores, key=scores.get)
        question = ("Two prompt programs were measured only on development examples, with [correct,total] counts "
                    + serialized(candidates) + ". Select the higher accuracy; return program and exact accuracy_fraction. "
                    "Do these development results alone establish improvement on a fresh final test panel? The final panel has not been queried.")
        answer = {"program": winner, "accuracy_fraction": str(scores[winner]), "fresh_test_improvement_proven": False,
                  "explanation": "Compare proportions, not raw correct counts when totals differ. Development scores guide selection, but repeated selection can overfit that panel. A fresh final evaluation remains necessary and must not be used to choose the prompt or training rate."}
        checks = {"development_scores": {key: str(value) for key, value in scores.items()}, "selected": winner}
    else:
        raise ValueError("Unknown predeclared task family")
    question += " Return one JSON object with the requested fields and a concise public explanation."
    return question, answer, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    selection = json.loads((root / "selection.json").read_text())
    if (root / "materials-manifest.json").exists():
        raise SystemExit("Material manifest already exists; inspect its receipts instead of regenerating")
    script_sha = sha(Path(__file__).read_bytes())
    selected_sha = sha((root / "selection.json").read_bytes())
    products = []
    seen_families = set()
    for source in selection["sources"]:
        ident = source["id"]
        reference_dir = root / "references" / ident
        reference = json.loads((reference_dir / "receipt.json").read_text())
        assert reference["selection_sha256"] == selected_sha
        assert sha((reference_dir / reference["filename"]).read_bytes()) == reference["sha256"]
        family = source_family(reference["url"])
        assert family and family not in seen_families
        seen_families.add(family)
        folder = root / "materials" / ident
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        count = 8 if source["role"] == "train" else 3
        for i in range(count):
            question, answer, checks = task(ident, i)
            receipt = {"task_id": f"{ident}-{i}", "task_origin": "locally_authored_teaching_fixture",
                       "upstream_benchmark_item": False, "builder_sha256": script_sha,
                       "source_sha256": reference["sha256"], "question_sha256": sha(question.encode()),
                       "answer_sha256": sha(serialized(answer).encode()), "check_results": checks,
                       "public_explanation_origin": "authored_solution_not_provider_reasoning"}
            receipt_raw = (serialized(receipt) + "\n").encode()
            (folder / f"{i}.receipt.json").write_bytes(receipt_raw)
            row = {"format": FORMAT, "id": f"{ident}-{i}", "question": question,
                   "answer": serialized(answer), "sources": [{"url": reference["url"], "sha256": reference["sha256"]}],
                   "verification": {"method": "trusted_local_fixture", "receipt_sha256": sha(receipt_raw)}}
            line = serialized(row)
            parse_record(line)
            records.append(line)
        raw = ("\n".join(records) + "\n").encode()
        document = folder / f"{ident}.jsonl"
        document.write_bytes(raw)
        products.append({"id": ident, "role": source["role"], "source_family": family,
                         "path": str(document.relative_to(root)), "sha256": sha(raw), "examples": count,
                         "reference_receipt": str((reference_dir / "receipt.json").relative_to(root)),
                         "training_eligible": False})
        print(ident, source["role"], count, "checked public examples", flush=True)
    manifest = {"schema_version": 1, "mode": "supervised", "selection_sha256": selected_sha,
                "builder_sha256": script_sha, "materials": products,
                "examples": sum(row["examples"] for row in products), "imported": False,
                "native_training_started": False,
                "independence_limit": "Different reference repositories and task families; shared local authorship is not proof of statistical independence or unseen pretraining content."}
    (root / "materials-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
