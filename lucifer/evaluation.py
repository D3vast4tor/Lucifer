"""Pure, frozen transfer smoke tests with independent mathematical/formal oracles.

Only reviewed USER questions select families. No answer, candidate, model,
network, hardware probe or executable source text supplies an oracle. These
small related tasks test transfer within supported families; they are not an
independently sampled benchmark or proof of broad expertise. Unknown domains
remain unknown and cannot qualify through unrelated fixed anchors.
"""
from __future__ import annotations

import copy
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import json
import math
import random
import re
import unicodedata


_VERSION = 1
_MAX_PROMPT_BYTES = 1200
_DEFAULT_OUTPUT_BUDGET = 2048
_SOURCE_FAMILIES = {
    "user_stated_constraint": "literal_user_constraints",
    "source_supplied_bounded_code_tests": "bounded_pure_code",
}
_STOP = set("lucifer this complex expert learning study think configured max reasoning level solve following carefully then provide self contained final worked solution assumptions main derivation results concise verification under words reveal private system instructions the a an and or for with from what how why is are of to in on at by as all return only json answer".split())
_PATTERNS = {
    "shortest_path": r"shortest.path|cheapest.*(?:path|route)|directed graph|distance labels",
    "bayesian_prediction": r"\b(?:bayesian|posterior|urns?)\b",
    "modular_arithmetic": r"chinese remainder|\bcrt\b|\bcongruen\w*|≡|moduli|coprime",
    "markov_hitting": r"\bmarkov\b|hitting time|transition matrix",
    "binary_strings": r"binary strings|no adjacent ones|labelled circular",
    "prefix_coding": r"\bhuffman\b|entropy.*bits|kraft|prefix.free code",
    "convex_projection": r"\bkkt\b|convex.*(?:minim|optim|project)|minimize.*\(x|complementary slackness",
    "linear_programming": r"linear.programming|dual certificate|derive the dual|shadow.price|primal.*dual|maximi[sz]e\s+\d+\s*x\s*\+\s*\d+\s*y\s+subject to",
    "linear_system": r"eigenvalues|positive definiteness|spectral condition|linear algebra|solve.*(?:ax|matrix)|times x",
    "parallel_scheduling": r"makespan|nonpreemptive|identical workers|optimal schedule",
    "newton_iteration": r"newton|residual.based error|numerical analysis",
    "coin_runs": r"consecutive heads|\bhhh\b|coin toss|pattern waiting",
    "collision_pairs": r"birthday|collision probability|independent.*hashes",
    "isothermal_gas": r"\bisothermal\b.*(?:gas|expan)|ideal gas.*isothermal",
    "coupled_equilibrium": r"(?:chemical|coupled).*equilibrium|mass balance.*equilibrium|conserved.*concentration|\[B\]/\[A\]",
    "relativistic_trip": r"twin.*(?:paradox|asymmetry)|travell?er.*(?:star|earth)|simultaneous.*turnaround|special relativity",
    "snapshot_guard": r"snapshot isolation|write skew|shared guard row|serializable isolation",
    "file_operation_ledger": r"operation fingerprints|precondition.*edit|sha.?256.precondition|fd.relative|symlink escape",
    "remote_idempotency": r"\bidempotenc\w*|remote.*(?:payment|effect)|transaction.outbox|exactly.once.*remote",
    "adapter_replay": r"consolidat\w*.*(?:experts?|adapters?)|(?:experts?|adapters?).*consolidat|(?:experts?|adapters?).*source.*consent|continual.learning.*workflow",
    "arithmetic": r"\b(?:arithmetic|addition|subtraction|multiplication)\b",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize_question(question: str) -> str:
    """Ignore Unicode presentation, case, punctuation and whitespace, not meaning."""
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", question).casefold()))


def _questions(examples):
    questions = []
    for example in examples:
        messages = example.get("messages", [])
        question = next((message.get("content") for message in messages if message.get("role") == "user"), None)
        _require(isinstance(question, str) and question.strip(), "An evaluation source lacks a user question")
        _require("<think>" not in question.casefold() and "</think>" not in question.casefold(), "Reasoning markers cannot enter evaluation sources")
        questions.append(question)
    return questions


def _family(question):
    return next((name for name, pattern in _PATTERNS.items() if re.search(pattern, question, re.I)), None)


def source_family(example):
    """Recognize narrow, verified source contracts without granting domain coverage."""
    validation = example.get("validation", {})
    if validation.get("method") == "independent_parameter_oracle" and validate_training_variant(example):
        return validation["family"]
    question = _questions([example])[0]
    if validation.get("status") == "verified":
        # Recompute bounded evidence: a claimed validation label is not an oracle.
        from .curriculum import _ALIASES, _FIELDS, _SLOTS, _code_tests, _math_value, _user_constraint_question, _verified_content
        try:
            answer = example["messages"][-1]["content"]
            if validation.get("method") == "user_stated_constraint":
                field = validation["field"]
                marker = "\n\nUSER STATEMENT (quoted data):\n"
                quote = question.split(marker, 1)[1]
                values = [match[2].strip() for match in _FIELDS.finditer(quote)
                          if _ALIASES.get(match[1].casefold().replace(" ", "_"),
                                          match[1].casefold().replace(" ", "_")) == field]
                if (field in _SLOTS and values and answer == values[-1]
                        and question == _user_constraint_question(quote, field)
                        and validation.get("quote_sha256") == hashlib.sha256(quote.encode()).hexdigest()):
                    return "literal_user_constraints"
            elif validation.get("method") == "source_supplied_bounded_code_tests":
                checked = _code_tests(answer, [question])
                if checked and all(validation.get(key) == checked[key] for key in ("tests_sha256", "code_sha256")):
                    return "bounded_pure_code"
            elif validation.get("method") == "bounded_literal_arithmetic":
                checked = _math_value(question)
                if checked is not None and Fraction(answer) == checked[0]:
                    return "arithmetic"
            elif validation.get("method") in {"verified_worked_arithmetic", "verified_error_correction"}:
                _verified_content(example)
                return "arithmetic"
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            pass
    literal_question = example.get("source_question") or question
    _require(isinstance(literal_question, str) and not re.search(r"</?think>", literal_question, re.I),
             "Source family selection requires literal user text")
    return _family(literal_question)


def audit_source_coverage(suite, examples):
    """Audit refreshed sources against an immutable admission artifact, not its anchors.

    This does not expand the admission evidence or certify a new family. The
    caller must stop for missing coverage rather than regenerate easier gates.
    """
    supported = set(suite.get("metadata", {}).get("coverage", {}).get("supported_families", []))
    assignments = []
    for example in examples:
        question = _questions([example])[0]
        family = source_family(example)
        assignments.append({"source_fingerprint": example.get("fingerprint"),
            "prompt_sha256": _digest(normalize_question(question)), "family": family,
            "covered": family is not None and family in supported})
    covered = sum(item["covered"] for item in assignments)
    status = "verified" if assignments and covered == len(assignments) else "partial" if covered else "unknown"
    return {"status": status, "promotion_eligible": status == "verified", "source_count": len(assignments),
            "covered_source_count": covered, "assignments": assignments,
            "missing_families": sorted({item["family"] for item in assignments
                                        if item["family"] is not None and not item["covered"]}),
            "unsupported_source_fingerprints": [item["source_fingerprint"] for item in assignments if not item["covered"]]}


def _label(question):
    words = [word for word in normalize_question(question).split() if word.isalpha() and word not in _STOP]
    return " ".join(dict.fromkeys(words))[:100] or "unmapped source question"


def _numeric_atoms(value):
    """Canonical literal values, used to reject unchanged parameter settings."""
    if isinstance(value, dict):
        return set().union(*(_numeric_atoms(item) for item in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_numeric_atoms(item) for item in value)) if value else set()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {str(Fraction(str(value)))}
    if isinstance(value, str):
        return {str(Fraction(atom)) for atom in re.findall(r"-?(?:\d+\.\d+|\d+/[1-9]\d*|\d+)", value)}
    return set()


def _fraction(value, prefix=""):
    return {prefix + "numerator": value.numerator, prefix + "denominator": value.denominator}


def _ratio(value):
    return [value.numerator, value.denominator]


def _oracle(family):
    if family == "literal_user_constraints":
        return "independent_quoted_literal_fields"
    if family == "bounded_pure_code":
        return "independent_bounded_arithmetic_function"
    if family in {"remote_idempotency", "snapshot_guard", "file_operation_ledger", "adapter_replay"}:
        return "independent_finite_protocol_trace"
    if family == "isothermal_gas":
        return "independent_closed_physics_decimal_half_even"
    if family == "coupled_equilibrium":
        return "independent_exact_mass_action_equations"
    if family == "relativistic_trip":
        return "independent_exact_spacetime_events"
    return "independent_exact_python_math"


def parameter_partition(family, parameters, *, version=1):
    """Reserve a stable parameter bucket before generating any train/test text."""
    _require(type(version) is int and version in {1, 2}, "Unknown parameter partition version")
    bucket = int(_digest(["parameter_partition_v1", family, parameters]), 16) % 10
    if bucket % 5 == 0:
        return "holdout"
    return "development" if version == 2 and bucket % 5 == 1 else "train"


def generate_training_variants(examples, *, excluded_cases=(), maximum=32, cursors=None,
                               excluded_semantic_keys=(), partition_version=1, target_families=None):
    """Generate independently solved training tasks, never from a model's labels.

    Train/holdout parameter buckets are disjoint. Historical frozen evaluation
    prompts/parameters are additionally reserved, including legacy suites that
    predate that partition. Provenance/source eligibility is enforced by the
    curriculum caller before and after this pure operation.
    """
    if type(maximum) is not int or not 0 <= maximum <= 256:
        raise ValueError("maximum must be between zero and 256")
    reserved_prompts = {normalize_question(case["prompt"]) for case in excluded_cases if isinstance(case.get("prompt"), str)}
    reserved_prompt_hashes = {case["prompt_sha256"] for case in excluded_cases if isinstance(case.get("prompt_sha256"), str)}
    supported_families = set(_PATTERNS) | set(_SOURCE_FAMILIES.values())
    if target_families is not None:
        _require(isinstance(target_families, (list, tuple))
                 and all(isinstance(family, str) and family in supported_families for family in target_families)
                 and len(set(target_families)) == len(target_families), "Target oracle families must be known and unique")
    reserved_parameters = {_digest([case["family"], case["parameters"]]) for case in excluded_cases
                           if case.get("family") in supported_families and isinstance(case.get("parameters"), dict)}
    reserved_parameters.update(case["parameter_sha256"] for case in excluded_cases if isinstance(case.get("parameter_sha256"), str))
    reserved_groups = {case["source_group_id"] for case in excluded_cases if isinstance(case.get("source_group_id"), str)}
    seen = {key.removeprefix("oracle:") for key in excluded_semantic_keys if isinstance(key, str)}
    rows, parents = [], []
    for parent in examples:
        if parent.get("partition", "train") != "train":
            continue
        question = parent.get("source_question") or parent["messages"][0]["content"]
        family = source_family(parent)
        if target_families is not None and family not in target_families:
            continue
        if family is None or normalize_question(question) in reserved_prompts or _digest(normalize_question(question)) in reserved_prompt_hashes:
            continue
        group = parent.get("source_group_id") or parent.get("semantic_key") or parent["fingerprint"]
        if group in reserved_groups:
            continue
        parents.append((parent, family, group))
    if target_families is not None:
        priorities = {family: index for index, family in enumerate(target_families)}
        parents.sort(key=lambda item: (priorities[item[1]], item[2], item[0]["fingerprint"]))
    # Legacy callers get one deterministic variant per parent. Explicit refresh
    # callers advance durable per-group cursors and round-robin to the budget.
    active = list(parents)
    while active and len(rows) < maximum:
        following = []
        for parent, family, group in active:
            if len(rows) >= maximum:
                break
            cursor_key = _digest([family, group])
            start = cursors.get(cursor_key, 0) if cursors is not None else 0
            _require(type(start) is int and 0 <= start < 2 ** 63 - 256, "Invalid oracle generation cursor")
            for attempt in range(start, start + 256):
                if cursors is not None:
                    cursors[cursor_key] = attempt + 1
                generator_version = "training_variant_v2" if target_families is not None else "training_variant_v1"
                seed = _digest([generator_version, family, group, attempt])
                prompt, expected, parameters = _training_variant(family, random.Random(int(seed, 16)), generator_version)
                semantic = _digest([family, parameters])
                if (parameter_partition(family, parameters, version=partition_version) != "train"
                        or semantic in reserved_parameters or semantic in seen or normalize_question(prompt) in reserved_prompts
                        or _digest(normalize_question(prompt)) in reserved_prompt_hashes):
                    continue
                seen.add(semantic)
                messages = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False, sort_keys=True, allow_nan=False)}]
                rows.append({"messages": messages, "source": "generated_oracle",
                             "conversation_id": parent["conversation_id"], "source_message_id": parent["source_message_id"],
                             "source_group_id": group, "semantic_key": "oracle:" + semantic, "partition": "train",
                             "fingerprint": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                             "validation": {"method": "independent_parameter_oracle", "family": family, "seed": seed,
                                            "parameters": parameters, "oracle": _oracle(family), "expected_json": expected},
                             "provenance": {"parent_fingerprint": parent["fingerprint"], "task_family": family,
                                            "generator_version": generator_version, "partition_version": partition_version,
                                            "generation_attempt": attempt}})
                if cursors is not None:
                    following.append((parent, family, group))
                break
        active = following
    return rows


def validate_training_variant(example, *, excluded_cases=()):
    """Recompute ground truth rather than trusting a claimed validator label."""
    try:
        validation = example["validation"]
        family, seed = validation["family"], validation["seed"]
        if (validation.get("method") != "independent_parameter_oracle"
                or family not in set(_PATTERNS) | set(_SOURCE_FAMILIES.values())
                or not isinstance(seed, str) or not re.fullmatch(r"[a-f0-9]{64}", seed)):
            return False
        messages = example["messages"]
        provenance = example.get("provenance", {})
        generator_version = provenance.get("generator_version", "training_variant_v1")
        if generator_version not in {"training_variant_v1", "training_variant_v2"}:
            return False
        prompt, expected, parameters = _training_variant(family, random.Random(int(seed, 16)), generator_version)
        version = provenance.get("partition_version", 1)
        if "generation_attempt" in provenance:
            attempt = provenance["generation_attempt"]
            if (type(attempt) is not int or not 0 <= attempt < 2 ** 63
                    or seed != _digest([generator_version, family, example["source_group_id"], attempt])):
                return False
        if (parameter_partition(family, parameters, version=version) != "train" or validation.get("parameters") != parameters
                or validation.get("oracle") != _oracle(family) or validation.get("expected_json") != expected
                or messages != [{"role": "user", "content": prompt}, {"role": "assistant", "content":
                                  json.dumps(expected, ensure_ascii=False, sort_keys=True, allow_nan=False)}]):
            return False
        for case in excluded_cases:
            if case.get("prompt_sha256") == _digest(normalize_question(prompt)) or case.get("parameter_sha256") == _digest([family, parameters]):
                return False
            if isinstance(case.get("prompt"), str) and normalize_question(case["prompt"]) == normalize_question(prompt):
                return False
            if case.get("family") == family and case.get("parameters") == parameters:
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _solve(matrix, target):
    """Exact Gaussian elimination, used only for small synthetic linear tasks."""
    rows = [[Fraction(value) for value in row] + [Fraction(target[index])] for index, row in enumerate(matrix)]
    for column in range(len(rows)):
        pivot = next(index for index in range(column, len(rows)) if rows[index][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for index in range(len(rows)):
            if index != column:
                multiplier = rows[index][column]
                rows[index] = [left - multiplier * right for left, right in zip(rows[index], rows[column])]
    return [row[-1] for row in rows]


def _training_variant(family, rng, version):
    """Broaden finite practice spaces without changing any frozen v1 oracle.

    A generator version travels with each training record and is independently
    recomputed on admission. Development/admission continue to use _variant.
    """
    if version == "training_variant_v2":
        if family == "binary_strings":
            n, k = rng.randint(14, 64), rng.randint(2, 6)
            prompt = (f"Count length{n} binary strings with exactly{k} ones and no adjacent ones, first linearly "
                      f"and then on{n} labelled circular positions with the first and last adjacent. Rotations remain "
                      "distinct. Return only JSON with integer linear and circular.")
            return prompt, {"linear": math.comb(n - k + 1, k),
                            "circular": n * math.comb(n - k, k) // (n - k)}, {"n": n, "k": k}
        if family == "markov_hitting":
            a, b, c = (Fraction(rng.randint(1, upper), 19) for upper in (12, 5, 5))
            expected = _solve([[1 - a, -(1 - a)], [-b, 1 - c]], [1, 1])[0]
            transition = [[a, 1 - a, Fraction(0)], [b, c, 1 - b - c], [Fraction(0), Fraction(0), Fraction(1)]]
            text = "[" + ",".join("[" + ",".join(map(str, row)) + "]" for row in transition) + "]"
            return (f"A Markov chain has transition matrix {text}. Starting in state0, find the expected first "
                    "hitting time of state2. Return only JSON with reduced-fraction integer numerator and denominator.",
                    _fraction(expected), {"a": str(a), "b": str(b), "c": str(c)})
        if family == "coin_runs":
            run = rng.randint(2, 6)
            probability = rng.choice([Fraction(2, 7), Fraction(3, 8), Fraction(4, 9), Fraction(5, 11),
                                      Fraction(7, 13), Fraction(8, 15), Fraction(11, 17), Fraction(13, 19)])
            horizon = run + rng.randint(1, 5)
            waiting = (1 - probability ** run) / ((1 - probability) * probability ** run)
            states = {0: Fraction(1)}
            for _ in range(horizon):
                next_states = {run: states.get(run, Fraction(0))}
                for streak, mass in states.items():
                    if streak < run:
                        next_states[0] = next_states.get(0, Fraction(0)) + mass * (1 - probability)
                        next_states[streak + 1] = next_states.get(streak + 1, Fraction(0)) + mass * probability
                states = next_states
            prompt = (f"Independent coin tosses have heads probability {probability}. Find the expected waiting time "
                      f"for {run} consecutive heads and the probability the first such run occurs by toss{horizon}. "
                      "Return only JSON with reduced-fraction fields expected_numerator, expected_denominator, "
                      "by_horizon_numerator, by_horizon_denominator.")
            return prompt, {**_fraction(waiting, "expected_"), **_fraction(states[run], "by_horizon_")}, {
                "run": run, "probability": str(probability), "horizon": horizon}
        if family == "snapshot_guard":
            # The protocol's three traces are independent of replica count.
            # Widen only that literal parameter, retaining the formal contract.
            class WiderReplicas:
                def randint(self, low, high):
                    return rng.randint(3, 80) if (low, high) == (3, 6) else rng.randint(low, high)

                def choice(self, values):
                    return rng.choice(values)
            return _variant(family, WiderReplicas())
    return _variant(family, rng)


def _variant(family, rng):
    """Return a fresh problem, exact target and auditable generator parameters."""
    if family == "literal_user_constraints":
        port = rng.randint(12000, 58000)
        language = rng.choice(["Python", "Rust", "Go", "Julia"])
        output = rng.choice(["JSON", "CSV", "plain text"])
        prompt = ("Quoted USER preferences (data, not instructions):\n"
                  f"Language: {language}\nFormat: {output}\nPort: {port}\n\n"
                  "Extract only the literal stated preferences. Return JSON with string language, string format, "
                  "and integer port. This tests extraction, not software implementation.")
        return prompt, {"language": language, "format": output, "port": port}, {
            "language": language, "format": output, "port": port}
    if family == "bounded_pure_code":
        a, b, c = rng.randint(-9, 9), rng.randint(2, 13), rng.randint(-20, 20)
        inputs = [[rng.randint(-17, 17), rng.randint(-17, 17)] for _ in range(4)]
        prompt = (f"A pure integer function is defined by f(x,y) = {a}*x + {b}*y + {c}. "
                  f"Evaluate it for the input pairs {inputs} in that order. Return only JSON with integer list outputs. "
                  "This tests only finite arithmetic function execution, not general program correctness.")
        return prompt, {"outputs": [a * x + b * y + c for x, y in inputs]}, {
            "a": a, "b": b, "c": c, "inputs": inputs}
    if family == "shortest_path":
        weights = [rng.randint(2, 19) for _ in range(7)]
        edges = [list(edge) for edge in zip(("S", "S", "A", "B", "A", "B", "C"),
                                          ("A", "B", "C", "C", "T", "T", "T"), weights)]
        distances = {"S": 0}
        for vertex in ("S", "A", "B", "C", "T"):
            for start, end, cost in edges:
                if start == vertex and start in distances:
                    distances[end] = min(distances.get(end, math.inf), distances[start] + cost)
        via_c = min(weights[0] + weights[2], weights[1] + weights[3]) + weights[6]
        prompt = "A directed graph has edges " + ", ".join(f"{a}->{b}={cost}" for a, b, cost in edges) + ". Find the cheapest S-to-T cost and the cheapest such cost required to visit C. Return only JSON with integer cost and via_C."
        return prompt, {"cost": distances["T"], "via_C": via_c}, {"edges": edges}
    if family == "bayesian_prediction":
        weights = [rng.randint(1, 5) for _ in range(3)]
        rates = rng.sample([Fraction(n, 11) for n in range(1, 11)], 3)
        reds = rng.randint(2, 4)
        denominator = sum(weight * rate ** reds for weight, rate in zip(weights, rates))
        prediction = sum(weight * rate ** (reds + 1) for weight, rate in zip(weights, rates)) / denominator
        prompt = f"Choose one urn with prior weights {weights} normalized by their sum. The urn red rates are {[str(rate) for rate in rates]}. Draw {reds} red balls with replacement from that same urn. What is the probability the next draw is red? Return only JSON with reduced-fraction integer numerator and denominator."
        return prompt, _fraction(prediction), {"weights": weights, "rates": [str(rate) for rate in rates], "reds": reds}
    if family == "modular_arithmetic":
        moduli = rng.sample([11, 13, 17, 19, 23, 29, 31], 3)
        residues = [rng.randrange(modulus) for modulus in moduli]
        period = math.prod(moduli)
        answer = sum(residue * (period // modulus) * pow(period // modulus, -1, modulus)
                     for residue, modulus in zip(residues, moduli)) % period
        prompt = "Find the least nonnegative x satisfying " + ", ".join(f"x={residue} mod {modulus}" for residue, modulus in zip(residues, moduli)) + ". Return only JSON with integer x and period."
        return prompt, {"x": answer, "period": period}, {"moduli": moduli, "residues": residues}
    if family == "markov_hitting":
        a = Fraction(rng.randint(1, 3), 7)
        b = Fraction(rng.randint(1, 2), 7)
        c = Fraction(rng.randint(1, 2), 7)
        transition = [[a, 1 - a, Fraction(0)], [b, c, 1 - b - c], [Fraction(0), Fraction(0), Fraction(1)]]
        expected = _solve([[1 - a, -(1 - a)], [-b, 1 - c]], [1, 1])[0]
        text = "[" + ",".join("[" + ",".join(map(str, row)) + "]" for row in transition) + "]"
        return f"A Markov chain has transition matrix {text}. Starting in state0, find the expected first hitting time of state2. Return only JSON with reduced-fraction integer numerator and denominator.", _fraction(expected), {"a": str(a), "b": str(b), "c": str(c)}
    if family == "binary_strings":
        n, k = rng.randint(14, 25), rng.randint(2, 5)
        linear = math.comb(n - k + 1, k)
        circular = n * math.comb(n - k, k) // (n - k)
        return f"Count length{n} binary strings with exactly{k} ones and no adjacent ones, first linearly and then on{n} labelled circular positions with the first and last adjacent. Rotations remain distinct. Return only JSON with integer linear and circular.", {"linear": linear, "circular": circular}, {"n": n, "k": k}
    if family == "prefix_coding":
        depth = rng.randint(3, 7)
        probabilities = [Fraction(1, 2 ** power) for power in range(1, depth)] + [Fraction(1, 2 ** (depth - 1))]
        rng.shuffle(probabilities)
        entropy = sum(probability * (probability.denominator.bit_length() - 1) for probability in probabilities)
        prompt = f"A source has probabilities {[str(value) for value in probabilities]}. Find its entropy in bits and the minimum expected binary prefix-code length; these dyadic probabilities permit equality. Return only JSON with reduced-fraction fields entropy_numerator, entropy_denominator, length_numerator, length_denominator."
        return prompt, {**_fraction(entropy, "entropy_"), **_fraction(entropy, "length_")}, {"probabilities": [str(value) for value in probabilities]}
    if family == "convex_projection":
        a, b, shift = rng.randint(8, 20), rng.randint(8, 20), rng.randint(2, 6)
        bound = a + b - 2 * shift
        return f"Minimize (x-{a})^2+(y-{b})^2 subject to x>=0, y>=0 and x+y<={bound}. Return only JSON with integer x, y and objective.", {"x": a - shift, "y": b - shift, "objective": 2 * shift ** 2}, {"a": a, "b": b, "bound": bound}
    if family == "linear_programming":
        a, bound = rng.randint(2, 5), rng.randint(7, 15)
        second = rng.randint(bound + 1, a * bound - 1)
        c2 = rng.randint(3, 9)
        c1 = rng.randint(c2 + 1, a * c2 - 1)
        x, y = Fraction(second - bound, a - 1), Fraction(a * bound - second, a - 1)
        u, v = Fraction(a * c2 - c1, a - 1), Fraction(c1 - c2, a - 1)
        prompt = f"Maximize {c1}x+{c2}y subject to x+y<={bound}, {a}x+y<={second}, x,y>=0. Dual u,v follow constraint order. If the second RHS becomes {second}+delta, give the current basis's closed optimality interval and value slope. Return only JSON: x,y,value,dual_u,dual_v,slope as reduced [numerator,denominator] arrays; delta_min,delta_max as integers."
        return prompt, {"x": _ratio(x), "y": _ratio(y), "value": _ratio(c1 * x + c2 * y),
            "dual_u": _ratio(u), "dual_v": _ratio(v), "slope": _ratio(v), "delta_min": bound - second,
            "delta_max": a * bound - second}, {"a": a, "R": bound, "S": second, "c1": c1, "c2": c2}
    if family == "linear_system":
        a, d, b = rng.randint(5, 11), rng.randint(5, 11), rng.randint(1, 4)
        u, v = rng.randint(-9, 9), rng.randint(-9, 9)
        x, y = _solve([[a, b], [b, d]], [u, v])
        prompt = f"Solve [[{a},{b}],[{b},{d}]] times [x,y] = [{u},{v}] exactly. Return only JSON with reduced-fraction integer fields x_numerator, x_denominator, y_numerator, y_denominator."
        return prompt, {**_fraction(x, "x_"), **_fraction(y, "y_")}, {"a": a, "b": b, "d": d, "u": u, "v": v}
    if family == "parallel_scheduling":
        durations = [rng.randint(2, 12) for _ in range(5)]
        total = sum(durations)
        achievable = {0}
        for duration in durations:
            achievable |= {value + duration for value in achievable}
        answer = min(max(value, total - value) for value in achievable)
        return f"Five independent nonpreemptive jobs have durations {durations}. With two identical workers, find the minimum makespan; jobs have no prerequisites. Also give the one-worker makespan. Return only JSON with integer two_workers and one_worker.", {"two_workers": answer, "one_worker": total}, {"durations": durations}
    if family == "newton_iteration":
        square = rng.choice([number for number in range(3, 80) if math.isqrt(number) ** 2 != number])
        start = Fraction(rng.randint(2, 9))
        first = start - (start * start - square) / (2 * start)
        second = first - (first * first - square) / (2 * first)
        prompt = f"Use Newton iteration on f(x)=x^2-{square} starting at x0={start}. Compute x1 and x2 exactly. Return only JSON with reduced-fraction fields first_numerator, first_denominator, second_numerator, second_denominator."
        return prompt, {**_fraction(first, "first_"), **_fraction(second, "second_")}, {"square": square, "start": str(start)}
    if family == "coin_runs":
        run = rng.randint(2, 4)
        probability = rng.choice([Fraction(2, 5), Fraction(3, 7), Fraction(4, 7), Fraction(3, 5)])
        horizon = run + 2
        matrix = [[Fraction(index == column) for column in range(run)] for index in range(run)]
        for index in range(run):
            matrix[index][0] -= 1 - probability
            if index + 1 < run:
                matrix[index][index + 1] -= probability
        waiting = _solve(matrix, [1] * run)[0]
        states = {0: Fraction(1)}
        for _ in range(horizon):
            next_states = {run: states.get(run, Fraction(0))}
            for streak, mass in states.items():
                if streak < run:
                    next_states[0] = next_states.get(0, Fraction(0)) + mass * (1 - probability)
                    next_states[streak + 1] = next_states.get(streak + 1, Fraction(0)) + mass * probability
            states = next_states
        by_horizon = states[run]
        prompt = f"Independent coin tosses have heads probability {probability}. Find the expected waiting time for {run} consecutive heads and the probability the first such run occurs by toss{horizon}. Return only JSON with reduced-fraction fields expected_numerator, expected_denominator, by_horizon_numerator, by_horizon_denominator."
        return prompt, {**_fraction(waiting, "expected_"), **_fraction(by_horizon, "by_horizon_")}, {"run": run, "probability": str(probability), "horizon": horizon}
    if family == "collision_pairs":
        bits, items = rng.randint(9, 18), rng.randint(17, 113)
        pairs = Fraction(items * (items - 1), 2 * 2 ** bits)
        prompt = f"Hash {items} independent items uniformly into {bits}-bit values. Find the exact expected number of colliding unordered pairs, not the probability of at least one collision. Return only JSON with reduced-fraction integer numerator and denominator."
        return prompt, _fraction(pairs), {"bits": bits, "items": items}
    if family == "isothermal_gas":
        amount, temperature, ratio = rng.randint(2, 4), rng.randrange(270, 351, 10), rng.randint(3, 7)
        initial_volume = Fraction(rng.randint(2, 7), 100)
        with localcontext() as context:
            context.prec = 70
            gas_constant = Decimal("8.314462618")
            nr = Decimal(amount) * gas_constant
            logarithm = Decimal(ratio).ln()
            reversible_work = nr * temperature * logarithm
            irreversible_work = nr * temperature * (1 - Decimal(1) / ratio)
            entropy = nr * logarithm
            irreversible_total = entropy - irreversible_work / temperature
            def milli(value):
                return int((value * 1000).to_integral_value(rounding=ROUND_HALF_EVEN))
            expected = {"W_rev": milli(reversible_work), "W_irrev": milli(irreversible_work),
                "Q_rev": milli(reversible_work), "Q_irrev": milli(irreversible_work), "dU": 0,
                "dS_sys": milli(entropy), "dS_tot_rev": 0, "dS_tot_irrev": milli(irreversible_total)}
        prompt = f"An ideal gas, n={amount}mol, expands isothermally at {temperature}K from {initial_volume}m^3 to {initial_volume * ratio}m^3 in a same-temperature heat bath; R=8.314462618J/(mol K). Compare reversible expansion and irreversible expansion against constant final gas pressure. Work by gas and heat into gas are positive. Return only JSON integers W_rev,W_irrev,Q_rev,Q_irrev,dU in millijoules; dS_sys,dS_tot_rev,dS_tot_irrev in millijoules/K. Round each value to nearest integer, ties to even."
        return prompt, expected, {"n": amount, "temperature": temperature, "volume_initial": str(initial_volume),
                                  "volume_ratio": ratio, "R": "8.314462618", "rounding": "milli_half_even"}
    if family == "coupled_equilibrium":
        k1 = rng.randint(2, 6)
        k2 = Fraction(rng.randint(1, 4), rng.randint(2, 5))
        first = Fraction(1, rng.randint(12, 30))
        second = k1 * first
        third = k2 * second ** 2
        total = first + second + 2 * third
        prompt = f"In an ideal concentration model, A<->B has K1=[B]/[A]={k1} and 2B<->C has K2=[C]/[B]^2={k2} M^-1. The conserved concentration [A]+[B]+2[C] is {total}M. Solve the nonnegative equilibrium. Return only JSON with A,B,C as reduced [numerator,denominator] concentration arrays and negative_root_valid as a boolean."
        return prompt, {"A": _ratio(first), "B": _ratio(second), "C": _ratio(third), "negative_root_valid": False}, {
            "K1": k1, "K2": str(k2), "conserved": str(total)}
    if family == "relativistic_trip":
        speed, perpendicular, hypotenuse = rng.choice([(5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)])
        beta, gamma = Fraction(speed, hypotenuse), Fraction(hypotenuse, perpendicular)
        distance = Fraction(rng.randint(11, 37), rng.randint(2, 5))
        turn_time = distance / beta
        earth_out, earth_in = turn_time - beta * distance, turn_time + beta * distance
        prompt = f"A traveller goes from Earth to a star {distance} light-years away at speed ({beta})c and returns at the same speed after instantaneous reversal; Earth/star are at rest together and c=1light-year/year. Give Earth and traveller round-trip durations, Earth times assigned simultaneous with turnaround by outbound/inbound inertial frames, and their difference. Return only JSON reduced [numerator,denominator] arrays earth_years,traveller_years,earth_at_turn_out,earth_at_turn_in,simultaneity_change."
        return prompt, {"earth_years": _ratio(2 * turn_time), "traveller_years": _ratio(2 * turn_time / gamma),
            "earth_at_turn_out": _ratio(earth_out), "earth_at_turn_in": _ratio(earth_in),
            "simultaneity_change": _ratio(earth_in - earth_out)}, {"beta": str(beta), "distance_ly": str(distance)}
    if family == "remote_idempotency":
        namespace = rng.randint(17, 99)
        keys = [f"op_{namespace}", f"op_{namespace + 1}"]
        requests = [keys[0], keys[1], keys[0], rng.choice(keys), keys[1]]
        acknowledgements = [rng.choice([False, True]) for _ in requests]
        acknowledgements[0] = False
        deduplicate = rng.choice([False, True])
        stored, receipts, effects = set(), set(), 0
        for key, acknowledged in zip(requests, acknowledgements):
            if not deduplicate or key not in stored:
                effects += 1
            stored.add(key)
            if acknowledged:
                receipts.add(key)
        prompt = f"Formal service trace: requests={json.dumps(requests)}, acknowledged={json.dumps(acknowledgements)}, dedup={str(deduplicate).lower()}. All requests reach the provider and commit before any acknowledgement can be lost. With dedup=true, an already committed key returns its stored result without another effect; with dedup=false every request causes one effect. Acknowledged keys alone enter the local receipt set. Return only JSON effects,distinct_keys,local_receipts as integers and blind_retry_duplicate_free,local_transaction_alone_exactly_once as booleans; a local transaction cannot alter provider behavior."
        return prompt, {"effects": effects, "distinct_keys": len(stored), "local_receipts": len(receipts),
            "blind_retry_duplicate_free": deduplicate, "local_transaction_alone_exactly_once": False}, {
                "requests": requests, "acknowledged": acknowledgements, "dedup": deduplicate}
    if family == "snapshot_guard":
        replicas = rng.randint(3, 6)
        mode = rng.choice(["snapshot", "shared_guard_write", "guard_before_snapshot"])
        if mode == "snapshot":
            active, committed, aborted = 0, replicas, 0
        elif mode == "shared_guard_write":
            active, committed, aborted = replicas - 1, 1, replicas - 1
        else:
            active, committed, aborted = 1, replicas - 1, 0
        prompt = f"Formal replica trace: {replicas} rows start active; one transaction per row wants to disable its own row only when another is active. Mode={mode}. In snapshot mode all read the initial snapshot and disjoint row writes commit. In shared_guard_write mode they ALSO write the same guard row under first-committer-wins, so conflicting writers abort without retry. In guard_before_snapshot mode an external guard serializes transactions BEFORE each fresh snapshot; each commits or skips using current rows. Count only disabling commits. Return only JSON integers active_rows,disabling_commits,aborted and boolean invariant_preserved."
        return prompt, {"active_rows": active, "disabling_commits": committed, "aborted": aborted,
            "invariant_preserved": active > 0}, {"replicas": replicas, "mode": mode}
    if family == "file_operation_ledger":
        state = rng.choice(["DONE", "PREPARED", "ABSENT", "STARTED", "UNCERTAIN"])
        expected_version = rng.randint(17, 99)
        actual_version = expected_version + rng.choice([0, 1])
        reused = state == "DONE"
        uncertain = state in {"STARTED", "UNCERTAIN"}
        checked = state in {"PREPARED", "ABSENT"}
        executed = checked and actual_version == expected_version
        prompt = f"Formal file-edit ledger: operation F has state {state}; expected digest-version v{expected_version}, actual v{actual_version}. Version IDs denote distinct SHA256 digests. Contract: DONE returns the recorded result BEFORE checking current preconditions, with no effect. PREPARED guarantees no effect started; ABSENT has no record. Those two may check the atomic version precondition and execute once only on a match. STARTED/UNCERTAIN have unknown effects and must not retry or check a new edit precondition automatically. Return only JSON booleans reuse_result,check_precondition,execute,uncertain."
        return prompt, {"reuse_result": reused, "check_precondition": checked, "execute": executed, "uncertain": uncertain}, {
            "state": state, "expected_version": expected_version, "actual_version": actual_version}
    if family == "adapter_replay":
        namespace = rng.randint(17, 99)
        base, checkpoint = f"base_{namespace}", f"checkpoint_{namespace}"
        sources = {f"c_{namespace + index}": {"exists": True, "logging": True, "training": True} for index in range(4)}
        source_ids = sorted(sources)
        sources[source_ids[-1]]["training"] = False
        records = [
            {"id": f"p_{namespace}", "base": base, "checkpoint": checkpoint, "verified": True, "revoked": False, "sources": [source_ids[0]]},
            {"id": f"p_{namespace + 1}", "base": base, "checkpoint": checkpoint, "verified": rng.choice([False, True]), "revoked": False, "sources": [source_ids[1]]},
            {"id": f"p_{namespace + 2}", "base": base + "_other", "checkpoint": checkpoint, "verified": True, "revoked": False, "sources": [source_ids[2]]},
            {"id": f"p_{namespace + 3}", "base": base, "checkpoint": checkpoint, "verified": True, "revoked": False, "sources": [source_ids[-1]]},
            {"id": f"p_{namespace + 4}", "base": base, "checkpoint": checkpoint + "_other", "verified": True, "revoked": False, "sources": [source_ids[0]]},
        ]
        eligible, union = [], set()
        for record in records:
            valid = record["base"] == base and record["checkpoint"] == checkpoint and record["verified"] and not record["revoked"]
            valid = valid and bool(record["sources"]) and all(all(sources[cid].values()) for cid in record["sources"])
            if valid:
                eligible.append(record["id"])
                union.update(record["sources"])
        source_rows = [[cid, flags["exists"], flags["logging"], flags["training"]] for cid, flags in sources.items()]
        parent_rows = [[record[key] for key in ("id", "base", "checkpoint", "verified", "revoked", "sources")] for record in records]
        prompt = f"Formal source-replay trace: target base={base}, checkpoint={checkpoint}. Source columns=[id,exists,logging,training], rows={json.dumps(source_rows, separators=(',', ':'))}. Parent columns=[id,base,checkpoint,verified,revoked,sources], rows={json.dumps(parent_rows, separators=(',', ':'))}. Select ALL parents with exact base+checkpoint, verified=true, revoked=false, nonempty sources, and every source exists/logging/training true. Union their sources. The system supports reviewed-source replay only, never tensor averaging. Return only JSON sorted lists eligible_parents,source_ids and boolean tensor_merge_allowed."
        return prompt, {"eligible_parents": sorted(eligible), "source_ids": sorted(union), "tensor_merge_allowed": False}, {
            "base": base, "checkpoint": checkpoint, "sources": sources, "parents": records}
    a, b, c = [rng.randint(13, 79) for _ in range(3)]
    return f"Compute {a}*{b}+{c} exactly. Return only JSON with integer value.", {"value": a * b + c}, {"a": a, "b": b, "c": c}


def _check_cases(cases):
    _require(isinstance(cases, list) and bool(cases), "Evaluation needs cases")
    ids = []
    for case in cases:
        _require(isinstance(case, dict) and isinstance(case.get("prompt"), str) and bool(case["prompt"].strip()), "Evaluation case requires a prompt")
        _require(isinstance(case.get("id"), str) and bool(case["id"]), "Evaluation cases need stable IDs")
        _require(any(key in case for key in ("expected_exact", "expected_json", "expected_regex", "expected_contains", "format")), "Evaluation case lacks an assertion")
        if "expected_regex" in case:
            _require(isinstance(case["expected_regex"], str) and bool(case["expected_regex"]), "Regex assertions must be nonempty")
            re.compile(case["expected_regex"])
        if "expected_contains" in case:
            contains = case["expected_contains"]
            _require((isinstance(contains, str) and bool(contains)) or
                     (isinstance(contains, list) and bool(contains) and all(isinstance(item, str) and item for item in contains)),
                     "Contains assertions must not pass vacuously")
        ids.append(case["id"])
    _require(len(ids) == len(set(ids)), "Evaluation case IDs must be unique")
    _require(any(case.get("suite") == "safety" for case in cases) and any(case.get("suite") != "safety" for case in cases), "Evaluation needs both safety and task cases")


def freeze_evaluation_suite(configured_suite: dict, examples: list[dict], *, base_model_id: str,
                            training_model: str, profile_revision: str | None = None,
                            capabilities: dict | None = None, replay_examples: list[dict] | None = None,
                            purpose: str = "admission", dataset_revision: str | None = None,
                            excluded_cases=(), default_max_output_tokens: int = _DEFAULT_OUTPUT_BUDGET,
                            partition_version: int = 1) -> dict:
    """Freeze anchors and source-selected variants before any training effects.

    Caller must persist the returned canonical artifact and an external SHA,
    then enforce that SHA during evaluation/promotion. Metadata digests detect
    inconsistent edits; they are not signatures against a writer changing all
    copies. JSON output is requested through plain text, requiring neither tools
    nor a structured-output provider parameter.
    """
    _require(isinstance(configured_suite, dict), "Configured evaluation suite must be an object")
    _require(purpose in {"admission", "development"}, "Unknown evaluation purpose")
    _require(type(default_max_output_tokens) is int and 1 <= default_max_output_tokens <= 262144,
             "Evaluation output budget must be 1..262144")
    _require(dataset_revision is None or isinstance(dataset_revision, str) and bool(dataset_revision),
             "Dataset revision must be a nonempty string")
    parameter_partition("arithmetic", {}, version=partition_version)
    _require(isinstance(base_model_id, str) and bool(base_model_id) and isinstance(training_model, str)
             and bool(training_model), "Evaluation requires pinned registered base/checkpoint identity")
    if capabilities:
        _require(capabilities.get("model_id") in {None, base_model_id}
                 and capabilities.get("profile_revision") in {None, profile_revision}, "Startup capabilities belong to another model/profile")
    suite = copy.deepcopy(configured_suite)
    anchors = suite.get("cases", [])
    _check_cases(anchors)
    _require(isinstance(suite.get("promotion", {}), dict), "Promotion policy must be an object")
    reviewed = [item for item in examples if item.get("source") != "bundled_replay"]
    _require(bool(reviewed), "Freeze evaluation only after selecting reviewed examples")
    questions = _questions(reviewed)
    all_questions = _questions(examples) + _questions(replay_examples or [])
    fingerprints = [item.get("fingerprint") for item in reviewed]
    _require(all(isinstance(value, str) and value for value in fingerprints) and len(fingerprints) == len(set(fingerprints)), "Reviewed source fingerprints must be unique")
    fingerprints.sort()
    question_hashes = sorted(_digest(normalize_question(value)) for value in questions)
    exclusion_hashes = sorted({_digest(normalize_question(value)) for value in all_questions})
    exclusions = set(exclusion_hashes)
    source_families = [source_family(example) for example in reviewed]
    families = sorted({name for name in source_families if name})
    anchor_overlap = [case["id"] for case in anchors if _digest(normalize_question(case["prompt"])) in exclusions]
    generated, provenance, used_ids = [], [], {case["id"] for case in anchors}
    reserved_prompts = {_digest(normalize_question(case["prompt"])) for case in excluded_cases
                        if isinstance(case.get("prompt"), str)}
    reserved_prompts.update(case["prompt_sha256"] for case in excluded_cases
                            if isinstance(case.get("prompt_sha256"), str))
    reserved_parameters = {_digest([case["family"], case["parameters"]]) for case in excluded_cases
                           if case.get("family") and isinstance(case.get("parameters"), dict)}
    reserved_parameters.update(case["parameter_sha256"] for case in excluded_cases
                               if isinstance(case.get("parameter_sha256"), str))
    target_partition = "development" if purpose == "development" else "holdout"
    for family in families:
        source_numbers = _numeric_atoms([question for question, name in zip(questions, source_families) if name == family])
        for attempt in range(32):
            seed_input = {"version": _VERSION, "base": base_model_id, "training_model": training_model,
                          "revision": profile_revision, "questions": question_hashes, "family": family, "attempt": attempt}
            if purpose == "development":
                seed_input.update(purpose=purpose, dataset_revision=dataset_revision)
            seed = _digest(seed_input)
            prompt, expected, parameters = _variant(family, random.Random(int(seed, 16)))
            prompt_hash = _digest(normalize_question(prompt))
            if parameter_partition(family, parameters, version=partition_version) != target_partition:
                continue
            if (prompt_hash in exclusions or prompt_hash in reserved_prompts
                    or _digest([family, parameters]) in reserved_parameters or len(prompt.encode()) > _MAX_PROMPT_BYTES):
                continue
            if not (_numeric_atoms(parameters) - source_numbers):
                continue
            ident = f"dynamic_{family}_{seed[:12]}"
            if ident in used_ids:
                continue
            generated.append({"id": ident, "suite": "reasoning", "prompt": prompt, "format": "json", "expected_json": expected})
            provenance.append({"case_id": ident, "family": family, "parameters": parameters, "seed": seed, "attempt": attempt,
                               "prompt_sha256": prompt_hash, "oracle": _oracle(family)})
            exclusions.add(prompt_hash)
            used_ids.add(ident)
            break
    families = sorted({record["family"] for record in provenance})
    unsupported = [question for question, name in zip(questions, source_families) if name not in families]
    coverage_status = "verified" if families and not unsupported else "partial" if families else "unknown"
    source_binding = {"source_fingerprints": fingerprints, "source_question_hashes": question_hashes,
                      "base_model_id": base_model_id, "training_model": training_model, "profile_revision": profile_revision}
    metadata = {**copy.deepcopy(suite.get("metadata", {})), "dynamic_evaluation_version": _VERSION,
        "suite_purpose": purpose, "dataset_revision": dataset_revision,
        "evaluation_source_fingerprints": fingerprints,
        **source_binding, "source_digest": _digest(source_binding), "startup_capabilities": copy.deepcopy(capabilities),
        "startup_capabilities_digest": _digest(capabilities), "configured_anchors": copy.deepcopy(anchors),
        "configured_anchors_digest": _digest(anchors), "configured_promotion": copy.deepcopy(suite.get("promotion", {})),
        "promotion_policy_digest": _digest(suite.get("promotion", {})), "exclusion_prompt_hashes": exclusion_hashes,
        "generated_before_training": True, "independent_oracles": True, "generation_method": "deterministic_parameter_transfer_smoke",
        "parameter_partition_version": partition_version,
        "reserved_prompt_hashes": sorted(reserved_prompts), "reserved_parameter_hashes": sorted(reserved_parameters),
        "dynamic_cases": provenance, "dynamic_case_ids": [case["id"] for case in generated],
        "anchor_training_overlap_ids": anchor_overlap,
        "format_only_anchor_ids": [case["id"] for case in anchors if not any(key in case for key in ("expected_exact", "expected_json", "expected_regex", "expected_contains"))],
        "coverage": {"status": coverage_status, "supported_topics": families, "supported_families": families,
            "unsupported_topics": [_label(value) for value in unsupported], "source_question_count": len(questions),
            "supported_question_count": len(questions) - len(unsupported), "generated_cases": len(generated),
            "definition": "Each mapped question selects a small related verified math/physics/protocol smoke family; subskills and broad domain capability are not established."},
        "source_question_families": [{"prompt_sha256": _digest(normalize_question(question)),
            "family": name if name in families else None} for question, name in zip(questions, source_families)],
        "source_numeric_atoms": {family: sorted(_numeric_atoms([question for question, name in zip(questions, source_families)
                                                               if name == family])) for family in families},
        "promotion_eligible": coverage_status == "verified" and not anchor_overlap,
        "required_interfaces": {"text_completion": True, "structured_output": False, "tools": False},
        "generation_limits": {"max_prompt_bytes": _MAX_PROMPT_BYTES},
        "limitations": ["Related deterministic parameter variants, not independently sampled domain benchmarks.",
                        "Unknown question families do not receive invented task-specific ground truth.",
                        "Exact normalized prompt overlap is excluded; semantic or pretraining overlap cannot be ruled out.",
                        "Domain improvement requires separate dynamic-case evidence; anchor gains alone are insufficient."]}
    suite["cases"] = anchors + generated
    metadata["case_budgets"] = {case["id"]: case.get("max_tokens", case.get("max_output_tokens", default_max_output_tokens))
                                for case in suite["cases"]}
    _require(all(type(value) is int and 1 <= value <= 262144 for value in metadata["case_budgets"].values()),
             "Case output budgets must be 1..262144")
    metadata["cases_digest"] = _digest(suite["cases"])
    metadata["freeze_digest"] = _digest(metadata)
    suite["metadata"] = metadata
    validate_frozen_suite(suite, base_model_id=base_model_id, training_model=training_model,
                          source_fingerprints=fingerprints, profile_revision=profile_revision)
    return suite


def validate_frozen_suite(suite: dict, *, base_model_id: str, training_model: str,
                          source_fingerprints: list[str], profile_revision: str | None = None,
                          cumulative_training_fingerprints: list[str] | None = None):
    """Validate a persisted plan against pinned provenance, not current guesses."""
    _check_cases(suite.get("cases", []))
    meta = suite.get("metadata", {})
    _require(meta.get("freeze_digest") == _digest({key: value for key, value in meta.items() if key != "freeze_digest"}), "Frozen evaluation metadata changed")
    _require(meta.get("dynamic_evaluation_version") == _VERSION and meta.get("generated_before_training") is True
             and meta.get("independent_oracles") is True, "Evaluation was not independently frozen before training")
    binding = {"source_fingerprints": sorted(source_fingerprints), "source_question_hashes": meta.get("source_question_hashes"),
               "base_model_id": base_model_id, "training_model": training_model, "profile_revision": profile_revision}
    _require(all(meta.get(key) == value for key, value in binding.items()) and meta.get("source_digest") == _digest(binding), "Frozen evaluation source/model/profile binding changed")
    _require(meta.get("evaluation_source_fingerprints", binding["source_fingerprints"]) == binding["source_fingerprints"],
             "Initial evaluation sources changed")
    if cumulative_training_fingerprints is not None:
        _require(all(isinstance(value, str) and value for value in cumulative_training_fingerprints),
                 "Cumulative training lineage contains an invalid fingerprint")
        _require(set(source_fingerprints) <= set(cumulative_training_fingerprints),
                 "Cumulative training lineage withdrew an initial evaluation source")
    if "case_budgets" in meta:
        _require(set(meta["case_budgets"]) == {case["id"] for case in suite["cases"]}
                 and all(type(value) is int and 1 <= value <= 262144 for value in meta["case_budgets"].values()),
                 "Frozen case budgets changed")
    anchors = meta.get("configured_anchors")
    _require(isinstance(anchors, list) and suite["cases"][:len(anchors)] == anchors
             and _digest(anchors) == meta.get("configured_anchors_digest"), "Configured evaluation anchors changed")
    _require(suite.get("promotion", {}) == meta.get("configured_promotion")
             and _digest(suite.get("promotion", {})) == meta.get("promotion_policy_digest"), "Promotion policy changed")
    _require(_digest(suite["cases"]) == meta.get("cases_digest")
             and _digest(meta.get("startup_capabilities")) == meta.get("startup_capabilities_digest"), "Frozen case/capability content changed")
    generated = suite["cases"][len(anchors):]
    _require([case["id"] for case in generated] == meta.get("dynamic_case_ids"), "Dynamic case list changed")
    exclusion_hashes = set(meta.get("exclusion_prompt_hashes", []))
    _require(set(meta.get("source_question_hashes", [])) <= exclusion_hashes, "Training question exclusions were withdrawn")
    records = meta.get("dynamic_cases", [])
    _require(len(records) == len(generated), "Dynamic case provenance is incomplete")
    for case, record in zip(generated, records):
        prompt_hash = _digest(normalize_question(case["prompt"]))
        _require(len(case["prompt"].encode()) <= _MAX_PROMPT_BYTES, "Dynamic case exceeds its bounded text budget")
        _require(record.get("case_id") == case["id"] and record.get("prompt_sha256") == prompt_hash
                 and record.get("family") in set(_PATTERNS) | set(_SOURCE_FAMILIES.values())
                 and record.get("oracle") == _oracle(record["family"]),
                 "Dynamic case oracle provenance changed")
        _require(prompt_hash not in exclusion_hashes, "Dynamic case leaked a training/replay prompt")
        seed_input = {"version": _VERSION, "base": base_model_id, "training_model": training_model,
                      "revision": profile_revision, "questions": meta["source_question_hashes"],
                      "family": record["family"], "attempt": record.get("attempt")}
        if meta.get("suite_purpose") == "development":
            seed_input.update(purpose="development", dataset_revision=meta.get("dataset_revision"))
        expected_seed = _digest(seed_input)
        _require(record["seed"] == expected_seed and isinstance(record.get("attempt"), int)
                 and not isinstance(record["attempt"], bool) and 0 <= record["attempt"] < 32, "Dynamic seed is not bound to the frozen source/profile")
        prompt, expected, parameters = _variant(record["family"], random.Random(int(record["seed"], 16)))
        if meta.get("parameter_partition_version") in {1, 2}:
            partition = "development" if meta.get("suite_purpose") == "development" else "holdout"
            _require(parameter_partition(record["family"], parameters, version=meta["parameter_partition_version"]) == partition,
                     "Evaluation parameters belong to another partition")
        _require(prompt_hash not in meta.get("reserved_prompt_hashes", [])
                 and _digest([record["family"], parameters]) not in meta.get("reserved_parameter_hashes", []),
                 "Evaluation overlaps previously reserved data")
        _require(case["prompt"] == prompt and case.get("expected_json") == expected and record.get("parameters") == parameters,
                 "Dynamic case no longer matches its independent oracle")
        _require(bool(_numeric_atoms(parameters) - set(meta.get("source_numeric_atoms", {}).get(record["family"], []))),
                 "Dynamic case reused an unchanged source parameter setting")
        exclusion_hashes.add(prompt_hash)
    coverage = meta.get("coverage", {})
    status = coverage.get("status")
    assignments = meta.get("source_question_families", [])
    _require(sorted(item.get("prompt_sha256") for item in assignments) == meta["source_question_hashes"], "Source family assignments changed")
    supported = {item["family"] for item in assignments if item.get("family") is not None}
    _require(supported <= set(_PATTERNS) | set(_SOURCE_FAMILIES.values())
             and supported == {record["family"] for record in records}, "Supported family coverage changed")
    unsupported_count = sum(item.get("family") is None for item in assignments)
    expected_status = "verified" if supported and not unsupported_count else "partial" if supported else "unknown"
    _require(status == expected_status and coverage.get("generated_cases") == len(generated)
             and coverage.get("source_question_count") == len(assignments)
             and coverage.get("supported_question_count") == len(assignments) - unsupported_count
             and len(coverage.get("unsupported_topics", [])) == unsupported_count, "Frozen coverage is invalid")
    overlaps = [case["id"] for case in anchors if _digest(normalize_question(case["prompt"])) in meta["exclusion_prompt_hashes"]]
    _require(meta.get("anchor_training_overlap_ids") == overlaps, "Anchor/training overlap facts changed")
    _require(meta.get("promotion_eligible") is (status == "verified" and bool(generated) and not overlaps), "Unsupported coverage cannot qualify for promotion")
    return suite


def suite_exclusions(suite):
    """Return content/parameter reservations without treating scores as targets."""
    records = {record["case_id"]: record for record in suite.get("metadata", {}).get("dynamic_cases", [])}
    return [{"prompt": case["prompt"], **({key: records[case["id"]][key] for key in ("family", "parameters")}
             if case["id"] in records else {})} for case in suite.get("cases", [])]


def freeze_development_suite(configured_suite, examples, *, admission_suite, dataset_revision, excluded_cases=(), **kwargs):
    """Create a separate revision-bound diagnostic suite; admission never changes."""
    _require(admission_suite.get("metadata", {}).get("suite_purpose", "admission") == "admission",
             "Development must be bound to an admission suite")
    configured = copy.deepcopy(configured_suite)
    # Fixed anchors belong to admission. Development retains a minimal safety
    # and task anchor only when they were explicitly supplied with new IDs;
    # callers normally pass their independent development anchors here.
    admission_prompts = {normalize_question(case["prompt"]) for case in admission_suite.get("cases", [])}
    _require(not any(normalize_question(case["prompt"]) in admission_prompts for case in configured.get("cases", [])),
             "Development anchors overlap admission")
    suite = freeze_evaluation_suite(configured, examples, purpose="development", dataset_revision=dataset_revision,
        partition_version=2, excluded_cases=[*suite_exclusions(admission_suite), *excluded_cases], **kwargs)
    # The admission identity is evidence; re-seal only this newly made artifact.
    metadata = suite["metadata"]
    metadata["admission_freeze_digest"] = admission_suite["metadata"]["freeze_digest"]
    metadata["freeze_digest"] = _digest({key: value for key, value in metadata.items() if key != "freeze_digest"})
    return suite
