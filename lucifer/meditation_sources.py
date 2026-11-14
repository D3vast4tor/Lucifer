"""Source-bound native-head datasets and privacy invalidation.

This module never trains, loads models, or treats tool/provider reasoning as data.
"""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil

from .curriculum import revalidate_example
from .memory import capture_source_snapshot, validate_source_snapshot


VERSION = 1
PARTITIONS = ("train", "development", "heldout")
KINDS = ("meditation_runs", "meditation_features", "meditation_datasets", "meditation_checkpoints",
         "base_candidates", "meditation_admissions", "meditation_evaluations", "meditation_operations")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _native_core(row):
    return {key: row[key] for key in ("native_source_version", "training_mode", "source_kind", "source_authority",
        "conversation_id", "source_message_ids", "source_bindings", "provenance", "messages", "text") if key in row}


def _native_row(store, cid, *, mode, source_kind, authority, message_ids, messages=None, text=None, provenance=None):
    payload = {"native_source_version": 1, "training_mode": mode, "source_kind": source_kind,
        "source_authority": authority, "conversation_id": cid, "source_message_ids": message_ids,
        "source_bindings": [capture_source_snapshot(store, cid, message_ids, require_training=True)],
        "provenance": provenance or {}}
    if messages is not None:
        payload["messages"] = messages
    if text is not None:
        payload["text"] = text
    content_hash = digest(payload)
    ident = "meditation_source_" + content_hash[:32]
    return {**payload, "id": ident, "candidate_id": ident, "content_hash": content_hash,
            "fingerprint": digest(text if text is not None else messages)}


def _native_rows(store, conversation_ids, mode):
    """Project only public text; consent authorizes the mode, never truth claims."""
    from .knowledge import excerpt_question
    from .learning import scrub
    rows = []
    for cid in conversation_ids:
        public = [row for row in store.list("messages", conversation_id=cid)
                  if row.get("role") in {"user", "assistant"} and row.get("source") != "knowledge_attachment"]
        reviews = {row.get("message_id"): row for row in store.list("feedback", conversation_id=cid)}
        if mode == "unsupervised":
            # Keep role text for language modelling; do not ingest arbitrary
            # message fields, tool messages, reasoning or failed partial output.
            for message in public:
                text = scrub(message.get("content", ""))
                if text:
                    rows.append(_native_row(store, cid, mode=mode, source_kind="chat",
                        authority="literal_user_text" if message["role"] == "user" else "public_model_final",
                        message_ids=[message["id"]], text=text,
                        provenance={"public_role": message["role"], "instructions_trusted": False}))
        else:
            user = None
            history = []
            for message in public:
                if message["role"] == "user":
                    user = message
                    history.append(message)
                    continue
                if user is None:
                    continue
                review = reviews.get(message["id"], {})
                correction = scrub(review.get("correction") or "")
                if review.get("label") in {"bad", "user_disliked", "reject"} and not correction:
                    history.append(message)
                    continue
                question, answer = scrub(user.get("content", "")), correction or scrub(message.get("content", ""))
                if not question or not answer:
                    continue
                authority = "user_correction" if correction else "reviewed_public_final" if review.get("label") in {
                    "good", "user_liked"} else "public_model_final"
                # Keep preceding public turns so references such as "continue"
                # retain their task context. Only the last assistant is a target;
                # the tokenizer masks every preceding turn. Bound whole turns,
                # never include provider reasoning or tool-result metadata.
                context, size = [], 0
                for prior in reversed(history):
                    text = scrub(prior.get("content", ""))
                    if not text:
                        continue
                    if context and (len(context) >= 32 or size + len(text) > 65536):
                        break
                    context.append((prior, {"role": prior["role"], "content": text}))
                    size += len(text)
                context.reverse()
                while context and context[0][0]["role"] != "user":
                    context.pop(0)
                rows.append(_native_row(store, cid, mode=mode, source_kind="chat", authority=authority,
                    message_ids=[prior["id"] for prior, _ in context] + [message["id"]],
                    messages=[turn for _, turn in context] + [{"role": "assistant", "content": answer}],
                    provenance={"public_final_message_id": message["id"], "instructions_trusted": False}))
                history.append(message)
        for document in store.list("knowledge_documents", conversation_id=cid):
            interacted = {binding["chunk"] for receipt in store.list("knowledge_interactions", conversation_id=cid)
                          if receipt.get("document_id") == document["id"] for binding in receipt.get("chunks", [])}
            for chunk in store.list("knowledge_chunks", document_id=document["id"]):
                if chunk["chunk"] not in interacted:
                    continue
                text = scrub(chunk["text"])
                if not text:
                    continue
                authority = "literal_document" if chunk["authority"] == "literal" else chunk["authority"]
                from .problem_solution import FORMAT, parse_record
                provided = (parse_record(chunk["text"]) if
                    document.get("metadata", {}).get("problem_solution_format") == FORMAT else None)
                if provided is not None and mode == "supervised":
                    authority = "document_provided_solution"
                kwargs = {"text": text} if mode == "unsupervised" else {"messages": [
                    {"role": "user", "content": provided["question"] if provided else excerpt_question(document, chunk)},
                    {"role": "assistant", "content": provided["answer"] if provided else text}]}
                rows.append(_native_row(store, cid, mode=mode, source_kind="document", authority=authority,
                    message_ids=[document["message_id"]], provenance={"knowledge_document_id": document["id"],
                        "chunk": chunk["chunk"], "document_sha256": document["sha256"], "instructions_trusted": False,
                        "authority_is_truth_verification": False,
                        **({"provided_solution_id": provided["id"], "provided_verification": provided["verification"],
                            "provided_sources": provided["sources"],
                            "provided_lineage": provided.get("lineage", [])} if provided else {})}, **kwargs))
    if mode == "supervised":
        from .memory import source_context_superseded
        rows = [row for row in rows if not source_context_superseded(store, row["source_bindings"][0])]
    return rows


def _validate_row(store, row, *, current=None):
    if row.get("native_source_version") != 1:
        return revalidate_example(store, row)
    if digest(_native_core(row)) != row.get("content_hash"):
        raise ValueError("Meditation public source content changed")
    if digest(row["text"] if "text" in row else row["messages"]) != row.get("fingerprint"):
        raise ValueError("Meditation public source fingerprint changed")
    for snapshot in row["source_bindings"]:
        validate_source_snapshot(store, snapshot, require_training=True,
                                 allow_superseded=row["training_mode"] == "unsupervised")
    current = current if current is not None else {item["id"]: item for item in
        _native_rows(store, [row["conversation_id"]], row["training_mode"])}
    observed = current.get(row["id"])
    if observed is None or _native_core(observed) != _native_core(row):
        raise ValueError("Meditation public source no longer matches its saved chat/document")
    return True


def _members(store, row, *, edges=None):
    members = {"conversation:" + row["conversation_id"], "example:" + row["fingerprint"]}
    for binding in row.get("source_bindings", []):
        members.add("conversation:" + binding["conversation_id"])
        members.update("document:" + item["sha256"] for item in binding.get("knowledge", []))
    document_id = row.get("provenance", {}).get("knowledge_document_id")
    if document_id:
        members.add("document:" + store.get("knowledge_documents", document_id)["sha256"])
    # A final answer's referenced sources belong to the same dependency group.
    for message in store.list("messages", conversation_id=row["conversation_id"]):
        if message["id"] not in row.get("source_message_ids", []):
            continue
        for ident in message.get("knowledge_source_ids", []):
            members.add("document:" + store.get("knowledge_documents", ident)["sha256"])
    # The shared curriculum already removes exact duplicate excerpt rows. Keep
    # their other conversation/document dependencies despite that deduplication.
    edges = _dependency_edges(store) if edges is None else edges
    while True:
        connected = {value for edge in edges if members.intersection(edge) for value in edge}
        if connected <= members:
            break
        members.update(connected)
    return members


def _dependency_edges(store):
    eligible = {item["id"] for item in store.list("conversations")
                if item.get("logging_enabled") and item.get("training_enabled")}
    from .problem_solution import FORMAT, parse_record, source_family
    edges = []
    chunks = {}
    for chunk in store.list("knowledge_chunks"):
        chunks.setdefault(chunk["document_id"], []).append(chunk)
    for doc in store.list("knowledge_documents"):
        if doc["conversation_id"] not in eligible:
            continue
        cid = "conversation:" + doc["conversation_id"]
        edges.append((cid, "document:" + doc["sha256"]))
        uris = [doc.get("source", "")]
        uris.extend(origin.get("source", "") for origin in doc.get("metadata", {}).get("import_origins", []))
        if doc.get("metadata", {}).get("problem_solution_format") == FORMAT:
            for chunk in chunks.get(doc["id"], []):
                record = parse_record(chunk["text"])
                uris.extend(ref["url"] for ref in record["sources"])
                # Source byte identities merge local copies and remote references.
                edges.extend((cid, "document:" + ref["sha256"]) for ref in record["sources"])
                edges.extend((cid, "provided_dataset:" + value) for value in record.get("lineage", []))
        edges.extend((cid, "source_family:" + family) for uri in uris if (family := source_family(uri)))
    return edges


def _group_rows(store, rows, *, source_roles=None):
    """Connected conversation/document components, including duplicate documents."""
    components = []
    edges = _dependency_edges(store)
    for row in rows:
        members = _members(store, row, edges=edges)
        attached = [component for component in components if component[0].intersection(members)]
        for component in attached:
            components.remove(component)
            members.update(component[0])
        components.append((members, [row] + [item for component in attached for item in component[1]]))
    # Existing reservations preserve roles through refreshes and subset selection.
    reservations = store.list("meditation_source_partitions")
    source_roles = source_roles or {}
    assigned, fresh = [], []
    for members, examples in components:
        previous = [record for record in reservations if members.intersection(record["members"])]
        roles = {record["partition"] for record in previous}
        roles.update(role for cid, role in source_roles.items() if "conversation:" + cid in members)
        if len(roles) > 1:
            raise ValueError("A source dependency connects conflicting reserved or requested training/development/heldout groups")
        all_members = members | {value for record in previous for value in record["members"]}
        ident = min((record["id"] for record in previous), default="meditation_group_" + digest(sorted(all_members))[:32])
        item = {"id": ident, "members": sorted(all_members), "examples": examples,
                "partition": next(iter(roles), None)}
        (assigned if roles else fresh).append(item)
    fresh.sort(key=lambda item: digest(item["members"]))
    # Precommit roles without looking at scores. Three independent groups are
    # the minimum needed even to represent all three roles; admission has its
    # own stricter group-count/statistical gate.
    used = {item["partition"] for item in assigned}
    for item in fresh:
        if "heldout" not in used and len(fresh) + len(assigned) >= 3:
            item["partition"] = "heldout"
        elif "development" not in used and len(fresh) + len(assigned) >= 3:
            item["partition"] = "development"
        elif "train" not in used:
            item["partition"] = "train"
        else:
            bucket = int(digest(item["members"])[:8], 16) % 10
            item["partition"] = "heldout" if bucket < 2 else "development" if bucket < 4 else "train"
        used.add(item["partition"])
        assigned.append(item)
    return assigned


async def prepare_meditation_sources(store, request, *, progress=None, run_id=None, reserve=True):
    """Prepare consented public sources without requiring a separate review step."""
    selected = request.get("conversation_ids")
    if selected is not None and (not isinstance(selected, list) or any(not isinstance(cid, str) for cid in selected)):
        raise ValueError("conversation_ids must be a list of saved conversation IDs")
    conversations = store.list("conversations")
    known = {row["id"] for row in conversations}
    if selected is not None and set(selected) - known:
        raise ValueError("A selected training conversation does not exist")
    eligible = [row["id"] for row in conversations if row.get("logging_enabled") and row.get("training_enabled")
                and (selected is None or row["id"] in selected)]
    if selected is not None and set(selected) - set(eligible):
        raise ValueError("Selected conversations require current logging and training consent")
    mode = request.get("mode", request.get("training_mode", "supervised"))
    if mode not in {"supervised", "unsupervised"}:
        raise ValueError("Meditation mode must be supervised or unsupervised")
    rows = _native_rows(store, eligible, mode)
    consumed = sorted({row["conversation_id"] for row in rows})
    source_roles = request.get("source_roles") or {}
    if (not isinstance(source_roles, dict) or set(source_roles) - set(consumed)
            or any(role not in PARTITIONS for role in source_roles.values())):
        raise ValueError("Source roles must name consumed eligible chats and train/development/heldout roles")
    groups = _group_rows(store, rows, source_roles=source_roles)
    partitions = {role: [] for role in PARTITIONS}
    references, group_manifest = [], []
    message_order = {row["id"]: index for index, row in enumerate(store.list("messages"))}
    for group in groups:
        seen = set()
        ordered = sorted(group["examples"], key=lambda row: (min(
            (message_order.get(ident, len(message_order)) for ident in row.get("source_message_ids", [])),
            default=len(message_order)), row.get("candidate_id", "")))
        for row in ordered:
            if row["fingerprint"] in seen:
                continue
            seen.add(row["fingerprint"])
            example = {**copy.deepcopy(row), "meditation_group_id": group["id"],
                       "meditation_partition": group["partition"]}
            partitions[group["partition"]].append(example)
            references.append({"candidate_id": row.get("candidate_id") or row["id"],
                "content_hash": row["content_hash"], "fingerprint": row["fingerprint"],
                "group": group["id"], "partition": group["partition"]})
        reservation = {key: group[key] for key in ("id", "members", "partition")}
        if reserve:
            store.put("meditation_source_partitions", reservation)
        group_manifest.append(reservation)
    snapshots = [capture_source_snapshot(store, cid, require_training=True) for cid in consumed]
    interactions = sorted((row for cid in consumed for row in store.list("knowledge_interactions", conversation_id=cid)),
                          key=lambda row: row["id"])
    manifest = {"schema_version": VERSION, "run_id": run_id, "source_conversation_ids": consumed,
        "selected_conversation_ids": sorted(eligible),
        "training_mode": mode,
        "source_snapshots": snapshots, "example_refs": sorted(references, key=lambda ref: ref["candidate_id"]),
        "groups": sorted(group_manifest, key=lambda group: group["id"]),
        "interaction_refs": [{"id": row["id"], "binding_sha256": row["binding_sha256"]} for row in interactions],
        "raw_thinking_retained": False, "raw_tool_traces_retained": False}
    manifest["manifest_sha256"] = digest(manifest)
    metrics = {"train": len(partitions["train"]), "development": len(partitions["development"]),
        "heldout": len(partitions["heldout"]), "independent_groups": len(groups), "source_interactions": len(interactions),
        "partition_coverage": all(partitions.values()), "training_mode": mode, "manual_review_required": False,
        "source_authorities": sorted({row["source_authority"] for row in rows}), "examples": len(rows),
        "selected_chats": len(eligible), "consumed_chats": len(consumed), "empty_selected_chats": len(eligible) - len(consumed)}
    return {"schema_version": VERSION, "examples": partitions["train"], "development_examples": partitions["development"],
        "holdout_examples": partitions["heldout"], "manifest": manifest, "source_conversation_ids": consumed,
        "metrics": metrics, "pending": []}


def revalidate_meditation_sources(store, bundle):
    manifest = bundle.get("manifest", {})
    if bundle.get("schema_version") != VERSION or manifest.get("schema_version") != VERSION:
        raise ValueError("Unknown Meditation source contract")
    if digest({key: value for key, value in manifest.items() if key != "manifest_sha256"}) != manifest.get("manifest_sha256"):
        raise ValueError("Meditation source manifest changed")
    if bundle.get("automatic_validation") is not None:
        from .behavioral_resources import validate_numeric_resources
        validate_numeric_resources(bundle["automatic_validation"])
        if manifest.get("automatic_validation_sha256") != bundle["automatic_validation"]["sha256"]:
            raise ValueError("Automatic validation differs from its source manifest binding")
    elif manifest.get("automatic_validation_sha256"):
        raise ValueError("Automatic validation resource is missing")
    for snapshot in manifest["source_snapshots"]:
        validate_source_snapshot(store, snapshot, require_training=True)
    for ref in manifest["interaction_refs"]:
        row = store.get("knowledge_interactions", ref["id"])
        if row.get("binding_sha256") != ref["binding_sha256"]:
            raise ValueError("A consumed document interaction changed")
        binding = {key: value for key, value in row.items() if key not in {"id", "binding_sha256", "created_at", "updated_at"}}
        from .knowledge import fingerprint
        if fingerprint(binding) != row["binding_sha256"]:
            raise ValueError("A consumed document interaction receipt changed")
        document = store.get("knowledge_documents", row["document_id"])
        if document["sha256"] != row["source_sha256"]:
            raise ValueError("A consumed document changed")
        for chunk in row["chunks"]:
            current = store.get("knowledge_chunks", f"{document['id']}:{chunk['chunk']}")
            if hashlib.sha256(current["text"].encode()).hexdigest() != chunk["sha256"]:
                raise ValueError("A consumed document chunk changed")
    refs, seen = [], set()
    modes = {row["training_mode"] for key in ("examples", "development_examples", "holdout_examples")
             for row in bundle[key] if row.get("native_source_version") == 1}
    current = {row["id"]: row for mode in modes
               for row in _native_rows(store, manifest["source_conversation_ids"], mode)}
    for role, key in (("train", "examples"), ("development", "development_examples"), ("heldout", "holdout_examples")):
        for row in bundle[key]:
            _validate_row(store, row, current=current)
            if row["fingerprint"] in seen or row["meditation_partition"] != role:
                raise ValueError("Meditation source partitions overlap or changed")
            seen.add(row["fingerprint"])
            refs.append({"candidate_id": row.get("candidate_id") or row["id"], "content_hash": row["content_hash"],
                "fingerprint": row["fingerprint"], "group": row["meditation_group_id"], "partition": role})
    if sorted(refs, key=lambda ref: ref["candidate_id"]) != manifest["example_refs"]:
        raise ValueError("Meditation examples no longer match the source manifest")
    groups = {group["id"]: group for group in manifest["groups"]}
    observed_groups = [{key: group[key] for key in ("id", "members", "partition")}
                       for group in _group_rows(store, list(current.values()))]
    observed_groups = {group["id"]: group for group in observed_groups}
    for ident, expected in groups.items():
        observed = observed_groups.get(ident)
        # New duplicate copies may expand a reserved component without
        # repartitioning consumed evidence or changing its original binding.
        if (observed is None or observed["partition"] != expected["partition"]
                or not set(expected["members"]) <= set(observed["members"])):
            raise ValueError("Meditation source dependency groups changed")
    for ref in refs:
        if ref["partition"] != groups[ref["group"]]["partition"]:
            raise ValueError("Meditation group partition changed")
    return True


def _worker_dataset(bundle):
    datasets = {}
    for partition, key in (("train", "examples"), ("development", "development_examples"), ("holdout", "holdout_examples")):
        datasets[partition] = [{**({"text": row["text"]} if "text" in row else {"messages": row["messages"]}),
            "group": row["meditation_group_id"], "fingerprint": row["fingerprint"]} for row in bundle[key]]
    if bundle.get("automatic_validation") is not None:
        from .behavioral_resources import validate_numeric_resources
        value = bundle["automatic_validation"]
        validate_numeric_resources(value)
        for partition, examples in value['datasets'].items():
            if datasets[partition]:
                raise ValueError('Automatic validation cannot replace an existing source partition')
            if bundle['manifest'].get('training_mode') == 'unsupervised':
                datasets[partition] = [{'text': '\n'.join(turn['content'] for turn in row['messages']),
                                        'group': row['group'], 'fingerprint': row['fingerprint']} for row in examples]
            else:
                datasets[partition] = copy.deepcopy(examples)
    return {"datasets": datasets}


def _sha_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def commit_source_revision(store, run_id, run_dir, bundle, previous_revision_refs=None):
    """Seal an append-only source revision after validating all consumed lineage.

    Previous references may point to the original run-root sources.json during
    migration. Optimizer state may refresh only after this proves that old
    training, development and heldout dependencies were retained unchanged.
    """
    from .adaptive import atomic_json, canonical_json
    run_dir = Path(run_dir)
    expected = store.path.parent / "meditation" / run_id
    if run_dir.is_symlink() or run_dir.resolve() != expected.resolve():
        raise ValueError("Source revision must stay in its owned Meditation run directory")
    root = run_dir.resolve()
    refs = copy.deepcopy(previous_revision_refs or [])
    revalidate_meditation_sources(store, bundle)
    mode = bundle["manifest"].get("training_mode", "supervised")
    new_examples = {ref["fingerprint"]: (ref["group"], ref["partition"], ref["content_hash"])
                    for ref in bundle["manifest"]["example_refs"]}
    previous_data = None
    for ref in refs:
        source = Path(ref.get("source_cache_path", ref.get("path", "")))
        expected_sha = ref.get("source_bundle_sha256", ref.get("sha256"))
        if source.is_symlink() or not source.resolve().is_relative_to(root) or not source.is_file():
            raise ValueError("A consumed Meditation source revision is unavailable or outside the run")
        if not expected_sha or _sha_file(source) != expected_sha:
            raise ValueError("A consumed Meditation source revision changed")
        previous = json.loads(source.read_text())
        revalidate_meditation_sources(store, previous)
        if previous["manifest"].get("training_mode", "supervised") != mode:
            raise ValueError("A Meditation run cannot change its consumed training mode")
        for row in previous["manifest"]["example_refs"]:
            binding = (row["group"], row["partition"], row["content_hash"])
            if new_examples.get(row["fingerprint"]) != binding:
                raise ValueError("Source refresh withdrew or repartitioned consumed Meditation evidence")
        if ref.get("source_manifest_sha256") and ref["source_manifest_sha256"] != previous["manifest"]["manifest_sha256"]:
            raise ValueError("A consumed source manifest changed")
        previous_data = hashlib.sha256(canonical_json(_worker_dataset(previous)) + b"\n").hexdigest()
        if ref.get("dataset_path"):
            dataset = Path(ref["dataset_path"])
            if (dataset.is_symlink() or not dataset.resolve().is_relative_to(root) or not dataset.is_file()
                    or _sha_file(dataset) != previous_data
                    or ref.get("dataset_sha256", previous_data) != previous_data):
                raise ValueError("A consumed Meditation token-input dataset changed")
    manifest_sha = bundle["manifest"]["manifest_sha256"]
    revision = root / "source-revisions" / manifest_sha
    if revision.is_symlink() or not revision.resolve().is_relative_to(root):
        raise ValueError("Invalid Meditation source revision directory")
    revision.mkdir(parents=True, exist_ok=True, mode=0o700)
    dataset_value = _worker_dataset(bundle)
    source_path, dataset_path = revision / "sources.json", revision / "dataset.json"
    for path, value in ((source_path, bundle), (dataset_path, dataset_value)):
        expected_bytes = canonical_json(value) + b"\n"
        if path.is_symlink():
            raise ValueError("Source revision artifacts cannot be symlinks")
        if path.exists():
            if not path.is_file() or _sha_file(path) != hashlib.sha256(expected_bytes).hexdigest():
                raise ValueError("Immutable Meditation source revision was overwritten")
        else:
            atomic_json(path, value)
        path.chmod(0o600)
    ref = {"source_cache_path": str(source_path), "source_bundle_sha256": _sha_file(source_path),
           "source_manifest_sha256": manifest_sha, "dataset_path": str(dataset_path),
           "dataset_sha256": _sha_file(dataset_path)}
    if not any(old.get("source_bundle_sha256", old.get("sha256")) == ref["source_bundle_sha256"]
               and old.get("source_cache_path", old.get("path")) == ref["source_cache_path"] for old in refs):
        refs.append(ref)
    return {**ref, "source_revision_refs": refs,
            "data_refreshed": previous_data is not None and previous_data != ref["dataset_sha256"]}


def _has_reference(item, *, conversation_id=None, message_id=None, candidate_id=None):
    from .memory import _candidate_matches, _source_matches
    if conversation_id is not None and _source_matches(item, conversation_id, message_id):
        return True
    if candidate_id is not None and _candidate_matches(item, candidate_id):
        return True
    return any(_has_reference(value, conversation_id=conversation_id, message_id=message_id, candidate_id=candidate_id)
               for value in item.values() if isinstance(value, dict)) or any(
        _has_reference(value, conversation_id=conversation_id, message_id=message_id, candidate_id=candidate_id)
        for values in item.values() if isinstance(values, list) for value in values if isinstance(value, dict))


def _purge_owned(store, record):
    """Erase source-bearing managed caches only, never original/active weights."""
    root = store.path.parent.resolve()
    owned_roots = [(root / name).resolve() for name in ("meditation", "datasets", "features", "evaluations")]
    paths = []
    for key in ("dataset_path", "feature_path", "features_path", "cache_path", "source_cache_path"):
        value = record.get(key)
        if isinstance(value, str):
            paths.append(value)
            if key == "source_cache_path" and Path(value).name == "sources.json":
                # Native orchestration commits the source bundle and its token
                # input side by side. Both contain revoked source material.
                paths.append(str(Path(value).with_name("dataset.json")))
    paths.extend(value for value in record.get("source_paths", []) if isinstance(value, str))
    # Worker diagnostic configurations contain real source token IDs even when
    # their numerical outputs are outside the main compact feature cache.
    run_ids = {record.get("run_id"), record.get("training_run_id"), record.get("id")}
    for run_id in run_ids:
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
            continue
        directory = root / "meditation" / run_id
        if directory.is_symlink() or directory.resolve().parent != (root / "meditation").resolve() or not directory.is_dir():
            continue
        paths.extend(str(path) for path in directory.glob("native-verify-*") if not path.is_symlink())
        paths.extend(str(path) for path in directory.glob("admission-*") if not path.is_symlink())
        paths.extend(str(directory / name) for name in ("features", "source-revisions", "behavioral") if (directory / name).exists())
        # Remove source-bearing configs while retaining learned heads, master
        # weights, moments and optimizer metadata for blocked historical records.
        for base, directories, files in os.walk(directory, followlinks=False):
            directories[:] = [name for name in directories if not (Path(base) / name).is_symlink()]
            if "worker_config.json" in files:
                paths.append(str(Path(base) / "worker_config.json"))
    for value in dict.fromkeys(paths):
        path = Path(value)
        if path.is_symlink():
            if any(path.parent.resolve().is_relative_to(base) for base in owned_roots):
                path.unlink(missing_ok=True)
            continue
        resolved = path.resolve()
        if not any(resolved.is_relative_to(base) and resolved != base for base in owned_roots):
            continue
        if resolved.is_file():
            resolved.unlink()
        elif resolved.is_dir():
            try:
                shutil.rmtree(resolved)
            except FileNotFoundError:
                pass
            except OSError as exc:
                # An owned worker may still be closing/creating a shard. Its
                # controller repeats the purge after all owned processes exit.
                if exc.errno != errno.ENOTEMPTY:
                    raise


def purge_revoked_meditation_artifacts(store, run_id):
    """Final idempotent cleanup after the owning worker has exited."""
    for kind in KINDS:
        for row in store.list(kind):
            if (row.get("id") == run_id or row.get("run_id") == run_id or row.get("training_run_id") == run_id):
                if row.get("source_revoked") or row.get("source_invalidated"):
                    _purge_owned(store, row)


def _redact_source_metadata(value):
    if isinstance(value, dict):
        source_keys = {"diagnostic_ids", "ids", "labels", "segments", "text", "messages", "token_ids", "tokens"}
        result = {}
        for key, item in value.items():
            if key in source_keys and not isinstance(item, (bool, int, float, type(None))):
                sealed = isinstance(item, dict) and item.get("unavailable") is True and set(item) == {"unavailable", "sha256"}
                result[key] = item if sealed else {"unavailable": True, "sha256": digest(item)}
            else:
                result[key] = _redact_source_metadata(item)
        return result
    if isinstance(value, list):
        return [_redact_source_metadata(item) for item in value]
    return value


def invalidate_meditation_sources(store, *, conversation_id=None, message_id=None,
                                  candidate_id=None, reason="source_changed"):
    """Invalidate consumed native lineage and descendants without model effects."""
    records = {kind: store.list(kind) for kind in KINDS}
    invalid = {(kind, row["id"]) for kind, rows in records.items() for row in rows
               if _has_reference(row, conversation_id=conversation_id, message_id=message_id, candidate_id=candidate_id)}
    while True:
        ids = {ident for _, ident in invalid}
        run_ids = {row.get("run_id") or row.get("training_run_id") for kind, rows in records.items()
                   for row in rows if (kind, row["id"]) in invalid}
        run_ids.discard(None)
        added = {(kind, row["id"]) for kind, rows in records.items() for row in rows
                 if row["id"] in run_ids or row.get("run_id") in ids or row.get("training_run_id") in ids
                 or row.get("parent_base_id") in ids or row.get("base_candidate_id") in ids
                 or row.get("candidate_id") in ids or row.get("candidate_base_id") in ids}
        if added <= invalid:
            break
        invalid |= added
    for kind, rows in records.items():
        for row in rows:
            if (kind, row["id"]) not in invalid:
                continue
            changes = {"source_invalidated": True, "source_revoked": True, "artifact_revoked": True,
                       "source_invalidation_reason": reason}
            if kind in {"meditation_admissions", "meditation_evaluations"}:
                changes["status"] = "revoked"
            elif kind == "base_candidates" and row.get("status") not in {"active", "archived"}:
                changes["status"] = "revoked"
            if conversation_id:
                changes["source_invalidation_conversation_id"] = conversation_id
            redacted = _redact_source_metadata(row)
            changes.update({key: value for key, value in redacted.items() if row.get(key) != value})
            if any(row.get(key) != value for key, value in changes.items()):
                store.update(kind, row["id"], **changes)
            _purge_owned(store, row)
    return {"meditation_records_invalidated": len(invalid)}
