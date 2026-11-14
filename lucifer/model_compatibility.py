"""One pure adapter compatibility decision for serving and registry projections."""
from __future__ import annotations


def adapter_compatibility(model, adapter):
    """Consume an initialized profile snapshot; never probe files or hardware."""
    if hasattr(model, "model_dump"):
        model = model.model_dump()
    if not model:
        return {"compatible": False, "reason": "Load a compatible base model"}
    if adapter.get("source_revoked") or adapter.get("source_invalidated"):
        return {"compatible": False, "reason": "Source consent or evidence is no longer valid"}
    backend = model.get("backend")
    if backend == "strata":
        return {"compatible": False, "reason": "This Strata runtime does not support adapters"}
    base = adapter.get("base_model_id") or adapter.get("base_model")
    if base and base not in {model.get("id"), model.get("path"), model.get("training_model")}:
        return {"compatible": False, "reason": "Expert belongs to a different base model"}
    actual = model.get("base_revision")
    expected = adapter.get("base_revision") or adapter.get("base_weight_revision")
    if actual and expected != actual or expected and expected != actual:
        return {"compatible": False, "reason": "Exact base weight revision is incompatible or unknown"}
    if backend == "demo":
        compatible = adapter.get("format") == "demo" or adapter.get("backend") == "demo"
        return {"compatible": compatible, "reason": "" if compatible else "Real adapters cannot use the demo backend"}
    if backend == "llama.cpp":
        compatible = str(adapter.get("path", "")).lower().endswith(".gguf")
        return {"compatible": compatible, "reason": "" if compatible else "This runtime requires a GGUF LoRA adapter"}
    if backend not in {"vllm"} and model.get("adapter_backend") != "vllm":
        return {"compatible": False, "reason": "Runtime has no supported adapter interface"}
    if not model.get("enable_lora", True):
        return {"compatible": False, "reason": "Adapters are disabled for this profile"}
    return {"compatible": True, "reason": ""}


def decorate_adapters(model, adapters):
    result = []
    for adapter in adapters:
        assessment = adapter_compatibility(model, adapter)
        result.append({**adapter, **assessment, "compatibility_reason": assessment["reason"],
                       "compatibility": assessment})
    return result
