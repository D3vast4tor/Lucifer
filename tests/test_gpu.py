import pytest

from lucifer.gpu import discover_gpus, training_placement


def inventory(backend="cuda", count=3):
    return {"gpus": [{"index": str(i), "id": f"{backend}-{i}", "backend": backend, "name": f"GPU {i}"} for i in range(count)]}


def test_all_devices_and_inherited_masks():
    result = training_placement(inventory=inventory(), environ={})
    assert result["device_count"] == 3
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2"
    result = training_placement(inventory=inventory(), environ={"CUDA_VISIBLE_DEVICES": "2"})
    assert result["device_count"] == 1 and result["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert training_placement(inventory=inventory(), environ={"CUDA_VISIBLE_DEVICES": "-1"})["device_count"] == 0


def test_uuid_selection_single_gpu_and_cpu():
    result = training_placement(["cuda-2", "cuda-0"], inventory=inventory(), environ={})
    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "2,0"
    result = training_placement(["2", "0"], multi_gpu=False, inventory=inventory(), environ={})
    assert result["device_count"] == 1 and result["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    result = training_placement([], inventory=inventory(), environ={})
    assert result["backend"] == "cpu" and result["device_count"] == 0
    with pytest.raises(ValueError, match="duplicates"):
        training_placement(["0", "0"], inventory=inventory(), environ={})
    with pytest.raises(ValueError, match="duplicate aliases"):
        training_placement(["0", "cuda-0"], inventory=inventory(), environ={})
    with pytest.raises(ValueError, match="unavailable"):
        training_placement(["9"], inventory=inventory(), environ={})


def test_rocm_and_mixed_vendors():
    result = training_placement(inventory=inventory("rocm", 2), environ={"ROCR_VISIBLE_DEVICES": "1"})
    assert result["env"] == {"HIP_VISIBLE_DEVICES": "1", "ROCR_VISIBLE_DEVICES": "1"}
    mixed = {"gpus": inventory("cuda", 1)["gpus"] + inventory("rocm", 1)["gpus"]}
    with pytest.raises(ValueError, match="cannot combine"):
        training_placement(inventory=mixed, environ={})
    assert training_placement(["cuda-0"], inventory=mixed, environ={})["backend"] == "cuda"


def test_inventory_lists_connected_without_claiming_compute(monkeypatch):
    from lucifer import gpu
    def run(args):
        if args[0] == "nvidia-smi":
            return "0, GPU-123, 00000000:01:00.0, NVIDIA RTX, 12000, 500, 11500\n"
        if args[0] == "lspci":
            return '0000:01:00.0 "VGA compatible controller" "NVIDIA Corporation" "RTX"\n0000:00:02.0 "Display controller" "Intel Corporation" "Integrated"\n'
        return ""
    monkeypatch.setattr(gpu, "_run", run)
    result = discover_gpus()
    assert len(result["gpus"]) == 2
    assert result["cuda_device_count"] == 1
    assert result["gpus"][0]["free_mb"] == 11500
    assert result["gpus"][1]["compute_available"] is False
