"""Optional CPU Torch/PEFT checks; no model download or GPU initialization."""
import pytest

from lucifer.train_worker import capture_embedding_precision, restore_frozen_embedding_precision

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")


class VocabularyModel(torch.nn.Module):
    is_loaded_in_4bit = True

    def __init__(self, tied):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 16, dtype=torch.bfloat16)
        self.norm = torch.nn.LayerNorm(16, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
        if tied:
            self.head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpointing_options = gradient_checkpointing_kwargs


@pytest.mark.parametrize("tied", [True, False])
def test_real_peft_preparation_restores_frozen_vocabulary_precision_and_gradients(tied):
    torch.manual_seed(7)
    model = VocabularyModel(tied)
    original_input, original_output = model.embedding.weight, model.head.weight
    recorded = capture_embedding_precision(model)
    peft.prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.embedding.weight.dtype == torch.float32
    assert model.head.weight.dtype == torch.float32
    assert model.norm.weight.dtype == torch.float32
    policy = restore_frozen_embedding_precision(recorded)
    assert model.embedding.weight is original_input and model.head.weight is original_output
    assert (model.embedding.weight is model.head.weight) is tied
    assert model.embedding.weight.dtype == model.head.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == model.norm.bias.dtype == torch.float32
    assert policy["unique_embedding_parameters"] == (1 if tied else 2)
    assert policy["embedding_bytes_saved"] == 32 * 16 * 2 * (1 if tied else 2)
    assert model.checkpointing_options == {"use_reentrant": False}
    assert all(not parameter.requires_grad and parameter.device.type == "cpu" for parameter in model.parameters())

    # A newly inserted trainable path still receives finite gradients through
    # the restored, frozen vocabulary head; the base matrices stay untouched.
    adapter = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    hidden = model.norm(model.embedding(torch.tensor([[1, 4, 7, 2]])))
    logits = model.head(hidden + adapter(hidden))
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 32), torch.tensor([4, 7, 2, 8]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(adapter.weight.grad).all()
    assert adapter.weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad is None and model.head.weight.grad is None
    assert model.norm.weight.grad is None


def test_precision_policy_refuses_to_recast_trainable_base_weights():
    model = VocabularyModel(True)
    with pytest.raises(ValueError, match="already frozen"):
        restore_frozen_embedding_precision(capture_embedding_precision(model))
