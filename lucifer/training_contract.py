"""Shared teacher contract, loaded once rather than on chat/token/render paths."""
from functools import lru_cache
from importlib.resources import files
import hashlib


@lru_cache(maxsize=1)
def _master():
    return files('lucifer.resources').joinpath('training_master_prompt.txt').read_text()


def master_prompt(mode):
    if mode == 'sleep':
        specific = ('SLEEP: Consolidate verified experience into compatible adapters/experts. '
                    'Preserve the selected student base, architecture, source consent and replay. '
                    'Do not claim that adapter learning replaces or updates the primary base weights.')
    elif mode == 'meditation':
        specific = ('MEDITATION: Prepare verified supervised targets or faithful unsupervised source text '
                    'for the PRIMARY model output head. Its backbone, embeddings, norms and other tensors '
                    'stay frozen. Do not substitute the compact Sleep student or a LoRA adapter. '
                    'The controller alone may admit an immutable base revision; activation is explicit.')
    else:
        raise ValueError('Unknown training mode')
    return _master().strip() + '\n\n' + specific


def prompt_receipt(mode):
    value = master_prompt(mode)
    return {'version': 1, 'mode': mode, 'sha256': hashlib.sha256(value.encode()).hexdigest()}
