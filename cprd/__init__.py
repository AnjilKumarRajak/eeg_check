"""Prior-Residual Decoding -- independent Claude implementation.

Preserves the proposal's immutable novelty: an EEG evidence encoder, an information-gain
objective over a frozen language prior, and prior-residual decoding.

See CLAUDE_ARCHITECTURE.md for the equations, the measured prior selection, and the list
of claims this implementation deliberately does not support.
"""
from .data import (PRDDataset, Sentence, assert_no_text_leakage, collate,
                   read_split, recover_word_spans, retokenize_split, split_stats)
from .encoder import EvidenceEncoder, RankBudgetedTilt
from .evaluate import DIResult, evaluate, zero_gamma_check
from .model import ModelConfig, PriorResidualModel
from .objective import assert_upper_bound, information_gain, sentence_information_gain
from .priors import BartEmptySourcePrior, CausalLMPrior, ReferenceLM, build_prior

__all__ = [
    "PRDDataset", "Sentence", "assert_no_text_leakage", "collate", "read_split",
    "recover_word_spans", "retokenize_split", "split_stats",
    "EvidenceEncoder", "RankBudgetedTilt",
    "DIResult", "evaluate", "zero_gamma_check",
    "ModelConfig", "PriorResidualModel",
    "assert_upper_bound", "information_gain", "sentence_information_gain",
    "BartEmptySourcePrior", "CausalLMPrior", "ReferenceLM", "build_prior",
]
