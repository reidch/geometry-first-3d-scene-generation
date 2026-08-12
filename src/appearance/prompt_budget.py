from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptBudgetResult:
    text: str
    token_count: int
    original_token_count: int
    compressed: bool


def _fallback_count(text: str) -> int:
    # Conservative approximation when tokenizer is unavailable.
    # CLIP tokenization often produces more tokens than whitespace splitting.
    words = text.replace(',', ' ').replace('.', ' ').replace(';', ' ').split()
    return max(1, int(round(len(words) * 1.35)))


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            # Counting by calling ``tokenizer(...)`` on an over-length CLIP prompt
            # makes Transformers emit its 77-token warning before we have a chance
            # to shorten the prompt. ``tokenize`` performs the same lexical split
            # without running model-length validation, so prompt budgeting remains
            # exact while the console only reports the final bounded prompt.
            pieces = tokenizer.tokenize(text)
            special = 0
            if hasattr(tokenizer, "num_special_tokens_to_add"):
                special = int(tokenizer.num_special_tokens_to_add(pair=False))
            return len(pieces) + special
        except Exception:
            try:
                # Compatibility fallback for unusual tokenizers that do not expose
                # ``tokenize``. ``verbose=False`` suppresses their length warning.
                ids = tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )
                return len(ids)
            except Exception:
                pass
    return _fallback_count(text)


def fit_prompt(text: str, max_tokens: int = 72, tokenizer=None) -> PromptBudgetResult:
    text = ' '.join(str(text).split())
    original = count_tokens(text, tokenizer)
    if original <= max_tokens:
        return PromptBudgetResult(text, original, original, False)

    # Remove lowest-priority trailing clauses first.
    clauses = [c.strip(' ,.;') for c in text.replace(';', '.').split('.') if c.strip()]
    kept = []
    for clause in clauses:
        candidate = '. '.join(kept + [clause])
        if count_tokens(candidate, tokenizer) <= max_tokens:
            kept.append(clause)
        else:
            break
    if kept:
        fitted = '. '.join(kept)
    else:
        words = text.split()
        lo, hi = 1, len(words)
        fitted = words[0]
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = ' '.join(words[:mid])
            if count_tokens(candidate, tokenizer) <= max_tokens:
                fitted = candidate
                lo = mid + 1
            else:
                hi = mid - 1

    final_count = count_tokens(fitted, tokenizer)
    return PromptBudgetResult(fitted, final_count, original, True)
