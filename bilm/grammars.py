"""Deterministic synthetic grammars and data generators for controlled evaluation."""
from __future__ import annotations

import numpy as np


def generate_repeating_pattern(pattern: bytes, length: int) -> bytes:
    """Generate a repeated pattern of given total length."""
    return (pattern * ((length // len(pattern)) + 1))[:length]


def generate_ab_pattern(length: int) -> bytes:
    """Alternating A B A B ... pattern."""
    return generate_repeating_pattern(b"AB", length)


def generate_abc_pattern(length: int) -> bytes:
    """Repeating ABC pattern."""
    return generate_repeating_pattern(b"ABC", length)


def generate_markov_chain(
    start_byte: int,
    transitions: dict[int, list[tuple[int, float]]],
    length: int,
    seed: int = 42,
) -> bytes:
    """
    Generate bytes from a Markov chain.
    transitions: {byte: [(next_byte, probability), ...]}
    """
    rng = np.random.default_rng(seed)
    result = bytearray()
    current = start_byte
    for _ in range(length):
        result.append(current)
        if current in transitions:
            candidates = transitions[current]
            targets = [t[0] for t in candidates]
            probs = [t[1] for t in candidates]
            current = int(rng.choice(targets, p=probs))
        else:
            current = start_byte
    return bytes(result)


def generate_simple_grammar(length: int, seed: int = 42) -> bytes:
    """
    Simple grammar: S -> aSb | ab
    Generates balanced a^n b^n sequences of exactly `length` bytes.
    """
    rng = np.random.default_rng(seed)
    result = bytearray()
    while len(result) < length:
        n = int(rng.integers(1, 8))
        pair_len = 2 * n
        if len(result) + pair_len <= length:
            result.extend(b"a" * n + b"b" * n)
        else:
            half = (length - len(result)) // 2
            result.extend(b"a" * half + b"b" * half)
            break
    return bytes(result)


def generate_nested_structure(length: int, seed: int = 42) -> bytes:
    """
    Nested bracket-like structure: { [ ( ... ) ] }
    """
    rng = np.random.default_rng(seed)
    pairs = [(ord("{"), ord("}")), (ord("["), ord("]")), (ord("("), ord(")"))]
    result = bytearray()
    while len(result) < length:
        depth = int(rng.integers(1, 4))
        open_sym, close_sym = pairs[int(rng.integers(0, len(pairs)))]
        result.extend([open_sym] * depth)
        result.extend([close_sym] * depth)
    return bytes(result[:length])


def generate_delayed_copy(length: int, delay: int = 10, seed: int = 42) -> bytes:
    """
    Delayed copy: output[t] = input[t - delay]
    Tests long-range context dependency.
    """
    rng = np.random.default_rng(seed)
    preamble = bytes(rng.integers(32, 127, size=delay, dtype=np.uint8))
    result = bytearray(preamble)
    for i in range(delay, length):
        result.append(result[i - delay])
    return bytes(result[:length])


def generate_language_with_syntax(length: int, seed: int = 42) -> bytes:
    """
    Mini-language with word-level structure.
    Vocabulary: 'the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran'
    Grammar: 'the' noun verb 'on' noun | 'the' noun verb
    """
    rng = np.random.default_rng(seed)
    nouns = [b"cat", b"dog", b"mat"]
    verbs = [b"sat", b"ran"]
    articles = [b"the"]
    prepositions = [b"on"]

    result = bytearray()
    while len(result) < length:
        noun1 = nouns[int(rng.integers(0, len(nouns)))]
        verb = verbs[int(rng.integers(0, len(verbs)))]
        if rng.random() < 0.5:
            noun2 = nouns[int(rng.integers(0, len(nouns)))]
            phrase = articles[0] + b" " + noun1 + b" " + verb + b" " + prepositions[0] + b" " + noun2 + b" "
        else:
            phrase = articles[0] + b" " + noun1 + b" " + verb + b" "
        result.extend(phrase)
    return bytes(result[:length])


GRAMMAR_REGISTRY: dict[str, callable] = {
    "ab_pattern": generate_ab_pattern,
    "abc_pattern": generate_abc_pattern,
    "simple_grammar": generate_simple_grammar,
    "nested_structure": generate_nested_structure,
    "delayed_copy_10": lambda length: generate_delayed_copy(length, delay=10),
    "delayed_copy_20": lambda length: generate_delayed_copy(length, delay=20),
    "language": generate_language_with_syntax,
}
