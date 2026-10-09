"""Unit tests for eval/quality.py's pure-text diversity diagnostics (distinct_n,
repetition_rate) -- no model/checkpoint needed for these, unlike the BLEU/chrF/
ROUGE-L path which requires an actual generation to score against a reference.
"""

from nepali.eval.quality import distinct_n, repetition_rate


def test_distinct_n_all_unique_words_is_one():
    texts = ["the quick brown fox jumps"]
    assert distinct_n(texts, n=1) == 1.0


def test_distinct_n_repeated_word_is_less_than_one():
    texts = ["the the the the"]
    # 4 unigrams total, 1 unique ("the") -> 1/4
    assert distinct_n(texts, n=1) == 0.25


def test_distinct_n_bigrams():
    texts = ["a b a b"]
    # bigrams: (a,b), (b,a), (a,b) -> 3 total, 2 unique -> 2/3
    assert abs(distinct_n(texts, n=2) - (2 / 3)) < 1e-9


def test_distinct_n_empty_input_is_zero():
    assert distinct_n([], n=1) == 0.0
    assert distinct_n([""], n=1) == 0.0


def test_repetition_rate_no_repeats():
    texts = ["one two three four five six seven eight"]
    assert repetition_rate(texts, n=4) == 0.0


def test_repetition_rate_detects_repeated_ngram():
    # "a b c d" repeated twice -> the 4-gram (a,b,c,d) appears at two starting
    # offsets, so this text does contain a repeated 4-gram.
    texts = ["a b c d a b c d"]
    assert repetition_rate(texts, n=4) == 1.0


def test_repetition_rate_averages_across_texts():
    texts = [
        "a b c d a b c d",  # has a repeated 4-gram
        "one two three four five six",  # does not
    ]
    assert repetition_rate(texts, n=4) == 0.5
