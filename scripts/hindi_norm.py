#!/usr/bin/env python3
"""
hindi_norm.py -- THE transcript convention for this project (frozen as v2, 2026-10-18).

Reference and hypothesis must BOTH go through normalize(). Never edit this per experiment:
if you change it, bump NORM_VERSION and re-score every cache (rescore.py does that offline).

v1 (what eval_asr.py used so far)
    NFC, lower-case Latin, drop punctuation and the danda, keep letters / combining marks /
    digits, Devanagari digits -> ASCII digits, delete zero-width joiners, collapse spaces.
v2 = v1 plus
    * nukta removed:        ज़ -> ज, फ़ -> फ, ड़ -> ड, ...  (annotators and Whisper disagree on it)
    * chandrabindu -> anusvara:  पाँच -> पांच, यहाँ -> यहां
NOT handled (deliberately): compound splitting/joining (पशु मेला vs पशुमेला), English words in
Latin vs Devanagari, and real spelling variants. Compound effects are covered by also reporting
CER (chars_nospace) next to WER. Decide the English-word rule separately and bump the version.
"""
import unicodedata

NORM_VERSION = "v2"
NUKTA = "़"
CHANDRABINDU = "ँ"
ANUSVARA = "ं"


def normalize_v1(text):
    """The original convention (identical to eval_asr.normalize at the time of Gate 1/2)."""
    out = []
    for c in unicodedata.normalize("NFC", text).lower():
        cat = unicodedata.category(c)
        if cat == "Nd":
            out.append(str(unicodedata.digit(c)))
        elif cat[0] in "LM":
            out.append(c)
        elif cat == "Cf":
            continue
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def normalize(text):
    """Frozen v2 convention. Use this everywhere (scoring and training targets for scoring)."""
    s = normalize_v1(text)
    # NFD first so precomposed nukta letters (U+0929, U+0931, U+0934, U+0958-095F) split into
    # base + U+093C, then drop the nukta and recompose.
    s = unicodedata.normalize("NFD", s).replace(NUKTA, "").replace(CHANDRABINDU, ANUSVARA)
    return " ".join(unicodedata.normalize("NFC", s).split())


def chars_nospace(text):
    """Character sequence for CER: v2 text with all spaces removed, so word splitting/joining
    (compounds, postpositions) does not count as an error."""
    return list(normalize(text).replace(" ", ""))


if __name__ == "__main__":
    tests = [
        ("हर चीज़ की फ़ोटो", "हर चीज की फोटो"),
        ("पाँच यहाँ", "पांच यहां"),
        ("पशु मेला", "पशु मेला"),
    ]
    for a, b in tests:
        assert normalize(a) == normalize(b), (a, b, normalize(a), normalize(b))
    assert normalize("क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़") == normalize("क ख ग ज ड ढ फ य")
    assert normalize("१२३ abc, ।") == "123 abc"
    assert normalize("पशु मेला") != normalize("पशुमेला")           # compounds NOT merged by WER
    assert chars_nospace("पशु मेला") == chars_nospace("पशुमेला")   # ... but CER ignores the split
    print("hindi_norm self-test ok, version", NORM_VERSION)
