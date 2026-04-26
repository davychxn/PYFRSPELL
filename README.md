# pyfrspell

[English](README.md) | [中文](README.cn.md) | [Francais](README.fr.md)

pyfrspell is a Python package for French lemma prediction and derivative form generation.
It supports:

- conjugation to lemma prediction
- noun form generation
- adjective form generation
- verb form generation

The package runs with ONNX Runtime and quantized INT8 models for high speed and small model footprint.

## Install

```bash
pip install pyfrspell
```

## Integrate Into Your Project

```python
from pyfrspell import FrSpell

predictor = FrSpell()

lemma = predictor.lemma("mangeons")
noun = predictor.noun_derive("chat", "THD_PLF")
adje = predictor.adje_derive("beau", "THD_F")
verb = predictor.verb_derive("manger", "FST_PL", "INDI", "PRES")

print(lemma)
print(noun)
print(adje)
print(verb)
```

Sample runtime output:

```txt
{'input': 'mangeons', 'lemma': 'manger', 'wordType': 'VERB', 'confidence': 0.9965, 'timeMs': 3.89}
{'lemma': 'chat', 'wordType': 'NOUN', 'person': 'THD_PLF', 'mode': 'ALL', 'tense': 'ALL', 'output': 'chattes', 'confidence': 0.9997, 'timeMs': 5.06}
{'lemma': 'beau', 'wordType': 'ADJE', 'person': 'THD_F', 'mode': 'ALL', 'tense': 'ALL', 'output': 'belle', 'confidence': 0.9999, 'timeMs': 3.08}
{'lemma': 'manger', 'wordType': 'VERB', 'person': 'FST_PL', 'mode': 'INDI', 'tense': 'PRES', 'output': 'mangeons', 'confidence': 0.9999, 'timeMs': 4.79}
```

## Prediction Parameters

Lemma prediction:

- API: `predictor.lemma(input_word)`
- `input_word`: string, inflected or conjugated word form, for example `mangeons`

Derive prediction:

- Noun API: `predictor.noun_derive(lemma, person)`
- Adjective API: `predictor.adje_derive(lemma, person)`
- Verb API: `predictor.verb_derive(lemma, person, mode, tense)`
- Generic API: `predictor.derive(lemma, word_type, person, mode, tense)`

Allowed `word_type` values:

- `NOUN` (noun)
- `ADJE` (adjective)
- `VERB` (verb)

Allowed `person` values:

- `FST` (1st person singular)
- `SND` (2nd person singular)
- `THD_M` (3rd person masculine singular)
- `THD_F` (3rd person feminine singular)
- `FST_PL` (1st person plural)
- `SND_PL` (2nd person plural)
- `THD_PLM` (3rd person masculine plural)
- `THD_PLF` (3rd person feminine plural)

Allowed `mode` values:

- `INDI` (indicative)
- `SUBJ` (subjunctive)
- `COND` (conditional)
- `PART` (participle)
- `IMPE` (imperative)
- `INFI` (infinitive)

Allowed `tense` values in current implementation:

- `PRES` (present)
- `IMPA` (imperfect)
- `FUTU` (future)
- `PASS` (past)

Note:

- The original grammar definition includes more tense names, but this package currently supports only `PRES`, `IMPA`, `FUTU`, `PASS`.
- For noun and adjective derive calls, `mode` and `tense` are not required in user input.

## Run Test

```bash
python smoke_test.py --verbose
```

This runs a local smoke test to validate import, model loading, and core APIs.

## Build and Upload to PyPI

1. Build package files:

```bash
python -m pip install --upgrade build
python -m build
```

2. Upload to PyPI:

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```

Build output:

- `dist/pyfrspell-<version>-py3-none-any.whl`
- `dist/pyfrspell-<version>.tar.gz`
