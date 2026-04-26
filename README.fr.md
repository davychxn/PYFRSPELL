# pyfrspell

Langue: [English](README.md) | [中文](README.cn.md) | [Francais](README.fr.md)

pyfrspell est un package Python pour la prediction de lemmes francais et la generation de formes derivees.
Il prend en charge:

- la prediction du lemme a partir d'une forme conjuguee
- la generation de formes nominales
- la generation de formes adjectivales
- la generation de formes verbales

Le package utilise ONNX Runtime et des modeles quantifies INT8 pour une execution rapide et une taille reduite.

## Installation

```bash
pip install pyfrspell
```

## Integrer Dans Votre Projet

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

Exemple de sortie:

```txt
{'input': 'mangeons', 'lemma': 'manger', 'wordType': 'VERB', 'confidence': 0.9965, 'timeMs': 3.89}
{'lemma': 'chat', 'wordType': 'NOUN', 'person': 'THD_PLF', 'mode': 'ALL', 'tense': 'ALL', 'output': 'chattes', 'confidence': 0.9997, 'timeMs': 5.06}
{'lemma': 'beau', 'wordType': 'ADJE', 'person': 'THD_F', 'mode': 'ALL', 'tense': 'ALL', 'output': 'belle', 'confidence': 0.9999, 'timeMs': 3.08}
{'lemma': 'manger', 'wordType': 'VERB', 'person': 'FST_PL', 'mode': 'INDI', 'tense': 'PRES', 'output': 'mangeons', 'confidence': 0.9999, 'timeMs': 4.79}
```

## Parametres De Prediction

Prediction de lemme:

- API: `predictor.lemma(input_word)`
- `input_word`: chaine de caracteres, forme flechie ou conjuguee, par exemple `mangeons`

Prediction de derive:

- API nom: `predictor.noun_derive(lemma, person)`
- API adjectif: `predictor.adje_derive(lemma, person)`
- API verbe: `predictor.verb_derive(lemma, person, mode, tense)`
- API generique: `predictor.derive(lemma, word_type, person, mode, tense)`

Valeurs autorisees pour `word_type`:

- `NOUN` (nom)
- `ADJE` (adjectif)
- `VERB` (verbe)

Valeurs autorisees pour `person`:

- `FST` (1re personne singulier)
- `SND` (2e personne singulier)
- `THD_M` (3e personne masculin singulier)
- `THD_F` (3e personne feminin singulier)
- `FST_PL` (1re personne pluriel)
- `SND_PL` (2e personne pluriel)
- `THD_PLM` (3e personne masculin pluriel)
- `THD_PLF` (3e personne feminin pluriel)

Valeurs autorisees pour `mode`:

- `INDI` (indicatif)
- `SUBJ` (subjonctif)
- `COND` (conditionnel)
- `PART` (participe)
- `IMPE` (imperatif)
- `INFI` (infinitif)

Valeurs autorisees pour `tense` dans l'implementation actuelle:

- `PRES` (present)
- `IMPA` (imparfait)
- `FUTU` (futur)
- `PASS` (passe)

Note:

- La definition grammaticale d'origine contient plus de noms de temps, mais cette implementation prend actuellement en charge seulement `PRES`, `IMPA`, `FUTU`, `PASS`.
- Pour les appels de derive nominal et adjectival, `mode` et `tense` ne sont pas requis en entree.

## Executer Le Test

```bash
python smoke_test.py --verbose
```

Cette commande execute un smoke test local pour verifier l'import, le chargement du modele et les API principales.

## Build Et Publication Sur PyPI

1. Construire les fichiers du package:

```bash
python -m pip install --upgrade build
python -m build
```

2. Publier sur PyPI:

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```

Artefacts generes:

- `dist/pyfrspell-<version>-py3-none-any.whl`
- `dist/pyfrspell-<version>.tar.gz`
