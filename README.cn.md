# pyfrspell

语言: [English](README.md) | [中文](README.cn.md) | [Francais](README.fr.md)

pyfrspell 是一个用于法语词元预测与派生词形生成的 Python 包。
支持以下能力：

- 由变位词形预测词元
- 名词词形生成
- 形容词词形生成
- 动词词形生成

本包基于 ONNX Runtime 与 INT8 量化模型，兼顾速度与模型体积。

## 安装

```bash
pip install pyfrspell
```

## 集成到项目

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

示例运行输出：

```txt
{'input': 'mangeons', 'lemma': 'manger', 'wordType': 'VERB', 'confidence': 0.9965, 'timeMs': 3.89}
{'lemma': 'chat', 'wordType': 'NOUN', 'person': 'THD_PLF', 'mode': 'ALL', 'tense': 'ALL', 'output': 'chattes', 'confidence': 0.9997, 'timeMs': 5.06}
{'lemma': 'beau', 'wordType': 'ADJE', 'person': 'THD_F', 'mode': 'ALL', 'tense': 'ALL', 'output': 'belle', 'confidence': 0.9999, 'timeMs': 3.08}
{'lemma': 'manger', 'wordType': 'VERB', 'person': 'FST_PL', 'mode': 'INDI', 'tense': 'PRES', 'output': 'mangeons', 'confidence': 0.9999, 'timeMs': 4.79}
```

## 参数说明

词元预测：

- API: `predictor.lemma(input_word)`
- `input_word`: 字符串，变位或屈折后的词形，例如 `mangeons`

派生预测：

- 名词 API: `predictor.noun_derive(lemma, person)`
- 形容词 API: `predictor.adje_derive(lemma, person)`
- 动词 API: `predictor.verb_derive(lemma, person, mode, tense)`
- 通用 API: `predictor.derive(lemma, word_type, person, mode, tense)`

允许的 `word_type`：

- `NOUN`（名词）
- `ADJE`（形容词）
- `VERB`（动词）

允许的 `person`：

- `FST`（第一人称单数）
- `SND`（第二人称单数）
- `THD_M`（第三人称阳性单数）
- `THD_F`（第三人称阴性单数）
- `FST_PL`（第一人称复数）
- `SND_PL`（第二人称复数）
- `THD_PLM`（第三人称阳性复数）
- `THD_PLF`（第三人称阴性复数）

允许的 `mode`：

- `INDI`（直陈式）
- `SUBJ`（虚拟式）
- `COND`（条件式）
- `PART`（分词）
- `IMPE`（命令式）
- `INFI`（不定式）

当前实现允许的 `tense`：

- `PRES`（现在时）
- `IMPA`（未完成过去时）
- `FUTU`（将来时）
- `PASS`（过去时）

说明：

- 原始语法定义包含更多时态名称，但当前实现仅支持 `PRES`、`IMPA`、`FUTU`、`PASS`。
- 对于名词和形容词派生，用户输入时不需要提供 `mode` 与 `tense`。

## 运行测试

```bash
python smoke_test.py --verbose
```

该命令会执行本地冒烟测试，验证导入、模型加载和核心 API 是否可用。

## 构建并上传到 PyPI

1. 构建发布文件：

```bash
python -m pip install --upgrade build
python -m build
```

2. 上传到 PyPI：

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```

构建产物：

- `dist/pyfrspell-<version>-py3-none-any.whl`
- `dist/pyfrspell-<version>.tar.gz`
