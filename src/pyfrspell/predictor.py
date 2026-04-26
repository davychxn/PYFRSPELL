"""Core FR-SPELL predictor implementation for Python."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import onnxruntime as ort


def _argmax(values: Sequence[float]) -> tuple[int, float]:
    best_idx = 0
    best_val = float(values[0])
    for idx in range(1, len(values)):
        val = float(values[idx])
        if val > best_val:
            best_val = val
            best_idx = idx
    return best_idx, best_val


def _softmax_at(logits: Sequence[float], idx: int) -> float:
    max_val = max(float(v) for v in logits)
    exps = [math.exp(float(v) - max_val) for v in logits]
    return exps[idx] / sum(exps)


def _to_stoi(itos: Sequence[str]) -> Dict[str, int]:
    return {token: idx for idx, token in enumerate(itos)}


def _decode_text_from_ids(ids: Iterable[int], itos: Sequence[str], specials: Dict[str, str]) -> str:
    output: List[str] = []
    for token_id in ids:
        token = itos[token_id]
        if token == specials["eos"]:
            break
        if token in (specials["pad"], specials["bos"], specials["unk"]):
            continue
        output.append(token)
    return "".join(output)


def _normalize_enum_token(value: Optional[str], fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return fallback


def _make_int64_tensor_2d(rows: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.int64)


def _make_int64_tensor_1d(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _resolve_default_model_paths() -> Dict[str, str]:
    base = files("pyfrspell").joinpath("models", "community")
    return {
        "lemma_model_path": str(base.joinpath("lemma_type_model.int8.onnx")),
        "lemma_vocab_path": str(base.joinpath("lemma_type_vocab.json")),
        "lemma_labels_path": str(base.joinpath("lemma_type_labels.json")),
        "derivative_model_path": str(base.joinpath("derive_form_model.int8.onnx")),
        "derivative_vocab_path": str(base.joinpath("derive_form_vocab.json")),
    }


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class _LemmaTypePredictor:
    model_path: str
    vocab_path: str
    labels_path: str
    max_decode_len: Optional[int] = None
    execution_providers: Optional[List[str]] = None

    def __post_init__(self) -> None:
        vocab = _load_json(self.vocab_path)
        labels = _load_json(self.labels_path)

        self.itos: List[str] = vocab["itos"]
        self.stoi = _to_stoi(self.itos)
        self.id_to_code: Dict[int, str] = {int(k): v for k, v in vocab["id_to_code"].items()}
        self.code_char_to_name: Dict[str, str] = labels.get("code_char_to_name", {})

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

        providers = self.execution_providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]

        self.decode_limit = self.max_decode_len or labels.get("max_decode_len", 32)
        self.specials = {
            "pad": self.pad_token,
            "bos": self.bos_token,
            "eos": self.eos_token,
            "unk": self.unk_token,
        }

    def _encode_word(self, word: str) -> List[int]:
        encoded = [self.stoi.get(char, self.unk_id) for char in word]
        encoded.append(self.eos_id)
        return encoded

    def predict(self, word: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        src_ids = self._encode_word(word)
        src_tensor = _make_int64_tensor_2d([src_ids])
        src_len_tensor = _make_int64_tensor_1d([len(src_ids)])

        tgt = [self.bos_id]
        lemma_token_ids: List[int] = []
        code_logits: Optional[np.ndarray] = None

        for _ in range(self.decode_limit):
            tgt_tensor = _make_int64_tensor_2d([tgt])
            feed: Dict[str, np.ndarray] = {}
            if "src" in self.input_names:
                feed["src"] = src_tensor
            if "src_len" in self.input_names:
                feed["src_len"] = src_len_tensor
            if "tgt_in" in self.input_names:
                feed["tgt_in"] = tgt_tensor

            output_values = self.session.run(None, feed)
            outputs = dict(zip(self.output_names, output_values))

            token_logits = np.asarray(outputs["token_logits"])
            code_logits = np.asarray(outputs["code_logits"]).reshape(-1)

            last_step = token_logits[0, -1, :]
            next_id, _ = _argmax(last_step)
            lemma_token_ids.append(next_id)

            if next_id == self.eos_id:
                break
            tgt.append(next_id)

        if code_logits is None:
            raise RuntimeError("Model did not produce code logits.")

        pred_lemma = _decode_text_from_ids(lemma_token_ids, self.itos, self.specials)
        code_best_idx, _ = _argmax(code_logits)
        pred_code = self.id_to_code.get(code_best_idx, "A")
        code_conf = _softmax_at(code_logits, code_best_idx)
        pred_type = self.code_char_to_name.get(pred_code, "UNKNOWN")

        return {
            "input": word,
            "lemma": pred_lemma,
            "wordType": pred_type,
            "confidence": code_conf,
            "timeMs": (time.perf_counter() - t0) * 1000.0,
        }

    def lemma(self, word: str) -> Dict[str, Any]:
        return self.predict(word)

    def predict_batch(self, words: Sequence[str]) -> List[Dict[str, Any]]:
        return [self.predict(word) for word in words]


@dataclass
class _DerivativeTypePredictor:
    model_path: str
    vocab_path: str
    max_decode_len: Optional[int] = None
    execution_providers: Optional[List[str]] = None

    def __post_init__(self) -> None:
        vocab = _load_json(self.vocab_path)
        self.itos: List[str] = vocab["itos"]
        self.stoi = _to_stoi(self.itos)

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

        providers = self.execution_providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.output_token_name = "token_logits" if "token_logits" in self.output_names else self.output_names[0]

        self.decode_limit = self.max_decode_len or 48
        self.specials = {
            "pad": self.pad_token,
            "bos": self.bos_token,
            "eos": self.eos_token,
            "unk": self.unk_token,
        }

    def _encode_text(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    @staticmethod
    def _build_source_text(lemma: str, word_type: str, person: str, mode: str, tense: str) -> str:
        return f"L:{lemma}|W:{word_type}|P:{person}|M:{mode}|T:{tense}"

    def predict(
        self,
        lemma: str,
        word_type: str,
        sentence_person: Optional[str] = None,
        sentence_mode: Optional[str] = None,
        sentence_tense: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        normalized_lemma = str(lemma or "").strip()
        normalized_word_type = _normalize_enum_token(word_type, "NONE")
        normalized_person = _normalize_enum_token(sentence_person, "ALL")

        is_noun_or_adje = normalized_word_type in ("NOUN", "ADJE")
        normalized_mode = _normalize_enum_token(sentence_mode, "ALL") if is_noun_or_adje else _normalize_enum_token(sentence_mode, "NONE")
        normalized_tense = _normalize_enum_token(sentence_tense, "ALL") if is_noun_or_adje else _normalize_enum_token(sentence_tense, "NONE")

        source = self._build_source_text(
            normalized_lemma,
            normalized_word_type,
            normalized_person,
            normalized_mode,
            normalized_tense,
        )

        src_ids = self._encode_text(source, add_bos=False, add_eos=True)
        src_tensor = _make_int64_tensor_2d([src_ids])
        src_len_tensor = _make_int64_tensor_1d([len(src_ids)])

        tgt = [self.bos_id]
        token_ids: List[int] = []
        last_step_logits: Optional[np.ndarray] = None

        for _ in range(self.decode_limit):
            tgt_tensor = _make_int64_tensor_2d([tgt])
            feed: Dict[str, np.ndarray] = {}
            if "src" in self.input_names:
                feed["src"] = src_tensor
            if "src_len" in self.input_names:
                feed["src_len"] = src_len_tensor
            if "tgt_in" in self.input_names:
                feed["tgt_in"] = tgt_tensor

            output_values = self.session.run(None, feed)
            outputs = dict(zip(self.output_names, output_values))

            token_logits = np.asarray(outputs[self.output_token_name])
            last_step_logits = token_logits[0, -1, :]
            next_id, _ = _argmax(last_step_logits)

            token_ids.append(next_id)
            if next_id == self.eos_id:
                break
            tgt.append(next_id)

        if last_step_logits is None:
            raise RuntimeError("Model did not produce token logits.")

        output_form = _decode_text_from_ids(token_ids, self.itos, self.specials)
        best_idx, _ = _argmax(last_step_logits)
        confidence = _softmax_at(last_step_logits, best_idx)

        return {
            "lemma": normalized_lemma,
            "wordType": normalized_word_type,
            "person": normalized_person,
            "mode": normalized_mode,
            "tense": normalized_tense,
            "output": output_form,
            "confidence": confidence,
            "timeMs": (time.perf_counter() - t0) * 1000.0,
        }

    def noun_derive(self, lemma: str, sentence_person: str, sentence_mode: Optional[str] = None, sentence_tense: Optional[str] = None) -> Dict[str, Any]:
        return self.predict(lemma, "NOUN", sentence_person, sentence_mode, sentence_tense)

    def adje_derive(self, lemma: str, sentence_person: str, sentence_mode: Optional[str] = None, sentence_tense: Optional[str] = None) -> Dict[str, Any]:
        return self.predict(lemma, "ADJE", sentence_person, sentence_mode, sentence_tense)

    def verb_derive(self, lemma: str, sentence_person: str, sentence_mode: str, sentence_tense: str) -> Dict[str, Any]:
        return self.predict(lemma, "VERB", sentence_person, sentence_mode, sentence_tense)


class FrSpell:
    """Top-level predictor API for lemma and derivative generation."""

    def __init__(
        self,
        *,
        lemma_model_path: Optional[str] = None,
        lemma_vocab_path: Optional[str] = None,
        lemma_labels_path: Optional[str] = None,
        derivative_model_path: Optional[str] = None,
        derivative_vocab_path: Optional[str] = None,
        lemma_max_decode_len: Optional[int] = None,
        derivative_max_decode_len: Optional[int] = None,
        execution_providers: Optional[List[str]] = None,
    ) -> None:
        defaults = _resolve_default_model_paths()

        self.lemma_predictor = _LemmaTypePredictor(
            model_path=lemma_model_path or defaults["lemma_model_path"],
            vocab_path=lemma_vocab_path or defaults["lemma_vocab_path"],
            labels_path=lemma_labels_path or defaults["lemma_labels_path"],
            max_decode_len=lemma_max_decode_len,
            execution_providers=execution_providers,
        )

        self.derivative_predictor = _DerivativeTypePredictor(
            model_path=derivative_model_path or defaults["derivative_model_path"],
            vocab_path=derivative_vocab_path or defaults["derivative_vocab_path"],
            max_decode_len=derivative_max_decode_len,
            execution_providers=execution_providers,
        )

        self.metadata = {
            "lemma": {
                "modelPath": self.lemma_predictor.model_path,
                "vocabSize": len(self.lemma_predictor.itos),
                "decodeLimit": self.lemma_predictor.decode_limit,
            },
            "derivative": {
                "modelPath": self.derivative_predictor.model_path,
                "vocabSize": len(self.derivative_predictor.itos),
                "decodeLimit": self.derivative_predictor.decode_limit,
            },
        }

    def lemma(self, word: str) -> Dict[str, Any]:
        return self.lemma_predictor.lemma(word)

    def derive(
        self,
        lemma: str,
        word_type: str,
        person: Optional[str] = None,
        mode: Optional[str] = None,
        tense: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.derivative_predictor.predict(lemma, word_type, person, mode, tense)

    def noun_derive(
        self,
        lemma: str,
        person: str,
        mode: Optional[str] = None,
        tense: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.derivative_predictor.noun_derive(lemma, person, mode, tense)

    def adje_derive(
        self,
        lemma: str,
        person: str,
        mode: Optional[str] = None,
        tense: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.derivative_predictor.adje_derive(lemma, person, mode, tense)

    def verb_derive(self, lemma: str, person: str, mode: str, tense: str) -> Dict[str, Any]:
        return self.derivative_predictor.verb_derive(lemma, person, mode, tense)
