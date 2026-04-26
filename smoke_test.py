"""Pre-upload smoke test for pyfrspell.

Run this script in a clean environment after installing the package:

    python smoke_test.py

It validates importability, model loading, and core public API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any, Dict


def _check_common_fields(result: Dict[str, Any], expected_fields: list[str]) -> None:
    missing = [field for field in expected_fields if field not in result]
    if missing:
        raise AssertionError(f"Missing fields: {missing}. Got keys: {sorted(result.keys())}")

    confidence = result.get("confidence")
    if confidence is None or not isinstance(confidence, (int, float)):
        raise AssertionError("'confidence' must be a number")
    if not (0.0 <= float(confidence) <= 1.0):
        raise AssertionError(f"'confidence' out of range [0, 1]: {confidence}")

    time_ms = result.get("timeMs")
    if time_ms is None or not isinstance(time_ms, (int, float)):
        raise AssertionError("'timeMs' must be a number")
    if float(time_ms) < 0:
        raise AssertionError(f"'timeMs' must be non-negative: {time_ms}")


def run_smoke_test(verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    from pyfrspell import FrSpell, __version__

    if verbose:
        print(f"pyfrspell version: {__version__}")

    predictor = FrSpell()
    results: Dict[str, Dict[str, Any]] = {}

    results["lemma"] = predictor.lemma("mangeons")
    _check_common_fields(
        results["lemma"],
        ["input", "lemma", "wordType", "confidence", "timeMs"],
    )
    if not isinstance(results["lemma"]["lemma"], str):
        raise AssertionError("Lemma output 'lemma' must be a string")

    results["noun_derive"] = predictor.noun_derive("chat", "THD_PLF")
    _check_common_fields(
        results["noun_derive"],
        ["lemma", "wordType", "person", "mode", "tense", "output", "confidence", "timeMs"],
    )

    results["adje_derive"] = predictor.adje_derive("beau", "THD_F")
    _check_common_fields(
        results["adje_derive"],
        ["lemma", "wordType", "person", "mode", "tense", "output", "confidence", "timeMs"],
    )

    results["verb_derive"] = predictor.verb_derive("manger", "FST_PL", "INDI", "PRES")
    _check_common_fields(
        results["verb_derive"],
        ["lemma", "wordType", "person", "mode", "tense", "output", "confidence", "timeMs"],
    )

    results["derive_generic"] = predictor.derive("manger", "VERB", "FST_PL", "INDI", "PRES")
    _check_common_fields(
        results["derive_generic"],
        ["lemma", "wordType", "person", "mode", "tense", "output", "confidence", "timeMs"],
    )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pyfrspell pre-upload smoke test")
    parser.add_argument("--json", action="store_true", help="Print result payload as JSON")
    parser.add_argument("--verbose", action="store_true", help="Print additional details")
    args = parser.parse_args()

    try:
        results = run_smoke_test(verbose=args.verbose)
        print("[PASS] pyfrspell smoke test completed successfully")
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        elif args.verbose:
            for name, payload in results.items():
                print(f"- {name}: {payload}")
        return 0
    except Exception as exc:
        print("[FAIL] pyfrspell smoke test failed")
        print(f"Reason: {exc}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
