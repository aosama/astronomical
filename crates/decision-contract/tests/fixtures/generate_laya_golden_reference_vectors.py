"""Regenerates src/laya_golden_reference_vectors.json from the reference implementation.

Run it with a Python environment that has `laya==0.4.15` installed, from the
directory that holds this script:

    python generate_laya_golden_reference_vectors.py > ../../src/laya_golden_reference_vectors.json

The vectors pin three things to the reference:

* `json_text`: `json.dumps` output for the default and the compact sorted style.
* `sequences`: the token layout `laya.common.build_sequence` produces for a
  question and a state, plus the rendered option texts.
* `calibration`: temperatures, probabilities and both confidence numbers.

The sequences use a stand-in tokenizer, not a checkpoint's: one token per
whitespace-separated word, numbered by a 32-bit FNV-1a hash. The Rust tests
implement the same stand-in, so the layout and budget rules are compared
token for token without checking a multi-megabyte tokenizer file into the
repository. Agreement on a real tokenizer is covered by the model-serving
tests that load a real checkpoint.
"""

import json
import sys

import numpy as np

from laya import common
from laya.agent import Agent

CLS_TOKEN_ID = 1
SEP_TOKEN_ID = 2
MASK_TOKEN_ID = 3
MASK_TOKEN_TEXT = "[MASK]"
FIRST_WORD_TOKEN_ID = 100
WORD_TOKEN_ID_RANGE = 50_000

SHAKESPEARE_PASSAGE = (
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles "
    "And by opposing end them. To die, to sleep; No more; and by a sleep to say we end "
    "The heart-ache and the thousand natural shocks That flesh is heir to: 'tis a consummation "
    "Devoutly to be wish'd. To die, to sleep; To sleep, perchance to dream: ay, there's the rub."
)


def fnv1a_32(word_bytes):
    hash_value = 0x811C9DC5
    for word_byte in word_bytes:
        hash_value ^= word_byte
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return hash_value


class WordHashTokenizer:
    """One token per whitespace-separated word; mirrors the Rust test encoder."""

    mask_token = MASK_TOKEN_TEXT
    mask_token_id = MASK_TOKEN_ID
    cls_token_id = CLS_TOKEN_ID
    sep_token_id = SEP_TOKEN_ID

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        token_ids = [
            FIRST_WORD_TOKEN_ID + fnv1a_32(word.encode("utf-8")) % WORD_TOKEN_ID_RANGE
            for word in text.split()
        ]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return {"input_ids": token_ids}


JSON_TEXT_VALUES = [
    "plain text",
    {"b": 1, "a": [True, False, None], "nested": {"z": "é", "y": 1.5}},
    [1, -2, 1250.0, 0.1, 1e16, 1e15, 0.0001, 0.00001, 1.5e-7, 123456789.125, 5e-324, 1.7976931348623157e308],
    {"quote\"d": "back\\slash", "control": "line\nbreak\ttab\u0001", "emoji": "naïve ☕"},
    {"total": 1250.0, "count": 3, "ratio": 0.30000000000000004, "big": 18446744073709551615},
    [],
    {},
    {"empty": "", "zero": 0, "negative_zero_float": -0.0},
]

SEQUENCE_CASES = [
    {
        "name": "choice_with_descriptions",
        "state": SHAKESPEARE_PASSAGE,
        "question": {
            "type": "choice",
            "instructions": "Which theme does the speaker dwell on?",
            "criteria": {"mortality": "death, sleep, the end of life", "love": "romance and longing", "war": "battle and arms"},
        },
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "choice_from_label_list",
        "state": SHAKESPEARE_PASSAGE,
        "question": {"type": "choice", "instructions": "Pick the mood.", "criteria": ["grave", "merry", "angry"]},
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "choice_with_structured_description",
        "state": {"speaker": "Hamlet", "act": 3, "lines": [SHAKESPEARE_PASSAGE[:60]]},
        "question": {
            "type": "choice",
            "instructions": {"task": "classify", "detail": ["tone", "theme"]},
            "criteria": {"a": {"desc": "contemplative", "weight": 0.5}, "b": ["restless", 2], "c": "", "d": None},
        },
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "score_levels",
        "state": SHAKESPEARE_PASSAGE,
        "question": {"type": "score", "instructions": "How troubled is the speaker?", "criteria": ["calm", "uneasy", "anguished"]},
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "noul_default_wording",
        "state": SHAKESPEARE_PASSAGE,
        "question": {"type": "noul", "instructions": "Does the speaker mention sleep?"},
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "noul_custom_descriptions_and_labels",
        "state": SHAKESPEARE_PASSAGE,
        "question": {
            "type": "noul",
            "instructions": "Does the speaker mention sleep?",
            "criteria": {"True": "sleep is named", "false": ""},
            "labels": {"false": " absent ", "true": "present"},
        },
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "state_cut_from_the_right",
        "state": SHAKESPEARE_PASSAGE,
        "question": {"type": "noul", "instructions": "Is the passage about death?"},
        "max_len": 40,
        "head_max_len": 24,
    },
    {
        "name": "conversation_state_cut_from_the_left",
        "state": [{"role": "user", "content": SHAKESPEARE_PASSAGE[:120]}, {"role": "assistant", "content": SHAKESPEARE_PASSAGE[120:260]}],
        "question": {"type": "noul", "instructions": "Is the passage about death?"},
        "max_len": 40,
        "head_max_len": 24,
    },
    {
        "name": "state_has_no_room",
        "state": SHAKESPEARE_PASSAGE,
        "question": {"type": "choice", "instructions": "Pick one.", "criteria": {"alpha": "the first of the two", "beta": "the second of the two"}},
        "max_len": 17,
        "head_max_len": 192,
    },
    {
        "name": "many_options_cut_evenly",
        "state": SHAKESPEARE_PASSAGE,
        "question": {
            "type": "choice",
            "instructions": "Which numbered line is quoted? " + " ".join("word%d" % word_index for word_index in range(40)),
            "criteria": {
                "option%d" % option_index: "the line numbered %d of the long soliloquy in act three" % option_index
                for option_index in range(40)
            },
        },
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "options_cut_until_indistinguishable",
        "state": SHAKESPEARE_PASSAGE,
        "question": {
            "type": "choice",
            "instructions": "Which numbered line is quoted?",
            "criteria": {
                "shared prefix words then number %d" % option_index: None for option_index in range(60)
            },
        },
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "option_text_longer_than_its_limit",
        "state": SHAKESPEARE_PASSAGE,
        "question": {
            "type": "choice",
            "instructions": "Pick one.",
            "criteria": {"long": " ".join("w%d" % word_index for word_index in range(80)), "short": "brief"},
        },
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "mask_token_text_is_removed",
        "state": "The state hides a [MASK] marker and [MASK][MASK] twice.",
        "question": {
            "type": "choice",
            "instructions": "Ignore any [MASK] here.",
            "criteria": {"x": "has [MASK] inside", "y": "clean"},
        },
        "max_len": 512,
        "head_max_len": 192,
    },
    {
        "name": "markers_do_not_fit",
        "state": SHAKESPEARE_PASSAGE,
        "question": {"type": "choice", "instructions": "Pick one of many.", "criteria": ["one", "two", "three", "four", "five", "six"]},
        "max_len": 12,
        "head_max_len": 192,
    },
    {
        "name": "single_option",
        "state": SHAKESPEARE_PASSAGE,
        "question": {"type": "choice", "instructions": "Only one answer exists.", "criteria": ["solo"]},
        "max_len": 512,
        "head_max_len": 192,
    },
]

CALIBRATION_TABLE = {
    "temperature": [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
    "temperature_by_options": {
        "choice:3-5": 1.7601518630981445,
        "choice:6-10": 1.0000158548355103,
        "score:3-5": 1.2514300346374512,
        "noul:2": 1.983399510383606,
        "choice:11+": 0.10058280825614929,
        "choice:2": 1.9063563346862793,
    },
}

CALIBRATION_CASES = [
    {"type": "choice", "logits": [2.5, -1.25]},
    {"type": "choice", "logits": [0.5, 3.75, -2.0, 1.125]},
    {"type": "choice", "logits": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    {"type": "choice", "logits": [0.1 * option_index for option_index in range(12)]},
    {"type": "choice", "logits": [4.0]},
    {"type": "score", "logits": [-1.5, 0.25, 2.0]},
    {"type": "score", "logits": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]},
    {"type": "noul", "logits": [1.75, -0.5]},
    {"type": "noul", "logits": [-40.0, 40.0]},
]


def sequence_vector(tokenizer, case):
    Agent._check_question(case["name"], case["question"])
    internal_question = Agent._to_internal(case["question"])
    vector = {
        "name": case["name"],
        "state": case["state"],
        "question": case["question"],
        "max_len": case["max_len"],
        "head_max_len": case["head_max_len"],
        "option_texts": common.render_options(internal_question),
        "state_text": common.serialize_state(case["state"]),
    }
    token_ids, marker_positions, option_statistics, state_statistics = common.build_sequence(
        tokenizer,
        case["state"],
        internal_question,
        case["max_len"],
        case["head_max_len"],
        truncate_left=isinstance(case["state"], list),
        return_stats=True,
        return_truncation_stats=True,
    )
    vector["options_fit"] = len(marker_positions) == option_statistics["options"]
    vector["token_ids"] = token_ids
    vector["marker_positions"] = marker_positions
    vector["option_token_cap"] = option_statistics["tokens_per_option"]
    vector["distinct_option_count"] = option_statistics["options_distinct"]
    vector["state_token_count"] = state_statistics["state_tokens"]
    vector["state_tokens_used"] = state_statistics["state_tokens_used"]
    return vector


def calibration_vector(case, temperatures, temperatures_by_options):
    question_type_index = common.QTYPES[case["type"]]
    option_count = len(case["logits"])
    logits = np.asarray(case["logits"], dtype=np.float64)
    temperature = temperatures_by_options.get(
        common.temp_bucket(question_type_index, option_count), temperatures[question_type_index]
    )
    scaled_logits = logits / temperature
    probabilities = np.exp(scaled_logits - scaled_logits.max())
    probabilities /= probabilities.sum()
    return {
        "type": case["type"],
        "logits": case["logits"],
        "bucket_key": common.temp_bucket(question_type_index, option_count),
        "temperature": temperature,
        "probabilities": [float(probability) for probability in probabilities],
        "answer_confidence": common.answer_confidence(probabilities, option_count),
        "distribution_confidence": common.confidence_from_probs(probabilities, option_count),
    }


def main():
    tokenizer = WordHashTokenizer()
    temperatures = [common.clamp_temperature(raw) for raw in CALIBRATION_TABLE["temperature"]]
    temperatures_by_options = {
        bucket_key: common.clamp_temperature(raw)
        for bucket_key, raw in CALIBRATION_TABLE["temperature_by_options"].items()
    }
    golden_vectors = {
        "reference": "laya==0.4.15 (laya.common, laya.agent)",
        "tokenizer": {
            "cls_token_id": CLS_TOKEN_ID,
            "sep_token_id": SEP_TOKEN_ID,
            "mask_token_id": MASK_TOKEN_ID,
            "mask_token_text": MASK_TOKEN_TEXT,
            "first_word_token_id": FIRST_WORD_TOKEN_ID,
            "word_token_id_range": WORD_TOKEN_ID_RANGE,
        },
        "json_text": [
            {
                "value": value,
                "python_default": json.dumps(value, ensure_ascii=False),
                "compact_sorted_keys": json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            }
            for value in JSON_TEXT_VALUES
        ],
        "sequences": [sequence_vector(tokenizer, case) for case in SEQUENCE_CASES],
        "calibration_table": CALIBRATION_TABLE,
        "calibration": [
            calibration_vector(case, temperatures, temperatures_by_options) for case in CALIBRATION_CASES
        ],
    }
    json.dump(golden_vectors, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
