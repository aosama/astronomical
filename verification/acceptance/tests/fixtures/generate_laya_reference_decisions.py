"""Regenerates the Laya reference decisions used by the real-checkpoint parity journey.

Run it with a Python environment that has `laya==0.4.15` installed:

    python generate_laya_reference_decisions.py CHECKPOINT_DIRECTORY > laya_reference_decisions.json

CHECKPOINT_DIRECTORY holds the original `convaiinnovations/laya` checkpoints: the
English one at its root and the multilingual one in `multilingual/`. Each case is
answered by the reference implementation in float32 on the CPU, and the fixture
records what the model read (token identifiers and marker positions), its raw
option logits, the temperature applied and the resulting probabilities.
"""

import json
import sys

import numpy as np
import torch

import laya
from laya import common
from laya.agent import Agent

ROMEO_OPENING = (
    "Two households, both alike in dignity, In fair Verona, where we lay our scene, "
    "From ancient grudge break to new mutiny, Where civil blood makes civil hands unclean. "
    "From forth the fatal loins of these two foes A pair of star-cross'd lovers take their life; "
    "Whose misadventur'd piteous overthrows Doth with their death bury their parents' strife."
)
HAMLET_SOLILOQUY = (
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles "
    "And by opposing end them. To die, to sleep; No more; and by a sleep to say we end "
    "The heart-ache and the thousand natural shocks That flesh is heir to."
)
LONG_STATE = " ".join([ROMEO_OPENING, HAMLET_SOLILOQUY] * 6)
HAMLET_GERMAN = (
    "Sein oder Nichtsein, das ist hier die Frage: Ob's edler im Gemüt, die Pfeil' und Schleudern "
    "des wütenden Geschicks erdulden, oder, sich waffnend gegen eine See von Plagen, "
    "durch Widerstand sie enden."
)
HAMLET_FRENCH = (
    "Être, ou ne pas être, telle est la question. Y a-t-il plus de noblesse d'âme à subir "
    "la fronde et les flèches de la fortune outrageante, ou bien à s'armer contre une mer de douleurs "
    "et à l'arrêter par une révolte ?"
)

THEME_CHOICE = {
    "type": "choice",
    "instructions": "Which theme does the passage dwell on most?",
    "criteria": {
        "mortality": "death, dying, the end of life",
        "feud": "two families in conflict",
        "comedy": "jokes, mirth and light entertainment",
    },
}
ENGLISH_QUESTIONS = {
    "theme": THEME_CHOICE,
    "speaker_count": {
        "type": "choice",
        "instructions": "Is this spoken by one person reflecting, or does it introduce a story?",
        "criteria": {"reflection": "one person thinking aloud", "prologue": "an introduction to a story"},
    },
    "play": {
        "type": "choice",
        "instructions": "Which play is the passage from?",
        "criteria": [
            "Romeo and Juliet", "Hamlet", "Macbeth", "Othello", "King Lear", "The Tempest",
            "Twelfth Night", "Julius Caesar", "Richard III", "Henry V", "As You Like It", "The Winter's Tale",
        ],
    },
    "gravity": {
        "type": "score",
        "instructions": "How grave is the mood of the passage?",
        "criteria": ["light-hearted", "serious", "tragic"],
    },
    "violence": {
        "type": "score",
        "instructions": "How much violence does the passage describe?",
        "criteria": ["none", "a hint", "some", "much", "constant"],
    },
    "mentions_death": {"type": "noul", "instructions": "Does the passage mention death?"},
    "mentions_sea_voyage": {
        "type": "noul",
        "instructions": "Does the passage describe a sea voyage?",
        "criteria": {"true": "a ship or a journey by sea is described", "false": "no ship or journey by sea appears"},
    },
}
MULTILINGUAL_QUESTIONS = {
    "theme": THEME_CHOICE,
    "gravity": ENGLISH_QUESTIONS["gravity"],
    "mentions_death": ENGLISH_QUESTIONS["mentions_death"],
}

CHECKPOINTS = [
    {
        "name": "english",
        "subfolder": None,
        "cases": [
            {"name": "romeo_opening", "state": ROMEO_OPENING, "questions": ENGLISH_QUESTIONS},
            {"name": "hamlet_soliloquy", "state": HAMLET_SOLILOQUY, "questions": ENGLISH_QUESTIONS},
            {"name": "long_state_is_cut", "state": LONG_STATE, "questions": ENGLISH_QUESTIONS},
            {
                "name": "structured_state",
                "state": {"play": "Hamlet", "act": 3, "scene": 1, "lines": HAMLET_SOLILOQUY, "weight": 0.5},
                "questions": {"theme": THEME_CHOICE, "mentions_death": ENGLISH_QUESTIONS["mentions_death"]},
            },
            {
                "name": "conversation_state",
                "state": [
                    {"role": "user", "content": ROMEO_OPENING},
                    {"role": "assistant", "content": "A prologue about two feuding houses."},
                    {"role": "user", "content": HAMLET_SOLILOQUY},
                ],
                "questions": {"theme": THEME_CHOICE},
            },
        ],
    },
    {
        "name": "multilingual",
        "subfolder": "multilingual",
        "cases": [
            {"name": "hamlet_english", "state": HAMLET_SOLILOQUY, "questions": MULTILINGUAL_QUESTIONS},
            {"name": "hamlet_german", "state": HAMLET_GERMAN, "questions": MULTILINGUAL_QUESTIONS},
            {"name": "hamlet_french", "state": HAMLET_FRENCH, "questions": MULTILINGUAL_QUESTIONS},
            {"name": "long_state", "state": LONG_STATE, "questions": MULTILINGUAL_QUESTIONS},
        ],
    },
]


def reference_case(agent, case):
    question_names = list(case["questions"])
    for question_name in question_names:
        Agent._check_question(question_name, case["questions"][question_name])
    internal_questions = {name: Agent._to_internal(case["questions"][name]) for name in question_names}
    items = agent._encode_state(case["state"], question_names, internal_questions)
    batch = common.collate_items([items], agent.tok.pad_token_id)
    with torch.no_grad():
        logits, _act_probabilities = agent._forward(batch)
    question_vectors = []
    for row_index, question_name in enumerate(question_names):
        item = items[row_index]
        option_count = len(item["markers"])
        question_type_index = common.QTYPES[internal_questions[question_name]["t"]]
        temperature = agent.temperature_by_options.get(
            common.temp_bucket(question_type_index, option_count), agent.temperature[question_type_index]
        )
        option_logits = np.asarray(logits[row_index][:option_count], dtype=np.float64)
        scaled_logits = option_logits / temperature
        probabilities = np.exp(scaled_logits - scaled_logits.max())
        probabilities /= probabilities.sum()
        question_vectors.append({
            "name": question_name,
            "token_ids": [int(token_id) for token_id in item["ids"]],
            "marker_positions": [int(marker) for marker in item["markers"]],
            "option_logits": [float(logit) for logit in option_logits],
            "temperature": float(temperature),
            "probabilities": [float(probability) for probability in probabilities],
        })
    return {
        "name": case["name"],
        "state": case["state"],
        "questions": case["questions"],
        "reference": question_vectors,
    }


def main():
    checkpoint_directory = sys.argv[1]
    fixture = {"reference": "laya==0.4.15, float32 on the CPU", "checkpoints": []}
    for checkpoint in CHECKPOINTS:
        agent = laya.load(checkpoint_directory, device="cpu", subfolder=checkpoint["subfolder"])
        sys.stderr.write("loaded %s dtype=%s amp=%s\n" % (checkpoint["name"], agent.dtype, agent.amp_enabled))
        fixture["checkpoints"].append({
            "name": checkpoint["name"],
            "max_len": agent.cfg.get("max_len"),
            "head_max_len": agent.cfg.get("head_max_len"),
            "cases": [reference_case(agent, case) for case in checkpoint["cases"]],
        })
    json.dump(fixture, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
