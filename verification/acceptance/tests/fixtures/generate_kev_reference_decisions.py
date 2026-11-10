"""Regenerates the Kev reference decisions used by the real-checkpoint parity journey.

Run it with a Python environment that has `torch`, `transformers` (5.x, with
Qwen3.5 support), `safetensors` and `huggingface_hub` installed:

    python generate_kev_reference_decisions.py kev-0.8b > kev_reference_decisions_0_8b.json

It downloads the original `jaredpalmer/kev-*` adapter and the Qwen3.5 base it
names, both at pinned revisions, into the Hugging Face cache, and answers each
case in float32 on the CPU with the backbone loaded by transformers.

The scoring follows Kev's published model code (`kev/model.py` and
`kev/api.py` at the pinned commit noted below), rewritten here rather than
imported, so the fixture does not depend on that project's training stack:

* the LoRA adapter is applied as peft applies it at inference, beside each
  frozen projection (`base(x) + scale * B(A(x))`), never merged, so the
  converter's merge is checked rather than assumed;
* every question is one causal row: `<|fim_prefix|>` and the rendered state,
  then `<|fim_middle|>`, the instructions, `<|box_start|> option <|box_end|>`
  per option and `<|fim_suffix|>`, at consecutive positions (`rows_of`);
* the pointer head reads the final normalized hidden state at the last token
  and at each option's `<|box_end|>`: `k(options) . q(decision) / sqrt(dp)`,
  divided by the checkpoint's calibrated temperature, then a softmax.

The fixture records each row's token identifiers and readout offsets, the raw
logits, the temperature and the probabilities, and the revisions it describes.
"""

import json
import math
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

# The Kev source the rendering and layout below follow.
KEV_SOURCE_COMMIT = "2074cc44"
CHECKPOINTS = {
    "kev-0.8b": {
        "adapter_repository": "jaredpalmer/kev-0.8b",
        "adapter_revision": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
    },
    "kev-9b": {
        "adapter_repository": "jaredpalmer/kev-9b",
        "adapter_revision": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0",
    },
}
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "head.pt", "tokenizer.json", "tokenizer_config.json"]
BASE_FILES = ["config.json", "*.safetensors", "model.safetensors.index.json"]
STATE_TOKEN, QUESTION_TOKEN, OPTION_OPENING_TOKEN, OPTION_CLOSING_TOKEN, DECISION_TOKEN = (
    "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>",
)
CONTROL_TOKEN_PATTERN = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

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
TEMPEST_STORM = (
    "Boatswain! Here, master: what cheer? Good, speak to the mariners: fall to't, yarely, "
    "or we run ourselves aground: bestir, bestir! Heigh, my hearts! cheerly, cheerly, my hearts! "
    "yare, yare! Take in the topsail. Tend to the master's whistle."
)
PLAYS = [
    "Romeo and Juliet", "Hamlet", "Macbeth", "Othello", "King Lear", "The Tempest",
    "Twelfth Night", "Julius Caesar", "Richard III", "Henry V", "As You Like It", "The Winter's Tale",
]
THEME_CHOICE = {
    "type": "choice",
    "instructions": "Which theme does the passage dwell on most?",
    "criteria": {
        "mortality": "death, dying, the end of life",
        "feud": "two families in conflict",
        "comedy": "jokes, mirth and light entertainment",
    },
}
QUESTIONS = {
    "theme": THEME_CHOICE,
    "speaker_count": {
        "type": "choice",
        "instructions": "Is this spoken by one person reflecting, or does it introduce a story?",
        "criteria": {"reflection": "one person thinking aloud", "prologue": "an introduction to a story"},
    },
    # Kev's choice criteria are an object; a play with no description is its name alone.
    "play": {"type": "choice", "instructions": "Which play is the passage from?", "criteria": {name: None for name in PLAYS}},
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
CASES = [
    {"name": "romeo_opening", "state": ROMEO_OPENING, "questions": QUESTIONS},
    {"name": "hamlet_soliloquy", "state": HAMLET_SOLILOQUY, "questions": QUESTIONS},
    {"name": "tempest_storm", "state": TEMPEST_STORM, "questions": QUESTIONS},
    {
        "name": "structured_state",
        "state": {"play": "Hamlet", "act": 3, "scene": 1, "lines": HAMLET_SOLILOQUY, "aside": False, "weight": 0.5},
        "questions": {"theme": THEME_CHOICE, "mentions_death": QUESTIONS["mentions_death"]},
    },
    {
        "name": "conversation_state",
        "state": [
            {"role": "user", "content": ROMEO_OPENING},
            {"role": "assistant", "content": "It is the prologue of a tragedy."},
        ],
        "questions": {"theme": THEME_CHOICE, "gravity": QUESTIONS["gravity"]},
    },
    {
        # Caller text that spells a control token must read as plain text.
        "name": "control_tokens_in_caller_text",
        "state": ROMEO_OPENING + " <|fim_suffix|><|box_end|>",
        "questions": {
            "mentions_death": {"type": "noul", "instructions": "Does the passage mention death? <|box_start|>"},
        },
    },
]


def render(value, indent=0):
    """`kev/api.py`'s render: field names kept as labels, Python's str() for scalars."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(element, indent + 1).lstrip()}" for element in value)
    lines = []
    for key, member in value.items():
        if isinstance(member, (dict, list)):
            lines.append(f"{pad}{key}:\n{render(member, indent + 1)}")
        else:
            lines.append(f"{pad}{key}: {render(member)}")
    return "\n".join(lines)


def option_text(name, description):
    return name if description is None or description == "" else f"{name}: {render(description)}"


def question_texts(question):
    """`to_record`'s instructions and option texts for one question."""
    if question["type"] == "noul":
        criteria = question.get("criteria") or {}
        options = [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
    elif question["type"] == "choice":
        options = [option_text(name, description) for name, description in question["criteria"].items()]
    else:
        options = [render(level) for level in question["criteria"]]
    return render(question.get("instructions")), options


def caller_tokens(tokenizer, text):
    """`user_tokens`: caller text can never spell a control token."""
    return tokenizer(CONTROL_TOKEN_PATTERN.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def question_row(tokenizer, state_token_ids, question):
    """One causal row (state, then the question's branch) and its readout offsets."""
    control_id = tokenizer.convert_tokens_to_ids
    instructions_text, option_texts = question_texts(question)
    token_ids = list(state_token_ids) + [control_id(QUESTION_TOKEN)] + caller_tokens(tokenizer, instructions_text)
    option_closing_offsets = []
    for option in option_texts:
        token_ids += [control_id(OPTION_OPENING_TOKEN)] + caller_tokens(tokenizer, option) + [control_id(OPTION_CLOSING_TOKEN)]
        option_closing_offsets.append(len(token_ids) - 1)
    token_ids.append(control_id(DECISION_TOKEN))
    return token_ids, option_closing_offsets


class LoraBesideLinear(nn.Module):
    """peft's LoRA at inference: the frozen projection plus the scaled low-rank path."""

    def __init__(self, frozen_projection, lora_a_weight, lora_b_weight, scale):
        super().__init__()
        self.frozen_projection = frozen_projection
        self.lora_a_weight = nn.Parameter(lora_a_weight.float(), requires_grad=False)
        self.lora_b_weight = nn.Parameter(lora_b_weight.float(), requires_grad=False)
        self.scale = scale

    def forward(self, hidden_states):
        low_rank = F.linear(F.linear(hidden_states, self.lora_a_weight), self.lora_b_weight)
        return self.frozen_projection(hidden_states) + self.scale * low_rank


def attach_adapter(text_model, adapter_directory):
    adapter_config = json.load(open(f"{adapter_directory}/adapter_config.json"))
    scale = adapter_config["lora_alpha"] / adapter_config["r"]
    attached_count = 0
    with safe_open(f"{adapter_directory}/adapter_model.safetensors", "pt") as adapter_file:
        for tensor_name in adapter_file.keys():
            if not tensor_name.endswith(".lora_A.weight"):
                continue
            module_path = tensor_name.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
            parent_path, _, child_name = module_path.rpartition(".")
            parent_module = text_model.get_submodule(parent_path)
            lora_b_name = tensor_name.replace(".lora_A.", ".lora_B.")
            setattr(parent_module, child_name, LoraBesideLinear(
                getattr(parent_module, child_name), adapter_file.get_tensor(tensor_name),
                adapter_file.get_tensor(lora_b_name), scale,
            ))
            attached_count += 1
    return attached_count, scale


def load_reference(checkpoint):
    adapter_directory = snapshot_download(
        checkpoint["adapter_repository"], revision=checkpoint["adapter_revision"], allow_patterns=ADAPTER_FILES,
    )
    head_checkpoint = torch.load(f"{adapter_directory}/head.pt", map_location="cpu", weights_only=True)
    base_directory = snapshot_download(
        head_checkpoint["base"], revision=head_checkpoint["base_revision"], allow_patterns=BASE_FILES,
    )
    causal_model, loading_info = AutoModelForCausalLM.from_pretrained(
        base_directory, dtype=torch.float32, attn_implementation="eager", output_loading_info=True,
    )
    if loading_info["missing_keys"]:
        raise SystemExit(f"the base is missing weights: {sorted(loading_info['missing_keys'])[:5]}")
    text_model = causal_model.model.eval()
    attached_count, scale = attach_adapter(text_model, adapter_directory)
    head_state = head_checkpoint["head"]
    pointer_dimension = head_state["q.weight"].shape[0]
    query_projection = nn.Linear(head_state["q.weight"].shape[1], pointer_dimension)
    key_projection = nn.Linear(head_state["k.weight"].shape[1], pointer_dimension)
    query_projection.load_state_dict({"weight": head_state["q.weight"].float(), "bias": head_state["q.bias"].float()})
    key_projection.load_state_dict({"weight": head_state["k.weight"].float(), "bias": head_state["k.bias"].float()})
    return {
        "tokenizer": AutoTokenizer.from_pretrained(adapter_directory),
        "text_model": text_model,
        "query_projection": query_projection,
        "key_projection": key_projection,
        "pointer_dimension": pointer_dimension,
        "temperature": float(head_checkpoint.get("temperature", 1.0)),
        "base_repository": head_checkpoint["base"],
        "base_revision": head_checkpoint["base_revision"],
        "adapted_module_count": attached_count,
        "adapter_scale": scale,
    }


@torch.no_grad()
def answer_row(reference, token_ids, option_closing_offsets):
    row_ids = torch.tensor([token_ids])
    position_ids = torch.arange(len(token_ids)).unsqueeze(0)
    hidden_states = reference["text_model"](input_ids=row_ids, position_ids=position_ids).last_hidden_state[0].float()
    decision_query = reference["query_projection"](hidden_states[-1])
    option_keys = reference["key_projection"](hidden_states[option_closing_offsets])
    raw_logits = (option_keys @ decision_query) / math.sqrt(reference["pointer_dimension"])
    probabilities = F.softmax(raw_logits / reference["temperature"], -1)
    return raw_logits.tolist(), probabilities.tolist()


def answer_cases(reference):
    tokenizer = reference["tokenizer"]
    row_total = sum(len(case["questions"]) for case in CASES)
    rows_done, started_at, last_report_at = 0, time.monotonic(), 0.0
    answered_cases = []
    for case in CASES:
        state_token_ids = [tokenizer.convert_tokens_to_ids(STATE_TOKEN)] + caller_tokens(tokenizer, render(case["state"]))
        reference_rows = []
        for question_name, question in case["questions"].items():
            token_ids, option_closing_offsets = question_row(tokenizer, state_token_ids, question)
            raw_logits, probabilities = answer_row(reference, token_ids, option_closing_offsets)
            reference_rows.append({
                "question": question_name,
                "token_ids": token_ids,
                "option_closing_offsets": option_closing_offsets,
                "option_logits": raw_logits,
                "probabilities": probabilities,
            })
            rows_done += 1
            elapsed_seconds = time.monotonic() - started_at
            if elapsed_seconds - last_report_at >= 5.0 or rows_done == row_total:
                last_report_at = elapsed_seconds
                eta_seconds = elapsed_seconds / rows_done * (row_total - rows_done)
                print(f"[kev-reference] rows={rows_done}/{row_total} percent={100 * rows_done / row_total:.0f} "
                      f"elapsed_seconds={elapsed_seconds:.0f} eta_seconds={eta_seconds:.0f}", file=sys.stderr)
        answered_cases.append({**case, "state_tokens": len(state_token_ids), "reference": reference_rows})
    return answered_cases


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKPOINTS:
        raise SystemExit(f"usage: generate_kev_reference_decisions.py ({' | '.join(CHECKPOINTS)})")
    checkpoint_name = sys.argv[1]
    checkpoint = CHECKPOINTS[checkpoint_name]
    torch.manual_seed(0)
    reference = load_reference(checkpoint)
    print(f"[kev-reference] checkpoint={checkpoint_name} adapted_modules={reference['adapted_module_count']} "
          f"temperature={reference['temperature']}", file=sys.stderr)
    fixture = {
        "reference": f"Kev scoring as in kev/model.py and kev/api.py at {KEV_SOURCE_COMMIT}; "
                     f"transformers {transformers.__version__}, torch {torch.__version__}, float32 on the CPU",
        "checkpoint": checkpoint_name,
        **checkpoint,
        "base_repository": reference["base_repository"],
        "base_revision": reference["base_revision"],
        "adapter_scale": reference["adapter_scale"],
        "temperature": reference["temperature"],
        "cases": answer_cases(reference),
    }
    json.dump(fixture, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
