"""Regenerates the Clef reference decisions used by the real-checkpoint parity journey.

Run it with a Python environment that has `torch`, `transformers` (5.x, with
Qwen3.5 support), `safetensors`, `numpy` and `huggingface_hub` installed:

    python generate_clef_reference_decisions.py > clef_reference_decisions_flash_8bit.json

It downloads `mlx-community/clef-flash-8bit` at a pinned revision into the
Hugging Face cache, the build Astronomical serves, and answers each case in
float32 on the CPU:

* the backbone is transformers' own Qwen3.5 text model, given the build's
  weights: each 8-bit affine tensor is dequantized exactly (`q * scale +
  bias`, groups of 64) into float32 when its decoder layer runs and freed
  after, the MLX conversion's conventions are undone (its five RMSNorm
  families store `1 + gamma` where Hugging Face stores `gamma`, and its
  convolution is `[channels, kernel, 1]`), and nothing is rounded;
* the joint head is `JointSchemaHead` from `Cloudflare/clef-flash`'s
  `joint_schema_model.py` (revision noted below), rewritten here rather than
  imported, in float32;
* the prompt follows that file's `encode_record`, and probabilities its
  plain softmax.

Both sides read identical weights, so the comparison measures the
implementation, not the 8-bit build: how far 8-bit Clef sits from the
bfloat16 release is a property of the build, not of Astronomical.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

BUILD_REPOSITORY = "mlx-community/clef-flash-8bit"
BUILD_REVISION = "5a195dc147926b63b1d139691ed4480ca9f92c0e"
# The `joint_schema_model.py` the head, prompt and answers follow.
CLEF_SOURCE_REPOSITORY = "Cloudflare/clef-flash"
CLEF_SOURCE_REVISION = "fb4417d31c1d0f9e7e77f1141b40aaede7491e00"
MAXIMUM_SEQUENCE_TOKENS = 16384
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
# The RMSNorm weights an MLX conversion stores as `1 + gamma`.
OFFSET_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)

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
        "name": "question_without_instructions",
        "state": TEMPEST_STORM,
        "questions": {"mentions_death": {"type": "noul"}, "sea": {"type": "noul", "instructions": {"ask": "Is there a ship?", "act": 1}}},
    },
]


# ---------------------------------------------------------------- the prompt


def render(value):
    """`render`: a string as written, anything else as compact sorted JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question):
    """`question_options`: Clef's own option order and descriptions."""
    if question["type"] == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question["type"] == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


def tokens(tokenizer, text):
    return tokenizer(text, add_special_tokens=False).input_ids


def encode_record(tokenizer, state, questions):
    """`encode_record`: one sequence, with the spans the head pools over."""
    schema_ids = tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    encoded_questions = []
    for question_index, (question_id, question) in enumerate(questions.items()):
        schema_ids += tokens(
            tokenizer,
            f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
        )
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids += tokens(tokenizer, render(instructions))
        question_end = len(schema_ids)
        schema_ids += tokens(tokenizer, "\nALLOWED OPTIONS:\n")
        option_spans, option_ids = [], []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids += tokens(tokenizer, f"OPTION {option_index + 1}: ")
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids += tokens(tokenizer, render(semantics))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids += tokens(tokenizer, "\n")
        schema_ids += tokens(tokenizer, "END FIELD\n")
        encoded_questions.append({
            "question_id": question_id,
            "type_id": QUESTION_TYPES[question["type"]],
            "question_span": (question_start, question_end),
            "option_spans": option_spans,
            "option_ids": option_ids,
        })
    prefix_ids = tokens(tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix_ids = tokens(tokenizer, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    state_ids = tokens(tokenizer, render(state))
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    state_ids = state_ids[: MAXIMUM_SEQUENCE_TOKENS - fixed_length]
    schema_offset = len(prefix_ids) + len(state_ids)
    for encoded_question in encoded_questions:
        start, end = encoded_question["question_span"]
        encoded_question["question_span"] = (start + schema_offset, end + schema_offset)
        encoded_question["option_spans"] = [(start + schema_offset, end + schema_offset) for start, end in encoded_question["option_spans"]]
    return prefix_ids + state_ids + schema_ids + suffix_ids, encoded_questions


# ------------------------------------------------- the build's weights, exact


class MlxBuildWeights:
    """The build's tensors, each read as an exact float32 PyTorch tensor in
    Hugging Face's conventions."""

    def __init__(self, build_directory):
        index = json.load(open(f"{build_directory}/model.safetensors.index.json"))
        self.shard_by_name = {name: f"{build_directory}/{shard}" for name, shard in index["weight_map"].items()}
        quantization = json.load(open(f"{build_directory}/config.json"))["quantization"]
        self.group_size = quantization["group_size"]
        self.bits = quantization["bits"]
        self.open_shards = {}

    def raw(self, tensor_name):
        shard_path = self.shard_by_name[tensor_name]
        if shard_path not in self.open_shards:
            self.open_shards[shard_path] = safe_open(shard_path, "pt")
        return self.open_shards[shard_path].get_tensor(tensor_name)

    def float32(self, mlx_name):
        """`mlx_name` (its weight, or the dequantized affine triple) in float32."""
        prefix = mlx_name.removesuffix(".weight")
        if f"{prefix}.scales" in self.shard_by_name and mlx_name.endswith(".weight"):
            return self.dequantized(prefix)
        tensor = self.raw(mlx_name).float()
        if mlx_name == "language_model.model.norm.weight" or mlx_name.endswith(OFFSET_NORM_SUFFIXES):
            tensor = tensor - 1.0
        if mlx_name.endswith(".conv1d.weight"):
            tensor = tensor.transpose(1, 2).contiguous()
        return tensor

    def dequantized(self, prefix, row_ids=None):
        """The affine triple under `prefix` as float32, all rows or `row_ids`."""
        packed_rows = self.raw(f"{prefix}.weight")
        scales = self.raw(f"{prefix}.scales").float()
        biases = self.raw(f"{prefix}.biases").float()
        if row_ids is not None:
            row_index = torch.tensor(row_ids)
            packed_rows, scales, biases = packed_rows[row_index], scales[row_index], biases[row_index]
        packed = packed_rows.numpy().view(np.uint32)
        values_per_word = 32 // self.bits
        level_mask = (1 << self.bits) - 1
        levels = np.empty((packed.shape[0], packed.shape[1] * values_per_word), dtype=np.float32)
        for slot in range(values_per_word):
            levels[:, slot::values_per_word] = (packed >> (slot * self.bits)) & level_mask
        grouped = torch.from_numpy(levels).view(levels.shape[0], -1, self.group_size)
        return (grouped * scales.unsqueeze(-1) + biases.unsqueeze(-1)).view(levels.shape[0], -1)


def hugging_face_name(mlx_name):
    if mlx_name.startswith("language_model.model."):
        return "model." + mlx_name.removeprefix("language_model.model.")
    return mlx_name.removeprefix("language_model.")


class LayerFromBuild:
    """Gives a decoder layer its parameters from the build when it runs and
    takes them back after, so only the running layer exists in float32; the
    rest of the time its parameters are shapes on the `meta` device."""

    def __init__(self, build_weights, layer_index, decoder_layer):
        self.build_weights = build_weights
        self.mlx_prefix = f"language_model.model.layers.{layer_index}."
        self.meta_parameters = dict(decoder_layer.named_parameters())

    def attach(self, decoder_layer):
        decoder_layer.register_forward_pre_hook(self.fill)
        decoder_layer.register_forward_hook(self.empty)

    def fill(self, decoder_layer, _layer_inputs):
        for parameter_name in self.meta_parameters:
            self.replace(decoder_layer, parameter_name, self.build_weights.float32(self.mlx_prefix + parameter_name))

    def empty(self, decoder_layer, _layer_inputs, _layer_outputs):
        for parameter_name, meta_parameter in self.meta_parameters.items():
            self.replace(decoder_layer, parameter_name, meta_parameter)

    @staticmethod
    def replace(decoder_layer, parameter_name, tensor):
        owner_path, _, leaf_name = parameter_name.rpartition(".")
        owner = decoder_layer.get_submodule(owner_path) if owner_path else decoder_layer
        owner._parameters[leaf_name] = tensor if isinstance(tensor, nn.Parameter) else nn.Parameter(tensor, requires_grad=False)


def load_backbone(build_directory, build_weights):
    text_configuration = Qwen3_5TextConfig(**json.load(open(f"{build_directory}/config.json"))["text_config"])
    with torch.device("meta"):
        causal_model = Qwen3_5ForCausalLM(text_configuration)
    text_model = causal_model.model
    text_model.embed_tokens.weight = nn.Parameter(build_weights.float32("language_model.model.embed_tokens.weight"), requires_grad=False)
    text_model.norm.weight = nn.Parameter(build_weights.float32("language_model.model.norm.weight"), requires_grad=False)
    rotary_embedding = text_model.rotary_emb
    inverse_frequencies, _ = rotary_embedding.compute_default_rope_parameters(rotary_embedding.config)
    rotary_embedding.inv_freq = inverse_frequencies.float()
    rotary_embedding.original_inv_freq = inverse_frequencies.float().clone()
    for layer_index, decoder_layer in enumerate(text_model.layers):
        LayerFromBuild(build_weights, layer_index, decoder_layer).attach(decoder_layer)
    return text_model.eval()


# --------------------------------------------- `JointSchemaHead`, in float32


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(0.0), nn.Linear(feedforward, width), nn.Dropout(0.0))

    def forward(self, queries, memory):
        normalized_memory = self.memory_norm(memory)
        routed, _ = self.attention(self.query_norm(queries), normalized_memory, normalized_memory, need_weights=False)
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(nn.Module):
    def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        for projection_name in ("memory", "question", "option_question", "global", "option_context", "option_lexical"):
            setattr(self, f"{projection_name}_projection", nn.Linear(hidden_size, width, bias=False))
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = nn.ModuleList([EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)])
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=width, nhead=heads, dim_feedforward=feedforward, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(nn.Linear(width * 4, width), nn.GELU(), nn.Dropout(0.0), nn.Linear(width, 1))
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden_states, input_ids, encoded_questions, output_rows):
        """One record: `hidden_states` `[L, H]`; returns one logit tensor per question."""
        sequence_hidden = self.hidden_norm(hidden_states)
        memory = self.memory_projection(sequence_hidden).unsqueeze(0)
        global_vector = sequence_hidden[-1]
        question_vectors = torch.stack([sequence_hidden[start:end].mean(0) for start, end in (q["question_span"] for q in encoded_questions)])
        type_ids = torch.tensor([q["type_id"] for q in encoded_questions])
        option_contexts, lexical_options, option_counts = [], [], []
        for encoded_question in encoded_questions:
            option_contexts.append(torch.stack([sequence_hidden[start:end].mean(0) for start, end in encoded_question["option_spans"]]))
            lexical_options.append(torch.stack([output_rows(input_ids[start:end]).mean(0) for start, end in encoded_question["option_spans"]]))
            option_counts.append(len(encoded_question["option_spans"]))
        option_queries = [
            self.option_context_projection(contexts) + self.option_lexical_projection(lexical)
            + self.option_question_projection(question_vectors[question_index]).unsqueeze(0)
            for question_index, (contexts, lexical) in enumerate(zip(option_contexts, lexical_options))
        ]
        routed_options = torch.cat(option_queries, 0).unsqueeze(0)
        for evidence_layer in self.evidence_layers:
            routed_options = evidence_layer(routed_options, memory)
        split_options = list(torch.split(routed_options[0], option_counts, 0))
        base_fields = self.question_projection(question_vectors)
        option_summaries = []
        for field, options in zip(base_fields, split_options):
            routing_weights = torch.softmax(options @ field / math.sqrt(options.shape[-1]), 0)
            option_summaries.append((routing_weights.unsqueeze(-1) * options).sum(0))
        fields = (base_fields + self.option_summary_norm(torch.stack(option_summaries))
                  + self.global_projection(global_vector).unsqueeze(0) + self.type_embedding(type_ids)).unsqueeze(0)
        for decoder_layer in self.layers:
            fields = decoder_layer(fields, memory)
        fields = self.field_norm(fields[0])
        question_logits = []
        for question_index, (field, lexical, routed) in enumerate(zip(fields, lexical_options, split_options)):
            anchor = F.normalize(question_vectors[question_index] + global_vector, dim=-1)
            prior = self.prior_logit_scale.clamp(max=math.log(100.0)).exp() * (F.normalize(lexical, dim=-1) @ anchor)
            options = self.option_norm(routed)
            repeated_field = field.unsqueeze(0).expand_as(options)
            cosine = F.cosine_similarity(repeated_field, options, dim=-1)
            features = torch.cat([repeated_field, options, repeated_field * options, (repeated_field - options).abs()], -1)
            residual = self.residual_scorer(features).squeeze(-1)
            joint = self.joint_logit_scale.clamp(max=math.log(100.0)).exp() * cosine + residual
            question_logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
        return question_logits


def load_head(build_directory):
    head = JointSchemaHead(**json.load(open(f"{build_directory}/joint_head_config.json")))
    with safe_open(f"{build_directory}/joint_head.safetensors", "pt") as head_file:
        head_state = {name: head_file.get_tensor(name).float() for name in head_file.keys()}
    head.load_state_dict(head_state, strict=True)
    return head.float().eval()


# ------------------------------------------------------------------- the run


@torch.no_grad()
def answer_case(tokenizer, backbone, head, build_weights, case):
    token_ids, encoded_questions = encode_record(tokenizer, case["state"], case["questions"])
    row_ids = torch.tensor([token_ids])
    hidden_states = backbone(input_ids=row_ids, position_ids=torch.arange(len(token_ids)).unsqueeze(0)).last_hidden_state[0].float()
    head_rows_needed = sorted({token_id for q in encoded_questions for start, end in q["option_spans"] for token_id in token_ids[start:end]})
    needed_rows = build_weights.dequantized("language_model.lm_head", head_rows_needed)
    rows_by_token = dict(zip(head_rows_needed, needed_rows))

    def output_rows(option_token_ids):
        return torch.stack([rows_by_token[int(token_id)] for token_id in option_token_ids])

    question_logits = head(hidden_states, torch.tensor(token_ids), encoded_questions, output_rows)
    reference_rows = []
    for encoded_question, logits in zip(encoded_questions, question_logits):
        reference_rows.append({
            "question": encoded_question["question_id"],
            "option_ids": encoded_question["option_ids"],
            "option_logits": logits.tolist(),
            "probabilities": logits.softmax(-1).tolist(),
        })
    return {**case, "token_ids": token_ids, "reference": reference_rows}


def main():
    torch.manual_seed(0)
    build_directory = snapshot_download(BUILD_REPOSITORY, revision=BUILD_REVISION)
    build_weights = MlxBuildWeights(build_directory)
    tokenizer = AutoTokenizer.from_pretrained(build_directory)
    backbone = load_backbone(build_directory, build_weights)
    head = load_head(build_directory)
    answered_cases, started_at, last_report_at = [], time.monotonic(), 0.0
    for case_index, case in enumerate(CASES, start=1):
        answered_cases.append(answer_case(tokenizer, backbone, head, build_weights, case))
        elapsed_seconds = time.monotonic() - started_at
        if elapsed_seconds - last_report_at >= 5.0 or case_index == len(CASES):
            last_report_at = elapsed_seconds
            print(f"[clef-reference] cases={case_index}/{len(CASES)} percent={100 * case_index / len(CASES):.0f} "
                  f"elapsed_seconds={elapsed_seconds:.0f} eta_seconds={elapsed_seconds / case_index * (len(CASES) - case_index):.0f}",
                  file=sys.stderr)
    fixture = {
        "reference": f"Clef as in {CLEF_SOURCE_REPOSITORY} joint_schema_model.py at {CLEF_SOURCE_REVISION[:8]}; "
                     f"transformers {transformers.__version__}, torch {torch.__version__}, float32 on the CPU",
        "build_repository": BUILD_REPOSITORY,
        "build_revision": BUILD_REVISION,
        "cases": answered_cases,
    }
    json.dump(fixture, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
