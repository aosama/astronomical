"""Regenerates the Kokoro-82M reference speech used by the real-checkpoint parity journey.

Run it with a Python environment that has `kokoro==0.9.4`, `misaki[en]==0.9.4`,
spaCy's `en_core_web_sm` and `huggingface_hub` installed:

    python generate_kokoro_reference_speech.py > kokoro_reference_speech.json

It downloads the original `hexgrad/Kokoro-82M` checkpoint and the `af_heart`
voice at the pinned revision below into the Hugging Face cache, and records
that revision and the package versions in the fixture. Each case is spoken by
the reference implementation in float32 on the CPU.

The vocoder's excitation adds Gaussian noise, and no two runtimes draw the
same random numbers, so the noise is switched off here and in the journey.
The random initial phase is zeroed too, only for a reproducible fixture: it
never reaches the averaged phase the waveform is built from. What is left is deterministic arithmetic, which is what a
port can be held to.

For each case the fixture records what the model read (Misaki's phonemes and
the token identifiers), how long it held each token (frames), the pitch and
energy curves it drew, and a fingerprint of the audio: the root-mean-square
level of every 600-sample frame and a 2,400-sample stretch from the middle.
The whole waveform would make the fixture megabytes long.
"""

import json
import sys

import numpy as np
import torch

import kokoro
import misaki
from huggingface_hub import hf_hub_download
from kokoro import KModel
from misaki import en

REFERENCE_REPOSITORY = "hexgrad/Kokoro-82M"
# The revision whose `kokoro-v1_0.pth` and voices `mlx-community/Kokoro-82M-bf16`
# carries as safetensors; pinning it keeps the fixture describing fixed weights.
REFERENCE_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
VOICE_NAME = "af_heart"
SAMPLES_PER_FRAME = 600
EXCERPT_SAMPLE_COUNT = 2400

# Short, mid-length and long lines, so the style row, the duration rounding
# and the vocoder all see more than one utterance length.
SHAKESPEARE_CASES = [
    ("romeo_soft", "But soft, what light through yonder window breaks?"),
    ("juliet_rose", "What's in a name? That which we call a rose by any other name would smell as sweet."),
    (
        "hamlet_question",
        "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles.",
    ),
]


def rounded(measured_numbers, digits=6):
    flat_numbers = np.asarray(measured_numbers, dtype=np.float64).ravel()
    return [float(f"{number:.{digits}g}") for number in flat_numbers]


class NoiselessExcitation:
    """Makes `torch.rand` and `torch.randn_like` return zeros while active.

    Kokoro's `SineGen` and `SourceModuleHnNSF` are the only callers during an
    evaluation-mode forward pass, so nothing else is affected.
    """

    def __enter__(self):
        self.original_rand = torch.rand
        self.original_randn_like = torch.randn_like
        torch.rand = noiseless_rand
        torch.randn_like = torch.zeros_like
        return self

    def __exit__(self, *exception):
        torch.rand = self.original_rand
        torch.randn_like = self.original_randn_like


def noiseless_rand(*shape, **options):
    return torch.zeros(*shape, **options)


def load_reference_model():
    configuration_path = hf_hub_download(REFERENCE_REPOSITORY, "config.json", revision=REFERENCE_REVISION)
    weights_path = hf_hub_download(REFERENCE_REPOSITORY, "kokoro-v1_0.pth", revision=REFERENCE_REVISION)
    voice_path = hf_hub_download(
        REFERENCE_REPOSITORY, f"voices/{VOICE_NAME}.pt", revision=REFERENCE_REVISION
    )
    model = KModel(repo_id=REFERENCE_REPOSITORY, config=configuration_path, model=weights_path).eval()
    voice_pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    return model, voice_pack


def record_pitch_and_energy(model):
    """Wraps the predictor's `F0Ntrain` so its curves can be read after a pass."""
    recorded_curves = {}
    original_curves = model.predictor.F0Ntrain

    def recording_curves(aligned_states, prosody_style):
        pitch_curve, energy_curve = original_curves(aligned_states, prosody_style)
        recorded_curves["pitch"] = pitch_curve.squeeze().numpy()
        recorded_curves["energy"] = energy_curve.squeeze().numpy()
        return pitch_curve, energy_curve

    model.predictor.F0Ntrain = recording_curves
    return recorded_curves


def audio_fingerprint(audio):
    frame_count = len(audio) // SAMPLES_PER_FRAME
    frames = audio[: frame_count * SAMPLES_PER_FRAME].reshape(frame_count, SAMPLES_PER_FRAME)
    excerpt_start = max(0, (len(audio) - EXCERPT_SAMPLE_COUNT) // 2)
    return {
        "sample_count": len(audio),
        "frame_levels": rounded(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))),
        "excerpt_start": excerpt_start,
        "excerpt": rounded(audio[excerpt_start : excerpt_start + EXCERPT_SAMPLE_COUNT]),
    }


def speak_case(model, voice_pack, recorded_curves, phonemizer, case):
    case_name, text = case
    phonemes, _ = phonemizer(text)
    token_ids = [model.vocab[symbol] for symbol in phonemes if symbol in model.vocab]
    reference_style = voice_pack[len(phonemes) - 1]
    with NoiselessExcitation():
        audio, frame_counts = model.forward_with_tokens(
            torch.LongTensor([[0, *token_ids, 0]]), reference_style, 1.0
        )
    return {
        "name": case_name,
        "text": text,
        "phonemes": phonemes,
        "token_ids": token_ids,
        "frame_counts": frame_counts.tolist(),
        "pitch_curve": rounded(recorded_curves["pitch"]),
        "energy_curve": rounded(recorded_curves["energy"]),
        "audio": audio_fingerprint(audio.numpy()),
    }


def main():
    model, voice_pack = load_reference_model()
    recorded_curves = record_pitch_and_energy(model)
    phonemizer = en.G2P(trf=False, british=False, fallback=None)
    cases = []
    for case_number, case in enumerate(SHAKESPEARE_CASES, start=1):
        print(f"speaking case {case_number}/{len(SHAKESPEARE_CASES)}: {case[0]}", file=sys.stderr)
        cases.append(speak_case(model, voice_pack, recorded_curves, phonemizer, case))
    fixture = {
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_revision": REFERENCE_REVISION,
        "voice_name": VOICE_NAME,
        "speed": 1.0,
        "versions": {"kokoro": kokoro.__version__, "misaki": misaki.__version__, "torch": torch.__version__},
        "cases": cases,
    }
    json.dump(fixture, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
