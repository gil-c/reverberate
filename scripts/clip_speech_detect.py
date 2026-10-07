"""Where a clip holds speech: three detectors, one row a clip, for the screen.

``reverberate.scenes.screen`` decides what a noise clip loses; it does not
listen. This script listens, and writes what it heard as the *detections* the
screen reads (``docs/formats/clip-library.md``, "Screening"). It runs in an
environment of its own, since none of the three models belongs to the
project's requirements:

    uv venv .venv/screen && uv pip install --python .venv/screen/bin/python \
        torch silero-vad transformers faster-whisper soundfile scipy numpy
    nice -n 19 .venv/screen/bin/python scripts/clip_speech_detect.py \
        --manifest src/reverberate/scenes/library/clarify_v1.json \
        --root <data root>/clips --out detections.json

or ``--list FILES.json``, a list of ``{"key": ..., "path": ...}``, for files
that are in no library yet.

The detectors (``docs/open-questions/clip-library-audit.md`` compares them):

- ``vad``: Silero VAD, the probability of speech every 32 ms. Fast, and it
  answers for a voice in the foreground; a murmur of many voices mostly
  passes under it.
- ``tags``: the Audio Spectrogram Transformer trained on AudioSet, the scores
  of the classes that say voice, singing, music or a programme, on windows of
  10 s every 5 s. It hears a crowd and a song, and says so by the window.
- ``words``: Whisper (``small``, through faster-whisper), the words it writes
  with their times and probabilities. A word it is sure of is speech one
  understands, which is what a separation model must not be taught to remove.

The first 300 s of a clip are read (``--seconds``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "reverberate.clip-detections"
SCHEMA_VERSION = 1
RATE_HZ = 16000
#: Silero reads 512 samples at 16 kHz at a time.
VAD_FRAME = 512
TAG_WINDOW_S, TAG_HOP_S = 10.0, 5.0
TAGGER = "MIT/ast-finetuned-audioset-10-10-0.4593"
#: The AudioSet classes kept, by the name the model gives them.
TAGS = (
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Speech synthesizer",
    "Whispering",
    "Shout",
    "Babbling",
    "Laughter",
    "Crowd",
    "Chatter",
    "Hubbub, speech noise, speech babble",
    "Children playing",
    "Singing",
    "Choir",
    "Rapping",
    "Humming",
    "Vocal music",
    "A capella",
    "Music",
    "Television",
    "Radio",
)


def load(path: Path, seconds: float) -> np.ndarray:
    """A file as mono float32 at 16 kHz, its first ``seconds``."""
    import soundfile
    from scipy.signal import resample_poly

    with soundfile.SoundFile(str(path)) as handle:
        rate = handle.samplerate
        samples = handle.read(int(seconds * rate), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != RATE_HZ:
        from math import gcd

        g = gcd(RATE_HZ, rate)
        mono = resample_poly(mono, RATE_HZ // g, rate // g).astype(np.float32)
    return np.ascontiguousarray(mono)


class Vad:
    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        self.torch = torch
        self.model = load_silero_vad()

    def __call__(self, mono: np.ndarray) -> list[float]:
        torch = self.torch
        self.model.reset_states()
        count = mono.size // VAD_FRAME
        out = []
        with torch.no_grad():
            for k in range(count):
                frame = torch.from_numpy(mono[k * VAD_FRAME : (k + 1) * VAD_FRAME])
                out.append(round(float(self.model(frame, RATE_HZ)), 2))
        return out


class Tagger:
    def __init__(self) -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.torch = torch
        self.extract = ASTFeatureExtractor.from_pretrained(TAGGER)
        self.model = ASTForAudioClassification.from_pretrained(TAGGER).eval()
        names = {name: index for index, name in self.model.config.id2label.items()}
        self.index = [int(names[tag]) for tag in TAGS]

    def __call__(self, mono: np.ndarray) -> list[dict[str, Any]]:
        torch = self.torch
        window, hop = int(TAG_WINDOW_S * RATE_HZ), int(TAG_HOP_S * RATE_HZ)
        starts = list(range(0, max(mono.size - window, 0) + 1, hop)) or [0]
        out = []
        for start in starts:
            piece = mono[start : start + window]
            inputs = self.extract(piece, sampling_rate=RATE_HZ, return_tensors="pt")
            with torch.no_grad():
                scores = torch.sigmoid(self.model(**inputs).logits[0])
            out.append(
                {
                    "start_s": round(start / RATE_HZ, 2),
                    "end_s": round(min(start + window, mono.size) / RATE_HZ, 2),
                    "scores": {
                        tag: round(float(scores[i]), 3)
                        for tag, i in zip(TAGS, self.index, strict=True)
                    },
                }
            )
        return out


class Words:
    def __init__(self, size: str, threads: int) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads)

    def __call__(self, mono: np.ndarray) -> dict[str, Any]:
        segments, info = self.model.transcribe(
            mono,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            beam_size=1,
        )
        # On a noise the recogniser writes anyway: one syllable a hundred
        # times, or the thanks that end a video. What tells such a segment
        # from speech is kept with it, and the screen reads it
        # (``reverberate.scenes.screen.sure_words``).
        out = []
        for segment in segments:
            out.append(
                {
                    "start_s": round(float(segment.start), 2),
                    "end_s": round(float(segment.end), 2),
                    "text": segment.text.strip(),
                    "no_speech": round(float(segment.no_speech_prob), 2),
                    "log_probability": round(float(segment.avg_logprob), 2),
                    "compression": round(float(segment.compression_ratio), 2),
                    "words": [
                        [
                            round(float(word.start), 2),
                            round(float(word.end), 2),
                            round(float(word.probability), 2),
                            word.word.strip(),
                        ]
                        for word in segment.words or []
                    ],
                }
            )
        return {
            "language": info.language,
            "language_probability": round(float(info.language_probability), 2),
            "segments": out,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--root", type=Path, help="the folder of libraries, <data root>/clips")
    parser.add_argument("--list", type=Path, help='[{"key": ..., "path": ...}, ...]')
    parser.add_argument(
        "--kind", default=None, help="of a manifest, only this kind: voice or noise"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--whisper", default="small")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip", nargs="*", default=[], choices=("vad", "tags", "words"))
    args = parser.parse_args()

    items: list[tuple[str, Path]] = []
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        for clip in manifest["clips"]:
            if args.kind in (None, clip["kind"]):
                items.append(
                    (clip["name"], args.root / manifest["library"] / f"{clip['name']}.wav")
                )
    if args.list:
        items += [(item["key"], Path(item["path"])) for item in json.loads(args.list.read_text())]

    import torch

    torch.set_num_threads(args.threads)
    done: dict[str, Any] = {}
    if args.out.exists():
        done = json.loads(args.out.read_text())["clips"]
    vad = None if "vad" in args.skip else Vad()
    tagger = None if "tags" in args.skip else Tagger()
    words = None if "words" in args.skip else Words(args.whisper, args.threads)
    began, heard = time.time(), 0.0
    for number, (key, path) in enumerate(items):
        if key in done:
            continue
        try:
            mono = load(path, args.seconds)
        except Exception as error:  # noqa: BLE001
            print(f"{key}: not read ({error})", flush=True)
            continue
        row: dict[str, Any] = {"duration_s": round(mono.size / RATE_HZ, 3)}
        if vad is not None:
            row["vad"] = {"frame_s": VAD_FRAME / RATE_HZ, "speech": vad(mono)}
        if tagger is not None:
            row["tags"] = tagger(mono)
        if words is not None:
            row["words"] = words(mono)
        done[key] = row
        heard += mono.size / RATE_HZ
        if number % 10 == 0 or number + 1 == len(items):
            print(
                f"{number + 1}/{len(items)} {key}: {heard / 60:.1f} min heard in "
                f"{(time.time() - began) / 60:.1f} min",
                flush=True,
            )
            _write(args.out, done, args)
    _write(args.out, done, args)


def _write(path: Path, clips: dict[str, Any], args: argparse.Namespace) -> None:
    tree = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "detectors": {
            "vad": "silero-vad",
            "tags": TAGGER,
            "words": f"faster-whisper {args.whisper}",
        },
        "seconds": args.seconds,
        "clips": clips,
    }
    scratch = path.with_suffix(".part")
    scratch.write_text(json.dumps(tree, separators=(",", ":")))
    scratch.replace(path)


if __name__ == "__main__":
    main()
