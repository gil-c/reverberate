"""The screen of a noise library: what speech a clip holds, and what becomes of it.

A noise that talks is two sources under one name. The scene's talkers are
its voices, each with its own place and its own clean reference; a voice
inside a noise clip has neither, and a separation model trained on it learns
that some speech is to be removed with the dishwasher. The first scene's
noises held such voices (``docs/open-questions/clip-library-audit.md``).

**This module does not listen.** Three detectors do, in an environment of
their own (``scripts/clip_speech_detect.py``), and write *detections*: for
every clip the probability of speech every 32 ms, the AudioSet scores of
the voice, music and programme classes on windows of 10 s, and the words a
recogniser wrote. From those and fixed :class:`Rules` this module

- marks the **passages** of speech (:func:`passages`),
- **decides** (:func:`decide`): a clip with none is kept as it is; a clip
  that is a programme or a song is not a noise and not deleted either, it is
  **routed** to *media voice*; any other clip **loses** its passages when
  enough of it remains, and is **rejected** when not,
- **removes** the passages (:func:`remove`): the pieces left are joined by
  equal power crossfades, a looping clip is closed again, the level the
  entry states is restored,
- and writes the screened library and its manifest, where every entry says
  what was done to it and from which file (:func:`screen_library`).

The decision is written, not the detectors' outputs: a screened clip is made
again from its source file and its entry alone (:func:`rebuild`), with no
model.

    python -m reverberate.scenes.screen screen --manifest M --detections D \
        --library NAME --out MANIFEST [--root DIR] [--out-root DIR] [--dry-run]
    python -m reverberate.scenes.screen rebuild --manifest SCREENED --source M \
        [--root DIR] [--out-root DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.scenes import clips as clip_library
from reverberate.scenes.clips import FULL_SCALE_SPL_1M_DB, HEADROOM_DBFS, RATE_HZ

__all__ = [
    "DETECTIONS_SCHEMA",
    "MEDIA_VOICE",
    "RULES",
    "Decision",
    "Rules",
    "decide",
    "evidence",
    "format_table",
    "is_media",
    "passages",
    "rebuild",
    "remove",
    "screen_clip",
    "screen_library",
    "sure_words",
]

DETECTIONS_SCHEMA = "reverberate.clip-detections"
#: Where a programme and a song go: a voice that is no one in the room.
MEDIA_VOICE = "media_voice"

#: The AudioSet classes that say someone speaks, as the tagger names them.
VOICE_TAGS = (
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Speech synthesizer",
    "Whispering",
    "Shout",
    "Chatter",
    "Hubbub, speech noise, speech babble",
    "Children playing",
)
#: Those that say someone sings.
SUNG_TAGS = ("Singing", "Choir", "Rapping", "Vocal music", "A capella")
#: Those that say a programme plays.
PROGRAMME_TAGS = ("Television", "Radio")
#: The subtypes of a recipe that are a programme whatever is detected.
PROGRAMME_SUBTYPES = ("television",)


@dataclass(frozen=True)
class Rules:
    """What counts as speech and what a clip may lose. Fixed; written in the manifest."""

    #: The voice detector: speech starts above ``vad_on`` and lasts until
    #: the probability falls under ``vad_off``.
    vad_on: float = 0.5
    vad_off: float = 0.35
    #: A run of the voice detector shorter than this is a click, seconds.
    vad_min_s: float = 0.1
    #: A window of the tagger is speech when one voice class scores this.
    tag_voice: float = 0.3
    #: A recogniser's word counts when it is this probable, in a segment
    #: that is speech (``no_speech`` under its limit), that the recogniser
    #: did not guess (mean log probability) and that does not repeat itself
    #: (compression ratio).
    word_probability: float = 0.5
    word_no_speech: float = 0.6
    word_log_probability: float = -1.0
    word_compression: float = 2.4
    #: The second opinion a run of the voice detector and a word need: the
    #: tagger's score of a voice class in a window over them, or, for a
    #: word, the voice detector's probability about it.
    second_tag: float = 0.05
    second_vad: float = 0.1
    #: Kept on either side of what was detected, seconds.
    pad_s: float = 0.3
    #: Two passages nearer than this are one, seconds.
    merge_s: float = 1.0
    #: A piece of noise shorter than this between two passages goes with
    #: them, seconds.
    min_piece_s: float = 2.0
    #: The pieces are joined over this long, at equal power, seconds.
    crossfade_s: float = 0.25
    #: What must remain of a clip, seconds and share: under either it is rejected.
    min_left_s: float = 5.0
    min_left_share: float = 0.25
    #: A clip is a programme or a song when this share of the tagger's
    #: windows says so: a programme class at ``tag_programme``, or music at
    #: ``tag_music`` with a sung class at ``tag_sung``.
    media_share: float = 0.3
    tag_programme: float = 0.3
    tag_music: float = 0.3
    tag_sung: float = 0.3


#: The rules a library is screened by unless others are handed in.
RULES = Rules()


@dataclass(frozen=True)
class Decision:
    """What becomes of one clip."""

    #: ``"kept"``, ``"cut"``, ``"routed"`` or ``"rejected"``.
    action: str
    #: The speech found, ``(start_s, end_s)`` of the source file.
    passages: tuple[tuple[float, float], ...]
    #: What is kept of the source file, ``(start_s, end_s)``, in order.
    pieces: tuple[tuple[float, float], ...]
    #: Why, in a few words.
    reason: str


# --------------------------------------------------------------------------
# passages: where the detectors heard speech
# --------------------------------------------------------------------------


def _vad_runs(detection: Mapping[str, Any], rules: Rules) -> list[tuple[float, float]]:
    vad = detection.get("vad")
    if not vad:
        return []
    step = float(vad["frame_s"])
    out, start = [], None
    for k, p in enumerate(vad["speech"]):
        if start is None and p >= rules.vad_on:
            start = k
        elif start is not None and p < rules.vad_off:
            out.append((start * step, k * step))
            start = None
    if start is not None:
        out.append((start * step, len(vad["speech"]) * step))
    return [(a, b) for a, b in out if b - a >= rules.vad_min_s]


def sure_words(
    detection: Mapping[str, Any], rules: Rules = RULES
) -> list[tuple[float, float, str]]:
    """The words the recogniser is sure of: ``(start_s, end_s, word)``.

    On a noise a recogniser writes anyway: a syllable a hundred times, or
    the thanks that end a video. Such a segment is told by what the
    recogniser itself reports of it, and none of its words count.
    """
    words = detection.get("words")
    if not words:
        return []
    out = []
    for segment in words["segments"]:
        if (
            float(segment["no_speech"]) >= rules.word_no_speech
            or float(segment["log_probability"]) <= rules.word_log_probability
            or float(segment["compression"]) >= rules.word_compression
        ):
            continue
        for start, end, probability, word in segment["words"]:
            if float(probability) >= rules.word_probability and any(c.isalnum() for c in word):
                out.append((float(start), float(end), str(word)))
    return out


def _score(window: Mapping[str, Any], tags: Sequence[str]) -> float:
    scores = window["scores"]
    return max((float(scores.get(tag, 0.0)) for tag in tags), default=0.0)


def _tag_runs(detection: Mapping[str, Any], rules: Rules) -> list[tuple[float, float]]:
    return [
        (float(w["start_s"]), float(w["end_s"]))
        for w in detection.get("tags") or ()
        if _score(w, VOICE_TAGS) >= rules.tag_voice
    ]


def _union(spans: Sequence[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def passages(detection: Mapping[str, Any], rules: Rules = RULES) -> list[tuple[float, float]]:
    """Where a clip holds speech, ``(start_s, end_s)``, in order and apart.

    What the three detectors found, each held to a second opinion. The
    voice detector fires on a piano's note and the recogniser writes a
    video's credits over a fugue; neither does so where the tagger hears
    any voice at all. So a run of the voice detector and a word count where
    the tagger's window gives a voice class ``second_tag`` at least, a
    score far under the one at which it would call the window speech by
    itself; a word also counts where the voice detector reached
    ``second_vad``. The times are theirs: a voice in the foreground has
    edges. A window the tagger calls speech is taken whole only where the
    other two placed nothing in it: that is a murmur of many voices, which
    has no edges to find.
    """
    duration = float(detection["duration_s"])
    windows = [
        (float(w["start_s"]), float(w["end_s"]), _score(w, VOICE_TAGS))
        for w in detection.get("tags") or ()
    ]

    def heard(a: float, b: float) -> bool:
        """Whether the tagger gives a voice its weak yes somewhere over ``a`` to ``b``."""
        if not windows:
            return True
        return any(c < b and d > a and score >= rules.second_tag for c, d, score in windows)

    vad = detection.get("vad") or {"frame_s": 1.0, "speech": []}
    step, frames = float(vad["frame_s"]), vad["speech"]

    def voiced(a: float, b: float) -> bool:
        first, last = int(max(a - rules.pad_s, 0.0) / step), int((b + rules.pad_s) / step) + 1
        return max(frames[first:last], default=0.0) >= rules.second_vad

    fine = [(a, b) for a, b in _vad_runs(detection, rules) if heard(a, b)]
    fine += [(a, b) for a, b, _ in sure_words(detection, rules) if heard(a, b) or voiced(a, b)]
    spans = list(fine)
    for a, b in _tag_runs(detection, rules):
        if not any(c < b and d > a for c, d in fine):
            spans.append((a, b))
    padded = [(max(a - rules.pad_s, 0.0), min(b + rules.pad_s, duration)) for a, b in spans]
    return [(round(a, 3), round(b, 3)) for a, b in _union(padded, rules.merge_s) if b > a]


def is_media(detection: Mapping[str, Any], rules: Rules = RULES) -> bool:
    """Whether the tagger hears a programme or a song in the clip."""
    windows = detection.get("tags") or ()
    if not windows:
        return False
    media = sum(
        1
        for w in windows
        if _score(w, PROGRAMME_TAGS) >= rules.tag_programme
        or (_score(w, ("Music",)) >= rules.tag_music and _score(w, SUNG_TAGS) >= rules.tag_sung)
    )
    return media / len(windows) >= rules.media_share


def evidence(detection: Mapping[str, Any], rules: Rules = RULES) -> dict[str, float]:
    """What each detector says of a clip, each on its own: shares of its time, and words."""
    duration = max(float(detection["duration_s"]), 1e-9)

    def share(spans: Sequence[tuple[float, float]]) -> float:
        return round(min(sum(b - a for a, b in _union(spans)) / duration, 1.0), 3)

    words = sure_words(detection, rules)
    return {
        "vad_share": share(_vad_runs(detection, rules)),
        "tag_share": share(_tag_runs(detection, rules)),
        "word_share": share([(a - 0.2, b + 0.2) for a, b, _ in words]),
        "words": float(len(words)),
        "speech_share": share(passages(detection, rules)),
    }


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------


def decide(detection: Mapping[str, Any], subtype: str = "", rules: Rules = RULES) -> Decision:
    """What becomes of a clip, from what was detected in it."""
    duration = float(detection["duration_s"])
    found = tuple(passages(detection, rules))
    whole = ((0.0, duration),)
    if subtype in PROGRAMME_SUBTYPES:
        return Decision("routed", found, whole, f"a {subtype} is a programme")
    if is_media(detection, rules):
        return Decision("routed", found, whole, "the tagger hears a programme or a song")
    if not found:
        return Decision("kept", (), whole, "no speech found")
    if subtype == "music":
        # A piece of music is not cut: with a voice in it, it is a song or a
        # presenter's piece, and without its bars it is no piece.
        return Decision("routed", found, whole, "music with a voice")
    pieces, at = [], 0.0
    for a, b in (*found, (duration, duration)):
        if a - at >= rules.min_piece_s:
            pieces.append((round(at, 3), round(a, 3)))
        at = b
    left = sum(b - a for a, b in pieces) - rules.crossfade_s * max(len(pieces) - 1, 0)
    if left < rules.min_left_s or left < rules.min_left_share * duration:
        return Decision("rejected", found, (), f"{max(left, 0.0):.1f} s would remain")
    return Decision("cut", found, tuple(pieces), f"{duration - left:.1f} s removed")


# --------------------------------------------------------------------------
# removal
# --------------------------------------------------------------------------


def remove(
    samples: np.ndarray,
    pieces: Sequence[Sequence[float]],
    *,
    crossfade_s: float,
    loop: bool = False,
    rate: int = RATE_HZ,
) -> np.ndarray:
    """``samples`` less what lies between ``pieces``, joined without a seam.

    Two pieces meet over ``crossfade_s`` at equal power, sine against
    cosine: they are two stretches of one noise, uncorrelated, and the
    level through the join is the level on either side. A clip that loops
    has its end folded onto its start the same way, so that it loops again;
    one that does not is faded at an end that was cut. The result is a
    whole number of milliseconds.
    """
    x = np.asarray(samples, dtype=float)
    fade = int(round(crossfade_s * rate))
    spans = [(int(round(a * rate)), min(int(round(b * rate)), x.size)) for a, b in pieces]
    if not spans or any(b - a < 2 * fade for a, b in spans):
        raise ValueError("a piece is shorter than two crossfades")
    rise = np.sin(0.5 * np.pi * (np.arange(fade) + 0.5) / fade) if fade else np.zeros(0)
    fall = rise[::-1]
    out = x[spans[0][0] : spans[0][1]].copy()
    for a, b in spans[1:]:
        piece = x[a:b]
        if fade:
            out[-fade:] = out[-fade:] * fall + piece[:fade] * rise
        out = np.concatenate([out, piece[fade:]])
    cut_start, cut_end = spans[0][0] > 0, spans[-1][1] < x.size
    if loop and (cut_start or cut_end or len(spans) > 1) and fade:
        out[:fade] = out[:fade] * rise + out[-fade:] * fall
        out = out[:-fade]
    elif not loop:
        edge = min(int(round(0.01 * rate)), out.size // 2)
        ramp = np.sin(0.5 * np.pi * (np.arange(edge) + 0.5) / edge) ** 2 if edge else np.zeros(0)
        if cut_start and edge:
            out[:edge] *= ramp
        if cut_end and edge:
            out[-edge:] *= ramp[::-1]
    return np.asarray(out[: out.size - out.size % (rate // 1000)])


def _level(samples: np.ndarray, how: str) -> float:
    if how == "active":
        return clip_library.active_speech_level(samples)[0]
    if how == "heard":
        return clip_library.heard_level_db(samples)
    return clip_library._db(float(np.sqrt(np.mean(samples * samples))))


def _apply(pcm: np.ndarray, clip: Mapping[str, Any], screen: Mapping[str, Any]) -> np.ndarray:
    """The screened clip's 16 bit samples from its source's and its ``screen`` entry."""
    x = remove(
        np.asarray(pcm).astype(float) / 32768.0,
        screen["pieces"],
        crossfade_s=float(screen["crossfade_s"]),
        loop=bool(clip.get("loop", False)),
    )
    x = x - float(np.mean(x))
    return clip_library._pcm(x * 10.0 ** (float(screen["gain_db"]) / 20.0))


def screen_clip(
    clip: Mapping[str, Any],
    pcm: np.ndarray,
    detection: Mapping[str, Any],
    library: str,
    rules: Rules = RULES,
) -> tuple[dict[str, Any] | None, bytes | None, dict[str, Any]]:
    """One clip of a library through the screen.

    Returns the entry of the screened library and its file, both ``None``
    for a clip that is rejected, and the ``screen`` record either way. A
    clip that is kept or routed keeps its file, byte for byte. One that is
    cut is brought back to the level its entry states, measured the way the
    entry says; where its peaks would then pass the headroom it is stored
    lower by whole decibels, as ``curate`` does, and its entry says so.
    """
    subtype = str(clip.get("subtype", ""))
    verdict = decide(detection, subtype, rules)
    record: dict[str, Any] = {
        "action": verdict.action,
        "reason": verdict.reason,
        "source": {"library": library, "name": clip["name"], "sha256": clip["sha256"]},
        "evidence": evidence(detection, rules),
        "passages": [list(p) for p in verdict.passages],
    }
    if verdict.action == "rejected":
        return None, None, record
    entry = json.loads(json.dumps(clip))
    if verdict.action == "routed":
        record["route"] = MEDIA_VOICE
    if verdict.action != "cut":
        entry["screen"] = record
        return entry, clip_library.wav_bytes(np.asarray(pcm, dtype="<i2")), record
    how = str(clip["level"]["measure"])
    spl = float(clip["level"]["spl_1m_db"])
    record["pieces"] = [list(p) for p in verdict.pieces]
    record["crossfade_s"] = rules.crossfade_s
    record["gain_db"] = 0.0
    raw = _apply(pcm, clip, record).astype(float) / 32768.0
    measured, peak = _level(raw, how), clip_library._db(float(np.max(np.abs(raw))))
    while peak + spl - FULL_SCALE_SPL_1M_DB - measured > HEADROOM_DBFS:
        spl -= 1.0
    record["gain_db"] = round(spl - FULL_SCALE_SPL_1M_DB - measured, 2)
    out = _apply(pcm, clip, record)
    payload = clip_library.wav_bytes(out)
    levels = clip_library._levels(out, "noise", ())
    if how == "heard":
        levels["heard_dbfs"] = round(clip_library.heard_level_db(out.astype(float) / 32768.0), 2)
    entry.update(
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        duration_s=round(out.size / RATE_HZ, 3),
        level={"measure": how, "spl_1m_db": spl, **levels},
        screen=record,
    )
    return entry, payload, record


def _read(path: Path) -> np.ndarray:
    import soundfile

    pcm, rate = soundfile.read(str(path), dtype="int16")
    if rate != RATE_HZ or pcm.ndim != 1:
        raise ValueError(f"{path} is not a clip of a library: mono at {RATE_HZ} Hz")
    return np.asarray(pcm)


def screen_library(
    manifest: Mapping[str, Any],
    detections: Mapping[str, Any],
    root: Path,
    library: str,
    out_root: Path | None = None,
    rules: Rules = RULES,
) -> dict[str, Any]:
    """The noises of a library through the screen: the screened library's manifest.

    The voices are not taken: a screened library is a library of noises,
    read beside the one that holds the voices. The files are written under
    ``out_root`` when one is given. A noise the detections do not hold is
    an error: nothing passes unheard.
    """
    if detections.get("schema") != DETECTIONS_SCHEMA:
        raise ValueError("the detections are not of scripts/clip_speech_detect.py")
    if library == manifest["library"]:
        raise ValueError("a screened library has a name of its own")
    source = str(manifest["library"])
    clips, rejected = [], []
    for clip in manifest["clips"]:
        if clip["kind"] != "noise":
            continue
        name = str(clip["name"])
        if name not in detections["clips"]:
            raise ValueError(f"no detection of {name}")
        detection = detections["clips"][name]
        heard = float(detection["duration_s"])
        if heard + 0.05 < float(clip["duration_s"]):
            raise ValueError(f"{name}: {heard} s were heard of {clip['duration_s']}")
        pcm = _read(clip_library.clip_path(root, source, name))
        entry, payload, record = screen_clip(clip, pcm, detection, source, rules)
        if entry is None or payload is None:
            rejected.append(
                {
                    "name": name,
                    "subtype": str(clip.get("subtype", "")),
                    "duration_s": clip["duration_s"],
                    **record,
                }
            )
            continue
        if out_root is not None:
            clip_library._write(clip_library.clip_path(out_root, library, name), payload)
        clips.append(entry)
    return {
        **{k: v for k, v in manifest.items() if k != "clips"},
        "library": library,
        "screen": {
            "source_library": source,
            "detectors": dict(detections.get("detectors", {})),
            "rules": asdict(rules),
            "rejected": rejected,
        },
        "clips": clips,
    }


def rebuild(screened: Mapping[str, Any], root: Path, out_root: Path) -> dict[str, str]:
    """A screened library's files from its source's files and its manifest: no detector.

    What was done to each clip, by name: ``"kept"`` (already there with the
    manifest's digest) or ``"built"``. A file whose digest would not be the
    manifest's is not written, and the names of such files are an error.
    """
    library = str(screened["library"])
    done, wrong = {}, []
    for clip in screened["clips"]:
        name, record = str(clip["name"]), clip["screen"]
        path = clip_library.clip_path(out_root, library, name)
        if path.is_file() and clip_library._sha256(path) == clip["sha256"]:
            done[name] = "kept"
            continue
        source = record["source"]
        origin = clip_library.clip_path(root, str(source["library"]), str(source["name"]))
        if clip_library._sha256(origin) != source["sha256"]:
            raise ValueError(f"{origin} is not the file {name} was screened from")
        pcm = _read(origin)
        out = _apply(pcm, clip, record) if record["action"] == "cut" else pcm
        payload = clip_library.wav_bytes(np.asarray(out, dtype="<i2"))
        if hashlib.sha256(payload).hexdigest() != clip["sha256"]:
            wrong.append(name)
            continue
        clip_library._write(path, payload)
        done[name] = "built"
    if wrong:
        raise ValueError("not the manifest's digest: " + ", ".join(wrong))
    return done


def format_table(screened: Mapping[str, Any]) -> str:
    """One row a clip of the source library: what was found and what was done."""
    rows = [
        (c["name"], c.get("subtype", ""), c["duration_s"], c["screen"]) for c in screened["clips"]
    ]
    rows += [(r["name"], r["subtype"], r["duration_s"], r) for r in screened["screen"]["rejected"]]
    width = max((len(r[0]) for r in rows), default=4)
    heads = ("s", "vad", "tags", "words", "speech")
    lines = [f"{'clip':<{width}}  {'subtype':<11}" + "".join(f"{h:>8}" for h in heads) + "  action"]
    for name, subtype, seconds, record in sorted(rows):
        e = record["evidence"]
        cells = f"{seconds:>8}{e['vad_share']:>8.2f}{e['tag_share']:>8.2f}{int(e['words']):>8}"
        lines.append(
            f"{name:<{width}}  {subtype:<11}{cells}{e['speech_share']:>8.2f}"
            f"  {record['action']}: {record['reason']}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reverberate.scenes.screen",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("screen", help="a library and its detections to a screened library")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--library", required=True, help="the screened library's name")
    p.add_argument("--out", type=Path, required=True, help="the screened manifest")
    p.add_argument("--root", type=Path, help="where the libraries are, <data root>/clips")
    p.add_argument("--out-root", type=Path, help="where the screened one goes, --root unless said")
    p.add_argument("--dry-run", action="store_true", help="decide and print; write no file")
    p = sub.add_parser("rebuild", help="a screened library's files from its source's")
    p.add_argument("--manifest", type=Path, required=True, help="the screened manifest")
    p.add_argument("--root", type=Path)
    p.add_argument("--out-root", type=Path)
    args = parser.parse_args(argv)

    if args.root is None:
        from reverberate.settings import data_root

        args.root = data_root() / "clips"
    out_root = args.out_root or args.root
    manifest = json.loads(args.manifest.read_text())
    if args.command == "rebuild":
        done = rebuild(manifest, args.root, out_root)
        print(f"{len(done)} clips: {sum(v == 'built' for v in done.values())} built")
        return 0
    detections = json.loads(args.detections.read_text())
    screened = screen_library(
        manifest, detections, args.root, args.library, None if args.dry_run else out_root
    )
    print(format_table(screened))
    before = sum(c["duration_s"] for c in manifest["clips"] if c["kind"] == "noise")
    after = sum(c["duration_s"] for c in screened["clips"])
    print(f"{before / 60:.1f} min of noise before, {after / 60:.1f} min after")
    if not args.dry_run:
        args.out.write_text(clip_library.dumps(screened))
    return 0


if __name__ == "__main__":
    sys.exit(main())
