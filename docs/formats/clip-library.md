# The clip library

Status: written with the first scene (lot L9). Made, fetched and checked by
`reverberate.scenes.clips`; read by the generator
(`reverberate.scenes.generate.load_clip_library`), by the audit
(`reverberate.viz.audit_dry`) and, through it, by the signal engine.

A recipe ([`scene-recipe.md`](scene-recipe.md)) holds no audio: an activity
interval names a clip by library, name and the SHA-256 of its file. A clip
library is those files and the manifest that describes them. The files are
derived and are stored nowhere: the manifest says what each is made of, and
`clips fetch` makes it again and checks the digest.

## The files

`<data root>/clips/<library>/<name>.wav`, the name holding one `/`:
`voice/s04_p260_03`, `noise/water_tap`.

| property | value |
| --- | --- |
| container | RIFF/WAVE, a 44 byte header and the samples |
| samples | 16 bit PCM, mono, 48 kHz |
| length | a whole number of milliseconds |
| identity | the SHA-256 of the file's bytes, the `sha256` of the manifest and of the recipe |

A clip is **dry**: what the source emits, not what a room makes of it. The
engine puts the room on.

## Levels

Stated once, here.

| quantity | value |
| --- | --- |
| the scale | a level of 0 dB re full scale stands for **86 dB SPL at 1 m** from the source, in free field (`FULL_SCALE_SPL_1M_DB`) |
| a voice | stored at an **active speech level of -26 dB re full scale** (`VOICE_ACTIVE_DBFS`), so 60 dB SPL at 1 m: a normal vocal effort (ISO 9921) |
| a noise | stored at the long term level its source has at 1 m on that scale, `level.spl_1m_db - 86` dB re full scale, **read on what a render holds of the clip** (`measure: "heard"`, below) |
| the engine | applies no normalisation: the source's `gain_db` and the interval's multiply the samples, and the free field gain is `1 / d`. A voice at `gain_db = 0` is therefore heard at 1 m at 60 dB SPL, if full scale of the output is read as 86 dB SPL |
| `gain_db` of a recipe | a departure from the stored level: `-6` is a quiet voice (54 dB at 1 m), and a noise at `-12` is 12 dB under what its entry says. The generator draws a voice's from -6 to 0 and a noise's from -6 to +6, about its stored level |

The **active speech level** is measured after ITU-T P.56, method B
(`active_speech_level`): the envelope is the rectified signal through two
low passes of 30 ms; the signal is active while the envelope is above a
threshold and for 200 ms after; the active level is the power over the
active time, at the threshold that lies 15.9 dB under it. The thresholds are
3 dB apart here, and interpolated. A programme's talk (the `television`
clips) is levelled the same way; every other noise on its long term level.

**A noise's long term level is read above the floor of the chain**
(`measure: "heard"`, `clips.heard_level_db`): the RMS of the clip through
the low cut every render applies, a Butterworth high pass of order 8 at
40 Hz (`HEARD_FROM_HZ`, `HEARD_ORDER`; a pack is valid from 45 Hz,
[`scene-pack.md`](scene-pack.md)). A recording made in a room can hold
most of its energy where no one hears and nothing is rendered, and a
level read on the whole band is then a level of that. The first library's
`noise/appliance_washing_machine` (DEMAND, a washing machine in its room)
holds 97.7 per cent of its energy between 10 and 20 Hz: -17.5 dB of its
level lies over 20 Hz and -24.7 dB over 40 Hz, so that, stored at 58 dB
SPL on its whole band, it was rendered 25 dB under that
(`docs/open-questions/chain-audit.md`, D12). A-weighting would answer the
same question and another one, how loud the clip is judged; the floor
alone is what the engine does to it, and leaves the entry's level a
physical one.

`measure: "rms"` is the long term level of the whole band, as every noise
of `clarify_v1` was levelled: its selection says so on each of them, so
that curating it again gives the files it gave, digest for digest, and
its entries keep `"measure": "rms"`. Of its 22 such noises the washing
machine is the one whose level is not what is heard, and no clip is
fetched or stored again for it: the scene that plays it sets its gain in
the pack (`python -m reverberate.render gains`, [`scene-pack.md`](scene-pack.md)).
A selection written from now on leaves `measure` out of a noise, which is
`"heard"`, and the entry then also states `level.heard_dbfs`.

A clip whose peaks would pass -1 dB re full scale at the level asked for is
stored lower, by whole decibels, and its entry states the level it has: a
kitchen's clatter has peaks 38 dB over its mean. `level.spl_1m_db` is always
what the file holds.

The levels at 1 m of the noises are typical values chosen for the first
scene, not measurements of the recorded machines; they are the selection's,
and are changed there.

## The manifest

One JSON document, kept in the package for the library the first scene
plays (`src/reverberate/scenes/library/clarify_v1.json`), written one clip a
line.

| key | type | meaning |
| --- | --- | --- |
| `schema` | string | `"reverberate.clip-library"` |
| `schema_version` | int | `1` |
| `library` | string | the name a recipe quotes, and the folder under `clips/` |
| `sample_rate_hz` | int | `48000` |
| `level` | object | `{"voice_active_dbfs": -26.0, "full_scale_spl_1m_db": 86.0}` |
| `datasets` | object | per source dataset: `licence`, `attribution` |
| `clips` | array | the clips, in the order of their names |

Each clip:

| key | type | meaning |
| --- | --- | --- |
| `name` | string | unique in the library |
| `kind` | string | `"voice"` or `"noise"` |
| `subtype` | string | for a noise: one of the recipe's subtypes |
| `speaker` | string | for a voice: who speaks. A voice of a scene keeps one speaker; speakers are taken in the order of these names |
| `sha256`, `bytes` | string, int | of the file |
| `duration_s` | float | of the file |
| `loop` | bool | whether the end runs into the start without a seam |
| `utterances` | array of `[start_s, end_s]` | for a voice: where the clip may be cut. In order, none overlapping, each end in a silence |
| `level` | object | `measure` (`"active"`, `"heard"` or `"rms"`), `spl_1m_db`, and what the file measures: `rms_dbfs`, `peak_dbfs`, for a noise levelled on what is heard `heard_dbfs`, for a voice `active_dbfs` and `activity`, and for a clip of sentences `source_floor_dbfs` and `source_decay_db` (below) |
| `licence`, `credit` | string | of this clip: the dataset's, or the clip's own where the dataset gives one a clip |
| `what` | string | what is heard, and where it comes from |
| `origin` | object | what the file is made of |

`origin.parts` are the members of the sibling project's library the clip is
cut out of, each `{dataset, clip_id, member_name, shard_path, payload_offset,
payload_length, crc32, extension, channel, start_s, end_s}`: one ranged read
of `payload_length` bytes at `payload_offset` of the object `shard_path`,
whose CRC-32 is checked (`reverberate.clarify_library.fetch_clip`).
`channel` is the channel taken, `null` for the mean of all; `start_s` and
`end_s` cut the member once it is at 48 kHz, `null` for its ends.

`origin.process` is what is done to them, in this order
(`reverberate.scenes.clips.build_clip`):

| key | meaning |
| --- | --- |
| | each part decoded, its channel taken, resampled to 48 kHz (`audio.resample_to`) when at another rate, cut |
| `part_fade_s` | each part faded in and out over this long, a raised cosine |
| `gap_s` | the parts joined with this much silence between two |
| `remove_dc` | the mean removed |
| `loop_crossfade_s` | the last stretch of this length folded onto the first, at equal power, so the clip loops; the clip is shorter by as much |
| `fade_s` | the whole faded in and out |
| | cut to a whole number of milliseconds |
| `gain_db` | the gain that brings it to its level, to 0.01 dB |

then rounded to 16 bits. Nothing is clipped: a clip that would be is refused.

### Voices

A voice clip is either **sentences** or a **passage**.

- *Sentences* (several parts): each member is one sentence. It is cut to
  its speech, what lies more than 30 dB under the talker's active level at
  either end going, less 50 ms before and 80 ms after; the sentences are
  joined by `gap_s` of silence, and an utterance is a sentence with half of
  the silence on either side. So utterances touch, and every cut falls in
  digital silence.
- *A passage* (one part): the utterances are found
  (`find_utterances`): speech between two pauses of 0.18 s or more, a pause
  being 18 dB under the active level; an utterance longer than 12 s is cut at
  its longest pause, again and again; each end is put on the quietest 10 ms
  of the pause.

### Loops and playlists

A clip that loops (`loop: true`) is a steady source: a machine, running
water, a street. One that does not is a piece with a beginning and an end,
faded: a programme's segment, a piece of music.

## What the generator does with it

`python -m reverberate.scenes generate ... --clips <manifest>`
(`reverberate.scenes.generate`):

- **A talk spurt is whole utterances.** A length is drawn from
  `sources.speech_s`; the spurt is the speaker's next utterances, as many in
  a row as come nearest that length, and it lasts what they last. It never
  crosses from one clip to the next. The speaker's clips are read on in the
  order of their names, and begin again when all were heard. A spurt the
  scene would end before is not started.
- **A noise** reads on through its clip from one time it is on to the next.
  Where its clip ends, a looping clip begins again, with no gap, and the
  source keeps that one clip for the whole scene; a clip that does not loop
  is followed by the next of its subtype. Which clip a subtype starts on is
  drawn, and two noises of one subtype start on two clips.
- **The subtypes of the noises are dealt** from those the library holds,
  none coming twice before all have come once.

A recipe's interval is therefore never longer than its clip, and a noise
that runs twenty minutes is as many intervals as it takes clips.

The engine fades an interval in and out over 5 ms, a raised cosine inside
the interval's own length (`reverberate.render.dry.EDGE_FADE_S`). For a
voice the cut is in silence and the fade changes nothing. A noise that comes
on or goes off was cut where it stood, at its own level on the first sample:
a step on 64 channels, heard as a click; the fade is the switch. Two
intervals of one source that meet on a sample are one interval in pieces (the
audit cuts a long one every 20 s) and are not faded where they meet.

## Commands

```
python -m reverberate.scenes clips fetch [--manifest M] [--root DIR] [--jobs N]
python -m reverberate.scenes clips check [--manifest M] [--root DIR]
python -m reverberate.scenes clips curate --selection S --out M [--root DIR] [--jobs N]
```

`--root` is `<data root>/clips` unless said, and the manifest the package's
`clarify_v1`. `fetch` and `curate` read the bucket (credentials as for every
other command, from the vault); `check` reads only the disk.

**`fetch`** keeps a file whose digest is the manifest's, builds every other
from its parts, and writes none whose digest is then another: it fails, and
names them. A member at 48 kHz is read, cut, scaled and rounded, which gives
the same bytes on any machine. A member at another rate is resampled, and an
MP3 is decoded, by libraries whose versions may differ by a least bit: such
a clip may be refused on another machine. The remedy is then to publish the
files themselves to the store; it was not needed on the machine the library
was made on.

**`check`** measures every file and holds it to fixed limits
(`reverberate.scenes.clips.LIMITS`); it prints one row a clip and fails when
one fails.

| column | what | limit |
| --- | --- | --- |
| `s` | duration | the manifest's |
| `peak` | largest sample, dB re full scale | at most -0.5 |
| `clip` | samples at either end of the scale | 0 |
| `rms` | long term level | for a noise, within 0.1 dB of `spl_1m_db - 86` |
| `active`, `act` | active speech level and activity factor | for a voice, within 0.1 dB of -26 |
| `dc` | the mean, dB re the clip's level | at most -40 |
| `floor` | the quietest 50 ms, dB re full scale | for a voice, at most -66: 40 dB under the speech |
| `decay` | the level 100 to 200 ms after speech stops, dB re the active level: the lower quartile of the pauses | for a voice, at most -28 |
| `edge` | the largest sample within 1 ms of where an utterance is cut | for a voice, at most -50 |
| `>8k` | the share of the power above 8 kHz, dB | none: it shows a source that was band limited |

`decay` is the estimate of how dry a voice is. Speech stops when its level
falls 18 dB under the active level for 0.2 s or more; in a room the level
100 to 200 ms later is what the room still rings with, in a chamber it is
the talker's breath and the recording's floor. The lower quartile is read,
because a room rings in every pause and a talker breathes in some. A pause
is 18 dB down by its definition; the limit is 10 dB further. Were the whole
of it reverberation decaying from the speech's own level, -28 dB at 150 ms
would be a reverberation time of 0.32 s: an upper bound, not a measurement.

A clip of sentences is silent between them by construction, so its `floor`
and `decay` as a file say nothing. They are measured when the clip is made,
on the members end to end before they are cut, written as
`level.source_floor_dbfs` and `level.source_decay_db`, and those are what
`check` prints and holds to the limits.

**`curate`** makes a manifest, and the files, from a selection: the same
document with, for each clip, `name`, `kind` (`"voice"` or a subtype),
`speaker`, `what`, `spl_1m_db`, `measure`, `process` without its gain, and
`parts` as `{dataset, shard, member, channel, start_s, end_s}`, the shard
being the catalogue's name. It finds the catalogue's rows, measures, sets
the gains and the cuts, and writes the utterances. The selection of
`clarify_v1` is kept beside its manifest.

## `clarify_v1`

Cut out of the audio library the sibling project keeps on the same bucket,
`library/shards/<dataset>/` with its catalogues under
`library/catalog/shards/<dataset>/`.

| what | from | licence |
| --- | --- | --- |
| nine voices, five women and four men, 5 to 10 minutes each: newspaper sentences, read | VCTK 0.92, the wide band microphone (`mic2`), sentences 25 and up, which differ from one speaker to the next; recorded in a hemi-anechoic chamber | CC BY 4.0 |
| `television`: five segments of two minutes, one talker each | EARS, free monologues, anechoic; the dataset removed their silences, which makes them a programme's talk and not a conversation's | CC BY-NC 4.0 |
| `music`: twelve excerpts of 30 s | FMA: Kimiko Ishizaka, the Open Goldberg Variations; Breuss Arrizabalaga Quintet | CC0 1.0; public domain |
| `appliance`: a washing machine; a range hood, an extractor fan | DEMAND `DWASHING`; Freesound, by FSD50K | CC BY-SA; CC0 1.0 |
| `water`: a tap, a shower, a bath filling | Freesound, by FSD50K | CC0 1.0 |
| `street`: a busy crossing | DEMAND `STRAFFIC` | CC BY-SA |
| `other`: food being prepared; frying, boiling | DEMAND `DKITCHEN`; Freesound, by FSD50K | CC BY-SA; CC0 1.0 |

The first three speakers in the order of their names hold five minutes and
are the near voices' of the first scene; the other six hold nine.

Not everything is as dry as a voice. The three DEMAND recordings were made
in the rooms themselves, a metre or more from the source, and carry those
rooms; the Freesound recordings are what their authors made. A noise that
is already in a room is then put in another by the engine. No dry recording
of a domestic machine is in the library these were taken from.
