# The dry audio: what the library holds, what is wrong with it, what to add

Status: an audit, lot L31, 2026-10-07. Nothing here changes what a scene
plays: `clarify_v1` is as it was. What is new is the screen
(`reverberate.scenes.screen`, [`clip-library.md`](../formats/clip-library.md),
"Screening"), the screened noise set `clarify_v1_screened`, and a plan the
owner decides on. Nothing was written to the bucket.

The owner listened to the first scene and found five things. In short:

| found | answer | where |
| --- | --- | --- |
| the noise tracks hold voices, which dominate them | SECTION2SUMMARY | 2 |
| is the speech spontaneous or read? | **read**: all nine voices are VCTK, newspaper sentences read in a hemi-anechoic chamber. The only free speech of the scene is its television, five EARS monologues | 1.3 |
| vocal effort: louder in noise, quiet and whispered in quiet | the bucket holds loud and whispered speech of 107 talkers (EARS, 11.5 h, anechoic, non commercial); it holds no Lombard speech and no soft speech. AVID and Lombard GRID fill both under CC BY 4.0 | 1.3, 4 |
| breathing and body noises are missing | the bucket holds them and the library took none: EARS non-verbal and vegetative sounds (3.2 h, anechoic), FSD50K breathing, coughs, footsteps, zips, doors (hours, per-clip licences). No class of chair or clothing sounds exists there; FoleySet is the candidate | 1.4, 4 |
| television and radio as a class apart | the screen routes a programme and a song to `media_voice`; real broadcast audio under a licence that allows a product does not exist in the open, and the class has to be made (4.5) | 2, 4 |

How to read the marks of section 4: **R** read on the dataset's own page
on 2026-10-07, **S** second hand (a search result or a third party's page),
**N** not verified.

## 1. What the database holds

The Clarify project's library is on the bucket this project uses, under
`library/`: 2 355 zip shards (`library/shards/<dataset>/`), one catalogue a
shard (`library/catalog/shards/`), one catalogue of all of them
(`library/catalog/global/clips-<build>.parquet`, 4 339 132 rows), sidecar
archives of the datasets' own metadata (`library/sidecars/`), and a
provenance record (`provenance/raw-sources/v2/`, 413 raw archives, each
with its source address and, for most, a licence checked on a page that is
named). It is **2.37 TB of audio, 4.34 million files, about 14 150 hours of
files** in 36 datasets.

How this was counted. The catalogue gives a file's name and bytes, not its
length or rate. The format of 16 files a group was read from their first
64 kB by ranged reads, and the hours are the bytes at that format (exact
for WAV, within a few per cent for FLAC and MP3, which are scaled from the
16 files). An array recording counts once a file, and a talker with two
microphones twice: the hours are hours of files. In all 0.33 GB of metadata
and 1.4 GB of audio samples were read; nothing was written.

### 1.1 Speech

| dataset | files | GB | hours of files | format | talkers | language | what it is | licence, as the bucket records it |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `vctk` | 88 328 | 11.7 | 82 (41 a microphone) | 48 kHz, 16 bit | 110 | English, many accents | **read** newspaper sentences, hemi-anechoic, two microphones | CC BY 4.0 |
| `ears` | 17 227 | 69.2 | 100 | 48 kHz, 32 bit | 107 | English | anechoic. **Free monologue** 32.5 h; emotional free speech 10.1 h; emotional sentences 7.8 h; read passages and sentences in seven styles (below); interjections 2.4 h, non-verbal 1.8 h, vegetative 1.4 h | CC BY-NC 4.0 |
| `librispeech` | 292 367 | 63.7 | 991 | 16 kHz, 16 bit | 2 484 | English | **read** audiobooks, the readers' own rooms | CC BY 4.0 |
| `libritts` | 170 042 | 47.7 | 276 | 24 kHz, 16 bit | 1 296 | English | **read** audiobooks (the clean subsets and the dev and test sets) | CC BY 4.0 |
| `mls_*` (7) | 1 473 249 | 376.7 | 6 131 | 16 kHz, 16 bit | 758 | Dutch 1 576 h, French 1 080, German 2 041, Italian 254, Polish 111, Portuguese 163, Spanish 906 | **read** audiobooks | not recorded (the dataset's page says CC BY 4.0: S) |
| `dns5-clean` | 1 122 847 | 884.7 | 2 560 | 48 kHz, 16 bit (much of it upsampled from 16 kHz by its makers) | not countable from the names | English 924 h (1 948 readers), German 987, Spanish 201, French 190, Italian 128, Russian 36 | **read** audiobooks; a second copy of VCTK (83 h); acted emotions 7.3 h; singing 2.9 h | not recorded: each part keeps its own |
| `aishell1`, `aishell3` | 229 960 | 47.9 | 265 | 16 and 44.1 kHz | 400, 218 | Mandarin | **read** | Apache 2.0 |
| `speech_commands` | 105 835 | 3.3 | 29 | 16 kHz | 2 618 | English | single words | CC BY 4.0 |
| `musan` speech | 426 | 7.0 | 60 | 16 kHz | not recorded | English mostly | read audiobooks 20 h; hearings of the United States government 40 h (speech in rooms, several talkers) | CC BY 4.0 |
| `ms_snsd` clean | 24 175 | 2.6 | 23 | 16 kHz | not recorded | English | read | MIT for the collection, parts keep theirs |
| `ami` | 2 039 | 138.0 | 1 198 | 16 kHz, 16 bit | not in the names | English, many non-native | **conversation**: 171 meetings of about 100 h; a headset a talker (687 files, 404 h) and two arrays (1 352 files) | CC BY 4.0 |
| `alimeeting` | 1 005 | 163.4 | 534 | 16 kHz | 537 by name | Mandarin | **conversation**: meetings; a headset a talker (768 files, 408 h), an 8 channel array (237 files, 126 h) | CC BY-SA 4.0 |
| `aishell4` | 211 | 51.9 | 120 | 16 kHz, 8 ch | not in the names | Mandarin | **conversation**: meetings, array only | CC BY-SA 4.0 |
| `easycom` | 1 336 | 37.9 | 22 | 48 kHz, 32 bit | not in the names | English | **conversation** at a table in restaurant noise played at about 71 dB SPL: a close microphone a talker (936 files, 15.4 h) and the glasses' 6 channels (6.6 h) | CC BY-NC 4.0 |
| `voices` | 20 248 | 37.1 | 322 | 16 kHz | 300 | English | LibriSpeech played by a loudspeaker in rooms: not dry | CC BY 4.0 |

Distinct talkers, where the names say: 110 + 107 + 2 484 + 758 + 618 +
2 618 + 537 = **about 7 200**, of whom **217 are dry at 48 kHz** (VCTK and
EARS). LibriTTS's talkers are LibriSpeech's, and the readers of
`dns5-clean` overlap them to an extent the names do not tell.

### 1.2 Noise, music, responses

| dataset | files | GB | hours | format | what it is | licence, as the bucket records it |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `demand` | 272 | 7.8 | 22.7 (1.4 a channel) | 48 kHz, 16 ch as files | 17 places, 5 min each: kitchen, living room, washing room, field, park, river, hallway, meeting, office, cafeteria, restaurant, station, square, traffic, bus, car, metro. Recorded in the place: not dry | CC BY-SA 3.0 |
| `dns5-noise` | 63 810 | 61.2 | 177 | 48 kHz | 10 s clips of AudioSet (by YouTube identifier) and Freesound, no label in the name | not recorded: AudioSet's clips are their uploaders' |
| `fsd50k` | 51 197 | 34.5 | 108.5 | 44.1 kHz, 16 bit | Freesound clips with 200 labels checked by people | per clip: CC0 19 873, CC BY 23 506, CC BY-NC 6 041, Sampling+ 1 777 |
| `esc50` | 2 000 | 0.9 | 2.8 | 44.1 kHz | 50 classes of 40 clips of 5 s | CC BY-NC 3.0 as a collection |
| `urbansound8k` | 8 732 | 7.1 | 9.2 | mixed | 10 street classes, 4 s | CC BY-NC 4.0 |
| `tau_urban_2019/2020/2022` | 268 305 | 104 | 169 | 48 kHz stereo; 44.1 kHz | 10 public scenes (airport, bus, metro, park, square, mall, streets, tram) in 10 s or 1 s pieces | non commercial |
| `wham_noise` | 1 578 | 81.4 | 78.5 | 48 kHz stereo, 24 bit | cafés, restaurants, bars, parks | not recorded (its page: CC BY-NC 4.0, S) |
| `musan` noise | 930 | 0.7 | 6.2 | 16 kHz | Freesound and Sound Bible clips | CC BY 4.0 |
| `ms_snsd` noise | 179 | 0.5 | 4.0 | 16 kHz | 14 named kinds: air conditioner, babble, neighbour, typing, vacuum, washer and others | parts keep theirs |
| `mimii_2020` | 30 987 | 10.1 | 88 | 16 kHz | six machines of a factory, 10 s | CC BY-NC-SA 4.0 |
| `noisex92` | 115 | 0.1 | 1.1 | 20 kHz | 15 military and factory noises of 4 min, 100 short sounds | not recorded |
| `rir_openslr28` | 61 260 | 3.8 | 26 | 16 kHz | simulated and real room responses, point source noises (MUSAN's) | Apache 2.0 |
| `fma` | 106 574 | 105.2 | 818 | MP3, 44.1 kHz stereo | 30 s of each of 106 574 tracks, with genre in the sidecar | per track |
| `musan` music | 660 | 4.9 | 42.7 | 16 kHz | whole pieces | CC BY 4.0 |
| `sadie-ii` | 193 218 | 0.7 | | | head responses, not audio to play | Apache 2.0 |

### 1.3 Spontaneous or read, and at what effort

**What a scene plays today is read.** The nine voices of `clarify_v1` are
VCTK: each talker reads newspaper sentences, one sentence a file, and the
library joins them with 0.45 s of silence. There is no turn taking, no
hesitation, no laughter, no breath between sentences (the library cuts each
sentence to its speech), and every voice is at one effort. Its
`television` is the one free speech it holds: EARS monologues, from which
the dataset removed the silences.

**Conversation exists in the bucket and none of it is dry.**

| source | hours | a channel a talker | why it does not serve as it is |
| --- | ---: | --- | --- |
| AMI headsets | about 100 h of meetings | yes | 16 kHz; a headset in a meeting room hears the others (crosstalk) and the room |
| AliMeeting headsets | about 100 h of meetings | yes | 16 kHz; Mandarin; the same crosstalk |
| EasyCom close microphones | 5.3 h of sessions | yes | 48 kHz, but restaurant noise plays in the room at 71 dB and leaks into every microphone |
| EARS free monologues | 32.5 h + 10.1 h | one talker | 48 kHz and anechoic: **spontaneous speech, and the only dry one**, but one person talking to no one, silences removed |

So a conversation between sources can be had today in one way only:
**turns built from EARS free speech**, each talker a source, the generator
placing the turns. What is lost is that the talkers do not answer one
another. That is a recipe matter (lot L30); the material is there: 107
talkers, about 18 minutes of free speech each.

**Vocal effort.** EARS reads one passage and a set of sentences in seven
styles, all 107 talkers, anechoic, 48 kHz:

| style | hours | what it serves |
| --- | ---: | --- |
| regular | 9.0 | the reference |
| **loud** | 5.5 | a raised voice. It is not Lombard speech: the talker was asked to speak loud, in silence |
| **whisper** | 6.0 | whisper in a quiet scene |
| fast, slow | 4.1, 8.2 | rate |
| high pitch, low pitch | 5.1, 5.6 | not an effort |

Nothing in the bucket is **Lombard speech** (spoken in noise, heard dry),
nothing is **soft speech** (voiced, quiet), and the levels of EARS's styles
are not calibrated against one another: how loud "loud" was is not written
(section 3 says what level to give each effort). FSD50K holds `Whispering`
(0.45 h), `Shout` (0.88 h), `Yell` (0.38 h): Freesound uploads, in rooms.
EasyCom's talkers do speak in 71 dB of noise, which is Lombard speech in
fact, with the noise in it.

### 1.4 Breaths, bodies, objects

In the bucket today, by FSD50K's own labels (a clip may carry several;
hours of clips), and beside them EARS and ESC-50:

| what | where | amount | dry? |
| --- | --- | --- | --- |
| the talkers' own sounds: coughing, throat clearing, sneezing, yawning, eating (`vegetative_*`); laughter, crying, cheering, yelling, screaming (`nonverbal_*`); fillers, greetings, agreement (`interjection_*`). **No class of breathing** | EARS, five or six files a talker | 1.4 h, 1.8 h, 2.4 h, 107 talkers | **anechoic** |
| breathing | FSD50K `Breathing` 658 clips, 1.23 h; ESC-50 40 clips | | as uploaded |
| cough, sneeze, sigh, gasp | FSD50K 385, 125, 136, 110 clips: 0.73, 0.17, 0.20, 0.17 h | | as uploaded |
| laughter | FSD50K 1 186 clips, 1.91 h | | as uploaded |
| chewing | FSD50K 256 clips, 0.80 h | | as uploaded |
| **footsteps** | FSD50K `Walk_and_footsteps` 580 clips, 1.47 h; `Run` 1.18 h; ESC-50 40 clips | | as uploaded, many outdoors |
| **clothing** | FSD50K `Zipper_(clothing)` 395 clips, 0.87 h. No class of cloth rustle | | as uploaded |
| **chair** | no class. `Squeak` 606 clips, 1.10 h, of which some are chairs | | |
| doors, drawers, cupboards | FSD50K `Door` 2.66 h, `Drawer_open_or_close` 0.47 h, `Cupboard_open_or_close` 0.19 h, `Knock` 0.54 h | | as uploaded |
| dishes, cutlery, glass | FSD50K 0.65 h, 0.45 h, 1.82 h | | as uploaded |
| keys, coins, paper, writing, typing | FSD50K 0.56, 0.75, 0.76 (crumpling), 1.02, 1.31 h | | as uploaded |

About 78 per cent of FSD50K is CC0 or CC BY (43 379 of 51 197 clips); the
rest is non commercial or Sampling+, and each clip's licence is in the
sidecar.

SECTION2

## 3. Levels

The owner found the noises too loud, after gains set by hand. A clip's
level has two halves: what the file holds, and what level at 1 m the
library says that stands for. The first is measured here; for the second
nothing in the database helps, and a table is proposed.

### 3.1 What the recordings hold

The sample of section 2.2, the first 180 s of each file, dB re full scale:
the long term level of the whole band, the level above 40 Hz (`heard`, what
a render keeps), the peak; median and range over the clips of a dataset.

| dataset | clips | whole band | above 40 Hz | peak | largest loss under 40 Hz |
| --- | ---: | --- | --- | --- | ---: |
| `demand` | 17 | -41.7 (-56.0 to -22.4) | -46.3 (-62.9 to -31.4) | -16.7 (-31.2 to -2.9) | 25.6 dB |
| `wham_noise` | 12 | -37.9 (-46.9 to -36.2) | -39.0 (-47.6 to -36.4) | -16.9 (-33.0 to -11.2) | 1.8 |
| `tau_urban_2019` | 40 | -47.5 (-59.1 to -28.0) | -49.1 (-60.6 to -31.5) | -33.5 (-46.6 to -15.5) | 9.3 |
| `mimii_2020` | 10 | -39.2 (-39.7 to -38.7) | -39.4 (-40.0 to -38.7) | -26.0 (-26.9 to -23.3) | 0.4 |
| `fsd50k` | 178 | -29.4 (-68.4 to -10.9) | -30.4 (-71.8 to -10.9) | -5.4 (-52.7 to 0.0) | 14.2 |
| `esc50` | 42 | -24.4 (-45.3 to -5.3) | -24.7 (-45.3 to -5.3) | -3.7 (-20.2 to 0.0) | 6.8 |
| `urbansound8k` | 38 | -22.6 (-49.9 to -12.8) | -23.0 (-51.0 to -12.8) | -7.5 (-36.9 to 0.0) | 5.4 |
| `dns5-noise` | 80 | -20.8 (-64.7 to -8.2) | -20.8 (-64.7 to -8.9) | -4.0 (-45.2 to 0.0) | 1.4 |
| `ms_snsd` noise | 40 | -21.2 (-39.9 to -12.3) | -21.8 (-39.9 to -13.5) | 0.0 (-8.5 to 0.0) | 9.0 |
| `musan` noise and music | 51 | -16.7 (-40.5 to -3.0) | -17.1 (-40.5 to -3.0) | 0.0 (-1.4 to 0.0) | 14.3 |
| `noisex92` | 15 | -22.6 (-32.8 to -19.6) | -24.0 (-32.8 to -19.6) | -8.0 (-18.3 to -6.0) | 7.7 |
| `fma` | 24 | -14.9 (-31.6 to -6.9) | -14.9 (-31.6 to -6.9) | -1.9 (-9.1 to +1.8) | 0.8 |

Three kinds of file:

- **Uploads and collections brought to full scale** (FSD50K, ESC-50,
  UrbanSound8K, MUSAN, MS-SNSD, DNS, FMA): the peak is at or near 0 dB. The
  level says how the uploader normalised, nothing of the source. A range of
  60 dB inside one dataset is that.
- **Field recordings at a fixed gain** (DEMAND, WHAM, TAU, MIMII): peaks
  well under full scale, and levels that differ from place to place.
  DEMAND's living room is 27 dB under its car and 22 dB under its traffic
  above 40 Hz, as one expects.
- **DEMAND under 40 Hz**: the washing room loses 25.6 dB, the office 15.5,
  the bus 13.8, the hallway 10.3. A level read on the whole band is not the
  level of what is heard (`clip-library.md`, "Levels"); every DEMAND clip
  must be levelled `heard`.

### 3.2 Is any level calibrated?

| dataset | calibrated? | what there is | mark |
| --- | --- | --- | --- |
| the bucket's catalogues and sidecars | **no** | no field of level anywhere: name, bytes, CRC | read here |
| DEMAND | no absolute level. Its paper says the recordings had no gain normalisation and a fixed preamplifier gain, the channels not calibrated against one another: **the levels of its 17 places against one another are true** | one known place would calibrate the other sixteen | R, https://zenodo.org/records/1227121/files/DEMAND.pdf |
| NOISEX-92 | **yes, as the level at recording**: babble 88 dB(A), F-16 cockpit 103, destroyer engine room 101, operations room 70, tank 114, and others; not for factory, car, white, pink | none of these is a home | R, https://spib.linse.ufsc.br/noise.html |
| MIMII | no; the microphone distance (50 cm, 10 cm for valves) | | R, https://ar5iv.labs.arxiv.org/html/1909.09347 |
| ESC-50, MS-SNSD, MUSAN, TAU 2019 | no mention | | R on each one's page |
| WHAM, FSD50K, DNS noise | no mention found | | S |
| EARS | no. Measured here on 8 talkers, active speech level against the same talker's regular reading: **loud +10.0 dB** (3.9 to 12.3), fast +3.1, **whisper -5.4 dB** (-10.4 to -2.1), coughing +12.8, sneezing +14.4, open laughter +16.2, yelling +21.1 (10.3 to 34.7), eating -5.0 | the loud voice is where a loud voice is against a normal one. The whisper is not: a whisper is some 20 dB under normal speech, so either the gain was raised or it is a stage whisper. **EARS's files do not give the efforts' levels** | measured |

So a class's level cannot be read from the data. It has to be stated, from
what is published on the sources themselves.

### 3.3 Proposed: a level at 1 m for each class

For the lot that writes the recipes. The library's convention stands: full
scale is 86 dB SPL at 1 m in free field, the engine adds the room.

How a published number becomes a level at 1 m:

- **A sound power level** (the EU energy label's "airborne acoustical noise
  emission", dB(A) re 1 pW) is the best kind: it is of the source and of no
  room. A source radiating freely gives `Lp(1 m) = LWA - 11 dB`; the engine
  adds the floor and the walls, so no more is added here.
- **A level at the user's place** in a room holds the room and an unknown
  distance: it bounds the source, it does not give it.
- **A level at the listening position** (television, music): the set is 2
  to 3 m away in a furnished room, where the direct sound is 6 to 10 dB
  under the level at 1 m and the room gives some of it back: the source at
  1 m is taken 5 dB above the level at the listener.

These numbers are **A-weighted**, and a library's noise is levelled on the
unweighted level above 40 Hz. For a hiss or a tap the two agree within a
decibel or two; for a washing machine or traffic the unweighted level is
several decibels above the A-weighted one, and a clip levelled unweighted
at a dB(A) value plays too quiet by as much. **A third measure is
proposed for noises, `"measure": "a"`**, the A-weighted long term level,
so that a published dB(A) is what the entry states. Not done here.

| class | proposed dB(A) at 1 m | range | from | mark |
| --- | ---: | --- | --- | --- |
| voice, relaxed | 54 | | ISO 9921 vocal efforts (read on a course sheet that quotes it); Pearsons 1977: casual 50, normal 58 | R |
| voice, normal | 60 | 55 to 61 | ISO 9921; conversation measured at home 55, in stores 61 | R |
| voice, raised | 66 | 62 to 66 | ISO 9921; Pearsons 65 (men), 62 (women) | R; Pearsons's table from memory |
| voice, loud | 72 | 71 to 76 | ISO 9921; Pearsons 76 (men), 72 (women) | R |
| voice, shouted | 82 | 78 to 89 | ISO 9921 very loud 78; Pearsons 89 (men), 82 (women) | R |
| voice, soft (voiced, quiet) | 50 | 45 to 52 | Pearsons casual 50; soft speech 42 to 47 dB(A) at 30 cm in one study, which is about 35 at 1 m for its quietest parts | R |
| voice, whisper | 38 | 30 to 45 | no source was found | **N** |
| voice in noise (Lombard) | normal up to an ambient level of about 50 dB(A) at the talker, then **+0.5 dB for each dB of noise**, up to loud | slope 0.3 to 1; onset 43 to 57 dB(A) | a review of the slope: 0.54 dB/dB, onsets 46 to 57 | R, https://pmc.ncbi.nlm.nih.gov/articles/PMC7316514 |
| television | 65 | 58 to 70 | preferred level at the listener about 60 dB(A) (tests from 50 to 70); 66 measured at home in Japan; men choose 5 dB more than women | S, S, R |
| radio in the background | 53 | 50 to 55 | 45 to 50 dB(A) at the listener | R, https://nonoise.org/library/household/index.htm |
| music, background | 55 | | 50 dB(A) at the listener | R, same page |
| music, listened to | 68 | 58 to 78 | chosen level on loudspeakers 65 dB; students at home: quiet 53.5, typical 63.5, loud 74.4 | R; S |
| washing machine, spinning | 64 | 61 to 70 | label classes: A under 73, B 73 to 77, C 77 to 81 dB(A) re 1 pW; three machines at 72, 75, 76 | R (regulation (EU) 2019/2014, annex II); S |
| washing machine, washing | 50 | | no number found; the label gives the spin only | **N** |
| tumble dryer | 53 | 49 to 57 | 62 to 66 dB(A) re 1 pW; new classes A at most 60, D over 68 | S |
| dishwasher | 34 | 28 to 40 | label classes: A under 39, B 39 to 45, C 45 to 51, D from 51 | R (2019/2017) |
| refrigerator | 25 | 19 to 31 | label classes: A under 30, B 30 to 36, C 36 to 42, D from 42 | R (2019/2016) |
| range hood | 45 | 30 to 53 | one hood: 41 lowest, 56 highest, 64 boost; hoods from 45 to 64 dB(A) re 1 pW | S |
| extractor fan, bathroom fan | 48 | | 54 to 55 dB(A) at the user, distance not stated | R, bounded |
| vacuum cleaner | 66 | 59 to 69 | at most 80 dB(A) re 1 pW since 2017 (regulation (EU) 666/2013) | R |
| air conditioner, indoor unit | 42 | up to 49 | at most 60 dB(A) re 1 pW up to 6 kW | R (206/2012) |
| microwave oven | 50 | | 55 to 59 dB(A) at the user | R, bounded |
| hair dryer | 72 | 68 to 78 | 80 to 95 dB(A) at the user's head | R, bounded |
| shower | 60 | | about 70 dB(A) in the room, distance not stated | S, bounded |
| tap, bath filling | 55 | | no number found | **N** |
| toilet flush | 62 | | about 75 dB(A) in the room | S, bounded |
| frying, boiling, dishes | 55 | | cooking and dining as activities: 69 dB(A) on average in the room | S, bounded |
| street through an open window, as a source at the window | 55 | 45 to 63 | the level outside less 10 dB (1.7 to 17.3 over 102 dwellings; 15.8 tilted, 27.8 closed), for 65 dB(A) at the wall of a busy street | R, https://pmc.ncbi.nlm.nih.gov/articles/PMC5800248 |
| cough | 72 | | "about 80 dB", distance not stated; EARS: 13 dB over the talker's speech | S; measured |
| breathing, footsteps, clothing, a chair | 30, 50, 35, 55 | | **no measurement was found**: placeholders, to be replaced by measured values | **N** |

Against this table `clarify_v1` states: range hood 60 (15 dB over), washing
machine 58, extractor fan 52, music 60, television 57 to 62 (under), street
58, shower 62, tap 58, bath 58, frying 55, boiling 50, kitchen 47. So the
library's own levels are within a few decibels of the table but for the
hood. What made the first scene's noises loud is rather the gains of its
pack (`render gains`: +13 dB on the washing machine, +11 dB on another)
and that every noise there sits at 50 to 62 dB at 1 m while a home's
steady machines are, for most of them, 25 to 50.

What the recipe side needs beyond the table:

- **A voice's level follows the scene.** A talker near a 64 dB(A) washing
  machine raises his voice by about 0.5 dB a decibel over 50: 7 dB, which
  is `raised`. The recipe can compute the ambient level at each talker from
  the noises' levels and distances, and pick the effort, and with it the
  clips of that effort, not only a gain: a loud voice is not a normal voice
  made louder (its spectrum tilts up).
- **Whisper and soft speech only in a quiet scene**, under about 40 dB(A)
  ambient.
- A level is a draw, not a constant: the ranges are there to be drawn
  from.

## 4. Gaps, and public data that would fill them

The project's purpose is a model for a hearing aid, which may be sold. So
the licence is read for three things: may it be used commercially, may a
model be trained on it, may it be passed on. No page read forbids training
in so many words; a licence that says *non commercial* forbids a commercial
model, or leaves it unsafe. The simulator already leans on non commercial
material: HSSD's dwellings, and EARS in `clarify_v1`'s television. Whether
that is acceptable is the owner's decision and comes before any of the
choices below (section 5, question 1).

### 4.1 Conversation with a clean channel a talker

| dataset | what it holds | licence, plain reading | mark | page |
| --- | --- | --- | --- | --- |
| **NOTSOFAR-1** (Microsoft) | 237 meetings of about 6 min, 35 talkers, 4 to 8 a meeting, 30 rooms, a head-worn close microphone each. Rate not stated on the page | the repository says CC BY 4.0: commercial use allowed. **The CHiME-8 page says the data is licensed for the challenge only.** The two disagree; ask before use. Needs an account | R, rate N | https://github.com/microsoft/NOTSOFAR1-Challenge , https://www.chimechallenge.org/challenges/chime8/task2/data |
| **DiPCo** (Amazon) | 10 dinners of 15 to 45 min, 4 talkers each, a close microphone each and 5 far arrays; 13.4 GB | CDLA Permissive 1.0: commercial use and passing on allowed | R, rate N | https://zenodo.org/records/8122551 |
| AMI | 100 h of meetings; headset and lapel a talker; recorded at 48 kHz, given at 16 kHz | CC BY 4.0: allowed. **In the bucket** | R | https://groups.inf.ed.ac.uk/ami/corpus/ |
| CHiME-5/6 | 20 dinners in real homes, 4 talkers, binaural microphones in the ears (so the room and the others are in them), 6 arrays; about 122 GB | CC BY-SA 4.0 on OpenSLR: allowed, share alike. The challenge's own data page did not load | R, page N | https://www.openslr.org/150/ |
| **Expresso** (Meta) | 40 h, 4 actors: 11 h read, 30 h **improvised dialogue with a channel an actor**, 48 kHz 24 bit, studio, 26 styles among them whisper and laughing; 36 GB | CC BY-NC 4.0: **non commercial** | R | https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset |
| EARS | 100 h, 107 talkers, 48 kHz, anechoic. Monologue, not dialogue | CC BY-NC 4.0: **non commercial**. **In the bucket** | R | https://github.com/facebookresearch/ears_dataset |
| EasyCom | 5 h 18 min, a close microphone a talker, 48 kHz, restaurant noise at about 71 dB SPL in the room, reverberation time 0.65 s | CC BY-NC 4.0: **non commercial**. **In the bucket** | R | https://github.com/facebookresearch/EasyComDataset |
| Seamless Interaction (Meta) | over 4 000 h of two people face to face | CC BY-NC 4.0: non commercial | S | https://ai.meta.com/research/seamless-interaction/ |
| Fisher, Switchboard | 984 h and 260 h of telephone calls, two channels, **8 kHz** | LDC: paid; the non member agreement allows research only, a product needs a for-profit membership; no passing on | R | https://catalog.ldc.upenn.edu/LDC2004S13 , https://catalog.ldc.upenn.edu/LDC97S62 |
| Mixer 6 | 594 talkers, 14 room microphones, 16 kHz | LDC members only | R | https://catalog.ldc.upenn.edu/LDC2013S03 |
| CANDOR | 1 656 video call conversations, 850 h: the audio went through a codec | by registration; no licence text was found on a page that loaded | S, licence N | https://www.betterup.com/research/candor-research |
| DailyTalk | 2 541 dialogues **read from scripts**: not spontaneous | CC BY-SA 4.0, and "for academic use": ambiguous | R | https://github.com/keonlee9420/DailyTalk |

**No open dataset is at once conversational, a channel a talker, dry, at
44.1 or 48 kHz, and free for a product.** The ones that fit by their sound
(Expresso, EARS) are non commercial; the ones that fit by their licence
(AMI, DiPCo, NOTSOFAR-1) are recorded in rooms at 16 kHz with the other
talkers in every channel.

### 4.2 Vocal effort

| dataset | what it holds | licence, plain reading | mark | page |
| --- | --- | --- | --- | --- |
| **AVID**, the Aalto Vocal Intensity Database (no "AVID-Lombard" was found: this is what the name refers to) | 50 talkers, 25 men and 25 women, **four efforts: soft, normal, loud, very loud**, fixed microphone distance and a calibration tone; 12.4 GB. Not Lombard speech | CC BY 4.0: allowed | R; rate and room N | https://zenodo.org/records/10524873 |
| **Lombard GRID** (Sheffield) | 54 talkers, 5 400 sentences, half spoken in noise and half plain. The sentences are the Grid commands, read; audio 651 MB | CC BY 4.0: allowed | R; rate and noise level N | https://spandh.dcs.shef.ac.uk/avlombard/ |
| Hurricane natural speech | one British man, sentences in quiet and under speech shaped noise at 84 dB(A), 48 kHz | CC BY-NC 4.0, seen second hand: non commercial | R, licence S | https://datashare.ed.ac.uk/handle/10283/3239 |
| DELNN, Radboud Lombard | Dutch and American talkers, plain and Lombard | restricted, academic use only | R, S | https://zenodo.org/records/4267819 , https://zenodo.org/record/4040685 |
| CHAINS (UCD) | 36 talkers: solo, in chorus, retelling, **whisper**, fast; a studio microphone in a booth | its own site refused the connection. A Creative Commons licence is reported second hand; the LDC copy is under LDC's terms | LDC page R, licence N | https://chains.ucd.ie , https://catalog.ldc.upenn.edu/LDC2008S09 |
| wTIMIT | 48 or 49 talkers, 450 sentences, normal and whispered, 44.1 kHz | no download page and no licence found | S, N | none verified |
| wSPIRE (IISc) | 88 talkers, neutral and whispered, recorded at 44.1 kHz and given at 16 kHz | no licence on its page | R | https://spire.ee.iisc.ac.in/src/wspire.php |
| EARS loud and whisper; Expresso whisper | section 1.3; section 4.1 | non commercial | R | above |

### 4.3 Breaths and non-verbal sounds

| dataset | what it holds | licence, plain reading | mark | page |
| --- | --- | --- | --- | --- |
| **VocalSound** | 21 024 clips of 3 365 people: laughter, sighs, coughs, throat clearing, sneezes, sniffs; 44.1 kHz. Recorded on telephones by the people themselves: not dry. No class of breathing | CC BY-SA 4.0: allowed, share alike | R | https://github.com/YuanGongND/vocalsound |
| **Coswara** (IISc) | over 2 200 people: **breathing, fast and slow**, coughs deep and shallow; recorded by the people themselves | CC BY 4.0: allowed | R | https://github.com/iiscleap/Coswara-Data |
| Nonspeech7k | 7 014 clips, 32 kHz, 0.5 to 4 s, from Freesound, YouTube and others; 2.5 GB | "non-commercial and academic research" only | R | https://zenodo.org/records/6967442 |
| NonverbalTTS | 6 260 clips cut from VoxCeleb and Expresso | its card says Apache 2.0, its sources are non commercial or YouTube: not to be relied on | R | https://huggingface.co/datasets/deepvk/NonverbalTTS |
| FSD50K, EARS | section 1.4 | per clip; non commercial | R | above |

### 4.4 Bodies and objects, dry

| dataset | what it holds | licence, plain reading | mark | page |
| --- | --- | --- | --- | --- |
| **FoleySet** (Georgia Tech, June 2026) | 10 000 clips in 9 kinds and 73 sub-kinds, among them **footsteps, cloth rustle, handling of objects**; a 2.2 GB archive. Said to be built from CC0 clips of Freesound | CC BY 4.0: allowed | R; the CC0 claim S; rate N | https://zenodo.org/records/20735877 |
| FSD50K | 51 197 clips, 108.3 h, 200 classes, 44.1 kHz | keep CC0 and CC BY for a product. **In the bucket** | R | https://zenodo.org/records/4060432 |
| DCASE 2023 task 7 | 4 850 clips of 4 s at **22.05 kHz**: footsteps 703, sneeze and cough 631 | its BBC part is for research only | R | https://dcase.community/challenge2023/task-foley-sound-synthesis |
| BBC Sound Effects | about 33 000 clips | personal, education and research; a product and the training of a model need a paid licence | S; the licence page did not load | https://sound-effects.bbcrewind.co.uk/licensing |

None of these promises a recording that is close and dry, and none has a
class for a chair.

### 4.5 Media voice

| dataset | what it holds | licence, plain reading | mark | page |
| --- | --- | --- | --- | --- |
| MUSAN | music, speech in 12 languages, noise; 11 GB. Not broadcast mixes | CC BY 4.0: allowed. **In the bucket** | R | https://www.openslr.org/17/ |
| OpenBMAT | 27 h of television broadcasts of 4 countries, in 1 647 excerpts of a minute | "only for non-profit purposes", on request with an academic affiliation | R | https://zenodo.org/records/3381249 |
| BAF | 57 h of television broadcasts of 23 countries, **8 kHz mono** | non commercial research, on request, no derivatives | R | https://zenodo.org/record/6868083 |
| TVSM (Netflix, Georgia Tech) | 1 600 h, **features only, no audio** | Apache 2.0, and of no use here | R | https://zenodo.org/record/7025971 |
| AudioSet | identifiers of YouTube videos and features; the audio is each uploader's | the metadata is CC BY 4.0, the audio is not Google's to license | R | https://research.google.com/audioset/download.html |
| DNS Challenge 5 | the noise is AudioSet, Freesound CC0 and DEMAND; its clean speech holds VoxCeleb2 and CREMA-D | "original licenses", part by part: mixed. **In the bucket** | R | https://github.com/microsoft/DNS-Challenge |

**No real television or radio audio is open for a product.** Every set of
broadcasts is non commercial, on request, at 8 kHz, or without audio.

## 5. What to add: the plan the owner decides on

Nothing below was downloaded beyond the samples the measurements needed,
and nothing was uploaded. Hours are what the source holds, not what a
library needs: a library of scenes takes minutes of each.

### 5.1 From the bucket, at no cost: a second library

| need | source in the bucket | amount | licence | class of the library |
| --- | --- | --- | --- | --- |
| spontaneous speech, dry | EARS `freeform_speech_*`, cut into turns at its pauses (`find_utterances` does it) | 32.5 h, 107 talkers | CC BY-NC 4.0 | `voice`, with a new `style: "spontaneous"` |
| emotional and lively speech | EARS `emo_*_freeform` | 10.1 h | CC BY-NC 4.0 | `voice`, `style` |
| raised voice | EARS `rainbow_*_loud`, `sentences_*_loud` | 5.5 h, 107 talkers | CC BY-NC 4.0 | `voice`, `effort: "loud"`, stored at the level of section 3.3 |
| whisper | EARS `rainbow_*_whisper`, `sentences_*_whisper` | 6.0 h, 107 talkers | CC BY-NC 4.0 | `voice`, `effort: "whisper"` |
| a talker's own sounds | EARS `vegetative_*` (coughing, throat, sneezing, yawning, eating), `nonverbal_laughter_*`, `interjection_filler` | 1.4 h, 0.6 h, 0.5 h | CC BY-NC 4.0 | a new kind, `body`, tied to the voice's `speaker`, so that a talker coughs with his own voice |
| breathing | FSD50K `Breathing`, CC0 and CC BY clips only, screened | 566 of 658 clips | CC0, CC BY | `body` |
| footsteps, zips, doors, drawers, dishes | FSD50K by label, CC0 and CC BY, screened, and listened to for dryness | section 1.4 | CC0, CC BY | `body` (footsteps, clothing), `noise/other` (objects) |
| more machines and water | FSD50K by label; ESC-50 `vacuum_cleaner`, `washing_machine` (non commercial) | section 1.4 | per clip | `noise/appliance`, `noise/water` |
| media voice | MUSAN speech (hearings, audiobooks) over MUSAN or FMA music, mixed here: a made programme | 60 h, 43 h | CC BY 4.0, per track | `media_voice` |
| conversation, as a comparison only | AMI headsets | 100 h | CC BY 4.0 | not for scenes: 16 kHz, crosstalk, a room |

### 5.2 From outside, to be fetched if the owner agrees

| rank | gap | dataset | size | licence | what it becomes |
| ---: | --- | --- | --- | --- | --- |
| 1 | soft, loud, very loud speech with a level reference | AVID (Aalto) | 12.4 GB, 50 talkers | CC BY 4.0 | `voice` with `effort`; its calibration tone gives the level differences between efforts, which EARS does not (section 3.2) |
| 2 | Lombard speech | Lombard GRID | 0.65 GB audio, 54 talkers | CC BY 4.0 | `voice`, `effort: "lombard"`. Read commands: it teaches the voice, not the wording |
| 3 | footsteps, cloth, handling | FoleySet | 2.2 GB | CC BY 4.0 | `body`, after listening for dryness and checking the sample rate |
| 4 | breathing | Coswara | not stated | CC BY 4.0 | `body`, screened; telephone recordings |
| 5 | laughter, sighs, throat, sniffs | VocalSound | 4.5 GB at 44.1 kHz | CC BY-SA 4.0 | `body`, screened; telephone recordings |
| 6 | conversation, a channel a talker | DiPCo; NOTSOFAR-1 once its licence is settled in writing | 13.4 GB; not stated | CDLA Permissive; CC BY 4.0 or challenge only | not dry: a study of turn taking (when talkers overlap, how long they pause) to drive the generator, rather than audio for scenes |
| 7 | improvised dialogue, a channel an actor, dry enough, with whisper | Expresso | 36 GB | CC BY-NC 4.0 | `voice`, `style: "dialogue"`: the best sounding source and a non commercial one |

### 5.3 What no dataset gives

| gap | why | what could be done |
| --- | --- | --- |
| a dry conversation at 48 kHz free for a product | section 4.1 | turns from monologues now; own recordings later (two talkers, two booths, two channels) |
| real television and radio | section 4.5 | a made programme (5.1); or a paid licence; or own recordings of a set playing |
| a chair, clothing in movement, dry and close | no class anywhere | own recordings: an hour with one microphone at 30 cm in a damped room covers it |
| whisper free for a product | CHAINS's licence could not be read; wTIMIT was not found | read CHAINS's licence on its own site when it answers |
| levels of body sounds | no measurement was found (section 3) | measure them when recording |

### 5.4 Questions for the owner

1. Is non commercial material acceptable in training data? EARS decides
   most of 5.1, and HSSD is already under such terms.
2. Is a made programme (speech over music, by this project) an acceptable
   `media_voice` to begin with?
3. Should a murmur of many voices stay a noise? The screen takes it for
   speech today (section 2.5).
4. May the samples of 5.2 be fetched (about 20 GB for ranks 1 to 5)?

## 6. Not verified

- **No clip was listened to.** Everything said of content comes from
  detectors, from transcripts read, and from the datasets' own labels.
- The hours are computed from bytes and from the format of 16 files a
  group; a group of mixed formats (UrbanSound8K, FMA) is a few per cent off.
- Talkers are counted from file names. AMI, AISHELL-4, EasyCom, MUSAN and
  the German and Spanish parts of `dns5-clean` do not name them.
- Licences in sections 1.1 and 1.2 are what the bucket's provenance record
  says, which names the page each was read on; they were not read again,
  except where section 4 marks R. `dns5-clean`, `dns5-noise`, `mls_*`,
  `noisex92`, `wham_noise` have no licence recorded.
- Pages that did not load: the CHiME-6 data page, the EARS project page,
  chains.ucd.ie, the CANDOR paper, the BBC Sound Effects licence, WHAM's
  site, the EUR-Lex texts (read on legislation.gov.uk instead), several
  papers of the Institute of Acoustics on domestic source levels.
- Not found at all: wTIMIT's home, a dataset named AVID-Lombard, measured
  levels of footsteps, clothing, chairs, a kitchen tap, frying.
- The sample rates of NOTSOFAR-1, DiPCo, AVID, Lombard GRID and FoleySet
  are not stated on the pages read.
- JVNV, ESD, SWARA, ALS Lombard, CASPER and the full duplex conversation
  sets were not examined.
