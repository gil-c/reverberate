# Small applications and their parts

Date: 2026-10-07

One page of `python -m reverberate.viz.serve_room` shows everything a pack
was computed on. It stays as it is: the inspector. A decision taken by ear
wants a page that holds what the decision needs and nothing else, and that
is deleted once the decision is taken. Such a page is a few hundred lines
because its parts exist: `reverberate.viz.parts` (the server's side) and
`reverberate/viz/parts/static/` (the page's side). No build step, no
dependency the inspector does not have.

| application | command | what it is for |
| --- | --- | --- |
| compare | `python -m reverberate.apps.compare FOLDER` | two to N renders of one short scene, by ear, by eye and blind |
| scene player (a first cut) | `python -m reverberate.apps.scene_player FOLDER` | a scene with its sources in sight and a fader on each |
| recipes (a first cut) | `python -m reverberate.apps.recipes FOLDER` | the recipes on disk, summarised and held to the rules |

Each binds the loopback address, takes `--port` (8780, 8781 and 8782 unless
said, the next free one when taken) and `--no-browser`.

## Compare

![Listening freely](apps/compare-free.jpg)

`FOLDER` is what `python -m reverberate.render check REF.h5 --against V1.h5
... --out FOLDER` writes (`formats/scene-signal.md`): `variants.json` (or
`ab.json` of two packs), the files of two ears, and with `--ambisonic` each
source's order 7 stem. From packs in one command, order 7 and the three
signals:

```
python -m reverberate.apps.compare --packs REF.h5 V1.h5 --names ref cheap \
    --window 242 262 --sources near_2 --out FOLDER
```

The screen holds:

- **the variants as buttons**, keys `1` to `N`. A press changes the render
  heard at the same instant, the two crossfaded over one block of 10.7 ms
  under the same head;
- **play, stop, loop** (space, `L`), a bar that is clicked to move and
  dragged to set the region the loop turns in, and the level, in dB over
  the default at which a pack's full scale stands for 98 dB SPL;
- **the head**: a dial dragged left and right to turn and up and down to
  tilt, the arrow keys by five degrees, `0` to centre. Order 7 is decoded on
  the page with the measured head (`viz/decoders.py`), under the scene's own
  head turned by what the listener adds: with nothing added it is what the
  kit's files of two ears hold. A folder of two ears alone is played as
  written and says that the head does not turn;
- **the signal**: speech (the recipe's clips), clicks, pink noise, where the
  folder holds them (`--signals`);
- **the sonograms**, one under the other on one time axis or each less the
  reference, on a scale that does not move;
- **what the kit measured**, the variant less the reference by third octave
  on the early and the late part of the response to an impulse and on the
  speech, and the largest of each either side of the crossover;
- **a blind test**.

![Each variant less the reference](apps/compare-difference.jpg)

### The blind test

![An ABX trial](apps/compare-blind.jpg)

- **ABX**: A and B are named, X is one of the two, drawn again for each
  trial. The listener hears all three as long as he likes and says which X
  is. Nothing is told after a trial. At the end: the score, and the
  probability of that score or a better one by guessing, the binomial's at
  one half a trial (12 right out of 16 is 3.8 per cent).
- **Ranking**: every variant under a name that says nothing, drawn again for
  each trial, put in the order preferred. At the end: how often each came
  first, its mean rank, and the probability that some variant comes first
  that often when the order is chance (one variant's, `1 / N` a trial, times
  the number of variants).

The key stays on the server: the page asks for `blind:X` and is never told
what it stands for; the sonograms are refused while a test runs and the
page hides what was measured. A test is written when its last trial is
answered, or when it is stopped, in `<FOLDER>_listening/blind-<kind>-<UTC
time>.json` (`--results` to write elsewhere):

| key | meaning |
| --- | --- |
| `schema`, `kind` | `"reverberate.apps.blind"`, `"abx"` or `"rank"` |
| `started`, `finished`, `complete`, `trials_planned` | when, and whether every trial was answered |
| `seed` | what the draws were made from: the same seed draws the same test |
| `items` | each variant's name and the item played for it |
| `context` | the folder, the recipe's identity, the window, the signal, the source, the level and the loop's region |
| `trials` | ABX: `x`, `answer`, `right`, `seconds`; ranking: `shown` (hidden name to variant), `ranking` (the preferred first), `seconds` |
| `result` | ABX: `answered`, `right`, `chance`; ranking: `first`, `mean_rank`, `chance` |

![The end of a test](apps/compare-result.jpg)

## The parts

Each is used alone. The page's side are ES modules served under `parts/`;
the inspector's decoder is served under `reuse/` and is not copied.

### The server (`viz/parts/server.py`)

```python
server = AppServer("mine", Path(__file__).parent / "static")
server.route("GET", "api/thing", lambda request: {"said": request.query.get("q")})
mount_decoders(server, scratch / "decoders", measured_head=None)
server.serve(8790)
```

A route answers what `json.dumps` takes, or a `Binary`; it raises
`HttpError` to refuse. `AppServer.handle(method, url, body)` answers with
no socket, which is how `tests/test_apps_parts.py` asks. A folder serves
nothing above itself.

### What is played (`viz/parts/media.py`, `routes.py`)

An **item** is one thing heard from its start to its end, made of **stems**
that are summed: order 7 signals (`formats/scene-signal.md`) or files of two
ears. `kit_folder(FOLDER)` reads a comparison into items, held to one length
where they are switched; `Library(items)` holds any others. `Media(library,
results).mount(server)` gives the page:

| route | answer |
| --- | --- |
| `GET api/items` | what there is to play |
| `GET api/frames?item=&start=&count=[&balance=]` | float32 `[frame][channel]`, the stems summed in float64 under a balance; one stem at a gain of one is its own samples |
| `GET api/sonogram?item=[&less=]` | float32 `[column][band]`, described in the `X-Sonogram` header |
| `GET api/levels?item=` | each stem's level every 50 ms |
| `GET`, `POST api/balance` | the track list's faders |
| `GET api/blind`, `POST api/blind/start`, `answer`, `stop` | the blind test |

### The player (`static/player.js`, `transport.js`)

```js
const player = createPlayer();
createTransport(element, player);      // the buttons, the bar, the level, the head's dial
await player.load(items);              // rows of api/items, of one length
player.setSceneHead(head);             // the scene's own head, or nothing
player.play(); player.select(items[1].id); player.turn({ yaw: 30 });
```

The audio thread is the inspector's worklet and the filters are turned by
its `rotateSpectra`; two ears go through the same thread with filters that
change nothing. `player.output()` is the node the two ears leave by.

### The sonogram (`viz/parts/sonogram.py`, `static/sonogram.js`)

A column every 10 ms, twelve bands an octave from 50 Hz to 16 kHz, of the
omnidirectional channel (or of the mean of two ears) at the page's level. A
cell is a mean square in dB re full scale: a sine of full scale reads -3 dB
in its band whatever else sounds. The colours run from -110 to -30 dB, and a
difference from -12 to +12 dB with grey where neither holds anything (60 dB
under the loudest cell of the two). `createSonogram(element, { player })`
draws them; a click moves the player and a drag sets its loop.

### The track list (`viz/parts/balance.py`, `static/track-list.js`)

A solo, a mute and a fader a source (-60 dB, where it is silent, to +12 dB),
its meter and its level over the window. Every change is saved at once in
`<FOLDER>_listening/balance.json` and the player asks its frames under it,
so the mix heard is the one the file holds; "Export" downloads the same
document. `balance.gains` is the one place a balance becomes factors.

| key | meaning |
| --- | --- |
| `schema`, `saved`, `context` | `"reverberate.apps.balance"`, when, and what it was set on |
| `sources` | for each source `gain_db`, `mute`, `solo` |
| `gains` | the factor each source's stem was multiplied by |

### The scene (`viz/parts/scene.py`, `static/scene-view.js`)

`scene_view(pack, window)` gives where the listener and each source are, a
value every 0.1 s, and the floor: the cells the pack's low band was solved
at. `createSceneView(element, scene)` draws a voice as a head in its colour
with its name, a noise as a cube, the listener as the white head; each
lights by its own stem's level under its fader, dark at -80 dB and full at
-45 dB.

### Numbers by third octave (`static/bands.js`)

`drawBands(element, { bandsHz, series, rangeDb, markHz })`: a line a series
on a logarithmic axis and a scale that does not follow the data.

## The scene player and the recipes, and what remains

![The scene player](apps/scene-player.jpg)

The scene player plays a folder of the kit that holds order 7 stems: the
scene, the transport and the head, a track list with a lane a source. It
remains to:

- play a whole scene from a pack, its stems rendered as they are asked for
  (`viz/audit_stems.py` does it for the inspector; here the window is what
  the kit rendered);
- draw the dwelling: the floor is the pack's cells, which are where the
  head goes, and no wall is drawn;
- a timeline that shows who speaks when beyond the lanes: the recipe's
  intervals, the listener's path, a click on a source to solo it;
- take what the recipes learn from a balance back to the generator.

![The recipes](apps/recipes.jpg)

The recipe manager lists the recipes under a folder and, for the one
chosen, shows `reverberate.scenes.describe` and every rule
`reverberate.scenes.validate` finds broken; with the dataset at hand it
draws one from a dwelling and a seed with the generator's defaults and
saves it in `--save-in`. It remains to:

- set the generator's parameters (the generator is being rewritten: the
  page keeps to the package's public functions and shows no parameter);
- check a recipe read from disk against its dwelling's floor, which needs
  the dataset;
- summarise many recipes side by side (how much speech, how many sources at
  once, the distances) and delete or rename one;
- open a recipe in the scene player.

## Writing another

A package under `reverberate/apps/` with a `static/` folder: build an
`AppServer`, mount a `Media` on what is played, add the routes the purpose
needs, and join the parts in one module of the page. `apps/scene_player` is
the shortest example: 98 lines of Python and 58 of JavaScript.
