#!/bin/zsh
# Everything W10 does once its response is home, as a script rather than as
# somebody's memory. The solve runs detached on a rented card for nearly three
# hours; whoever is watching may not be the one who started it.
set -u
cd /Users/gilles/Developer/worktrees/reverberate/ambisonics-binaural-conversion-c9cc9a
export REVERBERATE_DATA=/Users/gilles/Developer/reverberate/data PYTHONPATH=src
export PFFDTD_DIR=/Users/gilles/Developer/reverberate/data/vendor/pffdtd

RUNS=$REVERBERATE_DATA/runs
OLD=$RUNS/w37_bedroom_16k          # the name the run was started under
NEW=$RUNS/w10_bedroom_16k          # W37 belongs to another session's item

until grep -qE "FETCHED|MISMATCH|ENGINE GONE" /tmp/w10_fetch.log 2>/dev/null; do sleep 60; done
if ! grep -q "FETCHED" /tmp/w10_fetch.log; then
  echo "the response never came home; stopping rather than rendering nothing"
  grep -E "MISMATCH|ENGINE GONE" /tmp/w10_fetch.log
  exit 1
fi

# The run was started before the item was renumbered. Move it once, here, so the
# directory on disk and the registry row agree.
if [ -d "$OLD" ] && [ ! -d "$NEW" ]; then mv "$OLD" "$NEW"; echo "renamed to $(basename $NEW)"; fi

echo "rendering, encoding and decoding"
caffeinate -i .venv/bin/python -m reverberate.experiments.w10_render room \
  --run "$NEW" --low-cut 39.551 --audio 2>&1 | tail -3

.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["REVERBERATE_DATA"]) / "runs" / "w10_bedroom_16k"
r = json.load(open(run / "report.json"))
print(f"\nrun {r['run']}, room {r['room']}, reference {r['reference_run']}")
print("air:", r["air_absorption"].get("standard"), r["air_absorption"].get("humidity_percent"), "%")
usable = [x for x in r["direction_of_arrival"] if x["usable"]]
print("direction of arrival, usable bands:",
      [(x["band_hz"], x["error_deg"]) for x in usable])
print("effective order:", r["conditioning"]["effective_order"])
omni = r["omnidirectional"]
print("W channel T30 per band:", list(zip(omni["bands_hz"], omni["rt60_s"])))
for head, block in r["binaural_decodes"].items():
    row = block["measures"]["yaw_0"]
    print(f"{head}: ITD {row['itd_us']} us, ILD {row['ild_db']} dB, "
          f"late coherence {row['late_coherence']}")
print("artefacts:", r.get("artefacts"))
print("audio:", [a["path"] for a in r.get("audio", [])])
PY
echo "done. Publish with: --publish, and add the REGISTRY.md row."
