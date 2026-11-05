"""Write examples/predict_FD001_unit1.json: test engine 1 of FD001 as an API request body."""

import json
from pathlib import Path

from rul.config import SENSORS, SETTINGS, data_dir
from rul.data import read_split

raw = read_split(data_dir() / "test_FD001.txt")
engine = raw[raw["unit"] == 1]
body = {
    "subset": "FD001",
    "cycles": [
        {"settings": [float(r[s]) for s in SETTINGS], "sensors": [float(r[s]) for s in SENSORS]}
        for _, r in engine.iterrows()
    ],
}
out = Path("examples/predict_FD001_unit1.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(body) + "\n", encoding="utf-8")
print(f"wrote {out}: {len(body['cycles'])} cycles")
