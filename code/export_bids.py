#!/usr/bin/env python3
"""
Convert the raw EyeLink/Psychtoolbox session files into the BIDS-style dataset
shipped in ../data.

This script is included for transparency: it documents exactly how the raw
recordings were reduced to the fixation-level event tables that all analyses
read. It needs the raw session files, which are NOT part of this release (about
1.25 GB); the released dataset is its output. Pass the raw tree as an argument
to re-run it.

What it does
------------
* Reads one .mat per participant (the final block file, which accumulates all
  120 trials of the session).
* Reconstructs each fixation's onset as FixEndTimes - (entime - sttime), i.e.
  end-of-fixation minus its duration, relative to search-display onset.
* Writes one row per fixation to
  sub-<ID>/eyetrack/sub-<ID>_task-<task>_events.tsv, with the trial it belongs
  to, the condition, and the stimulus.
* Pseudonymises participants: the raw folders are named with initials plus a
  number, which are potentially identifying. They become sub-001, sub-002, ...
  in the order of the sorted raw folder names. The mapping is deliberately NOT
  written to the release.
* Collects age, sex, handedness and corrected-vision into participants.tsv.

Condition labelling
-------------------
Taken from Results.PrPath, the runtime record of which image was actually drawn
as the preview -- not from the stimulus filename. At collection time the scene
files were named Conferanceroom_* while the preview files were Conferenceroom_*,
so the experiment's filename-based pairing failed and the ten Conferanceroom
scenes were shown with a grey preview to every participant. The files have since
been renamed consistently, so filename matching would now report the INTENDED
design rather than what was displayed. PrPath is unaffected by the rename.

A note on `onset`
-----------------
BIDS expects `onset` to be relative to the start of the run. The raw files do
not record absolute trial onsets, only per-trial timings, so a run-relative
onset could only be reconstructed by assuming fixed inter-trial intervals. To
avoid inventing precision, `onset` here is relative to the onset of the search
display on that trial, which is the quantity that was actually measured and the
one every analysis uses. This is stated in the sidecar and in the dataset README.

Run: python3 export_bids.py /path/to/raw
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

OUT = Path(__file__).resolve().parent.parent / "data"

TASKS = {
    "facedetection": dict(raw="FacePrevGaze", stim_dur=6.0),
    "movingwindow": dict(raw="Moving_Window_Prew", stim_dur=20.0),
}
SKIP = re.compile(r"pilot|test|^999", re.I)

# Sessions that are not part of the sample: two moving-window sessions were
# recorded by accident and never analysed. They are identified by their 1-based
# position in the sorted list of that task's raw session folders and skipped
# BEFORE IDs are assigned, so the released IDs run without gaps.
NOT_IN_SAMPLE = {"facedetection": set(), "movingwindow": {12, 40}}


def scalar(x):
    a = np.asarray(x).ravel()
    return float(a[0]) if a.size else None


def text(x):
    a = np.asarray(x).ravel()
    return str(a[0]) if a.size else ""


def fixations(cell, endtimes):
    """(onset, duration, x, y) per fixation, onsets relative to display onset."""
    out = []
    fe = np.asarray(endtimes).ravel()
    for i in range(cell.size):
        e = cell[0, i]
        try:
            st, en = scalar(e.sttime), scalar(e.entime)
            gx, gy = scalar(e.gavx), scalar(e.gavy)
        except AttributeError:
            continue
        if st is None or en is None or i >= len(fe):
            continue
        dur = (en - st) / 1000.0
        out.append((float(fe[i]) - dur, dur, gx, gy))
    return out


def export_task(task, cfg, sub_offset, raw_root):
    raw = raw_root / cfg["raw"]
    stims = sorted(p.name for p in (raw / "Part_1").iterdir() if "with_face" in p.name)
    subs = [d for d in sorted((raw / "results").iterdir())
            if d.is_dir() and not SKIP.search(d.name)
            and list(d.glob("*_Block4_Part_*.mat"))]

    subs = [d for k, d in enumerate(subs, 1) if k not in NOT_IN_SAMPLE[task]]

    people = []
    for k, d in enumerate(subs):
        sid = f"sub-{sub_offset + k + 1:03d}"
        R = sio.loadmat(sorted(d.glob("*_Block4_Part_*.mat"))[0],
                        struct_as_record=False, squeeze_me=False)["Results"][0, 0]
        td = np.asarray(R.TrialDur).ravel()
        rows = []
        for t in range(td.size):
            cells = np.asarray(R.ImPath[0, t]).ravel()
            if cells.size < 3:
                continue
            stim = str(np.asarray(cells[-1]).ravel()[0])
            if stim not in stims:
                continue
            pr = np.asarray(R.PrPath[0, t]).ravel()
            cond = "no_preview" if (pr.size and "graypreview" in str(pr[0])) else "preview"
            timeout = int(float(td[t]) >= cfg["stim_dur"])

            search = fixations(R.FixData[0, t], R.FixEndTimes[0, t])
            prior = fixations(R.PriorFixData[0, t], R.PriorFixEndTimes[0, t]) \
                if hasattr(R, "PriorFixData") else []
            for phase, seq in (("preview", prior), ("search", search)):
                for j, (on, dur, gx, gy) in enumerate(seq):
                    last = (phase == "search" and j == len(seq) - 1)
                    rows.append(dict(
                        onset=round(on, 6), duration=round(dur, 6),
                        trial=t + 1, phase=phase, trial_type=cond,
                        stim_file=f"stimuli/{stim}",
                        fixation_index=j + 1,
                        gaze_x=None if gx is None else round(gx, 2),
                        gaze_y=None if gy is None else round(gy, 2),
                        is_detection=int(last and not timeout),
                        trial_duration=round(float(td[t]), 6),
                        timeout=timeout))
        ev = pd.DataFrame(rows)
        dest = OUT / sid / "eyetrack"
        dest.mkdir(parents=True, exist_ok=True)
        ev.to_csv(dest / f"{sid}_task-{task}_events.tsv", sep="\t",
                  index=False, na_rep="n/a")

        people.append(dict(
            participant_id=sid, task=task,
            age=int(scalar(R.Age)) if scalar(R.Age) else "n/a",
            sex=text(R.Sex).lower() or "n/a",
            handedness=text(R.Handedness).lower() or "n/a",
            corrected_vision=text(R.CorrectedVision).lower() or "n/a"))
        print(f"  {sid}  {len(ev):5d} fixation events")
    return people, len(subs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw_root", type=Path,
                    help="directory holding the raw session trees "
                         "(FacePrevGaze/ and Moving_Window_Prew/)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    people, off = [], 0
    for task, cfg in TASKS.items():
        print(f"{task}:")
        p, n = export_task(task, cfg, off, args.raw_root)
        people += p
        off += n
    pd.DataFrame(people).to_csv(OUT / "participants.tsv", sep="\t", index=False)
    print(f"\nwrote {len(people)} participants to {OUT/'participants.tsv'}")


if __name__ == "__main__":
    main()
