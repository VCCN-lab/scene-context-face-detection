# Scene previews and face detection

Analysis code for two gaze-contingent eye-tracking experiments on how a brief
preview of a scene changes how quickly observers find a face in it. The code
reproduces every reported number and figure from the published dataset.

Dataset: https://doi.org/10.5281/zenodo.24581856

## Data

The dataset is published on Zenodo: fixation-level event tables, participants
and face masks as a BIDS dataset, with the image features and CNN activations,
the encoding-model results, the Supplementary Tables and the figure source data
as BIDS derivatives. `code/fetch_data.py` downloads it into `data/`; its
`README.md` describes the layout. To use a copy stored elsewhere, set
`FACESCENES_DATA` to its path.

The scene photographs are not part of the dataset. They are adapted from the
Places365 dataset, whose image copyright stays with the original owners. No
analysis reads them; the image features and CNN activations computed from them
are in `data/derivatives/stimulus_features/`, and the face masks are in
`data/stimuli/masks/`.

## Repository contents

```
code/
  facescenes/          the analysis package
  run_*.py             one script per analysis, each runnable on its own
  fetch_data.py        downloads the dataset from Zenodo into data/
  export_bids.py       how the raw recordings were reduced to the dataset
  build_notebooks.py   regenerates the notebooks
notebooks/             one notebook per figure
figures/               every panel as PNG (300 dpi, RGB) and as vector PDF
tests/                 unit tests and consistency checks (pytest)
generate_images.sh     writes Dockerfile and Singularity.def
```

Two folders are created locally and are not part of the repository:

* `data/`: the dataset, as downloaded from Zenodo.
* `derivatives/`: the results of rerunning an analysis, in the layout of
  `data/derivatives/`. The released results in `data/derivatives/` are not
  modified.

## Quick start

The environment is managed with [uv](https://docs.astral.sh/uv/); `uv.lock` pins
every package version.

```bash
uv sync
cd code
uv run python fetch_data.py
uv run python run_behavioural_analyses.py
uv run python run_onset_gaze_control.py
uv run python run_reliability.py
uv run python run_encoding_analyses.py --force
uv run python run_contrast_test.py
uv run python run_masked_face_analyses.py
uv run python run_layer_depth.py
uv run python run_statistics_tables.py
```

Without uv, `pip install -r requirements.txt` installs the same pinned
versions, exported from `uv.lock`; then run the scripts with `python3`.

The released dataset is never modified. Scripts write their results to
`derivatives/`, and readers of the cached permutation results
(`run_statistics_tables.py`, the notebooks) take a rerun's from there when it
exists and the released ones from `data/derivatives/encoding/` otherwise.
Without `--force`, `run_encoding_analyses.py` reports the cached results;
with it, they are recomputed. `uv run pytest` compares every rerun result in
`derivatives/` with its released counterpart.

Then open the notebooks (`uv run jupyter lab`).

Each notebook writes its panels to `figures/` and the numbers behind each panel
to `derivatives/source_data/<panel>.tsv` in the same call, so a figure and its
source data cannot disagree. Panels carry no titles, as the journal requires;
set `SHOW_TITLES = True` in `facescenes/plotting.py` to label them by task.

## Tests

The tests read the released dataset, so download it first
(`uv run python code/fetch_data.py`).

```bash
uv run pytest
uv run pytest -m zenodo         # downloads the published dataset
uv run codespell                # spelling, configured in .codespellrc
uv run flake8                   # PEP8, configured in .flake8
```

The same checks run on GitHub for every pull request
(`.github/workflows/ci.yml`), the tests on Python 3.10 and 3.12.

`test_stats.py`, `test_encoding.py` and `test_behaviour.py` are unit tests on
synthetic data: permutation p-values, the paired and one-sample tests,
split-half reliability, the 2 x 2 ANOVA (every F must equal the squared t of the
matching contrast), the cross-validation splits (every image held out equally
often), the scoring and models, the trial reduction and the figure helpers.
`test_release.py` checks the released files against the code and against each
other: that the dataset meets the assumptions the code makes, that the cached
predictivity is reproduced from the current data, that the hard-coded
reliabilities match a fresh estimate, that the Supplementary Tables and the
figures' source data agree with the caches, and that the main statistics
reported in the paper are reproduced from the data.

## Containers

`Dockerfile` and `Singularity.def` build the same environment: the code, the
notebooks and the Python packages locked in `uv.lock`, on
`python:3.12-slim-bookworm`. Both are generated from one specification by
`bash generate_images.sh`, which runs
[Neurodocker](https://github.com/ReproNim/neurodocker) through uvx; edit the
script, not the recipes. The dataset is not part of either image. No
pre-built images are published; build them from the definitions:

```bash
# Docker: build, then start JupyterLab on port 8888 with the dataset mounted
docker build -t facescenes .
docker run -p 8888:8888 -v "$PWD/data":/opt/facescenes/data:ro facescenes

# Apptainer: the image is read only, so bind a clone of this repository
# (with data/ in it) over /opt/facescenes to write results
apptainer build facescenes.sif Singularity.def
apptainer exec --bind "$PWD":/opt/facescenes facescenes.sif \
    python /opt/facescenes/code/run_behavioural_analyses.py
```

Both images start JupyterLab. To run something else in the Docker image, pass
`--entrypoint`, e.g. `--entrypoint python ... code/run_reliability.py`. Without
the mount, `python code/fetch_data.py` in a JupyterLab terminal downloads the
dataset into the running container.

## The two experiments

Both used the same 120 indoor scenes, each containing one face, and both were
gaze-contingent: the display ended the moment the observer fixated the face.
Participant groups were entirely separate.

**Face detection.** The whole scene is visible. 38 participants, 6 s limit.

**Moving window.** Only a small aperture around fixation is visible, so the
scene must be explored serially. 38 participants, 20 s limit.

On half the trials the scene appeared briefly as a preview before the search
display; on the other half a grey placeholder appeared instead. Which images
were previewed was counterbalanced between two participant groups, so each image
contributes preview data from one group and no-preview data from the other.

## How detection latency is defined

One definition, in `facescenes/data.py`, used by every figure and script:

* latency is the onset of the **last fixation** of the trial — the display was
  gaze-contingent, so that fixation is the one that landed on the face;
* fixation onsets are reconstructed as end-of-fixation minus duration;
* fixations beginning within **100 ms** of search-display onset are discarded,
  since they carry over from the preceding fixation-dot period;
* **timeout** trials are dropped: the face was never fixated, so no latency
  exists (0.5% of trials in the face detection task, 0.4% in the moving window);
* there is **no upper cut-off**.

## Gaze position at target onset

`facescenes/onset_gaze.py`, shown in `supplementary_onset_gaze.ipynb`. A control
for the alternative reading of the preview benefit: that the preview lets
observers move their eyes toward the upcoming face before the search display
appears, so they simply start closer to it. Every other analysis here looks only
at eye movements *after* onset and cannot address that.

Eye position at onset comes from the fixation in progress when the display
appeared. Recording starts at display onset, so that fixation's start time is
clipped to the event and its mean position reflects the oculomotor state
established beforehand — saccade latencies of at least ~150 ms rule out a
target-driven change within it. It is an event-based estimate: the fixation-dot
interval was not sampled, so no continuous pre-target trace exists.

The result is the same in both experiments. The preview does displace the eye
(+0.70° from centre in face detection, +3.24° in moving window, both p < .001),
but the displacement is **not toward the face** (−0.36°, p = .48; +0.07°,
p = .87), and the benefit survives adjusting single-trial latency for both
distances within participant (−12.8 → −13.8 ms; −376 → −265 ms). Granting the
confound the largest displacement toward the face the confidence intervals
allow, at the observed exchange rate between distance and latency, bounds what
it could explain at 7.2 ms of a 12.8 ms effect and 93 ms of a 376 ms one.

A trial needs a fixation of at least 40 ms beginning within 100 ms of onset to
supply the estimate. That drops 0.3% of trials in face detection and 18% in the
moving window, where the eye more often moves at once. The loss is balanced
across conditions (p = .73 and .48), and every result is unchanged when the
100 ms criterion is removed; `run_onset_gaze_control.py` reports both checks.

Both experiments are analysed on all successful trials. The face detection task
is additionally reported on trials reached in a single fixation, where latency is
one saccade launched from the onset position and the bound tightens to 3.8 ms.

## Predictivity

Encoding-model performance is the **signed** squared Spearman correlation
between predicted and observed held-out log latency, divided by the split-half
reliability of that condition.

The modelled quantity is each image's **mean log latency** (the log of its
geometric mean across participants), and the split-half reliability is computed
on exactly that quantity, so the noise ceiling matches what the models predict.
Averaging logs rather than logging an arithmetic mean also keeps an image's
value from being dominated by the occasional very slow trial, since there is no
upper latency cut-off. Figure 1d/e plots the same geometric means on log axes;
its R² ceiling is the product of the two conditions' Pearson split-half
reliabilities on that scale (`config.R2_CEILING`).

It is called *predictivity* rather than explained variance throughout, and both
words in that sentence carry weight. It is not the coefficient of determination
of a fitted model: it measures rank agreement between prediction and observation
on images the model never saw. And it is *signed*, because a cross-validated
model can predict worse than chance — negative values are meaningful and are
plotted as such rather than hidden by squaring.

Because the measure is rank-based, single-feature predictivity is invariant to
any monotone transform of a predictor.

## Exclusions

The exclusions are defined in `code/facescenes/config.py`
(`EXCLUDE_PARTICIPANTS`, `EXCLUDE_STIMULI`), each with a comment giving its
reason. No participants are excluded.

**Stimuli, face detection task only.** Ten Conferanceroom scenes: a
filename-spelling mismatch at collection time meant the experiment's pairing
failed and these were shown with a grey preview to *every* participant in *both*
groups, so they have no preview data at all. Because each image draws its preview
data from only one group, five broken images per group amounts to ten unusable
images. Plus two scenes whose pooled latency exceeds 3 SD (Artstudio_06,
Restaurant_16); neither exceeds 3 SD in the moving-window task, which excludes no
stimuli. That leaves 108 of 120 images.

## Inference

All statistical tests are at the **image** level, not across cross-validation
folds. The 100 folds are overlapping resamples of the same images, so their
spread is resampling noise rather than sampling variability; tested on
pure-noise data, fold-level tests reject 23–96% of the time instead of 5%. Fold
spread appears in the figures only as a descriptive error bar.

## Cross-validation

Balanced repeated two-fold splits over images: the images are shuffled, cut in
half, and each half serves as the test set once. With 50 repetitions every image
is held out exactly 50 times, over 100 folds. This is preferable to 100
independent random half-splits, where each image is held out a random number of
times. The default is 5000 permutations.

## Apparatus

3840 × 2160 display, 600 × 340 mm, 60 Hz, viewed from 460 mm, giving 51.383
pixels per degree. Scenes were presented at 2400 × 1800 pixels. EyeLink 1000,
monocular. These values are read from the session files and recorded in
`config.DISPLAY`.

The pixels-per-degree figure is the scale at the centre of the display, which is
the right one for distances measured from fixation; dividing total pixels by
total degrees would give ~58, but that averages over a tangent-distorted axis.
The 600 mm panel width is the display's own EDID report and has not been
measured, so the absolute degree values in the onset-gaze control carry that
uncertainty. No inference does: the constant cancels from every paired test and
rescales the regression slopes and the orienting bound by the same factor.

## Notes on the data

`data/` follows BIDS with one documented departure: `onset` in the event tables
is relative to the onset of the search display on that trial, not to the start of
the run. The raw files record only per-trial timings, so a run-relative onset
could be produced only by assuming fixed inter-trial intervals, and inventing
that precision seemed worse than the deviation. This is stated in the event
sidecars.

Participants are pseudonymised. The raw folders were named with initials and a
number; the mapping to `sub-001` … `sub-076` is deliberately not included.
`sub-001`–`sub-038` did the face detection task and `sub-039`–`sub-076` the
moving-window task.
`data/participants.tsv` lists each participant's ID and task; age, sex,
handedness and corrective lenses are released per task only, in
`data/participants_summary.tsv`, written by `export_bids.py`.

## Licence

The code (`code/`, `notebooks/`) is released under the MIT License; see
`LICENSE`. The data in `data/` are licensed as stated in
`data/dataset_description.json`.
