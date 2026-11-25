# Scene previews and face detection: two gaze-contingent eye-tracking experiments

Fixation-level eye-tracking data from two experiments on how a brief preview of
a scene changes how quickly observers find a face in it, with the image
features, model results, supplementary tables and figure source data derived
from them.

Analysis code: https://github.com/VCCN-lab/scene-context-face-detection

## Experiments

Both experiments used the same 120 indoor scenes, each containing one face, and
both were gaze-contingent: the display ended the moment the observer fixated
the face. The participant groups were separate.

* **Face detection** (`task-facedetection`, `sub-001` to `sub-038`): the whole
  scene is visible; 6 s limit.
* **Moving window** (`task-movingwindow`, `sub-039` to `sub-076`): only a small
  aperture around fixation is visible; 20 s limit.

On half the trials the scene appeared briefly as a preview before the search
display; on the other half a grey placeholder appeared instead. Which images
were previewed was counterbalanced between two participant groups.

Display: 3840 x 2160 pixels, 600 x 340 mm, 60 Hz, viewed from 460 mm (51.383
pixels per degree at the centre). Scenes were shown at 2400 x 1800 pixels.
Eye tracker: EyeLink 1000, monocular.

## Contents

```
dataset_description.json, participants.tsv/.json
task-<task>_events.json        column descriptions of the event tables
sub-<label>/eyetrack/          one event table per participant, one row per fixation
stimulus_order.tsv             image index, scene, no-face version and face mask
stimuli/masks/                 binary face masks, one per scene
derivatives/
  stimulus_features/           image predictors and CNN activations
  encoding/                    encoding-model results and permutation tests
  tables/                      the Supplementary Tables
  source_data/                 the numbers plotted in every figure panel
```

Each folder in `derivatives/` is a BIDS derivative dataset with its own
`dataset_description.json`; every file in it has a JSON sidecar of the same
name describing its columns or arrays. Tables are tab-separated with `n/a` for
missing values.

`onset` in the event tables is relative to the onset of the search display on
that trial, not to the start of the run: the raw files record only per-trial
timings. This is stated in the event sidecars.

The scene photographs are not included. They are adapted from the Places365
dataset, whose image copyright stays with the original owners, and are
available from the OSF project (https://doi.org/10.17605/OSF.IO/5AYX8). The
`stim_file` column of the event tables names them.

Participants are pseudonymised.

## Licence

CC-BY-4.0. Please cite the accompanying paper.
