# Model C labelling rules (YOLO26 infrastructure, instance segmentation)

Rules agreed in step 12 (2026.10.19). Follow them on every image so labels stay consistent. If a case is not covered, skip the object and note it here at the end of the session.

## Classes

Class ids follow `classes` in `ml/configs/infrastructure_yolo26.yaml`. Hotkeys in Label Studio are 1 to 4.

| id | class | label it | do not label |
|----|-------|----------|--------------|
| 0 | `gate_exit` | Openings people pass through: gates and gaps in fences, railings or walls, park and court entrances, stair or escalator heads of footbridges, subway entrances, building entrances where visible from above. Draw the opening itself as a small polygon (width of the gap, short depth). | Road junctions, zebra crossings with no barrier gap, open ground with no barrier around it. |
| 1 | `barrier` | Fences (court fences included), railings along pavements and road medians, crowd control barriers, continuous hedge lines people cannot step over. | Painted lines, kerbs, single planters (those are obstruction). |
| 2 | `wall` | Building footprints (the roof outline seen from above) and solid free standing walls. Buildings that touch with no ground visible between them may be one polygon. | Covered footbridges, station canopies and other roofs people walk under. |
| 3 | `obstruction` | Stalls and market tents, kiosks, all vehicles (cars, taxis, buses, trucks, vans, also in traffic lanes; rule changed by the user after batch 1 review), single planters, benches, fountains, statues, large bins. | Trees, umbrellas held by people, bicycles and motorbikes under ~15 px. |

Crowds are never labelled: an area packed with people is not an `obstruction`, even when goods lie among them (crowd density is Model A's job). Label only the physical objects you can see, such as stall roofs and tables.

Not labelled at all: trees (people walk under them), water and ponds, footbridges and canopies (also the railings along a footbridge deck), roads, people.

Cases met in batch 1: construction hoarding along a pavement is `barrier`; a fence or railing in a road median is `barrier`; market umbrellas fixed over a stall belong to the stall polygon; a raised planter box is `obstruction`, a long planter strip along a pavement is `barrier`.

## How to draw

* Polygons follow the outline at full resolution; zoom in for edges. Enough points to follow the shape, no more.
* One polygon per object. Exceptions: touching buildings (one `wall` polygon) and touching stalls with no walkable gap between them (one `obstruction` polygon per row; a gap people can walk through starts a new polygon).
* Thin objects (railings, fences, hedges): draw a narrow polygon along the line, at least about 4 px wide. A railing that bends or breaks at a gap becomes separate polygons, and the gap is a `gate_exit` if people pass through it.
* Skip objects whose longest side is under about 15 px (single poles, bollards, small bins).
* Occlusion: label only the visible part. Do not guess what is under trees or awnings.
* Image border: run the polygon to the edge of the image.
* Dusk and night frames: label only outlines you can see clearly.
* Unsure about the class after a few seconds: skip the object rather than guess.
* An image with nothing to label is still submitted (it becomes a background image).

## Workflow

Images: DroneCrowd frames picked with `pick_frames.py` (`sheet` makes contact sheets, `copy clip:frame ...` copies frames to `data/infra_labels/images/`). Own footage frames go into the same folder later. Everything under `data/` is gitignored.

Start Label Studio (own venv `.venv-label`, Label Studio 1.23.2; its database lives in `data/label_studio/`):

```powershell
$env:LABEL_STUDIO_BASE_DATA_DIR="$PWD\data\label_studio"
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD\data\infra_labels"
.\.venv-label\Scripts\label-studio start --port 8080 --no-browser
```

Then open http://localhost:8080. The project `Sentinel Model C` uses `label_studio_config.xml` as its labelling interface and a Local Files source storage on `data/infra_labels/images` (absolute path, "Treat every bucket object as a source file" on). After adding images to the folder, press Sync on that storage.

Drafts (optional, saves most of the clicking): `sam_draft.py tiles <stems>` draws a pixel grid on each image (`data/infra_labels/tiles/`), prompts per object are written to a YAML file in `data/infra_labels/prompts/` (box for SAM 2.1, line for thin railings, poly or rect as given; see the script docstring), and `sam_draft.py run <prompts.yaml>` turns them into Label Studio predictions (`data/infra_labels/images/_drafts/`, overlays in `drafts_overlay/`). They are imported as predictions with model version `sam_draft` through `/api/predictions`. A draft is never a label: open each task, fix or delete polygons, add what is missing, then Submit.

Model drafts (from step 13 on, the default): a `yolo26s-seg` trained on the labels so far drafts the next batch. Run `ls_to_yolo.py` on the latest export, copy the new frames in, then `push.py labels -m msg` (uploads `images/*.jpg` + `labels/*.txt` as the private Kaggle dataset `sentinel-infra-labels`), wait until `kaggle datasets status` says ready, set `NAME` in `kaggle/infra_draft/run.py` and `push.py kernel eval/infrastructure_yolo26/kaggle/infra_draft`. The kernel trains on the labelled images and predicts every image without a label file. Download with `kaggle kernels output ahmar004/sentinel-infra-draft -p runs/kaggle/infra_draft/<NAME>`, copy `_drafts/<NAME>.json` into `images/_drafts/` and import it as predictions (model version `draft_<NAME>`). Same script locally on CPU: `draft_model.py train --out DIR` and `draft_model.py predict DIR/weights/best.pt --name NAME` (about 2 min per epoch on 30 images at imgsz 1280).

Missed vehicles: `sam_draft.py vehicles data/infra_labels/export.json --extra <boxes.yaml>` drafts vehicles not yet covered by an obstruction polygon, from the COCO model (from straight above it calls cars "cell phone", so that class counts) plus SAM on hand boxes per image (the COCO model misses most top down cars, so hand boxes do most of the work). Output `images/_drafts/vehicles.json` + `vehicles_overlay/`.

Review checklist per image: every building footprint is `wall`; railing lines follow the real railing; no label on footbridges, trees or roads; gaps in railings that people use are `gate_exit`; dark areas in night frames are only labelled where outlines are clear.

Export: Project, Export, JSON, save as `data/infra_labels/export.json`, then

```powershell
.\.venv\Scripts\python eval/infrastructure_yolo26/ls_to_yolo.py data/infra_labels/export.json --overlays data/infra_labels/overlays
```

writes YOLO segmentation labels to `data/infra_labels/labels/`, prints polygons per class and draws the labels back onto the images in `data/infra_labels/overlays/` for a visual check.

## Training split and fine tune (step 14)

Train and val are split by scene: clips that show the same place (`data/manifests/infra_scene_groups.yaml`, every clip once) go wholly to one side. `split_scenes.py` (seed 0) writes `data/manifests/infra_split.json` with about 20% of the images and of each class in val, at least one instance of every class and one night clip. Rerun it after new labels, and add new clips (own footage) to the groups file first.

Fine tune: `push.py labels`, `push.py src`, wait until both datasets are ready, then `push.py kernel eval/infrastructure_yolo26/kaggle/infra_train` (`EPOCHS` 3 for a pipeline check, 150 for the full run). It runs `train.py` (yolo26s-seg from COCO, imgsz 1280, batch 4, patience 50) and writes the Ultralytics run (`weights/best.pt`, `results.csv`, curves) to `run/`. Download with `kaggle kernels output ahmar004/sentinel-infra-train -p runs/kaggle/infra_train/<name>`.
