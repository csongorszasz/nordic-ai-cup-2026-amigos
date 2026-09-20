# Provenance, attribution and licensing

This file accompanies our Nordic AI Cup 2026 submission. It records where every
third-party asset came from, what licence it carries, which of those licences we
consider settled and which we consider a grey area, and how we kept training data
separate from validation and evaluation data.

We would rather over-disclose than have a reviewer find something unexplained.

## What we built

| Challenge | Our solution | Entry point |
| --- | --- | --- |
| Drone Flyby | YOLO11s detector trained entirely on synthetic 4K renders, plus a camera-steering policy | `drone-flyby/src/api.py`, `drone-flyby/STATUS.md` |
| Medical Appointment | Local instruction-tuned LLM (Gemma 4) over Whisper transcripts, answering ten yes/no questions with word-aligned evidence quotes | `medical-appointment/example.py`, `medical-appointment/docs/state-of-knowledge.md` |
| Survival Simulator | Learned policy with imitation warm-start and PPO | `survival-simulator/README.md` |

Each challenge folder has its own README plus a running experiment log
(`STATUS.md` or `docs/tries.md`) recording what we measured, including the
approaches that failed.

## Third-party data

**Helsinki 3D reality mesh (2017).** Background terrain for the drone-flyby
synthetic training data. © City of Helsinki, licensed **CC BY 4.0**.
Downloaded by `drone-flyby/training/fetch_city_mesh.py`, which documents the
source URL and tile list. The committed sample scene in
`drone-flyby/data/helsinki/` is a derivative render of this mesh and is
redistributed here under the same CC BY 4.0 terms, with attribution to the
City of Helsinki.

**Organiser-supplied data.** The frames, annotations and API templates that came
with the competition repository are unchanged from upstream except where a
challenge README says otherwise.

## Third-party 3D models

The drone-flyby synthetic data generator renders 3D meshes over the Helsinki
terrain. The authoritative per-class record is `drone-flyby/models/<class>/credits.txt`;
`drone-flyby/models/MODELS.md` explains how each was chosen and its caveats. The
original meshes are **not** redistributed (they are gitignored); what is committed
is the recoloured, rescaled derivative used by the generator, in
`drone-flyby/datasets/model_match/_baked/`.

We deliberately rejected any asset under CC BY-NC, CC BY-NC-ND, "Editorial Uses
Only", or with no licence set, and any view-only Sketchfab model. `credits.txt`
records the rejections alongside the choices.

### Settled: Creative Commons Attribution (CC BY)

Commercial use permitted, attribution required — given here and in `credits.txt`.

| Class | Model | Author |
| --- | --- | --- |
| `helicopter` | Mi-28N Havoc | Rukh3D |
| `jammer` | GTK Boxer armoured personnel carrier | Muhamad Mirza Arrafi |
| `large_launcher` | MIM-104 Patriot air defence system | Chenzoss |
| `medium_launcher` | MIM-23 Hawk SAM launcher | Dominik Biały |
| `medium_plane` | P-51 Mustang | UlissesVinicios |
| `mine_roller` | PT-34-85 mine-clearing vehicle | 42manako |
| `small_launcher` | Rapier missile launcher | Revada |
| `small_plane` | Yak-9 | Starpovich |
| `small_tower` | Old Wooden Watchtower (House 3) | Blenderust |
| `tank` | Low Poly Churchill VII | LowPolyCount |

`small_tower` additionally uses a procedural tower we generated ourselves
(`drone-flyby/training/procedural_models.py`); no third-party rights apply.

### Unverified: CGTrader "free"

CGTrader free models are usually Royalty Free but some are Editorial Uses Only,
and the licence line cannot be read programmatically. We recorded the source URL
but did **not** confirm the licence text for these three:

| Class | Model |
| --- | --- |
| `hangar` | NATO aircraft shelter v2 |
| `jet_plane` | A-7 Corsair II |
| `large_tower` | Old wooden watchtower (low poly) |

### Grey area: third-party intellectual property

We are flagging these three explicitly rather than leaving them to be discovered.
All three are disclosed in `drone-flyby/models/MODELS.md`, which we wrote during
the hackathon, before this review was requested.

| Class | Model | Problem |
| --- | --- | --- |
| `condor` | "Battlefield 2042 Condor" (CGTrader) | The design is EA/DICE's and the mesh appears to be extracted from the game. Whatever the upload page states, the uploader has no rights to grant. |
| `spacecraft` | Star Wars TIE/in Interceptor (Sketchfab, marked CC BY) | The uploader cannot license Lucasfilm/Disney intellectual property under CC BY. |
| `ta-ta` | Star Wars AT-AT Walker (CGTrader "free") | Same: Lucasfilm/Disney intellectual property. |

Two points of context, offered as context and not as a defence:

1. The `spacecraft` and `ta-ta` object classes **in the competition's own dataset**
   are a TIE fighter and an AT-AT. We matched meshes to the classes we were given,
   so the grey area is not one we introduced.
2. `MODELS.md` records our own mitigation, written at the time: if anything we
   built were to be published, swap the Condor for a generic tiltrotor and
   re-render. We have not done so here because the submitted model was trained
   with these meshes, and substituting them now would mean shipping something
   other than the artefact that actually produced our score.

If the jury would prefer a bundle with these three derivatives removed, we can
provide one; the detector weights would be unchanged, but the synthetic data
generator would no longer run end to end.

## Training, validation and evaluation hygiene

**Drone Flyby.** The submitted detector, `allbg_e25`
(`runs/synth400_11s_allbg_0919-2339/weights/epoch25_int8_openvino_model`), was
trained on 400 purely synthetic 4K frames and nothing else. No recorded
validation or evaluation frame was used as training data for it.

We did record the evaluator's own views during validation runs, and hand-labelled
the Copenhagen flight, in order to measure our models offline. Those labels live in
`drone-flyby/datasets/copenhagen_test/` and are marked, in the repository's
`.gitignore`, as test data only.

One exception is disclosed in full: `drone-flyby/training/copenhagen_train.py` cuts
YOLO training crops from that recorded validation flight. It was written at
2026-09-20 12:02 CEST, after the submitted model had been selected (STATUS.md,
"THE DECISION (2026-09-20 ~10:40 CEST)") and after our single evaluation attempt had
been spent. The model it produced is a separate run directory
(`runs/synth400_11s_cph_0920-1203`) and was never served to the evaluator or the
validator. The submitted weights predate that script by about twelve hours. The
script's own docstring repeats this; the git history corroborates the ordering. We
kept it in the repository rather than deleting it because it is a legitimate
post-hoc study of the sim-to-real gap and we did not want an omission to look like
concealment.

**Medical Appointment.** Request captures are debug-only and were never used as a
training source; `medical-appointment/docs/next-steps.md` records this as a
standing rule. Few-shot demonstrations are drawn from training conversations only
and are held disjoint from whatever is being scored.

**Survival Simulator.** Holdout seeds are version-controlled rather than secret,
so our reported holdout numbers can be re-run.

## Not in this repository

Large or regenerable artefacts are gitignored, so the repository is code and small
data. Missing pieces and how to rebuild them:

- Model weights and training runs (`drone-flyby/runs/`, `*.pt`) — retrain with
  `drone-flyby/training/train_yolo.py`, or fetch with `drone-flyby/training/fetch_data.sh`.
- Original downloaded 3D meshes (`drone-flyby/models/**`) — source URLs are in each
  `credits.txt`.
- Helsinki terrain tiles (`drone-flyby/backgrounds/`) — `fetch_city_mesh.py`.
- Held-out synthetic test flights (`drone-flyby/data/holdout/`, `data/woodland/`) —
  `synth_dataset.py` and `holdout_scene.py`; ~6.5 GB.
- Recorded evaluator views — `fetch_data.sh`.

## Infrastructure

The drone-flyby endpoint was served from an Azure VM; deployment scripts in
`drone-flyby/` contain that VM's public IP address and its `azureuser` login,
which used key-based authentication. No credentials, keys or tokens are committed
anywhere in this repository. The medical endpoint was served from NTNU's IDUN
cluster behind a tunnel, as described in `medical-appointment/docs/local-serving.md`.
