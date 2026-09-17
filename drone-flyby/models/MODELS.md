# 3D model hunt — status list

Why: the same 3D models appear in every scene, only the ground under them changes.
If we have the models we can render each class from any angle, at any light, and stop
depending on 25 frames of Helsinki. Sprite cut-outs get us part of the way; models get
us the rest.

**Two of the classes are Star Wars assets** (`spacecraft` is a TIE fighter, `ta-ta` is an
AT-AT walker), so the organisers were shopping for free models like we are. Sketchfab
"downloadable + free" is the first place to look for the rest.

- Reference sheet of every class as it appears from above: [`reference_sheet.png`](reference_sheet.png)
  (one sprite per class, nearest-neighbour upscaled). Per-class sheets are in `sprites/review/`
  after `python training/extract_sprites.py`.
- Sizes below are measured from the ground-truth boxes at 0.21 m/px, so they are the real
  extent of the object on the ground, longest side first. Use them to sanity-check a
  candidate: a "tank" that is 30 m long is the wrong model.
- Priority is worst first: low AP in our last local run, few sprites, or a sprite bank
  too rough to paste convincingly.

## Where to put what you find

```
models/<class>/<anything>.glb     # or .obj/.fbx; .glb keeps textures in one file
models/<class>/credits.txt        # source URL, author, licence, date downloaded
```

`.glb` and friends are gitignored (too big); the credits file is committed. Licence rule:
CC0 or CC-BY is fine for training and we credit the author. Anything "editorial use only",
"no derivatives", or a marketplace EULA is not worth the argument — skip it.

## Status

| # | Class | Size (m) | Sprites | Local AP | Looks like | Status |
|---|---|---|---|---|---|---|
| 1 | `mine_roller` | 12.6 x 11.1 | 2 | 0.505 | camo cylinder on a vehicle, mine-clearing roller drum | **wanted** — worst AP, only 2 sprites |
| 2 | `medium_launcher` | 9.9 x 9.3 | 10 (3 suspect) | 0.505 | small dark launcher vehicle, hard to read | **wanted** — worst AP, rough sprites |
| 3 | `small_launcher` | 6.3 x 4.6 | 25 | 0.810 | pale green box on a flatbed, tiny | **wanted** — too small to cut cleanly |
| 4 | `ta-ta` | 6.7 x 3.6 | 25 | 0.758 | four-legged walker (AT-AT) | **wanted** — search "AT-AT", likely the same free asset |
| 5 | `large_tower` | 13.9 x 12.6 | 19 | 0.771 | camo cylinder/mast with a light band | **wanted** |
| 6 | `hangar` | 38.9 x 25.0 | 6 | 0.832 | dark curved roof, Quonset/arched hangar | **wanted** — only 6 sprites, biggest false-positive source (roads/roofs) |
| 7 | `medium_plane` | 11.8 x 10.3 | 5 | 1.000 | dark green single-prop aircraft, WW2 style | **wanted** — only 5 sprites |
| 8 | `small_plane` | 10.5 x 9.0 | 9 | 0.857 | prop aircraft, red nose, green camo | **wanted** |
| 9 | `condor` | 36.3 x 34.9 | 11 (3 suspect) | 1.000 | big grey X-shaped four-engine aircraft/cargo drone | **wanted** |
| 10 | `small_tower` | 12.6 x 12.0 | 20 | 0.909 | green square platform on a pale base | wanted |
| 11 | `jammer` | 9.0 x 6.7 | 13 | 0.921 | green boxy truck with a flat box body | wanted |
| 12 | `helicopter` | 24.4 x 19.7 | 19 (4 suspect) | 1.000 | dark green helicopter, rotors visible | wanted |
| 13 | `tank` | 10.5 x 9.9 | 25 | 0.930 | camo tank, barrel forward | wanted |
| 14 | `large_launcher` | 31.3 x 23.1 | 25 | 1.000 | camo TEL, long missile tube on a truck | wanted |
| 15 | `jet_plane` | 17.0 x 16.2 | 22 | 1.000 | white/grey swept-wing jet | wanted |
| 16 | `spacecraft` | 10.3 x 9.2 | 23 | 0.882 | TIE fighter/interceptor | **have** — [Sketchfab TIE-in interceptor](https://sketchfab.com/3d-models/star-wars-tiein-interceptor-80171ec2930b4949836bcf24d9694c41), needs `credits.txt` + download into `models/spacecraft/` |

Local AP is from the Helsinki set, where the model memorised the scene, so treat 1.000 as
"nothing to learn here yet" rather than "solved". Live validation was 0.0069 overall.

## Places to look

| Source | Licence | Notes |
|---|---|---|
| [Sketchfab](https://sketchfab.com/search?features=downloadable&licenses=322a749bcfa841b29dff1e8a1bb74b0b&type=models) | CC-BY / CC0 filters | Where the Star Wars ones almost certainly came from. Filter downloadable + licence. |
| [Poly Pizza](https://poly.pizza) | CC-BY / CC0 | Ex-Google Poly library, low-poly, lots of vehicles and aircraft. |
| [Quaternius](https://quaternius.com) | CC0 | Clean low-poly military/vehicle packs, whole sets at once. |
| [Kenney](https://kenney.nl/assets) | CC0 | Tanks, towers, aircraft in a consistent style. |
| [OpenGameArt](https://opengameart.org) | mixed, check per asset | Older, uneven quality. |
| [NASA 3D Resources](https://nasa3d.arc.nasa.gov/models) | public domain | Good for aircraft/spacecraft shapes. |

## How to check a candidate before spending time on it

1. Open it in any viewer, look straight down. From above is the only view that matters.
2. Compare against the class in `reference_sheet.png`: silhouette first, colour second.
   Scale and texture we can adjust; the wrong outline stays wrong.
3. Check the real extent against the Size column.
4. Drop it in `models/<class>/`, write `credits.txt`, tick it off here.

## After we have them

Not built yet: a `trimesh`/`pyrender` script to render each model top-down over a range of
yaw, pitch (the camera is near-nadir but not exactly), sun angle and scale, producing RGBA
sprites in the same layout as `sprites/`, so `training/synth_dataset.py` can paste them
straight onto the aerial backgrounds. Worth writing once 4 or 5 models are in, not before.
