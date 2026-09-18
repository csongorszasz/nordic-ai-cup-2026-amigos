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
| 1 | `mine_roller` | 12.6 x 11.1 | 2 | 0.505 | camo cylinder on a vehicle, mine-clearing roller drum | **downloaded** — PT-34-85, see below |
| 2 | `medium_launcher` | 9.9 x 9.3 | 10 (3 suspect) | 0.505 | dark cluster with barrels: a Hawk launcher | **downloaded** — MIM-23 Hawk, see below |
| 3 | `small_launcher` | 6.3 x 4.6 | 25 | 0.810 | featureless pale green blob at 23x30 px; real shape unknowable | **downloaded** — Rapier launcher as a stand-in, see below |
| 4 | `ta-ta` | 6.7 x 3.6 | 25 | 0.758 | four-legged walker (AT-AT) | **candidate found** — AT-AT (CGTrader), licence to check |
| 5 | `large_tower` | 13.9 x 12.6 | 19 | 0.771 | brown/camo rectangular structure, pale band across the middle | **2 candidates** — CGTrader watchtower preferred, see below |
| 6 | `hangar` | 38.9 x 25.0 | 6 | 0.832 | dark curved roof, Quonset/arched hangar | **downloaded + fitted** — NATO shelter, IoU 0.91 (`--drop Cube.032 Lamps`); Sketchfab hangar kept as alt |
| 7 | `medium_plane` | 11.8 x 10.3 | 5 | 1.000 | dark green single-prop aircraft, WW2 style | **downloaded** — P-51 Mustang (CC-BY), see below |
| 8 | `small_plane` | 10.5 x 9.0 | 9 | 0.857 | prop aircraft, red nose, green camo | **downloaded** — Yak-9, see below |
| 9 | `condor` | 36.3 x 34.9 | 11 (3 suspect) | 1.000 | big grey X-shaped four-engine aircraft/cargo drone | **candidate found** — BF2042 Condor, licence to check |
| 10 | `small_tower` | 12.6 x 12.0 | 20 | 0.909 | dark green square roof on a pale base | **fitted: procedural tower** (`training/procedural_models.py tower`, no licence needed), IoU 0.85, colour error 25: platform + raised cabin, cabin leans off the platform like the real one. Watchtower kept as alt (roof covered the platform) |
| 11 | `jammer` | 9.0 x 6.7 | 13 | 0.921 | green boxy truck with a flat box body | **downloaded** — GTK Boxer, see below |
| 12 | `helicopter` | 24.4 x 19.7 | 19 (4 suspect) | 1.000 | dark green helicopter, rotors visible | **downloaded + fitted** — Mi-28N, IoU 0.51 with `--up z --drop rotor_Body`: fuselage pose right, the real rotor turns between frames, so add it back at random angles when generating |
| 13 | `tank` | 10.5 x 9.9 | 25 | 0.930 | camo tank, barrel forward | **downloaded + fitted** — Churchill VII, IoU 0.82; hull fits but the short gun misses the sprites' long barrel, and camo is not reproduced |
| 14 | `large_launcher` | 31.3 x 23.1 | 25 | 1.000 | camo TEL, long missile tube on a truck | **downloaded + fitted** — Patriot (Chenzoss), IoU 0.69; pose right, loses on the raised missile canisters and camo |
| 15 | `jet_plane` | 17.0 x 16.2 | 22 | 1.000 | white/grey swept-wing jet | **downloaded + fitted** — A-7 Corsair II, IoU 0.84, pose matches all 22 frames |
| 16 | `spacecraft` | 10.3 x 9.2 | 23 | 0.882 | TIE fighter/interceptor | **downloaded + fitted** — TIE/in Interceptor, IoU 0.83, same pose on all frames |

Local AP is from the Helsinki set, where the model memorised the scene, so treat 1.000 as
"nothing to learn here yet" rather than "solved". Live validation was 0.0069 overall.

## Candidates so far

Recorded with `python training/add_model.py <class> <url>`, which looks the model up on
Sketchfab and writes both `models/registry.json` and `models/<class>/credits.txt`. Add every
candidate as you find it, even an uncertain one: the URL is the part that gets lost.

| Class | Model | Author | Licence | Status |
|---|---|---|---|---|
| `mine_roller` | [PT-34-85 mine clearing vehicle](https://sketchfab.com/3d-models/pt-34-85-mine-clearing-vehicle-b845931a1ba64b318ad01c3968bba18f) | 42manako | CC-BY (credit required, commercial ok) | downloaded (`models/mine_roller/pt34/`) — T-34 hull with roller drum, extent close to the measured 12.6 x 11.1 m |
| `spacecraft` | [Star Wars: TIE/in Interceptor](https://sketchfab.com/3d-models/star-wars-tiein-interceptor-80171ec2930b4949836bcf24d9694c41) | Daniel | CC-BY (credit required) | candidate — silhouette matches the sprites |
| `medium_launcher` | [MIM-23 Hawk SAM (game-ready)](https://sketchfab.com/3d-models/mim-23-hawk-sam-air-defence-system-game-ready-8728909b6ce24ef8baeffabbf5bae8f4) | Dominik Biały | CC-BY (credit required) | downloaded (`models/medium_launcher/hawk/`) — the dark cluster with barrels in the sprites; scale to 9.9 x 9.3 m. Was first filed under `small_launcher` by mistake |
| `hangar` | [Hangar](https://sketchfab.com/3d-models/hangar-c3e821610c644ade9878aa56af867e05) | Vitor Augusto | CC-BY (credit required) | candidate — scale it to the measured 38.9 x 25.0 m; this class is 185x119 px, so the roof shape does matter |
| `large_tower` | [Old Wooden Watchtower (House 3)](https://sketchfab.com/3d-models/old-wooden-watchtower-house-3-49b77f82b0944d5188c04c3fc205a499) | Blenderust | CC-BY (credit required) | candidate — scale to 13.9 x 12.6 m. From above the sprites read as a brown/camo rectangular structure with a pale band across the middle, more camouflaged shelter than open tower, so check the render before trusting it |
| `small_tower` | [Old Wooden Watchtower (House 3)](https://sketchfab.com/3d-models/old-wooden-watchtower-house-3-49b77f82b0944d5188c04c3fc205a499) | Blenderust | CC-BY (credit required) | candidate — Juan's pick for this class: tint green, scale to 12.6 x 12.0 m |
| `large_tower` | [Old wooden watchtower (low poly)](https://www.cgtrader.com/free-3d-models/exterior/other/old-wooden-watchtower-low-poly) | CGTrader | free — **check the licence line** | candidate, Juan's preferred — closer to the sprites than the Sketchfab watchtower |
| `large_launcher` | [MIM-104 Patriot Air Defense System](https://sketchfab.com/3d-models/mim-104-patriot-air-defense-system-977f1f08a2014da99138a3364b7a56cd) | Chenzoss | CC-BY (credit required) | **preferred** — downloadable, 120k faces, scale to 31.3 x 23.1 m |
| `large_launcher` | [MIM-104 Patriot SAM](https://sketchfab.com/3d-models/mim-104-patriot-surface-to-air-missile-sam-7a64d0af78514a159877edab1ab2bccb) | Muhamad Mirza Arrafi | CC-BY (credit required) | alternative — 13.6k faces, lighter. A third Patriot Juan found first is view-only and was rejected |
| `helicopter` | [Mi-28N Havoc](https://sketchfab.com/3d-models/mi-28n-havoc-3e80c95bbadf46abbafba9d9a08afc68) | Rukh3D | CC-BY (credit required) | candidate — 25k faces, scale to 24.4 x 19.7 m. Replaces a view-only Mi-28 with no licence set |
| `jet_plane` | [A-7 Corsair II (with shelter bonus)](https://www.cgtrader.com/free-3d-models/aircraft/military-aircraft/a7-corsair-ii-aircraft-with-weapons-and-shelter-bonus) | CGTrader | free — **check the licence line** | candidate — scale to 17.0 x 16.2 m. The bundled shelter may also serve `hangar` |
| `medium_plane` | [P-51 Mustang](https://sketchfab.com/3d-models/p-51-mustang-36f0f3e71d2a4c18b479db1ae8f9e7a7) | UlissesVinicios | CC-BY (credit required) | not used — 5.7k faces. Downloaded instead: [an.s055163's P-51](https://sketchfab.com/3d-models/p-51-mustang-dbb4a717a4c141f9bf0869bf1ce74529) (CC-BY, 9.3k faces) in `models/medium_plane/p51/`. The Mustang first found is CC BY-NC-ND and was rejected |
| `small_plane` | [Yak-9](https://sketchfab.com/3d-models/yak-9-08ea8d09a2b943bead5814e1aa712684) | Starpovich | CC-BY (credit required) | downloaded (`models/small_plane/yak9/`) — 12k faces; the sprites have a red nose and red on the tail, scale to 10.5 x 9.0 m |
| `tank` | [Low Poly Churchill VII](https://sketchfab.com/3d-models/low-poly-churchill-vii-tank-ww2-bd140fe8aa32438ab712ce47762d968c) | LowPolyCount | CC-BY (credit required) | **preferred** — the [Black Prince](https://sketchfab.com/3d-models/black-prince-aa7487b728e34f5ea2196c57585182cb) Juan matched to the sprites is built on this hull, but is view-only with no licence, so it stays as a shape reference. 10k faces, scale to 10.5 x 9.9 m, repaint camo |
| `tank` | [Tank T-10M](https://sketchfab.com/3d-models/tank-t-10m-9aeda33a945c42f0bdebe3d1ef91da06) | yanix | CC-BY (credit required) | fallback — 500k faces |
| `ta-ta` | [Star Wars AT-AT Walker](https://www.cgtrader.com/free-3d-models/space/other/star-wars-at-at-walker-68bd9d4c-a316-4517-8b6c-b6feb3b1da77) | CGTrader | free — **check the licence line** | candidate — 32x17 px on screen, so the silhouette of the legs is all that matters. Star Wars IP, same grey area as the TIE |
| `small_launcher` | [Rapier Missile Launcher](https://sketchfab.com/3d-models/rapier-missle-launcher-047f034901d24a94af4cb090d671a72b) | Revada | CC-BY (credit required) | downloaded (`models/small_launcher/rapier/`) — a stand-in: the class is a 23x30 px blob, so any launcher of the right footprint works. Scale to 6.3 x 4.6 m, tint pale green |
| `jammer` | [Ukrainian Patria AMV](https://sketchfab.com/3d-models/ukrainian-patria-amv-49992289be014290b54f1e8345ac78b3) | 42manako | CC-BY (credit required) | not downloaded — successor of the XA-180 Juan picked, same author, but the XA-180 is CC BY-NC and was rejected. Scale to 9.0 x 6.7 m. Used instead: [GTK Boxer](https://sketchfab.com/3d-models/gtk-boxer-armored-personnel-carriers-e74b3f67d954401ab825bb1053dfd194) by Muhamad Mirza Arrafi (CC-BY), a literally box-shaped hull, in `models/jammer/boxer/` |
| `hangar` | [NATO aircraft shelter v2](https://www.cgtrader.com/free-3d-models/military/other/nato-aircraft-shelter-v2) | CGTrader | free — **check the licence line** | candidate — hardened shelter with an arched roof, closer to the sprites than a plain hangar |
| `condor` | [Battlefield 2042 Condor](https://www.cgtrader.com/free-3d-models/military/military-vehicle/battlefield-2042-condor-flight) | CGTrader | free — **check the licence line**, and see the game-asset caveat | candidate — quad-rotor VTOL, matches the X-shaped 4-engine sprite |

**Two things to check on any CGTrader model**, because they are not uniform like Sketchfab's:

0. **That it is actually downloadable.** Sketchfab shows view-only models in the same search
   results; the API field is `isDownloadable`, and on the page it is the absence of a Download
   button. `add_model.py` records it, so check the output.
1. **No "NoDerivs" or "NonCommercial".** Sketchfab's CC BY-NC-ND and similar variants look
   like CC-BY in a hurry. NoDerivs forbids exactly what we do (scale, tint, render into
   sprites), and a competition with prizes is arguably commercial. `add_model.py` prints the
   full licence label; anything other than plain CC Attribution, CC0 or Sketchfab Standard
   gets rejected.
2. **The licence line on the page.** CGTrader free models are usually "Royalty Free", but some
   are "Editorial Uses Only", which excludes training data. Record what it says with
   `--licence`; `add_model.py` cannot read CGTrader.
3. **Whether it is a game rip.** "Battlefield 2042 Condor" is EA/DICE's design, extracted from
   their game; the uploader has no rights to grant, whatever the page's licence box says. The
   same is true of the organisers' own `spacecraft` and `ta-ta`, which are Star Wars assets, so
   this is a competition-wide grey area rather than something we invented. For a weekend
   hackathon it is a small risk; if anything we build gets published, swap it for a generic
   tiltrotor (search "V-22 Osprey", "quadrotor VTOL transport") and re-render.

One mesh can serve more than one class. `small_tower` and `large_tower` are nearly the same
size (12.6 x 12.0 vs 13.9 x 12.6 m) and differ mainly in colour — green roof against brown
camo — so the same watchtower is registered for both, to be rendered with a per-class tint.
The renderer therefore needs a hue/saturation override per class, not just scale and angle.

**`small_launcher` has no recoverable shape.** All 25 cut-outs are the same pale green blob at
23x30 px (6.3 x 4.6 m). The Rapier is a stand-in chosen for footprint and colour; the cut-outs
remain the ground truth for what the camera actually sees.

### How exact does a model have to be?

It depends entirely on how many pixels the class covers, so check the Size column before
spending an hour on a search:

- **Under ~35 px** (`small_launcher` 23x30, `ta-ta` 32x17, `jammer` 43x32): there is no
  structure to see. Every one of the 25 `small_launcher` cut-outs is a featureless green
  blob, which is why that class has no model. Footprint, colour and rough aspect ratio are all the detector can learn, so any
  plausible vehicle of the right size does the job.
- **Over ~100 px** (`hangar` 185x119, `condor` 173x166, `large_launcher` 149x110,
  `helicopter` 116x94): the silhouette is clearly visible and the model matters. These are
  worth the hunt, and they are also where the false positives come from.

`status` in the registry is `candidate` until the `.glb` is actually in `models/<class>/`,
then `downloaded`; `reference` for the right shape that we cannot use (view-only, no licence),
kept to compare against; `rejected` (with a note) for ones that turned out wrong, so nobody
re-finds them.

## Places to look

| Source | Licence | Notes |
|---|---|---|
| [Sketchfab](https://sketchfab.com/search?features=downloadable&licenses=322a749bcfa841b29dff1e8a1bb74b0b&type=models) | CC-BY / CC0 filters | Where the Star Wars ones almost certainly came from. Filter downloadable + licence. |
| [Poly Pizza](https://poly.pizza) | CC-BY / CC0 | Ex-Google Poly library, low-poly, lots of vehicles and aircraft. |
| [Quaternius](https://quaternius.com) | CC0 | Clean low-poly military/vehicle packs, whole sets at once. |
| [Kenney](https://kenney.nl/assets) | CC0 | Tanks, towers, aircraft in a consistent style. |
| [OpenGameArt](https://opengameart.org) | mixed, check per asset | Older, uneven quality. |
| [NASA 3D Resources](https://nasa3d.arc.nasa.gov/models) | public domain | Good for aircraft/spacecraft shapes. |
| [Fab](https://fab.com) | free section incl. CC-BY, plus paid | Epic's merged Unreal Marketplace + Quixel + Sketchfab store. Military vehicle packs turn up regularly. |

Not usable, so nobody wastes an evening on it: **Steam Workshop**. Items are licensed for use
inside their own game (Steam Subscriber Agreement plus the game's terms), there is no purchase
that grants wider rights, and they ship as game packages (`.gma`, Source `.mdl`, Unity bundles,
Arma `.pbo`) rather than meshes. Much of the military content there is itself ripped from other
games. Paid marketplaces (TurboSquid, CGTrader, Unity Asset Store) are legitimate but pointless
for us: our objects render at 20-180 px, so a 200k-face asset shows nothing a 3k-face one does not.

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
straight onto the aerial backgrounds.

It should read `registry.json`, and per class it needs:

- **scale** to the measured ground size in the table (a model's own units mean nothing);
- **a hue/saturation tint**, so one mesh can cover two classes (see the towers above);
- **downsampling to the class's real pixel size**, because a crisp render at 400 px looks
  nothing like the same object at 60 px in the challenge imagery.
