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
| 1 | `mine_roller` | 12.6 x 11.1 | 2 | 0.505 | camo cylinder on a vehicle, mine-clearing roller drum | **candidate found** — PT-34-85, see below |
| 2 | `medium_launcher` | 9.9 x 9.3 | 10 (3 suspect) | 0.505 | small dark launcher vehicle, hard to read | **wanted** — worst AP, rough sprites |
| 3 | `small_launcher` | 6.3 x 4.6 | 25 | 0.810 | featureless green blob at 23x30 px | **candidate found** — MIM-23 Hawk, see below |
| 4 | `ta-ta` | 6.7 x 3.6 | 25 | 0.758 | four-legged walker (AT-AT) | **wanted** — search "AT-AT", likely the same free asset |
| 5 | `large_tower` | 13.9 x 12.6 | 19 | 0.771 | brown/camo rectangular structure, pale band across the middle | **2 candidates** — CGTrader watchtower preferred, see below |
| 6 | `hangar` | 38.9 x 25.0 | 6 | 0.832 | dark curved roof, Quonset/arched hangar | **2 candidates** — Sketchfab hangar + NATO shelter, see below |
| 7 | `medium_plane` | 11.8 x 10.3 | 5 | 1.000 | dark green single-prop aircraft, WW2 style | **wanted** — only 5 sprites |
| 8 | `small_plane` | 10.5 x 9.0 | 9 | 0.857 | prop aircraft, red nose, green camo | **wanted** |
| 9 | `condor` | 36.3 x 34.9 | 11 (3 suspect) | 1.000 | big grey X-shaped four-engine aircraft/cargo drone | **candidate found** — BF2042 Condor, licence to check |
| 10 | `small_tower` | 12.6 x 12.0 | 20 | 0.909 | dark green square roof on a pale base | **candidate found** — same watchtower mesh, tinted green |
| 11 | `jammer` | 9.0 x 6.7 | 13 | 0.921 | green boxy truck with a flat box body | wanted |
| 12 | `helicopter` | 24.4 x 19.7 | 19 (4 suspect) | 1.000 | dark green helicopter, rotors visible | wanted |
| 13 | `tank` | 10.5 x 9.9 | 25 | 0.930 | camo tank, barrel forward | wanted |
| 14 | `large_launcher` | 31.3 x 23.1 | 25 | 1.000 | camo TEL, long missile tube on a truck | wanted |
| 15 | `jet_plane` | 17.0 x 16.2 | 22 | 1.000 | white/grey swept-wing jet | wanted |
| 16 | `spacecraft` | 10.3 x 9.2 | 23 | 0.882 | TIE fighter/interceptor | **candidate found** — TIE/in Interceptor, see below |

Local AP is from the Helsinki set, where the model memorised the scene, so treat 1.000 as
"nothing to learn here yet" rather than "solved". Live validation was 0.0069 overall.

## Candidates so far

Recorded with `python training/add_model.py <class> <url>`, which looks the model up on
Sketchfab and writes both `models/registry.json` and `models/<class>/credits.txt`. Add every
candidate as you find it, even an uncertain one: the URL is the part that gets lost.

| Class | Model | Author | Licence | Status |
|---|---|---|---|---|
| `mine_roller` | [PT-34-85 mine clearing vehicle](https://sketchfab.com/3d-models/pt-34-85-mine-clearing-vehicle-b845931a1ba64b318ad01c3968bba18f) | 42manako | CC-BY (credit required, commercial ok) | candidate — T-34 hull with roller drum, extent close to the measured 12.6 x 11.1 m |
| `spacecraft` | [Star Wars: TIE/in Interceptor](https://sketchfab.com/3d-models/star-wars-tiein-interceptor-80171ec2930b4949836bcf24d9694c41) | Daniel | CC-BY (credit required) | candidate — silhouette matches the sprites |
| `small_launcher` | [MIM-23 Hawk SAM (game-ready)](https://sketchfab.com/3d-models/mim-23-hawk-sam-air-defence-system-game-ready-8728909b6ce24ef8baeffabbf5bae8f4) | Dominik Biały | CC-BY (credit required) | candidate — 3 rails on a trailer, ~5 m, fits the 6.3 x 4.6 m footprint |
| `hangar` | [Hangar](https://sketchfab.com/3d-models/hangar-c3e821610c644ade9878aa56af867e05) | Vitor Augusto | CC-BY (credit required) | candidate — scale it to the measured 38.9 x 25.0 m; this class is 185x119 px, so the roof shape does matter |
| `large_tower` | [Old Wooden Watchtower (House 3)](https://sketchfab.com/3d-models/old-wooden-watchtower-house-3-49b77f82b0944d5188c04c3fc205a499) | Blenderust | CC-BY (credit required) | candidate — scale to 13.9 x 12.6 m. From above the sprites read as a brown/camo rectangular structure with a pale band across the middle, more camouflaged shelter than open tower, so check the render before trusting it |
| `small_tower` | whichever watchtower wins for `large_tower` | see that row | see that row | candidate — same mesh, rendered tinted green and scaled to 12.6 x 12.0 m |
| `large_tower` | [Old wooden watchtower (low poly)](https://www.cgtrader.com/free-3d-models/exterior/other/old-wooden-watchtower-low-poly) | CGTrader | free — **check the licence line** | candidate, Juan's preferred — closer to the sprites than the Sketchfab watchtower |
| `hangar` | [NATO aircraft shelter v2](https://www.cgtrader.com/free-3d-models/military/other/nato-aircraft-shelter-v2) | CGTrader | free — **check the licence line** | candidate — hardened shelter with an arched roof, closer to the sprites than a plain hangar |
| `condor` | [Battlefield 2042 Condor](https://www.cgtrader.com/free-3d-models/military/military-vehicle/battlefield-2042-condor-flight) | CGTrader | free — **check the licence line**, and see the game-asset caveat | candidate — quad-rotor VTOL, matches the X-shaped 4-engine sprite |

**Two things to check on any CGTrader model**, because they are not uniform like Sketchfab's:

1. **The licence line on the page.** CGTrader free models are usually "Royalty Free", but some
   are "Editorial Uses Only", which excludes training data. Record what it says with
   `--licence`; `add_model.py` cannot read CGTrader.
2. **Whether it is a game rip.** "Battlefield 2042 Condor" is EA/DICE's design, extracted from
   their game; the uploader has no rights to grant, whatever the page's licence box says. The
   same is true of the organisers' own `spacecraft` and `ta-ta`, which are Star Wars assets, so
   this is a competition-wide grey area rather than something we invented. For a weekend
   hackathon it is a small risk; if anything we build gets published, swap it for a generic
   tiltrotor (search "V-22 Osprey", "quadrotor VTOL transport") and re-render.

One mesh can serve more than one class. `small_tower` and `large_tower` are nearly the same
size (12.6 x 12.0 vs 13.9 x 12.6 m) and differ mainly in colour — green roof against brown
camo — so the same watchtower is registered for both, to be rendered with a per-class tint.
The renderer therefore needs a hue/saturation override per class, not just scale and angle.

### How exact does a model have to be?

It depends entirely on how many pixels the class covers, so check the Size column before
spending an hour on a search:

- **Under ~35 px** (`small_launcher` 23x30, `ta-ta` 32x17, `jammer` 43x32): there is no
  structure to see. Every one of the 25 `small_launcher` cut-outs is a featureless green
  blob. Footprint, colour and rough aspect ratio are all the detector can learn, so any
  plausible vehicle of the right size does the job.
- **Over ~100 px** (`hangar` 185x119, `condor` 173x166, `large_launcher` 149x110,
  `helicopter` 116x94): the silhouette is clearly visible and the model matters. These are
  worth the hunt, and they are also where the false positives come from.

`status` in the registry is `candidate` until the `.glb` is actually in `models/<class>/`,
then `downloaded`; use `rejected` (with a note) for ones that turned out wrong, so nobody
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
