# Hologram Glitch Band

A hand-tracked video effect. Hold up both hands and a shape is drawn through
four fingertips - each hand's thumb and index finger. Whatever falls inside it
comes back printed, screened, dithered or run through worn tape. Move your fingers and the
shape stretches and skews with them; counter-rotate your hands and it twists
like a ribbon, pinching shut and crossing over into an X. Tap thumb to index on
both hands to change the effect.

That is one of three modes. Press `n` for the others: a 3D box with a rose in
it, and a slit-scan that makes every column of the picture a different moment.

Inspired by [this clip by @Vanny_Dev](https://vt.tiktok.com/ZSqrgJaqu/), rebuilt
on the current MediaPipe API with gesture control, per-hand bands, recording and
a config file.

## Running it

Everything is already installed in `.venv`. Double-click **`run.bat`**, or from a
terminal:

```bash
run.bat          # Windows
./run.sh         # Git Bash, WSL, macOS, Linux
```

Then raise both hands where the camera can see them. Either launcher checks the
environment, downloads the hand model if it is missing, prints the controls, and
passes any arguments straight through - `run.bat --effect vhs` works.

To skip the launcher:

```bash
.venv\Scripts\python.exe run.py
```

Other entry points:

```bash
.venv\Scripts\python.exe run.py --selftest        # probe the camera, no window
.venv\Scripts\python.exe run.py --list-effects    # print the effect names
.venv\Scripts\python.exe run.py --effect vhs     # start on a given effect
.venv\Scripts\python.exe run.py --camera 1        # use a different camera
.venv\Scripts\python.exe tests\test_offline.py    # run the checks, no camera needed
```

## Controls

| | |
|---|---|
| **move your thumbs and index fingers** | reshape the polygon |
| **counter-rotate your hands** | twist the shape, crossing it into an X |
| **tap thumb to index on both hands** | next effect |
| `n` | **next mode** - shape, cube, slit-scan |
| `b` | jump straight to **cube mode**, and back |
| **open both palms** (cube mode) | grow a rose inside the box |
| **one open palm alone** (cube mode) | the rose resting on your hand, no box |
| **index + middle finger** (cube mode, box down) | a red orb off your fingertips |
| **move a hand across** (slit-scan mode) | drag the scan through time |
| `g` | bloom - bright areas bleed light |
| `[` `]` | previous / next effect |
| `1`-`9` | pick an effect directly |
| `p` | one shape per hand, each an ellipse across that hand's pinch (shape mode) |
| `k` / `h` | toggle the hand skeleton / the overlay |
| `?` | key list |
| `m` | mirror |
| `s` | screenshot to `captures/` |
| `r` | start/stop recording an MP4 to `captures/` |
| `c` | reload `config.toml` without restarting |
| `q` or `Esc` | quit |

Screenshots and recordings are saved without the overlay, so clips show only the
effect.

## Modes

Press `n` to walk through them. Each one is a whole look with its own state, its
own gestures and its own line in the readout; the app keeps an ordered list and
knows nothing about what any of them does.

| | |
|---|---|
| **shape** | the original - a band through four fingertips, effect inside |
| **cube** | a 3D box between your hands, with a rose blooming in it |
| **slitscan** | every column of the picture from a different moment |

`b` still jumps straight to cube mode and back, since it predates the list.
`[modes]` in `config.toml` sets which modes exist, which order `n` walks them in,
which one the app starts on, and which of them the orb and the bloom ride along
with.

This was a single `cube_mode` boolean until there were more than two looks to
tell apart. Two modes fit in a boolean; what did not fit was the eleven places
that had quietly come to read "not cube" as "shape" - the orb's gate, the
bloom's gate, the readout, and four scalars written in one branch and read stale
in the other.

### Cube mode

Press `b`, or `n` until you reach it. A wireframe box hangs between your hands, with faint rays running
from your fingertips to its nearest corner. Hold up **one** open palm on its own
and you get the rose by itself, hovering above your hand - no box, no rays, no
effect window, since the box needs two hands to have anything to hang between.

The rose sits **centred over the middle of the hand and clear of its top edge**,
measured up the screen rather than along the hand's own axis. Offsetting along
the wrist-to-fingertip vector instead swings it off to one side the moment the
hand tilts, and puts it in front of the hand rather than above it. It is sized
from that hand's own scale, so it keeps its proportion at any distance. Move your hands to carry and resize
it, turn them like a wheel to spin it, raise one to tip it. The current effect is
applied to whatever the box covers, so it reads as a window cut through the
picture rather than an overlay. Open both palms and a rose blooms inside.

From [this clip by @fiqtor](https://vt.tiktok.com/ZSqhveavn/).

The rose is a **real model, drawn as a glowing point cloud** - its ~80,000
vertices become particles, splatted and bloomed. That is the only practical way
to use a model that size here: rasterising 80,000 faces in Python is hopeless,
but splatting the same vertices as points is one vectorised pass.

Import a model once with:

```bash
.venv\Scripts\python.exe scripts\import_rose.py "path\to\rose.obj"
```

which writes `models/rose_points.npy`. **Without that file the app falls back to
a generated rose**, so a fresh checkout still works.

Three things the importer has to get right, all found by measuring the model
rather than assuming:

- **It is a whole scene, not a flower.** Groups are classified by their
  material's texture: three grass backdrop planes are dropped, the stem and the
  flower kept. Matching on texture rather than group name survives a model whose
  objects are named differently.
- **Framing is on the flower, not the plant.** The model is 86 units tall and 68
  of that is bare stem, so normalising the whole thing shrinks the bloom to a
  27-pixel dot. Scaling to the flower fills the box with it and lets the stem
  run out of the bottom.
- **The stem is kept whole while the flower is strided down.** The flower has
  sixty times the stem's vertices packed into a fifth of the height; sampling
  them together leaves the stem too sparse to see at all.

Points are accumulated with `np.bincount` over flattened pixel indices - one
pass for the whole cloud, where `cv2.circle` per point would be hopeless and
`np.add.at` is far slower. Accumulating rather than overwriting is the point:
where the model's surface is dense the splats stack and burn brighter, so the
form shows through without any surface being drawn. Nearer points are weighted
brighter, then the buffer is bloomed and pushed through a black-to-magenta-to-
white ramp.

The generated fallback rose is drawn as **glowing strokes, never filled**. An
earlier version filled flat-shaded polygons, and every complaint about it - hard
facets, visible seams, dead colour - followed from that one choice:
`fillConvexPoly` paints each face a single colour and cannot interpolate shading
across it. Strokes have no interior to facet. Its petals sit in concentric
whorls of 3, 5 and 8 rather than one even spiral, and only their outlines are
stroked; adding ribs turned the flower into a ball of wire.

### Time slit-scan

Every column of the picture comes from a different moment, and your hand drags
the scan across. A ring of the last 30 frames is kept at half resolution - one
second, 20.7 MB - and each column is read from a different one of them, so a
movement crossing the frame is smeared out into a record of *when* it happened
rather than where it is.

It opens on a normal picture: the ring starts full of copies of the first frame
and dissolves into the effect over its first second, rather than flashing black.

This is the heaviest of the modes and the only one holding a big buffer, so it
is the first to turn down if the budget will not take it - halving
`[slitscan] scale` quarters both the memory and the work.

## The red orb

In **cube mode**, hold up your **index and middle fingers** and a sphere of red
energy appears off their ends - Gojo's Red. A burning core inside a red ball,
with three great circles turning around it.

It is real 3D, drawn through the same pipeline as the rose: a point cloud in
unit space, rotated and projected by `scene3d.project`, splatted into a
supersampled buffer, bloomed, and pushed through a colour ramp.

It works identically in any mode - the draw sits outside the modes entirely -
but shape mode already puts an effect around the same hand and the two together
are a lot, so it rides along only with the modes named under `[modes] orb`,
which is cube by default. Add or remove names there to change that. Leaving one
of those modes hides it by feeding it no hands rather than dropping the render,
so a lit orb eases out instead of snapping off mid-glow.

It used to be a `cube_only` boolean on `[orb]`, which worked while there were
exactly two modes to be on one side or the other of.

**The box takes priority.** Both hands up in cube mode draws the box, and while
it is there the orb stays down whatever `[modes] orb` says - the box already fills the space between your
hands, and the orb over the top of it is two effects fighting for the same
picture. Drop to one hand and the orb comes back. The rose alone on an open palm
is not the box, so the orb still works beside that.

Nothing clashes with the other gestures - the rose wants four or more fingers
open, the effect switch wants both thumbs tapped to their index fingers, and
this wants exactly index and middle. The thumb is ignored on purpose: people
hold this sign with it tucked in or cocked out and mean the same thing by it.

**The renderer accumulates, so density is brightness and the radial
distribution of the cloud is the entire look.** Points are placed at
`r = u ** core_bias`: at 1/3 that fills the volume evenly, and above it density
piles into the middle, which is the core burning through. A further `shell` of
the points goes on the surface, on a golden-angle lattice rather than at random,
because the silhouette is the one place clumping would show.

Four things this took to get right, none of them obvious:

- **The rings are handed over in arcs, not whole.** The glow renderer shades one
  depth per curve, so a ring given to it in one piece comes out flat at its mean
  depth. Cut into eight arcs, its near side burns and its far side falls away,
  and the ring reads as passing behind the ball. There is no depth buffer here
  and everything is additive, so that banding *is* the occlusion. Measured, it
  is worth between 2.6x and 14x of near-to-far contrast; whole, 0.9x.
- **The rings have a colour ramp of their own.** That renderer's nearest band
  burns at about 222 of 255, and on the sphere's ramp - which runs out to
  near-white for its core - that lands pale pink. A ramp topping out red instead
  keeps them red however hot they get: 5.9:1 red over blue, against 1.4:1 on the
  sphere's.
- **The point count follows the projected area, not the point size.** The same
  cloud packed into a small ball piles every point onto the same few pixels and
  burns out to a white marble; spread over a large one it thins to isolated
  specks. Scaling the *count* by the square of the radius holds the density, and
  so the look, steady at any distance from the camera. Raising the gain instead
  does not work - it only saturates each point one at a time.
- **What makes it read as a ball is not the near-to-far weighting.** That is the
  obvious answer and it is wrong: on a symmetric sphere every sightline holds a
  matched near and far point, so the weighting cancels to under a tenth across
  the whole disc. It is the radial density and the rings that put it in space.

## Bloom

Press `g`. A bloom pass over the finished frame: the bright parts bleed light,
which lifts everything drawn without changing any of it. Modelled on
TouchDesigner's Bloom TOP, from [this clip by @will](https://vt.tiktok.com/ZSqkMkhyb/).

It runs after the modes named under `[modes] bloom` - shape by default. The
slit-scan is left out because it rewrites every pixel of the frame, and blooming
a picture that is already all effect lifts the whole thing into haze. This was a
`shape_only` boolean on `[bloom]` until there were more than two modes for it to
choose between.

It picks out what is brighter than a threshold, blurs that at **several radii and
averages them**, and adds it back. The spread matters: one radius gives a single
flat smear, where a tight blur inside a wide one reads as a hot core with a halo
around it.

Two numbers found by measuring rather than guessing:

- **Threshold 0.45.** At 0.62 the pass was barely visible at all; by 0.30 the
  whole background lifted into haze. 0.45 glows the bright parts and leaves the
  rest alone.
- **7.9 ms, down from 48 ms.** The first version did the black point, gamma,
  threshold and knee as float maths per pixel. All of those are functions of a
  single pixel's brightness and nothing else, so they collapse into one
  256-entry lookup table - the same trick the print effects use. The
  giveaway was that the blur count barely changed the cost: one radius was
  40.7 ms against 41.8 ms for three, so the blurs were never the problem.

## Effects

Eight of them. Cycle with `[` `]`, pick one with the number keys, or tap both
thumbs to their index fingers to step forward without touching the keyboard.

| | |
|---|---|
| `risograph` | two spot inks screened into dots and printed slightly out of register, on paper that shows through |
| `cyanotype` | the blueprint - one Prussian-blue ramp from shadow through cyan to paper white |
| `quadtree` | blocks that keep splitting only where there is detail to justify it |
| `vhs` | worn tape - colour smeared sideways and lagging the edges, bands sliding, a head crossing |
| `ascii` | the picture retyped as characters |
| `halftone` | a printer's dot screen |
| `thermal` | heat colormap |
| `dither` | ordered dither against a Bayer matrix - two tones, scattered pixels |

Every one is tunable under `[effects.<name>]` in
[`config.toml`](config.toml) - ink colours, dot size, how far the riso misses
register, how hard the tape bleeds.

**Three of them are the same machine wearing different clothes.** Risograph,
halftone and dither all tile a small threshold matrix across the region and make
one uint8 compare against it. The only difference is where the low thresholds
sit in the cell: clustered in the middle gives a printer's dot that grows round
from the centre, spread evenly across it gives the pixel spray of a 1-bit
display. Both matrices are generated rather than written out by hand - the
clustered one by ranking the cell by a spot function - so any cell size works.

Everything that is a function of a size or a setting rather than of the picture
is built once and kept: the screens, the glyph atlas ASCII indexes into, the
colour ramps, the ink density curves. Nothing in the per-frame path allocates a
table.

Two that are worth a note of their own:

- **ASCII is one gather.** The region is shrunk to a grid of cells, each cell's
  brightness picks a glyph, and the whole picture is assembled by indexing the
  atlas with that grid in a single operation. Drawing the characters one at a
  time would be tens of thousands of `putText` calls a second.
- **Quadtree tests variance at block resolution, not pixel resolution.** Mean
  and mean-of-squares come from two `INTER_AREA` resizes, the split test happens
  on those small arrays, and only the mask of who settled is scaled back up. The
  one thing done at full size is copying the colour that wins.

## Tuning

Everything adjustable lives in [`config.toml`](config.toml) - band size, pinch
range, smoothing, colours, per-effect parameters, camera resolution. Edit it
while the app is running and press `c` to reload.

Two settings are worth playing with first. `point_smoothing` under `[tracking]`
- higher tracks your fingers more sharply, lower is calmer but lags. And
`tap_threshold` under `[gestures]` - raise it if taps are hard to trigger, lower
it if the effect changes by accident.

The twist has nothing to tune: it is only where your fingertips are.

Every mode has its own section: `[box]` for the cube - size, how far turning
your hands turns it, the fingertip rays, the rose's petal count and bloom speed
- and `[slitscan]` for the slit-scan. `[modes]` decides which of them exist at
all and in what order `n` walks them.

The slit-scan carries a `scale`, the fraction of the frame it works at. It is
the one mode that touches every pixel rather than masking itself to a small
region, and for it the *upscale* back to full size costs more than the effect
itself - so that one number is the lever that matters if it runs slow, and it
sets the size of the history buffer too.

To see any of it without standing in front of the camera - `--twist 180` gives a
full X, and the contact sheet includes whole simulated frames for every mode:

```bash
.venv\Scripts\python.exe scripts\preview_effects.py --twist 150
.venv\Scripts\python.exe scripts\preview_cube.py --petals 30
.venv\Scripts\python.exe scripts\preview_cube.py --modes-only
```

## How it works

1. MediaPipe's `HandLandmarker` returns 21 points per hand. Only two are used
   for the shape: the thumb tip and the index tip.
2. Those four points (two per hand) are each smoothed on their own, then joined
   **by which fingertip they are** - index to index, thumb to thumb. Those two
   edges are rails, so counter-rotating your hands drags one across the other
   and the shape crosses into an X, the way a twisted ribbon does.

   Sorting the corners by angle about their centre would be the obvious way to
   guarantee a tidy polygon, and an earlier version did exactly that. It also
   makes twisting impossible - the crossing can never appear, because the sort
   quietly undoes it every frame.
3. Because the shape is defined by points rather than a rotated rectangle, it
   has no angle, length or thickness, and none of the angle-wrapping care a
   rotating rectangle needs applies. The twist is not a value that gets tracked
   or smoothed either; it is just where your fingers are.
4. The effect runs on the shape's bounding box only, then is composited back
   through a filled-polygon mask, so cost scales with the shape rather than the
   frame. Filling uses `fillPoly`, never `fillConvexPoly`: the quad goes concave
   when you bend a finger in, and crosses itself entirely once twisted past the
   pinch. `fillPoly`'s even-odd rule renders both, filling a crossed shape as
   the two lobes of an X; the convex version fills the whole hull and would
   erase the twist.
5. The wrist-to-knuckle distance gives a hand size, and the thumb-to-index gap
   is divided by it - so a tap reads the same whether you are near the camera or
   far from it. Everything sized in those units - the orb, the solo rose -
   holds its proportion at any distance.
6. All of that is one **mode**. A mode owns its own state, its own gestures, the
   keys that belong to it and the line it writes in the readout; the main loop
   holds an ordered list of them and does no more than `update` then `render`
   the current one. It is the reason adding a look does not mean touching the
   loop, the overlay, or either of the two gates that decide whether the orb and
   the bloom come along.

## Notes on the build

**Python 3.12.** mediapipe publishes no wheels for 3.13 or newer.

**No `opencv-python` in requirements.** mediapipe already depends on
`opencv-contrib-python`; installing both puts two copies of the `cv2` extension
in one environment and fails in confusing ways. The contrib build has
everything used here.

**The Tasks API, not `mp.solutions`.** The original video uses
`mp.solutions.hands`. That module does not exist in mediapipe 1.0 - the whole
legacy `solutions` package is gone - so the code that inspired this would not
run today. `hand_landmarker.task` is fetched by `scripts/download_model.py`.

**Performance.** The first working version ran at 12 fps. Three changes took it
to 29-30 fps on a 30 fps camera, which is the camera's own ceiling:

| | before | after |
|---|---|---|
| effect pass | 30 ms | 4.6 ms |
| effect pass, large shape | 129 ms | 22 ms |
| mask composite | 6.2 ms | 0.06 ms |
| edge glow | 6.8 ms | 1.0 ms |

- Effects stay in `uint8` and colourise through a lookup table instead of
  multiplying float BGR, and anything blurred is blurred at a fraction of the
  resolution. A soft glow has no fine detail to lose.
- Compositing uses `cv2.copyTo` rather than a boolean numpy assignment.
- Frames are read on a background thread. `capture.read()` blocks for a whole
  frame interval, and inline that wait *added* to the processing time instead of
  overlapping it. Only the newest frame is kept, so falling behind drops frames
  instead of building up lag.

Worth being precise about the last one: it was worth 1.2x when the effects were
slow, and is worth about 1.04x now that they aren't, because the loop has become
camera-bound. It stays because it is what keeps a big shape or a heavier effect
from eating into the frame budget, and that now overlaps the camera wait rather
than adding to it. Across a 420x240 shape the current set costs:

| | | | |
|---|---|---|---|
| `halftone` 0.11 ms | `dither` 0.12 ms | `ascii` 0.45 ms | `thermal` 0.73 ms |
| `cyanotype` 0.85 ms | `risograph` 1.64 ms | `vhs` 3.15 ms | `quadtree` 3.75 ms |

What is left is mostly MediaPipe inference at ~21-26 ms per frame.

Twisting costs nothing. An earlier version swirled the pixels inside the shape
through `cv2.remap`, which ran to 12 ms a frame before optimisation. Twisting
the shape itself is just four points in a different order.

Cube mode costs about **8.9 ms** a frame with the point-cloud rose open, against
3.5 ms with it closed. The knobs, in the order worth reaching for: `--points`
when importing the model, then `rays`, then `effect_inside`.

The orb costs **2.9-4.6 ms** across normal hand distances. Both of its renderers
build a buffer the size of the orb's own bounding box, so the cost goes as the
square of the radius - which is why `max_radius` caps it, holding the worst case
near 10 ms instead of the 14 ms a hand against the lens would otherwise reach.

Splatting the cloud used to walk that buffer three times over - a multiply, a
clip and a cast, each allocating megabytes of float64 - where `convertScaleAbs`
does all three in one pass. That one line took the orb from 5.8 ms to 4.0 at a
64 px radius, and the rose came along with it, since both go through the same
renderer.

The point cloud is loaded once from `.npy`. Parsing 13 MB of OBJ text on every
start would add seconds to launch for a fixed answer, which is the whole reason
the importer is a separate step.

## Troubleshooting

**"Could not read from camera"** - something else has the webcam. Close Teams,
Zoom or the Camera app, or try `--camera 1`.

**"Hand landmark model not found"** - run
`.venv\Scripts\python.exe scripts\download_model.py`.

**Hands not detected** - MediaPipe needs reasonable light and the whole hand in
frame. Lower `detection_confidence` in `config.toml` if it is still missing them.

**Effect switching triggers by accident** - lower `tap_threshold` under
`[gestures]` so your fingers have to come closer, raise `cooldown`, or set
`enabled = false` and use `[` and `]` instead.

**Taps don't register** - raise `tap_threshold`. The default 0.38 means your
fingertips must come within about a third of your hand's wrist-to-knuckle length
of each other.

**Nothing appears with both hands up** - the shape needs both a thumb tip and an
index tip per hand. If it only sees one hand you get that hand's ellipse instead;
the `hands` count in the overlay tells you what it is tracking.

**The shape starts out already crossed** - your hands are held mirrored. Turn
one over and it opens out. The overlay shows the relative rotation of your two
hands, with an `X` next to it while the shape is crossed.
