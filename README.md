<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media_kit/wordmark-light.svg">
    <img src="media_kit/wordmark-dark.svg" alt="Moonlake" width="360" />
  </picture>

  <h3>Simulation environments for physical AI.</h3>

  <p>
    <a href="https://moonlakeai.mintlify.site/introduction">Documentation</a>
    &nbsp;|&nbsp;
    <a href="https://discord.gg/ZJZB2vymnY">Discord</a>
  </p>
</div>

# SimEnvBuilder

**SimEnvBuilder** is a cookbook for evaluating robot manipulation policies
on articulated, simulation-ready assets. It bundles 20 articulated assets
from the Moonlake asset library, derives tasks and prompts from each asset's
articulation, composes
[Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena) scenes around
them, and rolls out open-source generalist policies.

## Key features

- **Policy evaluation on articulated assets.** Each asset's authored joints
  define the prompts and success thresholds; the scorer grades rollouts
  from joint state. Runs produce videos and progress metrics, and failures
  report how far the policy got.
- **20 ready-to-run tasks.** Each bundled asset ships with a registered
  task. One command rolls out the policy, with optional background,
  lighting, and placement variations; a local dashboard shows per-episode
  progress and milestones.
- **Asset quality checks.** One command grades any USD or glTF package
  on physics authoring, mesh hygiene, collision proxies, UVs, materials,
  and articulation, producing a per-asset JSON report and optional
  turntable renders for visual inspection.
- **Bring your own assets.** Any USD with physics articulation (joints,
  limits, colliders, mass) plugs into the same pipeline, or generate a
  sim-ready articulated asset with the
  [Moonlake Asset API](https://moonlakeai.mintlify.site/introduction).
- **Enterprise services.** Moonlake provides customized digital twin creation,
  deformable-object simulation, pre-deployment studies, and accurate
  simulation environments for your deployment site. Contact
  [contact@moonlakeai.com](mailto:contact@moonlakeai.com).

## Example task rollouts

"open the dishwasher"

<img src="docs/media/rollouts/open_dishwasher_contact.gif" width="100%">

"open the soda maker gas cylinder door"

<img src="docs/media/rollouts/open_soda_maker_contact.gif" width="100%">

"lift the paper cutter blade"

<img src="docs/media/rollouts/lift_guillotine_blade_contact.gif" width="100%">

Each sheet shows the external, base, and wrist cameras. The rollouts run
PI-0.5 (`pi05_droid_jointpos`) on a Franka (DROID).

## Installation

Requirements:

- Linux x86_64
- an NVIDIA GPU (CUDA 12.8+ driver; a single-GPU run peaks at ~23 GB VRAM;
  an RTX 5090 is recommended)
- [uv](https://docs.astral.sh/uv/)
- [git-lfs](https://git-lfs.com)

Optional:

- [Blender](https://www.blender.org/download/) 4.5+, on `PATH` or set via
  `$BLENDER`, only needed for `check-asset --render` and `preview-gifs`.

### Simulation environment

```bash
sudo apt install git-lfs && git lfs install
git clone git@github.com:MoonlakeAI/sim-builder-poc.git
cd sim-builder-poc
uv sync
```

### Policy server

The policy server is a separate uv project under `policies/openpi` with its
own environment, since openpi runs on JAX and python 3.11.

```bash
cd policies/openpi && uv sync && cd ../..
```

## Quickstart

Terminal 1: the policy server. The first run downloads the ~12 GB PI-0.5
checkpoint:

```bash
cd policies/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run openpi-server
```

Terminal 2: evaluate the open-dishwasher task for 5 episodes. Isaac Lab
uses ~8 GB of VRAM:

```bash
uv run sim-env-builder rollout --task open-dishwasher
```

The memory fraction caps the policy server's share of GPU memory so the
simulator fits beside it.

The run writes its outputs to `outputs/open-dishwasher/<timestamp>/` and
prints a verdict line on stdout, e.g.
`[FAIL] 0/5 episodes succeeded | best: max_door_angle_deg=29.3`.

To browse the results, run the dashboard and open the printed URL:

```bash
uv run sim-env-builder dashboard
```

Per-episode milestone JSON:

```json
{
  "instruction": "open the dishwasher",
  "target_milestone": "open the dishwasher door",
  "success": true,
  "milestones": {"open the dishwasher door": true, "pull out the dishwasher lower rack": false},
  "progress": {"max_door_angle_deg": 52.3, "door_pivot_progress": 0.58}
}
```

The steps below explain what the quickstart ran and how to change it.

## Workflow

### 1. Select an asset and inspect its articulation

The repository bundles 20 articulated USD packages from the Moonlake asset
library in `assets/library/`.

| preview | asset | articulation |
|---|---|---|
| <img src="assets/library/previews/blender.png" width="160"> | `blender` | dial turn, pitcher lock, blade spin, lid lift, cap lift |
| <img src="assets/library/previews/broadcast_spreader.png" width="160"> | `broadcast_spreader` | bail squeeze, gate lever, gate plate, gate rod, lid hinge, spinner, wheel left, wheel right |
| <img src="assets/library/previews/dishwasher.png" width="160"> | `dishwasher` | door, button x6, detergent lid, lower rack, upper rack |
| <img src="assets/library/previews/espresso_machine.png" width="160"> | `espresso_machine` | button press x4, drip tray slide, drip tray grate lift, portafilter twist, portafilter extract, steam dial turn x2, steam wand swing, water tank remove |
| <img src="assets/library/previews/glass_electric_kettle.png" width="160"> | `glass_electric_kettle` | kettle, lid, lid button, power switch |
| <img src="assets/library/previews/hole_punch.png" width="160"> | `hole_punch` | alignment guide, chip tray, lever, punch pin x3 |
| <img src="assets/library/previews/multi_cooker_crisper_lid.png" width="160"> | `multi_cooker_crisper_lid` | button x2, dial, lid hinge, latch, pot lift |
| <img src="assets/library/previews/multimeter_probes.png" width="160"> | `multimeter_probes` | button x2, dial |
| <img src="assets/library/previews/office_printer.png" width="160"> | `office_printer` | button power, button small x10, output tray, paper tray, scanner lid |
| <img src="assets/library/previews/pancake_air_compressor.png" width="160"> | `pancake_air_compressor` | coupler sleeve x2, drain valve, regulator knob |
| <img src="assets/library/previews/paper_guillotine_cutter.png" width="160"> | `paper_guillotine_cutter` | blade hinge, paper stop slide |
| <img src="assets/library/previews/pill_bottle.png" width="160"> | `pill_bottle` | cap press, cap twist |
| <img src="assets/library/previews/popcorn_machine.png" width="160"> | `popcorn_machine` | control switch x3, glass door, kernel tray, kettle, kettle lid |
| <img src="assets/library/previews/salad_spinner.png" width="160"> | `salad_spinner` | basket lift, basket spin, lid lift, brake press, pump press |
| <img src="assets/library/previews/slushie_machine.png" width="160"> | `slushie_machine` | drip tray x2, tank x2, auger x2, dispense handle x2, lid x2 |
| <img src="assets/library/previews/soda_maker.png" width="160"> | `soda_maker` | bottle, carbonating button, drip tray, gas cylinder door |
| <img src="assets/library/previews/stacking_modular_toolbox.png" width="160"> | `stacking_modular_toolbox` | lid hinge x3, front latch x6, side lock x4, carry handle, telescopic handle, wheel x2 |
| <img src="assets/library/previews/stand_mixer.png" width="160"> | `stand_mixer` | bowl twist, head tilt, beater detach, speed slide |
| <img src="assets/library/previews/toilet.png" width="160"> | `toilet` | flush lever, lid hinge, seat hinge, tank lid lift |
| <img src="assets/library/previews/vending_machine.png" width="160"> | `vending_machine` | door, coin return press, keypad button press x12, retrieval flap |

The `suggest` command derives testable prompts from any asset's joints.

#### Example: task suggestion for the office printer

```bash
uv run sim-env-builder suggest assets/library/office_printer.usdz
```

```
Articulation of assets/library/office_printer.usdz:
  button_power_pivot       prismatic  range [0, 0.002] m
  button_small_01_pivot    prismatic  range [0, 0.002] m
  [... button_small_02 through button_small_10 ...]
  output_tray_pivot        prismatic  range [0, 0.1] m
  paper_tray_pivot         prismatic  range [0, 0.25] m
  scanner_lid_pivot        revolute   range [0, 80] deg

Derived milestones (threshold = value that counts as achieved):
  open the office printer scanner lid           scanner_lid_pivot >= 40 deg
  pull out the office printer paper tray        paper_tray_pivot >= 0.125 m
  pull out the office printer output tray       output_tray_pivot >= 0.05 m
  press the office printer button power         button_power_pivot >= 0.0016 m
  [... 10 more button milestones ...]

Suggested prompts to test:
  "open the office printer scanner lid"
  "pull out the office printer paper tray"
  "pull out the office printer output tray"
  "press the office printer button power"
```

Each suggested prompt maps to a milestone that the rollout scorer grades.

### 2. Check the asset's quality

To check the asset quality from multiple perspectives, `check-asset`
takes in one USD or glTF package and produces a JSON report covering
tests on geometry, UV layout, materials, collision proxies, physics
authoring and articulation.

```bash
uv run sim-env-builder check-asset assets/library/pill_bottle.usdz
```

| section | covers |
|---|---|
| `sim_ready` | colliders, mass, friction, joint parameters authored and plausible |
| `mesh` | topology, degeneracy, orientation, self-intersection, floaters, density |
| `proxy` | dedicated collision geometry: present, light, closed, tracks the render surface |
| `uv` | distortion, island count, texel density, seam placement |
| `materials` | PBR surfaces, texture resolution, duplicates, seam continuity, baked-in lighting |
| `articulation` | joint graph shape, limits, transforms, skin weights |

Per-check statuses and metrics land in
`outputs/asset_checks/<asset>/report.json`.

Visual inspection of asset quality is just as important as quantitative metrics.
Add `--render preview` to preview stills of rendered turntables and `--render loop`
to output the full assembled + exploded turntables for viewing.
Requires Blender accessible via your CLI to use.

What's equally important is what's NOT included in these static checks:
adherence to an articulation manifest, articulation and physics tests in
an actual Isaac simulation, and validation of metric measurements. These
require a detailed asset contract or a full simulation environment.

### 3. Compose the scene

An environment places the selected asset into a scene with a Franka
(DROID) facing it. Background, lighting, and placement are set per run
with CLI flags:

| variation | flag | options |
|---|---|---|
| background HDRI | `--hdr` | `home_office` (default), `hotel_room`, `garage`, `""` = plain dome |
| lighting | `--lighting` | `default`, `dim`, `bright` |
| placement | `--placement` | `floor` (asset on the ground), `table` (asset on Arena's maple table) |

Each placement comes with pose defaults that work out of the box. Other
options:

- `--asset_x` / `--asset_y` / `--asset_z`: adjust the asset's position
  (meters, robot at the origin), e.g. `--asset_x 0.8` to move the asset
  farther from the robot
- `--instruction "..."`: evaluate a different prompt phrasing against the
  same milestone criterion
- `--episodes` / `--steps`: run length
- `--seed`: RNG seed
- `--num-envs`: parallel environments
- `--viz`: open the viewport
- environment parameters such as the success threshold
  (`--openness_threshold`)

### 4. Run the evaluation

One ready-made task per bundled asset. Success: the graded joint reaches
≥50% of its authored range (e.g. `open-dishwasher`: `door_pivot` ≥45°).
Each GIF previews the graded joint motion.

| preview | task | asset | placement | instruction |
|---|---|---|---|---|
| <img src="docs/media/tasks/open_dishwasher.gif" width="160"> | `open-dishwasher` | dishwasher | floor | open the dishwasher |
| <img src="docs/media/tasks/open_toilet.gif" width="160"> | `open-toilet` | toilet | floor | open the toilet lid |
| <img src="docs/media/tasks/open_vending_machine.gif" width="160"> | `open-vending-machine` | vending machine | floor | open the vending machine door |
| <img src="docs/media/tasks/open_popcorn_machine.gif" width="160"> | `open-popcorn-machine` | popcorn machine | floor | open the popcorn machine door |
| <img src="docs/media/tasks/open_slushie_machine.gif" width="160"> | `open-slushie-machine` | slushie machine | floor | lift the slushie machine lid |
| <img src="docs/media/tasks/open_spreader.gif" width="160"> | `open-spreader` | broadcast spreader | floor | open the spreader lid |
| <img src="docs/media/tasks/open_toolbox.gif" width="160"> | `open-toolbox` | stacking modular toolbox | floor | open the toolbox top lid |
| <img src="docs/media/tasks/rotate_compressor_knob.gif" width="160"> | `rotate-compressor-knob` | pancake air compressor | floor | rotate the air compressor regulator knob |
| <img src="docs/media/tasks/open_printer.gif" width="160"> | `open-printer` | office printer | table | open the printer lid |
| <img src="docs/media/tasks/tilt_mixer.gif" width="160"> | `tilt-mixer` | stand mixer | table | tilt the mixer head up |
| <img src="docs/media/tasks/open_kettle.gif" width="160"> | `open-kettle` | glass electric kettle | table | open the kettle lid |
| <img src="docs/media/tasks/open_multi_cooker.gif" width="160"> | `open-multi-cooker` | multi cooker crisper lid | table | open the multi cooker lid |
| <img src="docs/media/tasks/open_soda_maker.gif" width="160"> | `open-soda-maker` | soda maker | table | open the soda maker gas cylinder door |
| <img src="docs/media/tasks/lift_blender_lid.gif" width="160"> | `lift-blender-lid` | blender | table | lift the blender lid |
| <img src="docs/media/tasks/lift_spinner_lid.gif" width="160"> | `lift-spinner-lid` | salad spinner | table | lift the salad spinner lid |
| <img src="docs/media/tasks/rotate_espresso_dial.gif" width="160"> | `rotate-espresso-dial` | espresso machine | table | rotate the espresso machine steam dial |
| <img src="docs/media/tasks/lift_guillotine_blade.gif" width="160"> | `lift-guillotine-blade` | paper guillotine cutter | table | lift the paper cutter blade |
| <img src="docs/media/tasks/press_hole_punch.gif" width="160"> | `press-hole-punch` | hole punch | table | press down the hole punch lever |
| <img src="docs/media/tasks/twist_pill_cap.gif" width="160"> | `twist-pill-cap` | pill bottle | table | twist open the pill bottle cap |
| <img src="docs/media/tasks/rotate_multimeter_dial.gif" width="160"> | `rotate-multimeter-dial` | multimeter probes | table | rotate the multimeter dial |

To evaluate a different asset, register an environment for it; see
[Adding assets](#adding-assets).

With the policy server from the quickstart running:

```bash
uv run sim-env-builder rollout --task <task> --episodes 5

# with variations
uv run sim-env-builder rollout --task <task> \
    --hdr garage --lighting dim

# batch several tasks (or `all`) in one Isaac Sim boot
uv run sim-env-builder rollout --task open-dishwasher,open-printer --episodes 1
```

### 5. Review results in the dashboard

```bash
uv run sim-env-builder dashboard              # serves outputs/ at http://localhost:8090
uv run sim-env-builder dashboard --port 9000  # serve on a different port
```

The dashboard visualizes each episode's progress and milestones.

## Adding assets

To add a new asset, you can use any existing sim-ready asset online: any
USD with physics articulation (joints, limits, colliders, mass) works.
Alternatively, use the
[Moonlake Asset API](https://moonlakeai.mintlify.site/introduction) to
generate a high-fidelity, sim-ready asset with articulation built in.

1. Create an [API key](https://app.moonlakeai.com/3d-agent-api) and generate
   an asset (see the
   [generation guide](https://moonlakeai.mintlify.site/guides/generate-assets)).
2. Add a task entry in `src/sim_env_builder/tasks.py` (asset name,
   instruction, placement flags), then follow the [workflow](#workflow)
   above.

## Contact us

- For technical questions and feature requests, use GitHub
  [Issues](https://github.com/MoonlakeAI/sim-builder-poc/issues).
- For discussion with fellow users, join our
  [Discord channel](https://discord.gg/ZJZB2vymnY).
- To use Moonlake's logo, refer to our [media kit](media_kit/).
- For collaborations and partnerships, contact
  [contact@moonlakeai.com](mailto:contact@moonlakeai.com).
