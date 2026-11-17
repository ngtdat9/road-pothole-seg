# Project 2 — Autonomous Drone Road-Following and Pothole Mapping (Simulation) — Project Plan

**Type:** Personal portfolio project (robotics, autonomous inspection, sim-to-real)
**Timeline:** 7 weeks, solo, part-time (~10–12 hours/week), 2026-12-22 to 2027-02-08
**Depends on:** `road-pothole-seg` (Project 1), installed at a pinned version — never copied
**Simulator:** PX4 SITL + Gazebo + ROS 2 Humble on Ubuntu 22.04 (dual boot on the PC)
**Data:** Auto-labelled frames from randomized Gazebo worlds built from CC0 asphalt textures and pothole cutouts from Project 1's training split

## 1. Problem & Motivation

**Current situation:** Road inspection is done by people driving or walking the
road. Drones are increasingly used for infrastructure inspection, but usually
flown by a pilot and reviewed by hand afterwards.

**Problem:** Piloted flights and manual review don't scale to a whole road
network, and a pilot can't steer and look for small defects at the same time.

**Impact:** Inspection stays slow and expensive, so damage is found late.

**Opportunity:** A drone that follows a road on its own using a segmentation model,
and logs every pothole with its position, turns inspection into a repeatable,
unattended job. Reusing an existing perception package shows how one model can
serve both a ground camera and an aerial platform.

**One-sentence goal:**
Make a simulated drone follow a road on its own using an existing road-and-pothole
segmentation package, and produce a map of the potholes it flies over.

## 2. Scope — in tiers

The drone phase is the riskiest part of the work, so success is defined in tiers.
**The Minimum tier is a finished, publishable project.** Everything above it is
improvement, not a requirement.

| Tier | Includes | Target date |
|---|---|---|
| **Minimum** | PX4 + Gazebo + ROS 2 running; one generated road world; drone follows the road using Project 1's road mask; potholes logged to CSV with approximate ground positions | End of Week 9 (Jan 18) |
| **Full** | Randomized train/test worlds; projected labels; fine-tuned aerial model (released as Project 1 v1.1); geo-referenced pothole map scored against ground truth | End of Week 11 (Feb 1) |
| **Stretch** | Four-test-set sim-to-real study; real aerial/UAV test images; procedural-pothole ablation; altitude and speed sweeps; junctions; depth | Only if time allows |

**Cut order if time runs short:** Stretch first, then the number of worlds and the
four-test-set study. Never cut the working flight.

| Out of scope (all tiers) |
|---|
| A real drone |
| Obstacle avoidance and path planning beyond following one road |
| Changes to Project 1's code inside this repo (changes go to Project 1 as a new release) |
| Own road photography |

## 3. Dataset Quality Checklist (sim data)

| Criterion | Assessment |
|---|---|
| Relevant | Frames come from the drone's own downward camera in the same simulator used for flying |
| Sufficient | 500–800 training frames over ~20 worlds (Full tier) — enough to fine-tune, not to train from scratch |
| **Representative — gap** | Simulated rendering, flat ground, simplified lighting. Real-road performance cannot be claimed from sim results alone; stated plainly in the README |
| Diverse | Randomized per world: road curvature, asphalt texture, pothole cutouts, scale, rotation, colour, sun direction, altitude, small tilts |
| Balanced — caveat | Pothole count per world is controlled (15–25), but road still dominates pixels |
| Accurate | Labels are projected from the generator's own label image, so they are exact up to pose timing error; 30 frames checked by overlay |
| Consistent | All frames from the same camera model and resolution |
| Well-Annotated | Pixel-accurate road and pothole masks by construction |
| Ethically Collected | CC0 textures and CC BY pothole cutouts, sources in `sim/assets/LICENSES.md`. No human subjects |
| Reproducible | Every world regenerated from its seed; drone poses logged per frame |

**Leakage rule:** pothole cutouts come from Project 1's **training split only**, and
test worlds use cutouts and asphalt textures never used in training worlds.

## 4. Goal (technical)

Install `road-pothole-seg` at a pinned version and build only the drone-specific
parts around it: a road-centerline extractor, a velocity controller, pixel-to-ground
geometry, ROS 2 nodes (perception, follower, mapper), and a world generator that
builds randomized Gazebo worlds from real pixels. Fly a PX4 SITL quadrotor along a
curved road using the package's road mask, log each pothole's ground position,
and (Full tier) close the dashcam-to-drone gap by fine-tuning on auto-labelled sim
frames and scoring the pothole map against ground truth.

### 4.1 Sim-to-real strategy

| Gap | Project 1 training data | Gazebo |
|---|---|---|
| **Viewpoint** | Dashcam or handheld, low angle | Straight down from 8–16 m |
| **Appearance** | Real asphalt and lighting | Rendered textures, simpler lighting |

**Build the sim from real pixels.** `sim/generate_world.py --seed N`:
1. Draws a random smooth road centerline (~300 m) with varying curvature.
2. Paints it with a random CC0 asphalt texture, lane lines and edges.
3. Pastes 15–25 potholes (cutouts, plus procedural ones) with random rotation,
   scale (30–120 cm) and brightness, with blended edges.
4. Writes a label image (0 = background, 1 = road, 2 = pothole) and `potholes_gt.csv`.
5. Applies colour, brightness, dirt and patch randomization.
6. Cuts the image into 20 × 20 m tiles at ~2 cm per pixel (1,000 × 1,000 px — small
   enough for the RX 580) and writes an `.sdf` world with a randomized sun.

At 10 m, a 60° camera at 640 px sees ~11.5 m of ground, about 1.8 cm per pixel —
hence the 2 cm texture resolution.

| Sim asset | Source |
|---|---|
| Road surface | CC0 asphalt textures (Poly Haven, ambientCG), 10–20, seamless |
| Potholes | Cutouts from Project 1 training images using their masks; near-top-down only, oblique ones straightened with a perspective warp |
| Extra potholes | Procedural: noise-shaped blobs filled with darkened asphalt or gravel, edge shadow, lighter rim |
| Surroundings | One low-resolution grass or aerial texture plane |

**Labels by projection.** For each saved frame, the logged drone position, heading
and altitude plus the camera field of view give the patch of the label image
under the camera, cut and rotated into a YOLO mask. Gazebo's segmentation camera is
not used: potholes baked into the road texture would all be labelled "road".

**Test sets (Stretch tier reports all four; Full tier reports the first three):**

| Test set | Shows |
|---|---|
| Frames from held-out sim test worlds | Viewpoint + appearance gap |
| Project 1 real test split, after fine-tuning | Whether sim training hurt real performance |
| Project 1 real test split, before fine-tuning | Baseline |
| Real aerial/UAV pothole images (if found) | Viewpoint gap alone |

## 5. Resource checklist

| Resource | Status |
|---|---|
| Time | 7 weeks part-time |
| Project 1 package | `road-pothole-seg v1.0`, released |
| Simulation machine | PC (RX 580 8 GB, 12 GB RAM), Ubuntu 22.04 dual boot on the HDD (300 GB free, ~100 GB used) |
| Simulation fallback | Rented cloud Ubuntu machine with an NVIDIA GPU — decided at end of Week 7 |
| Development machine | Mac — `drone_nav` logic and tests, generator script |
| Training compute | Colab — fine-tuning only (Week 10) |
| Sim assets | CC0 textures + Project 1 cutouts — no own photography |
| ROS 2 / PX4 / Gazebo skills | New — Weeks 6–7 budgeted for learning |

**Compute note:** the model runs on CPU through ONNX Runtime at 320–480 px and
5–10 Hz, which is enough for 2–3 m/s. RAM is the tight limit: close other apps
while the sim runs and keep 8 GB of swap. The HDD slows builds and launches, not
the running sim.

## 6. Repository structure

```
pothole-drone/
├── CLAUDE.md
├── README.md
├── pyproject.toml             # depends on road-pothole-seg @ pinned tag
├── configs/
│   ├── flight.yaml            # altitude, speeds, controller gains
│   └── camera.yaml            # resolution, FOV
├── drone_nav/                 # pure Python, no ROS — testable on the Mac
│   ├── types.py               # RoadLine, Command
│   ├── centerline.py          # road mask → angle error, lateral offset
│   ├── controller.py          # RoadLine → Command, smoothing, lost-road behaviour
│   └── geometry.py            # pixel → ground position, metres per pixel
├── ros2_ws/src/pothole_drone_ros/
│   ├── perception_node.py     # camera → road_pothole_seg → masks, potholes
│   ├── follower_node.py       # road mask → drone_nav → PX4 offboard setpoints
│   ├── mapper_node.py         # potholes + pose → pothole_map.csv
│   └── launch/
├── sim/
│   ├── assets/{asphalt,cutouts}/{train,test}/
│   ├── assets/LICENSES.md
│   ├── generate_world.py
│   ├── procedural_pothole.py
│   └── worlds/                # generated, one folder per seed (gitignored)
├── training/
│   ├── scripts/
│   │   ├── make_cutouts.py    # from Project 1's training split
│   │   └── project_labels.py  # frame pose + label image → YOLO mask
│   └── notebooks/finetune_aerial.ipynb
├── tests/                     # pytest for drone_nav and sim scripts
└── docs/
    ├── setup_ubuntu.md        # every install command, in order
    ├── results.md
    ├── log.md
    └── figures/
```

### Interfaces

```python
from road_pothole_seg import Segmenter, analyse_potholes, PotholeTracker  # Project 1

@dataclass
class RoadLine:
    angle_error: float      # radians, 0 = road straight ahead
    offset: float           # -1 to 1, 0 = centred
    valid: bool             # False when too little road is visible

@dataclass
class Command:
    forward: float          # m/s
    lateral: float          # m/s
    yaw_rate: float         # rad/s

def extract_centerline(road_mask: np.ndarray) -> RoadLine: ...
def compute_command(line: RoadLine, state: ControllerState) -> Command: ...
def pixel_to_ground(px: tuple[int, int], pose: DronePose, cam: CameraModel) -> tuple[float, float]: ...
def metres_per_px(altitude: float, cam: CameraModel) -> float: ...
```

If Project 1 lacks something this repo needs, add it to Project 1 as an optional
addition, release a minor version, and update the pin. Never patch it here.

## 7. Week-by-week plan

### Week 6 (Dec 22–28) — Simulator installation
- Back up the PC, shrink the HDD partition by 100 GB, turn off Windows Fast Startup,
  install Ubuntu 22.04 with 8 GB of swap.
- Install ROS 2 Humble, PX4 (pinned release tag) with its setup script, the
  uXRCE-DDS agent, and a workspace with `px4_msgs` matching the PX4 version.
- Fly the default quadrotor from QGroundControl, then run PX4's offboard example
  from a ROS 2 node.
- Write every command into `docs/setup_ubuntu.md` as you go.
- Deliverable: a drone that flies from your own ROS 2 node

### Week 7 (Dec 29–Jan 4) — World generator and drone camera
- Download 10–20 CC0 asphalt textures; write `make_cutouts.py` and build 50–100
  cutouts from Project 1's training split; split textures and cutouts into train/test.
- Write `generate_world.py` and `procedural_pothole.py`; generate one world and
  check it in Gazebo from above.
- Add a downward camera, bridge it to ROS 2, view it in RViz2, and log the drone
  pose with every saved frame. Write a one-command launch file.
- Measure real-time factor and camera frame rate.
- **Decision point:** if real-time factor stays below ~0.5 after lowering
  resolution and world detail, move to the cloud fallback.
- Deliverable: drone hovering at 10 m over a generated road, saving frames with poses

### Week 8 (Jan 5–11) — Road-following logic, offline
- Save 200 frames along the road. Run Project 1's `Segmenter` on them as-is.
  If its road mask is too poor from above, use the generator's projected road
  label as a temporary mask so control work isn't blocked.
- Implement `extract_centerline` and `compute_command` (smoothing, P-control on yaw
  rate and lateral speed, slower in sharp turns, lost-road behaviour).
- Unit-test on the Mac with saved frames: straight, left curve, right curve, no road.
- Deliverable: tested `drone_nav`, no simulator required

### Week 9 (Jan 12–18) — Closing the loop (**Minimum tier**)
- Write the perception and follower nodes; add arming, takeoff, and a safety stop
  after 3 s without a valid road line.
- Tune gains from 1 m/s upward; log cross-track error against the known centerline.
- Add basic pothole logging: `analyse_potholes` + `PotholeTracker` + `pixel_to_ground`
  → CSV of positions.
- Record a full-road flight video.
- Deliverable: **Minimum tier done.** Target: 3 completed runs in a row at 2+ m/s,
  mean cross-track error under 1 m. If the term ends here, write the README and
  release `v1.0-minimum`.

### Week 10 (Jan 19–25) — Domain gap and fine-tuning (Full tier)
- Generate ~20 training worlds and 3 test worlds with separate assets.
- Fly each at 2–3 altitudes; save 500–800 training frames plus held-out test frames.
- Write `project_labels.py`; overlay 30 masks on frames and check alignment.
- Record "before" metrics with Project 1 v1.0 on sim test frames.
- Fine-tune on a mix of sim and real images on Colab; evaluate on the Section 4.1
  test sets.
- Release the fine-tuned weights as **Project 1 v1.1**; update this repo's pin.
- Deliverable: before/after table

### Week 11 (Jan 26–Feb 1) — Pothole map (Full tier)
- Use the v1.1 model's road mask for following.
- Write the mapper node: merge repeated sightings, convert area to m², write
  `pothole_map.csv`, plot it against `potholes_gt.csv`.
- Score flights on the 3 test worlds: recall, false positives, mean position error.
- Deliverable: **Full tier done.** Target: 80%+ recall, mean position error under 1.5 m

### Week 12 (Feb 2–8) — Report and release
- Finish whichever experiments the tier reached; write the error analysis.
- README: demo GIF, architecture diagram linking to Project 1, setup tested from a
  clean install, results, limitations (sim-only claims stated plainly), credits.
- Record a 2–3 minute demo video; tag `v1.0`; add a "Used by" link in Project 1.
- Deliverable: public release

### 7.1 Experiments

| # | Tier | Question | Metric |
|---|---|---|---|
| 1 | Minimum | Does the drone follow the road? | Completion rate over 5 runs, cross-track error |
| 2 | Full | How big is the dashcam-to-drone gap? | mAP and road IoU on sim test worlds, before vs after |
| 3 | Full | Does sim fine-tuning hurt real images? | mAP on Project 1 real test split, before vs after |
| 4 | Full | How good is the map? | Recall, false positives, position error |
| 5 | Stretch | Viewpoint gap alone | mAP on real aerial/UAV images |
| 6 | Stretch | Do procedural potholes help? | mAP with vs without |
| 7 | Stretch | Altitude (8/12/16 m) and speed (1/2/3 m/s) | Cross-track error, recall |

### 7.2 Risks

| Risk | Response |
|---|---|
| PX4 / ROS 2 / Gazebo install problems | Follow the official guide exactly, pin versions, log every command |
| Sim too slow on the PC | Lower resolution, headless recording, cloud fallback after Week 7 |
| Out of RAM | Close other apps, small model input, 8 GB swap |
| Project 1 model fails from above | Expected. Temporary projected road mask in Week 8; fine-tuning in Week 10 |
| Projected labels misaligned | Check pose timing and FOV; overlay 30 frames before training |
| Generated worlds look fake | Blend edges, many textures, colour randomization; judge by sim-test mAP |
| Large textures exhaust GPU memory | 1,000 px tiles, only tiles near the road |
| Momentum loss after Project 1 | Week 6 starts right after the v1.0 release, even if the first session is just backing up the PC |

## 8. Using Claude Code for this project

Install on both the Mac and the Ubuntu side of the PC:
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

### CLAUDE.md
```markdown
# Project context
Simulated PX4 drone (Gazebo, ROS 2 Humble, Ubuntu 22.04) that follows a road and
maps potholes, using the road-pothole-seg package (Project 1) for perception.
Scope is tiered (Minimum / Full / Stretch). See Project2_Drone_Plan.md.

# Architecture rules
- Perception comes only from road_pothole_seg at the pinned version in
  pyproject.toml. Never copy or patch its code here; needed changes go to
  Project 1 as a new release
- drone_nav/ is pure Python: never import rclpy, ROS messages or Gazebo there
- ROS 2 nodes only get data, call road_pothole_seg / drone_nav, and publish —
  no model or control logic in nodes

# Sim data rules
- Worlds are generated only by sim/generate_world.py from a seed; never edit by hand
- Pothole cutouts come from Project 1's training split only
- Test worlds use cutouts and textures never used in training worlds
- Labels come from projecting the world's label image with the logged drone pose,
  not from Gazebo's segmentation camera

# Conventions
- Gains, speeds, altitude and camera settings in configs/, never hardcoded
- Type hints everywhere; every drone_nav module has a test runnable on the Mac
- PX4 pinned to one release tag; uXRCE-DDS with matching px4_msgs
- Machines: Mac = drone_nav and generator, PC Ubuntu = sim (CPU inference),
  Colab = fine-tuning only
- Commit after each working milestone
```

### Workflow
1. Paste the week's section and start with one module or node.
2. In Weeks 6–7, ask Claude Code to explain each install step before you run it
   and to add it to `docs/setup_ubuntu.md`.
3. Paste full ROS 2 / PX4 logs and tracebacks rather than describing errors.
4. Ask it to explain controller and geometry choices — useful for the README.
5. Run long flights and Colab jobs directly; bring logs back for analysis.

## 9. References
- Project 1: `road-pothole-seg` (your GitHub)
- Reference repo: https://github.com/parth12062005/Drone_Road_Following
- PX4 ROS 2 user guide: https://docs.px4.io/main/en/ros2/user_guide
- Poly Haven (CC0 textures): https://polyhaven.com/textures
- ambientCG (CC0 textures): https://ambientcg.com
