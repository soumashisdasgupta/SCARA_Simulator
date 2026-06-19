# SCARA_Simulator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-orange?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Phase](https://img.shields.io/badge/Phase-1%20of%203-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A 4-DoF SCARA Robot Pick-and-Place Simulator**  
*From inverse kinematics math to animated arm control — built modularly for real hardware integration.*

</div>

---

## Overview

This project simulates a **SCARA (Selective Compliance Assembly Robot Arm)** designed for autonomous pick-and-place and assembly operations. It combines 2-link planar inverse kinematics, an interactive matplotlib GUI, and a smooth pick-and-place animation engine — all designed to evolve into a full hardware-in-the-loop control system across three development phases.

The robot is inspired by real industrial SCARA systems and targets operations such as:
- Component sorting and placement
- Precision insertion (peg-in-hole)
- Fastener tightening with closed-loop torque control

---

## Robot Specifications

| Parameter | Value |
|---|---|
| **Type** | SCARA — Selective Compliance Assembly Robot Arm |
| **Degrees of Freedom** | 4 DoF |
| **Joint 1 (θ₁)** | Shoulder — rotational (planar XY) |
| **Joint 2 (θ₂)** | Elbow — rotational (planar XY) |
| **Joint 3 (Z)** | Linear vertical axis (prismatic) |
| **Joint 4 (θ₄)** | Wrist yaw — end-effector rotation |
| **Upper Arm (L1)** | 300 mm |
| **Forearm (L2)** | 200 mm |
| **Inner reach** | 100 mm (dead zone) |
| **Outer reach** | 500 mm |
| **End-Effector** | Pneumatic suction cup / mechanical gripper |
| **MCU Platform** | Raspberry Pi / STM32 *(Phase 2, TBD)* |

---

## Project Structure

```
SCARA_Simulator/
│
├── simulation/                  # Phase 1 — standalone Python scripts
│   └── scara_kinematics.py      # ★ Interactive pick-and-place visualiser
│
├── kinematics/                  # Phase 1 — pure math modules (no hardware deps)
│   ├── forward_kinematics.py
│   ├── inverse_kinematics.py
│   ├── trajectory.py
│   └── workspace.py
│
├── vision/                      # Phase 2 — computer vision pipeline
│   ├── capture.py
│   ├── color_filter.py          # BGR → HSV color masking
│   ├── contour_detect.py
│   └── coordinate_map.py        # Pixel → mm coordinate transform
│
├── drivers/                     # Phase 2 — low-level hardware abstraction
│   ├── motor_driver.py
│   ├── stepper.py
│   ├── gripper.py
│   └── current_monitor.py       # Torque sensing via current spike detection
│
├── comms/                       # Phase 2 — communication protocols
│   ├── uart_interface.py
│   ├── i2c_interface.py
│   └── spi_interface.py
│
├── state_machine/               # Phase 3 — high-level robot behaviour FSM
│   ├── states.py
│   ├── transitions.py
│   └── orchestrator.py
│
├── config/
│   └── robot_params.yaml        # All tunable parameters — no magic numbers in code
│
├── tests/
│   └── test_ik.py               # IK unit tests (pytest)
│
├── docs/                        # Design documents & schematics
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/soumashisdasgupta/SCARA_Simulator.git
cd SCARA_Simulator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the simulator

```bash
# Interactive pick-and-place GUI (default)
python simulation/scara_kinematics.py

# Pre-set initial and final positions via CLI
python simulation/scara_kinematics.py --ix 250 --iy 100 --fx -200 --fy 300

# Reachable workspace heatmap
python simulation/scara_kinematics.py --workspace

# Static grid of 7 IK demo targets
python simulation/scara_kinematics.py --demo-grid
```

---

## Simulator Features

### Interactive GUI (Pick & Place Mode)

The main window is split into a **sidebar control panel** and a **live arm plot**.

```
┌──────────────────────┬─────────────────────────────────────────┐
│  [ Pick & Place      │                                         │
│    Control ]         │         SCARA Arm Plot                  │
│                      │                                         │
│  INITIAL POSITION    │   · Workspace annulus                   │
│  X slider + textbox  │   · Blue arm  = Initial position        │
│  Y slider + textbox  │   · Green arm = Final ghost (dashed)    │
│                      │   · Gold crosshair = Final target       │
│  FINAL POSITION      │   · Animated sweep on PLACE             │
│  Fx textbox          │                                         │
│  Fy textbox          │                                         │
│                      │                                         │
│  [ PLACE ]           │                                         │
│                      │                                         │
│  Live Info Panel     │                                         │
│  θ₁, θ₂, FK error   │                                         │
│                      │                                         │
│  [ Reset ]           │                                         │
└──────────────────────┴─────────────────────────────────────────┘
```

| Control | Function |
|---|---|
| **X / Y sliders** | Drag to move the Initial position in real-time |
| **X / Y text boxes** | Type exact mm coordinates + Enter |
| **Fx / Fy text boxes** | Type Final destination coordinates + Enter |
| **PLACE button** | Animates arm from Initial → Final with ease-in/out |
| **Live Info panel** | Shows θ₁, θ₂, reach status, FK round-trip error |
| **Reset button** | Restores both positions to defaults |

### Animation
- **60 frames** at ~18 ms/frame (~1 second total)
- **Sine ease-in/out** profile — smooth acceleration, no jerk
- **Colour transition** blue → green during the sweep
- **"Done! Enter new targets."** banner on completion
- **Error banner** if either position is outside the reachable workspace

---

## Inverse Kinematics Math

For a 2-link planar arm with link lengths **L1** and **L2**:

$$\cos(\theta_2) = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 \cdot L_1 \cdot L_2}$$

$$\theta_2 = \pm \arccos(c_2) \quad \text{(elbow-up / elbow-down)}$$

$$\theta_1 = \text{atan2}(y,\, x) \;-\; \text{atan2}(L_2 \sin\theta_2,\; L_1 + L_2 \cos\theta_2)$$

**FK verification** (round-trip error < 10⁻¹³ mm):

$$x' = L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2)$$
$$y' = L_1 \sin\theta_1 + L_2 \sin(\theta_1 + \theta_2)$$

Reachable workspace:

$$|L_1 - L_2| \;\leq\; \sqrt{x^2 + y^2} \;\leq\; L_1 + L_2$$
$$100\,\text{mm} \;\leq\; r \;\leq\; 500\,\text{mm}$$

---

## Development Roadmap

### Phase 1 — Mathematical Modelling & Simulation *(current)*
- [x] 2-link planar IK/FK engine
- [x] Reachability & joint-limit checking
- [x] Interactive matplotlib GUI with sidebar
- [x] Pick-and-place animation (ease-in/out)
- [x] `config/robot_params.yaml` — single source of truth for all parameters
- [ ] Full `kinematics/` module (importable classes)
- [ ] Reachable workspace heatmap (`simulation/workspace_plot.py`)
- [ ] Trajectory / S-curve generator
- [ ] Unit test suite (`pytest tests/`)

### Phase 2 — Hardware & Communication
- [ ] Motor/actuator selection (stepper vs. servo)
- [ ] Wiring schematic planning (power rails, signal lines)
- [ ] Communication protocol definition (UART / I²C / SPI)
- [ ] Low-level driver stubs (`drivers/`)
- [ ] Computer vision pipeline (`vision/`) — HSV filter → contour → (X,Y) mm
- [ ] Pixel-to-mm coordinate mapping (camera homography calibration)

### Phase 3 — Control Loop & Path Planning
- [ ] PID controller per joint (with anti-windup)
- [ ] S-curve trajectory integration with motor drivers
- [ ] Finite State Machine: `IDLE → PERCEIVE → PICK → PLACE → FASTEN → VERIFY`
- [ ] Closed-loop torque cutoff (current spike detection for screw tightening)
- [ ] Full autonomous pick-and-place cycle test

---

## Configuration

All physical parameters live in [`config/robot_params.yaml`](config/robot_params.yaml).  
Edit this file to adapt the simulator to a different physical build — **no source code changes needed**.

```yaml
links:
  L1: 300.0   # Upper-arm length (mm)
  L2: 200.0   # Forearm length   (mm)

joints:
  theta1: { min_deg: -150.0, max_deg: 150.0 }
  theta2: { min_deg: -150.0, max_deg: 150.0 }
  z_axis: { stroke_mm: 100.0 }
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | Array math, trig |
| `matplotlib` | ≥ 3.7 | GUI, animation, plots |
| `PyYAML` | ≥ 6.0 | Config file parsing |
| `scipy` | ≥ 1.11 | Trajectory filtering *(future)* |

Phase 2 hardware dependencies (`pyserial`, `smbus2`, `spidev`, `RPi.GPIO`) are listed but commented out in `requirements.txt`.

---

## License

This project is licensed under the **MIT License**.

---

<div align="center">
  <sub>Built as part of the SCARA Pick-and-Place Robot Arm project.</sub>
</div>
