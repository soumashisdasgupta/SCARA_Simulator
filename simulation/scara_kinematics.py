"""
scara_kinematics.py
===================
Phase-1 SCARA Pick-and-Place Simulation — Interactive Visualiser.

Sidebar controls
----------------
  INITIAL POSITION
    · X / Y sliders   — drag to move the start point live
    · X / Y text boxes — type exact mm values + Enter

  FINAL POSITION
    · Fx / Fy text boxes — type the target destination coordinates

  PLACE button — animates the arm smoothly from Initial → Final,
                 then shows "Done!  Enter new targets."

  Reset button — clears both positions back to defaults

Modes (CLI)
-----------
  python simulation/scara_kinematics.py               # pick-and-place UI
  python simulation/scara_kinematics.py --ix 300 --iy 0 --fx 0 --fy 350
  python simulation/scara_kinematics.py --demo-grid   # static IK grid
  python simulation/scara_kinematics.py --workspace   # workspace heatmap

Link lengths:  L1 = 300 mm  |  L2 = 200 mm
Workspace   :  100 mm (inner dead-zone) – 500 mm (outer reach)

Author : Soumashis Dasgupta
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.widgets as mwidgets
from matplotlib.animation import FuncAnimation
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_L1: float = 300.0
DEFAULT_L2: float = 200.0

JOINT1_LIMIT_DEG: tuple[float, float] = (-150.0, 150.0)
JOINT2_LIMIT_DEG: tuple[float, float] = (-150.0, 150.0)

FK_TOLERANCE: float = 1e-9

DEFAULT_INIT_X: float = 300.0
DEFAULT_INIT_Y: float =   0.0
DEFAULT_FINAL_X: float =   0.0
DEFAULT_FINAL_Y: float = 350.0

ANIM_FRAMES:   int   = 60     # number of interpolation steps
ANIM_INTERVAL: int   = 18     # ms per frame  (~55 fps feel)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_DARK        = "#0d1117"
SIDEBAR_BG     = "#161b22"
GRID_COLOR     = "#21262d"
WORKSPACE_FILL = "#161b22"
WORKSPACE_EDGE = "#30363d"

COLOR_INIT  = "#58a6ff"   # initial-arm blue
COLOR_FINAL = "#3fb950"   # final-arm green (used for ghost)
COLOR_TGT_I = "#58a6ff"   # initial pos marker
COLOR_TGT_F = "#e3b341"   # final pos marker (gold)
COLOR_BASE  = "#e3b341"   # shoulder
COLOR_ANIM  = "#a5d6ff"   # arm during animation (lighter blue)
COLOR_TEXT  = "#c9d1d9"
COLOR_DIM   = "#484f58"
COLOR_WARN  = "#d29922"
COLOR_GOOD  = "#3fb950"
COLOR_ERR   = "#f85149"
COLOR_BTN   = "#238636"   # Place button green
COLOR_BTN_H = "#2ea043"

JOINT_RADIUS = 8.0

# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JointAngles:
    theta1_rad: float
    theta2_rad: float
    config: str

    @property
    def theta1_deg(self) -> float:
        return math.degrees(self.theta1_rad)

    @property
    def theta2_deg(self) -> float:
        return math.degrees(self.theta2_rad)


@dataclass
class ArmConfig:
    L1: float
    L2: float

    @property
    def outer_radius(self) -> float:
        return self.L1 + self.L2

    @property
    def inner_radius(self) -> float:
        return abs(self.L1 - self.L2)

# ──────────────────────────────────────────────────────────────────────────────
# KINEMATICS CORE
# ──────────────────────────────────────────────────────────────────────────────

def is_reachable(x: float, y: float, arm: ArmConfig) -> bool:
    r_sq = x ** 2 + y ** 2
    return arm.inner_radius ** 2 <= r_sq <= arm.outer_radius ** 2


def ik_solve(
    x: float, y: float, arm: ArmConfig,
) -> tuple[Optional[JointAngles], Optional[JointAngles]]:
    """
    2-link planar IK.  Returns (elbow_up, elbow_down); either may be None.

    cos(θ₂) = (x²+y² − L1²−L2²) / (2·L1·L2)
    θ₁      = atan2(y,x) − atan2(L2·sin θ₂,  L1+L2·cos θ₂)
    """
    if not is_reachable(x, y, arm):
        return None, None

    c2 = (x ** 2 + y ** 2 - arm.L1 ** 2 - arm.L2 ** 2) / (2.0 * arm.L1 * arm.L2)
    c2 = max(-1.0, min(1.0, c2))

    solutions: list[Optional[JointAngles]] = []
    for theta2, label in [(+math.acos(c2), "elbow_up"),
                          (-math.acos(c2), "elbow_down")]:
        k1 = arm.L1 + arm.L2 * math.cos(theta2)
        k2 = arm.L2 * math.sin(theta2)
        theta1 = math.atan2(y, x) - math.atan2(k2, k1)

        if not (JOINT1_LIMIT_DEG[0] <= math.degrees(theta1) <= JOINT1_LIMIT_DEG[1]):
            solutions.append(None); continue
        if not (JOINT2_LIMIT_DEG[0] <= math.degrees(theta2) <= JOINT2_LIMIT_DEG[1]):
            solutions.append(None); continue

        solutions.append(JointAngles(theta1, theta2, label))

    return solutions[0], solutions[1]


def fk_solve(angles: JointAngles, arm: ArmConfig) -> tuple[float, float]:
    t1, t2 = angles.theta1_rad, angles.theta2_rad
    return (arm.L1 * math.cos(t1) + arm.L2 * math.cos(t1 + t2),
            arm.L1 * math.sin(t1) + arm.L2 * math.sin(t1 + t2))


def fk_elbow(angles: JointAngles, arm: ArmConfig) -> tuple[float, float]:
    return (arm.L1 * math.cos(angles.theta1_rad),
            arm.L1 * math.sin(angles.theta1_rad))


def verify_fk(x: float, y: float, angles: JointAngles, arm: ArmConfig) -> float:
    fx, fy = fk_solve(angles, arm)
    return math.hypot(fx - x, fy - y)


def best_solution(
    x: float, y: float, arm: ArmConfig
) -> Optional[JointAngles]:
    """Return the elbow-up solution, or elbow-down if up is infeasible."""
    eu, ed = ik_solve(x, y, arm)
    return eu if eu is not None else ed

# ──────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _draw_arm(
    ax: plt.Axes,
    angles: JointAngles,
    arm: ArmConfig,
    color: str,
    alpha: float = 1.0,
    lw_upper: float = 4.5,
    lw_lower: float = 3.5,
    linestyle: str = "-",
    zorder: int = 4,
) -> list:
    """Draw arm links + joints; return list of artists for later removal."""
    artists = []
    ex, ey   = fk_elbow(angles, arm)
    ee_x, ee_y = fk_solve(angles, arm)

    artists += ax.plot([0, ex], [0, ey],
                       color=color, linewidth=lw_upper,
                       linestyle=linestyle, solid_capstyle="round",
                       alpha=alpha, zorder=zorder)
    artists += ax.plot([ex, ee_x], [ey, ee_y],
                       color=color, linewidth=lw_lower,
                       linestyle=linestyle, solid_capstyle="round",
                       alpha=alpha, zorder=zorder)
    for (px, py), r in [((ex, ey), JOINT_RADIUS),
                         ((ee_x, ee_y), JOINT_RADIUS * 0.75)]:
        c = plt.Circle((px, py), r, color=color, alpha=alpha, zorder=zorder + 1)
        ax.add_patch(c)
        artists.append(c)

    return artists


def _draw_position_marker(
    ax: plt.Axes,
    x: float, y: float,
    arm: ArmConfig,
    color: str,
    label: str,
    zorder: int = 6,
) -> list:
    """Draw a crosshair + dot + label at (x, y)."""
    artists = []
    cross = arm.outer_radius * 0.028
    artists += ax.plot([x - cross, x + cross], [y, y],
                       color=color, linewidth=1.8, zorder=zorder)
    artists += ax.plot([x, x], [y - cross, y + cross],
                       color=color, linewidth=1.8, zorder=zorder)
    dot = plt.Circle((x, y), cross * 0.4, color=color, zorder=zorder + 1)
    ax.add_patch(dot)
    artists.append(dot)
    ann = ax.annotate(
        f" {label}\n ({x:.0f}, {y:.0f}) mm",
        xy=(x, y),
        xytext=(x + cross * 2.5, y + cross * 2.5),
        color=color, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
    )
    artists.append(ann)
    return artists


def _draw_workspace_annulus(ax: plt.Axes, arm: ArmConfig) -> None:
    outer = mpatches.Annulus(
        (0, 0), r=arm.outer_radius,
        width=arm.outer_radius - arm.inner_radius,
        facecolor=WORKSPACE_FILL, edgecolor=WORKSPACE_EDGE,
        linewidth=1.2, zorder=1,
    )
    ax.add_patch(outer)
    ax.annotate(f"R={arm.outer_radius:.0f}mm",
                xy=(arm.outer_radius * 0.64, arm.outer_radius * 0.64),
                color=COLOR_DIM, fontsize=7.5, ha="center")
    ax.annotate(f"r={arm.inner_radius:.0f}mm",
                xy=(arm.inner_radius * 0.45, arm.inner_radius * 0.45),
                color=COLOR_DIM, fontsize=7.5, ha="center")

# ──────────────────────────────────────────────────────────────────────────────
# INTERACTIVE PICK-AND-PLACE VISUALISER
# ──────────────────────────────────────────────────────────────────────────────

class ScaraPickPlaceViz:
    """
    Interactive Pick-and-Place visualiser.

    Left sidebar:
      · Initial Position  — X/Y sliders + text boxes (live arm preview)
      · Final Position    — Fx/Fy text boxes           (golden crosshair)
      · PLACE button      — animates arm Initial → Final
      · Status panel      — joint angles, reachability, done message
      · Reset button      — restore defaults

    Main plot:
      · Workspace annulus (static)
      · Arm at Initial position (blue, live)
      · Ghost arm at Final position (green, dashed, shown when valid)
      · Animated sweep on Place
    """

    _SLIDE_MARGIN = 1.05   # slider range = outer_radius × this

    def __init__(
        self, arm: ArmConfig,
        init_x: float, init_y: float,
        final_x: float, final_y: float,
    ) -> None:
        self.arm     = arm
        self.init_x  = init_x
        self.init_y  = init_y
        self.final_x = final_x
        self.final_y = final_y

        self._arm_artists:    list = []   # initial-position arm
        self._ghost_artists:  list = []   # final-position ghost arm
        self._marker_artists: list = []   # crosshair markers
        self._overlay_artist: Optional[plt.Text] = None  # Done / Error banner
        self._anim: Optional[FuncAnimation] = None
        self._animating: bool = False

        self._build_figure()
        self._build_sidebar()
        self._build_main_axes()
        self._full_redraw()

    # ── Figure layout ──────────────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self.fig = plt.figure(
            figsize=(15, 10),
            facecolor=BG_DARK,
            num="SCARA — Pick & Place Simulator",
        )
        gs = gridspec.GridSpec(1, 2, figure=self.fig,
                               width_ratios=[3, 7], wspace=0.0)
        self.ax_sb   = self.fig.add_subplot(gs[0])
        self.ax_main = self.fig.add_subplot(gs[1])

        self.ax_sb.set_facecolor(SIDEBAR_BG)
        self.ax_sb.set_xticks([])
        self.ax_sb.set_yticks([])
        for sp in self.ax_sb.spines.values():
            sp.set_edgecolor("#30363d")

        self.ax_main.set_facecolor(BG_DARK)
        self.fig.patch.set_facecolor(BG_DARK)

    # ── Sidebar helpers ────────────────────────────────────────────────────────

    def _sb_ax(self, y0: float, h: float, indent: float = 0.05) -> plt.Axes:
        """Create a widget sub-axes positioned inside the sidebar."""
        pos = self.ax_sb.get_position()
        left   = pos.x0 + indent * pos.width
        width  = pos.width * (1 - 2 * indent)
        bottom = pos.y0 + y0 * pos.height
        height = h * pos.height
        return self.fig.add_axes([left, bottom, width, height])

    def _divider(self, y: float) -> None:
        self.ax_sb.plot([0.04, 0.96], [y, y],
                        color="#30363d", linewidth=0.8,
                        transform=self.ax_sb.transAxes, clip_on=False)

    def _sb_label(self, y: float, text: str,
                  size: float = 9.0, color: str = COLOR_TEXT,
                  bold: bool = False) -> None:
        self.ax_sb.text(
            0.07, y, text,
            transform=self.ax_sb.transAxes,
            fontsize=size, color=color, va="center",
            fontweight="bold" if bold else "normal",
        )

    def _sb_section(self, y: float, text: str, color: str = COLOR_TEXT) -> None:
        self.ax_sb.text(
            0.5, y, text,
            transform=self.ax_sb.transAxes,
            ha="center", fontsize=9.5, color=color,
            fontweight="bold", va="center",
        )

    def _make_textbox(self, y0: float, initial: str,
                      indent: float = 0.28) -> mwidgets.TextBox:
        ax = self._sb_ax(y0, 0.036, indent=indent)
        tb = mwidgets.TextBox(ax, "", initial=initial,
                              color="#21262d", hovercolor="#2d333b",
                              label_pad=0.0)
        tb.label.set_visible(False)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        return tb

    def _make_slider(self, y0: float, lo: float, hi: float,
                     val: float, color: str) -> mwidgets.Slider:
        ax = self._sb_ax(y0, 0.036)
        sl = mwidgets.Slider(ax, "", lo, hi, valinit=val,
                             color=color, initcolor=color,
                             track_color="#21262d")
        sl.label.set_visible(False)
        sl.valtext.set_color(color)
        sl.valtext.set_fontsize(7.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        return sl

    # ── Build sidebar ──────────────────────────────────────────────────────────

    def _build_sidebar(self) -> None:
        arm   = self.arm
        s_lim = arm.outer_radius * self._SLIDE_MARGIN

        # ── Header ────────────────────────────────────────────────────────────
        self.ax_sb.text(
            0.5, 0.968, "[ Pick & Place Control ]",
            transform=self.ax_sb.transAxes,
            ha="center", va="top",
            fontsize=12, fontweight="bold", color=COLOR_TEXT,
        )
        self.ax_sb.text(
            0.5, 0.935,
            f"L1 = {arm.L1:.0f} mm   |   L2 = {arm.L2:.0f} mm\n"
            f"Workspace: {arm.inner_radius:.0f} – {arm.outer_radius:.0f} mm",
            transform=self.ax_sb.transAxes,
            ha="center", va="top",
            fontsize=8, color=COLOR_DIM, linespacing=1.6,
        )
        self._divider(0.900)

        # ════════════════════════════════════════════════════════════════════
        #  INITIAL POSITION
        # ════════════════════════════════════════════════════════════════════
        self._sb_section(0.878, "INITIAL POSITION", color=COLOR_INIT)

        # X
        self._sb_label(0.848, "X  (mm)")
        self.sl_ix = self._make_slider(0.808, -s_lim, s_lim,
                                       self.init_x, COLOR_INIT)
        self.tb_ix = self._make_textbox(0.766, f"{self.init_x:.1f}")

        # Y
        self._sb_label(0.736, "Y  (mm)")
        self.sl_iy = self._make_slider(0.696, -s_lim, s_lim,
                                       self.init_y, COLOR_INIT)
        self.tb_iy = self._make_textbox(0.654, f"{self.init_y:.1f}")

        self._divider(0.628)

        # ════════════════════════════════════════════════════════════════════
        #  FINAL POSITION
        # ════════════════════════════════════════════════════════════════════
        self._sb_section(0.607, "FINAL POSITION", color=COLOR_TGT_F)

        self._sb_label(0.574, "Fx  (mm)")
        self.tb_fx = self._make_textbox(0.534, f"{self.final_x:.1f}")

        self._sb_label(0.504, "Fy  (mm)")
        self.tb_fy = self._make_textbox(0.464, f"{self.final_y:.1f}")

        self._divider(0.438)

        # ════════════════════════════════════════════════════════════════════
        #  PLACE BUTTON
        # ════════════════════════════════════════════════════════════════════
        ax_place = self._sb_ax(0.355, 0.068, indent=0.06)
        self.btn_place = mwidgets.Button(
            ax_place, "PLACE",
            color=COLOR_BTN, hovercolor=COLOR_BTN_H,
        )
        self.btn_place.label.set_color("white")
        self.btn_place.label.set_fontsize(13)
        self.btn_place.label.set_fontweight("bold")

        self._divider(0.338)

        # ════════════════════════════════════════════════════════════════════
        #  LIVE INFO PANEL
        # ════════════════════════════════════════════════════════════════════
        self._sb_section(0.320, "Live Info", color=COLOR_DIM)
        self._info_text = self.ax_sb.text(
            0.07, 0.302, "",
            transform=self.ax_sb.transAxes,
            va="top", fontsize=7.8, color=COLOR_TEXT,
            fontfamily="monospace", linespacing=1.7,
        )

        self._divider(0.072)

        # ════════════════════════════════════════════════════════════════════
        #  RESET BUTTON
        # ════════════════════════════════════════════════════════════════════
        ax_rst = self._sb_ax(0.016, 0.044, indent=0.12)
        self.btn_reset = mwidgets.Button(
            ax_rst, "Reset to defaults",
            color="#21262d", hovercolor="#2d333b",
        )
        self.btn_reset.label.set_color(COLOR_DIM)
        self.btn_reset.label.set_fontsize(8)

        # ── Wire callbacks ────────────────────────────────────────────────
        self.sl_ix.on_changed(self._cb_sl_ix)
        self.sl_iy.on_changed(self._cb_sl_iy)
        self.tb_ix.on_submit(self._cb_tb_ix)
        self.tb_iy.on_submit(self._cb_tb_iy)
        self.tb_fx.on_submit(self._cb_tb_fx)
        self.tb_fy.on_submit(self._cb_tb_fy)
        self.btn_place.on_clicked(self._cb_place)
        self.btn_reset.on_clicked(self._cb_reset)

    # ── Build main axes (static elements) ─────────────────────────────────────

    def _build_main_axes(self) -> None:
        ax  = self.ax_main
        arm = self.arm
        lim = arm.outer_radius * 1.18

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)
        ax.tick_params(colors=COLOR_DIM, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COLOR)
        ax.set_xlabel("X  (mm)", color=COLOR_DIM, fontsize=9)
        ax.set_ylabel("Y  (mm)", color=COLOR_DIM, fontsize=9)

        _draw_workspace_annulus(ax, arm)
        ax.axhline(0, color=GRID_COLOR, linewidth=0.8, zorder=1)
        ax.axvline(0, color=GRID_COLOR, linewidth=0.8, zorder=1)

        ax.add_patch(plt.Circle((0, 0), JOINT_RADIUS * 1.4,
                                color=COLOR_BASE, zorder=5))
        ax.annotate("Base", xy=(JOINT_RADIUS * 2.5, JOINT_RADIUS * 2.5),
                    color=COLOR_BASE, fontsize=8, fontweight="bold")

        legend_els = [
            mpatches.Patch(facecolor=COLOR_INIT,  label="Initial position  (blue)"),
            mpatches.Patch(facecolor=COLOR_TGT_F, label="Final target        (gold)"),
            mpatches.Patch(facecolor=COLOR_FINAL, label="Final arm ghost  (green)"),
            mpatches.Patch(facecolor=COLOR_BASE,  label="Base / Shoulder"),
        ]
        ax.legend(handles=legend_els, loc="lower right",
                  facecolor="#161b22", edgecolor="#30363d",
                  labelcolor=COLOR_TEXT, fontsize=8.5)

        ax.set_title(
            f"SCARA Pick & Place Simulator"
            f"   |   L1={arm.L1:.0f} mm   L2={arm.L2:.0f} mm",
            color=COLOR_TEXT, fontsize=11, fontweight="bold", pad=12,
        )

    # ── Artist management ──────────────────────────────────────────────────────

    def _clear(self, attr: str) -> None:
        for a in getattr(self, attr, []):
            try:
                a.remove()
            except Exception:
                pass
        setattr(self, attr, [])

    def _clear_overlay(self) -> None:
        if self._overlay_artist is not None:
            try:
                self._overlay_artist.remove()
            except Exception:
                pass
            self._overlay_artist = None

    # ── Redraw routines ────────────────────────────────────────────────────────

    def _full_redraw(self) -> None:
        """Recompute IK for both positions and refresh all dynamic artists."""
        if self._animating:
            return
        self._clear_overlay()
        self._draw_initial_arm()
        self._draw_final_ghost()
        self._draw_markers()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _draw_initial_arm(self) -> None:
        self._clear("_arm_artists")
        sol = best_solution(self.init_x, self.init_y, self.arm)
        if sol is not None:
            self._arm_artists += _draw_arm(
                self.ax_main, sol, self.arm,
                color=COLOR_INIT, alpha=1.0, zorder=4,
            )

    def _draw_final_ghost(self) -> None:
        self._clear("_ghost_artists")
        sol = best_solution(self.final_x, self.final_y, self.arm)
        if sol is not None:
            self._ghost_artists += _draw_arm(
                self.ax_main, sol, self.arm,
                color=COLOR_FINAL, alpha=0.30,
                lw_upper=3.5, lw_lower=2.5,
                linestyle="--", zorder=3,
            )

    def _draw_markers(self) -> None:
        self._clear("_marker_artists")
        if is_reachable(self.init_x, self.init_y, self.arm):
            self._marker_artists += _draw_position_marker(
                self.ax_main, self.init_x, self.init_y,
                self.arm, COLOR_TGT_I, "Initial", zorder=6,
            )
        if is_reachable(self.final_x, self.final_y, self.arm):
            self._marker_artists += _draw_position_marker(
                self.ax_main, self.final_x, self.final_y,
                self.arm, COLOR_TGT_F, "Final", zorder=6,
            )

    def _update_info(self) -> None:
        arm = self.arm
        ix, iy = self.init_x, self.init_y
        fx, fy = self.final_x, self.final_y

        def pos_lines(x, y, tag):
            r = math.hypot(x, y)
            reach = is_reachable(x, y, arm)
            sol = best_solution(x, y, arm)
            lines = [
                f"{tag}:  ({x:+.1f}, {y:+.1f}) mm",
                f"  r={r:.1f}  {'OK' if reach else 'OUT OF REACH'}",
            ]
            if sol:
                lines.append(
                    f"  t1={sol.theta1_deg:+.1f}  t2={sol.theta2_deg:+.1f}"
                )
            return lines

        all_lines = (
            pos_lines(ix, iy, "INIT ") +
            [""] +
            pos_lines(fx, fy, "FINAL")
        )
        self._info_text.set_text("\n".join(all_lines))
        self.fig.canvas.draw_idle()

    def _show_overlay(self, msg: str, color: str) -> None:
        self._clear_overlay()
        self._overlay_artist = self.ax_main.text(
            0.5, 0.06, msg,
            transform=self.ax_main.transAxes,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor=BG_DARK, edgecolor=color, alpha=0.92),
            zorder=20,
        )

    # ── ANIMATION ─────────────────────────────────────────────────────────────

    def _cb_place(self, event) -> None:
        if self._animating:
            return

        sol_i = best_solution(self.init_x, self.init_y, self.arm)
        sol_f = best_solution(self.final_x, self.final_y, self.arm)

        if sol_i is None:
            self._show_overlay("Initial position is unreachable!", COLOR_ERR)
            self.fig.canvas.draw_idle()
            return
        if sol_f is None:
            self._show_overlay("Final position is unreachable!", COLOR_ERR)
            self.fig.canvas.draw_idle()
            return

        # Hide static artists during animation
        for a in self._arm_artists + self._ghost_artists + self._marker_artists:
            try:
                a.set_visible(False)
            except Exception:
                pass
        self._clear_overlay()

        self._animating = True
        arm = self.arm

        t1_i, t2_i = sol_i.theta1_rad, sol_i.theta2_rad
        t1_f, t2_f = sol_f.theta1_rad, sol_f.theta2_rad

        # We keep one set of arm artists that we mutate each frame
        # Pre-create them at the start position
        anim_sol = JointAngles(t1_i, t2_i, "elbow_up")
        self._anim_artists: list = _draw_arm(
            self.ax_main, anim_sol, arm,
            color=COLOR_ANIM, alpha=1.0, zorder=4,
        )

        N = ANIM_FRAMES

        def _update(frame: int):
            # Remove previous animated arm
            for a in self._anim_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            self._anim_artists.clear()

            t = frame / max(N - 1, 1)
            # Smooth ease in-out (sine easing)
            t_ease = 0.5 - 0.5 * math.cos(math.pi * t)

            th1 = t1_i + t_ease * (t1_f - t1_i)
            th2 = t2_i + t_ease * (t2_f - t2_i)
            interp = JointAngles(th1, th2, "elbow_up")

            # Color transitions blue → green
            def lerp_hex(c1, c2, frac):
                r1, g1, b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
                r2, g2, b2 = int(c2[1:3],16), int(c2[3:5],16), int(c2[5:7],16)
                r = int(r1 + frac*(r2-r1))
                g = int(g1 + frac*(g2-g1))
                b = int(b1 + frac*(b2-b1))
                return f"#{r:02x}{g:02x}{b:02x}"

            col = lerp_hex(COLOR_ANIM, COLOR_FINAL, t_ease)
            self._anim_artists = _draw_arm(
                self.ax_main, interp, arm,
                color=col, alpha=1.0, zorder=4,
            )

            if frame == N - 1:
                self._on_anim_done(sol_f)

            return self._anim_artists

        self._anim = FuncAnimation(
            self.fig, _update,
            frames=N, interval=ANIM_INTERVAL,
            blit=False, repeat=False,
        )
        self.fig.canvas.draw_idle()

    def _on_anim_done(self, final_sol: JointAngles) -> None:
        """Called after the last animation frame."""
        self._animating = False

        # Restore ghost / markers hidden during anim
        for a in self._ghost_artists + self._marker_artists:
            try:
                a.set_visible(True)
            except Exception:
                pass

        self._show_overlay("Done!   Enter new targets.", COLOR_GOOD)
        self._update_info()
        self.fig.canvas.draw_idle()

    # ── Callbacks — Initial sliders/textboxes ──────────────────────────────────

    def _cb_sl_ix(self, val: float) -> None:
        self.init_x = round(val, 1)
        self.tb_ix.set_val(f"{self.init_x:.1f}")
        self._full_redraw()

    def _cb_sl_iy(self, val: float) -> None:
        self.init_y = round(val, 1)
        self.tb_iy.set_val(f"{self.init_y:.1f}")
        self._full_redraw()

    def _cb_tb_ix(self, text: str) -> None:
        try:
            self.init_x = float(text)
        except ValueError:
            return
        self.sl_ix.set_val(self.init_x)   # triggers _cb_sl_ix → _full_redraw

    def _cb_tb_iy(self, text: str) -> None:
        try:
            self.init_y = float(text)
        except ValueError:
            return
        self.sl_iy.set_val(self.init_y)

    # ── Callbacks — Final textboxes ────────────────────────────────────────────

    def _cb_tb_fx(self, text: str) -> None:
        try:
            self.final_x = float(text)
        except ValueError:
            return
        self._full_redraw()

    def _cb_tb_fy(self, text: str) -> None:
        try:
            self.final_y = float(text)
        except ValueError:
            return
        self._full_redraw()

    # ── Callbacks — Reset ──────────────────────────────────────────────────────

    def _cb_reset(self, event) -> None:
        if self._animating:
            return
        self.init_x  = DEFAULT_INIT_X
        self.init_y  = DEFAULT_INIT_Y
        self.final_x = DEFAULT_FINAL_X
        self.final_y = DEFAULT_FINAL_Y
        self.tb_fx.set_val(f"{self.final_x:.1f}")
        self.tb_fy.set_val(f"{self.final_y:.1f}")
        self.sl_ix.set_val(DEFAULT_INIT_X)   # triggers redraw chain
        self.sl_iy.set_val(DEFAULT_INIT_Y)

    def show(self) -> None:
        plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# STATIC MODES (unchanged from previous version)
# ──────────────────────────────────────────────────────────────────────────────

DEMO_TARGETS: list[tuple[float, float]] = [
    (350.0,   0.0),
    (300.0, 150.0),
    (  0.0, 400.0),
    (-200.0, 300.0),
    (150.0, 250.0),
    (480.0,  50.0),
    (110.0,  20.0),
]


def plot_workspace_grid(arm: ArmConfig, resolution: int = 60) -> plt.Figure:
    lim = arm.outer_radius * 1.1
    xs = np.linspace(-lim, lim, resolution)
    ys = np.linspace(-lim, lim, resolution)
    X, Y = np.meshgrid(xs, ys)
    reachable = np.vectorize(lambda x, y: is_reachable(x, y, arm))(X, Y).astype(float)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.contourf(X, Y, reachable, levels=1,
                colors=[WORKSPACE_FILL, "#1f6feb"], alpha=0.6)
    ax.contour(X, Y, reachable, levels=1,
               colors=[WORKSPACE_EDGE], linewidths=1.5)
    ax.set_title(
        f"SCARA Reachable Workspace\n"
        f"L1={arm.L1:.0f} mm  |  L2={arm.L2:.0f} mm  |  "
        f"R: {arm.inner_radius:.0f}–{arm.outer_radius:.0f} mm",
        color=COLOR_TEXT, fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("X  (mm)", color=COLOR_DIM)
    ax.set_ylabel("Y  (mm)", color=COLOR_DIM)
    ax.tick_params(colors=COLOR_DIM)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COLOR)
    ax.annotate(
        f"Outer: {arm.outer_radius:.0f} mm\nInner: {arm.inner_radius:.0f} mm",
        xy=(0.02, 0.04), xycoords="axes fraction",
        color=COLOR_TEXT, fontsize=9,
        bbox=dict(boxstyle="round", facecolor=BG_DARK,
                  edgecolor=WORKSPACE_EDGE, alpha=0.9),
    )
    fig.tight_layout()
    return fig


def run_demo_grid(arm: ArmConfig) -> None:
    n    = len(DEMO_TARGETS)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5),
                              facecolor=BG_DARK)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (tx, ty) in enumerate(DEMO_TARGETS):
        ax = axes_flat[idx]
        eu, ed = ik_solve(tx, ty, arm)
        _draw_workspace_annulus(ax, arm)
        if eu:
            _draw_arm(ax, eu, arm, COLOR_INIT, alpha=1.0)
        if ed:
            _draw_arm(ax, ed, arm, COLOR_FINAL, alpha=0.6, linestyle="--")
        ax.plot(tx, ty, "+", color=COLOR_TGT_F,
                markersize=14, markeredgewidth=2, zorder=6)
        ax.add_patch(plt.Circle((0, 0), 6, color=COLOR_BASE, zorder=5))
        ax.set_facecolor(BG_DARK)
        lim = arm.outer_radius * 1.15
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)
        ax.tick_params(colors=COLOR_DIM, labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COLOR)
        ax.set_title(f"({tx:.0f}, {ty:.0f}) mm",
                     color=COLOR_TEXT, fontsize=8)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"SCARA IK — Demo Target Grid\nL1={arm.L1:.0f} mm  |  L2={arm.L2:.0f} mm",
        color=COLOR_TEXT, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SCARA Phase-1 Pick-and-Place Visualiser",
    )
    p.add_argument("--L1",  type=float, default=DEFAULT_L1)
    p.add_argument("--L2",  type=float, default=DEFAULT_L2)
    p.add_argument("--ix",  type=float, default=DEFAULT_INIT_X,  help="Initial X (mm)")
    p.add_argument("--iy",  type=float, default=DEFAULT_INIT_Y,  help="Initial Y (mm)")
    p.add_argument("--fx",  type=float, default=DEFAULT_FINAL_X, help="Final X (mm)")
    p.add_argument("--fy",  type=float, default=DEFAULT_FINAL_Y, help="Final Y (mm)")
    p.add_argument("--demo-grid",  action="store_true")
    p.add_argument("--workspace",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arm  = ArmConfig(L1=args.L1, L2=args.L2)

    print(f"\n  SCARA Pick & Place Simulator — Phase 1")
    print(f"  L1={arm.L1:.0f} mm  L2={arm.L2:.0f} mm "
          f"| Workspace: {arm.inner_radius:.0f}–{arm.outer_radius:.0f} mm\n")

    if args.workspace:
        plt.show(plot_workspace_grid(arm))
        return

    if args.demo_grid:
        run_demo_grid(arm)
        return

    viz = ScaraPickPlaceViz(
        arm,
        init_x=args.ix, init_y=args.iy,
        final_x=args.fx, final_y=args.fy,
    )
    viz.show()


if __name__ == "__main__":
    main()
