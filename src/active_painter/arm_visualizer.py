from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from .arm_control import scripted_contact_pressure, scripted_pose
from .arm_sim import ArmPainterSim, ArmPose
from .config import PainterConfig
from .version import package_version


class ArmPainterVisualizer:
    def __init__(self, sim: ArmPainterSim, speed: float = 1.0) -> None:
        self.sim = sim
        self.speed = speed
        self.paused = False
        self.max_speed = False
        self.sim_time = 0.0
        self.start = time.perf_counter()
        self.last = self.start
        self.code_version = package_version()

        self.fig = plt.figure(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title(f"Active-Inference Arm Painter v{self.code_version} - native Python sim")
        grid = self.fig.add_gridspec(3, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.12, 0.38])
        self.ax3d = self.fig.add_subplot(grid[:, 0], projection="3d")
        self.ax_canvas = self.fig.add_subplot(grid[0, 1])
        self.ax_button = self.fig.add_subplot(grid[1, 1])
        self.ax_text = self.fig.add_subplot(grid[2, 1])
        self.fast_button = Button(self.ax_button, "Max speed: off")
        self.fast_button.on_clicked(self.toggle_max_speed)
        self.ax_text.axis("off")

        self.arm_line, = self.ax3d.plot([], [], [], color="#5ad1c4", linewidth=5, marker="o", markersize=8, zorder=20)
        self.tip_dot, = self.ax3d.plot([], [], [], marker="o", color="#f0734e", markersize=7, zorder=30)
        self.canvas_image = self.ax_canvas.imshow(
            1.0 - self.sim.canvas.observed_tone(),
            cmap="gray_r",
            vmin=0,
            vmax=1,
            origin="upper",
            interpolation="bilinear",
        )
        self.canvas_surface = None
        self._canvas_x, self._canvas_y, self._canvas_z = self._canvas_mesh()
        self.readout = self.ax_text.text(0.0, 1.0, "", va="top", family="monospace", fontsize=10)
        self._setup_axes()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _setup_axes(self) -> None:
        c = self.sim.canvas
        self.ax3d.set_title("4-DOF arm + soft contact canvas")
        self.ax3d.set_xlabel("X in")
        self.ax3d.set_ylabel("Depth to canvas in")
        self.ax3d.set_zlabel("Height in")
        self.ax3d.set_xlim(-18, 18)
        self.ax3d.set_ylim(0, 28)
        self.ax3d.set_zlim(-12, 14)
        self.ax3d.view_init(elev=18, azim=-72)
        self.ax3d.grid(False)
        for axis in (self.ax3d.xaxis, self.ax3d.yaxis, self.ax3d.zaxis):
            axis.pane.set_alpha(0.0)
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))

        xs = [-c.width / 2, c.width / 2, c.width / 2, -c.width / 2, -c.width / 2]
        ys = [c.distance] * 5
        zs = [-c.height / 2, -c.height / 2, c.height / 2, c.height / 2, -c.height / 2]
        self._draw_canvas_surface()
        self.ax3d.plot(xs, ys, zs, color="#222222", linewidth=1.5, zorder=2)

        self.ax_canvas.set_title("Visible pigment on canvas")
        self.ax_canvas.set_xticks([])
        self.ax_canvas.set_yticks([])
        self.ax_canvas.grid(False)

        self.fig.suptitle(
            f"Active-Inference Arm Painter v{self.code_version} - conventional arm simulation below the painting policy",
            fontsize=12,
        )
        self.fig.tight_layout()

    def _canvas_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.sim.canvas
        n = c.config.canvas_size
        x = np.linspace(-c.width / 2, c.width / 2, n)
        z = np.linspace(c.height / 2, -c.height / 2, n)
        xx, zz = np.meshgrid(x, z)
        yy = np.full_like(xx, c.distance)
        return xx, yy, zz

    def _canvas_facecolors(self) -> np.ndarray:
        tone = self.sim.canvas.observed_tone()
        rgb = 1.0 - tone[..., None]
        alpha = np.ones((*tone.shape, 1), dtype=np.float32)
        return np.concatenate([rgb, rgb, rgb, alpha], axis=-1)

    def _draw_canvas_surface(self) -> None:
        if self.canvas_surface is not None:
            self.canvas_surface.remove()
        stride = max(1, self.sim.canvas.config.canvas_size // 96)
        self.canvas_surface = self.ax3d.plot_surface(
            self._canvas_x,
            self._canvas_y,
            self._canvas_z,
            facecolors=self._canvas_facecolors(),
            rstride=stride,
            cstride=stride,
            shade=False,
            linewidth=0,
            antialiased=False,
            alpha=1.0,
            zorder=-10,
        )

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "f":
            self.toggle_max_speed()
        elif event.key == "c":
            self.sim.canvas.clear()
        elif event.key == "r":
            self.sim.reset_pose()
            self.sim.canvas.clear()
            self.start = time.perf_counter()
            self.sim_time = 0.0
        elif event.key == "b":
            self.sim.brush_tone = 1.0
        elif event.key == "w":
            self.sim.brush_tone = 0.0
        elif event.key == "p":
            self.sim.paint_enabled = not self.sim.paint_enabled

    def toggle_max_speed(self, _event=None) -> None:  # type: ignore[no-untyped-def]
        self.max_speed = not self.max_speed
        self.fast_button.label.set_text(f"Max speed: {'on' if self.max_speed else 'off'}")

    def _advance_simulation(self, wall_dt: float) -> int:
        if self.paused:
            return 0

        fixed_dt = 1.0 / 240.0
        if self.max_speed:
            steps = 240
        else:
            steps = max(1, int(np.ceil(min(1.0 / 20.0, wall_dt) * self.speed / fixed_dt)))

        for _ in range(steps):
            self.sim_time += fixed_dt * self.speed
            self.sim.set_target(scripted_pose(self.sim_time))
            self.sim.intended_contact_pressure = scripted_contact_pressure(self.sim_time)
            self.sim.step(fixed_dt)
        return steps

    def update(self, _frame: int) -> tuple[object, ...]:
        now = time.perf_counter()
        dt = min(1.0 / 20.0, now - self.last)
        self.last = now

        simulated_steps = self._advance_simulation(dt)

        self.canvas_image.set_data(1.0 - self.sim.canvas.observed_tone())
        self._draw_canvas_surface()

        points = self.sim.kinematics.joint_points(self.sim.actual_pose)
        self.arm_line.set_data(points[:, 0], points[:, 1])
        self.arm_line.set_3d_properties(points[:, 2])
        self.arm_line.set_zorder(20)
        tip = points[-1]
        self.tip_dot.set_data([tip[0]], [tip[1]])
        self.tip_dot.set_3d_properties([tip[2]])
        self.tip_dot.set_zorder(30)

        contact = self.sim.contact
        pose = self.sim.actual_pose
        current = self.sim.plant.telemetry.current
        self.readout.set_text(
            "\n".join(
                [
                    f"coverage    {self.sim.canvas.material_coverage():.3f}",
                    f"contact     {contact.pressure:.3f}  force {contact.force:.2f} N",
                    f"brush width {contact.brush_width_px:.2f} px  tone {'black' if self.sim.brush_tone else 'white'}",
                    f"pose deg    yaw {pose.yaw:6.1f}  pitch {pose.pitch:6.1f}",
                    f"            roll {pose.roll:6.1f}  elbow {pose.elbow:6.1f}",
                    f"current A   yaw {current['yaw']:5.2f}  pitch {current['pitch']:5.2f}",
                    f"            roll {current['roll']:5.2f}  elbow {current['elbow']:5.2f}",
                    f"sim time    {self.sim_time:6.2f} s  mode {'max' if self.max_speed else 'realtime'}",
                    f"steps/frame {simulated_steps:4d}",
                    "",
                    "keys: space pause | f max | r reset | c clear | b/w tone | p paint",
                ]
            )
        )
        return self.arm_line, self.tip_dot, self.canvas_image, self.readout

    def run(self) -> None:
        FuncAnimation(self.fig, self.update, interval=33, blit=False, cache_frame_data=False)
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sim = ArmPainterSim(PainterConfig(canvas_size=args.canvas_size))
    ArmPainterVisualizer(sim, speed=args.speed).run()


if __name__ == "__main__":
    main()
