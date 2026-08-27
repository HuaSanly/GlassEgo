"""Object-centric QA exports for HumanEgo-compatible object poses."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class ObjectPoseQAExporter:
    """Render static objects and propagated hand paths in the anchor frame."""

    COLORS = (
        (0.18, 0.80, 0.44),
        (0.20, 0.60, 0.86),
        (0.61, 0.35, 0.71),
        (0.91, 0.30, 0.24),
    )
    HAND_COLORS = {
        "left": (0.90, 0.25, 0.78),
        "right": (0.10, 0.70, 0.90),
    }

    def __init__(
        self,
        output_dir: str | Path,
        png_path: str | Path | None = None,
        ply_path: str | Path | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.ply_path = Path(ply_path) if ply_path else self.output_dir / "object_centric.ply"
        self.png_path = Path(png_path) if png_path else self.output_dir / "object_centric.png"

    def export(self, triangulation_document: dict, pose_document: dict) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ply_path.parent.mkdir(parents=True, exist_ok=True)
        self.png_path.parent.mkdir(parents=True, exist_ok=True)
        world_to_anchor = np.asarray(
            pose_document["world_to_anchor"],
            dtype=np.float64,
        )
        objects = self._objects_in_anchor(
            triangulation_document,
            world_to_anchor,
        )
        trajectories = self._hand_trajectories(pose_document)
        self._write_ply(objects, trajectories)
        self._write_png(objects, trajectories, pose_document["anchor_key"])
        return {
            "ply": str(self.ply_path),
            "png": str(self.png_path),
            "trajectory_points": {
                side: len(points) for side, points in trajectories.items()
            },
        }

    @staticmethod
    def _objects_in_anchor(triangulation_document, world_to_anchor):
        objects = {}
        for key, value in sorted(triangulation_document["objects"].items()):
            points_world = np.asarray(value["points_3d_world"], dtype=np.float64)
            points_anchor = (
                world_to_anchor[:3, :3] @ points_world.T
            ).T + world_to_anchor[:3, 3]
            object_to_anchor = world_to_anchor @ np.asarray(
                value["object_to_world_matrix"],
                dtype=np.float64,
            )
            objects[key] = {
                "points": points_anchor,
                "pose": object_to_anchor,
            }
        return objects

    @staticmethod
    def _hand_trajectories(pose_document):
        trajectories = {"left": [], "right": []}
        for frame in pose_document["frames"]:
            for side in trajectories:
                pose = frame["hands"][side]["T_hand_to_anchor"]
                if pose is not None:
                    trajectories[side].append(
                        np.asarray(pose, dtype=np.float64)[:3, 3]
                    )
        return {
            side: np.asarray(points, dtype=np.float64).reshape(-1, 3)
            for side, points in trajectories.items()
        }

    def _write_ply(self, objects, trajectories):
        points = []
        colors = []
        for index, value in enumerate(objects.values()):
            object_points = value["points"]
            points.extend(object_points)
            colors.extend([self.COLORS[index % len(self.COLORS)]] * len(object_points))
            self._append_axes(points, colors, value["pose"], 0.05)
        self._append_axes(points, colors, np.eye(4), 0.10)
        for side, trajectory in trajectories.items():
            dense = self._densify_trajectory(trajectory, 0.005)
            points.extend(dense)
            colors.extend([self.HAND_COLORS[side]] * len(dense))

        points_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        colors_array = np.clip(
            np.rint(np.asarray(colors, dtype=np.float64) * 255.0),
            0,
            255,
        ).astype(np.uint8)
        with self.ply_path.open("w", encoding="ascii") as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write(f"element vertex {len(points_array)}\n")
            stream.write("property float x\nproperty float y\nproperty float z\n")
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            stream.write("end_header\n")
            for point, color in zip(points_array, colors_array):
                stream.write(
                    f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                    f"{color[0]} {color[1]} {color[2]}\n"
                )

    def _write_png(self, objects, trajectories, anchor_key):
        figure = Figure(figsize=(8, 8), dpi=180)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111, projection="3d")
        all_points = []
        for index, (key, value) in enumerate(objects.items()):
            points = value["points"]
            color = self.COLORS[index % len(self.COLORS)]
            axes.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=5,
                color=color,
                label=f"{key} (Anchor)" if key == anchor_key else key,
            )
            self._draw_axes(axes, value["pose"], 0.05, 1.0)
            all_points.append(points)
        for side, trajectory in trajectories.items():
            if not len(trajectory):
                continue
            axes.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=self.HAND_COLORS[side],
                linewidth=1.5,
                label=f"{side.title()} Hand",
            )
            all_points.append(trajectory)
        self._draw_axes(axes, np.eye(4), 0.10, 2.0)
        radius = 0.2
        if all_points:
            radius = max(radius, float(np.max(np.abs(np.concatenate(all_points)))))
        axes.set_xlim(-radius, radius)
        axes.set_ylim(-radius, radius)
        axes.set_zlim(-radius, radius)
        axes.set_box_aspect((1, 1, 1))
        axes.set_xlabel("X (m)")
        axes.set_ylabel("Y (m)")
        axes.set_zlabel("Z (m)")
        axes.set_title(f"Object-Centric Scene ({anchor_key} as Static Origin)")
        axes.legend(loc="upper right", fontsize="small")
        figure.tight_layout()
        figure.savefig(self.png_path, bbox_inches="tight")

    @staticmethod
    def _append_axes(points, colors, pose, length):
        axis_colors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        origin = pose[:3, 3]
        for axis, color in enumerate(axis_colors):
            endpoint = origin + pose[:3, axis] * length
            samples = np.linspace(origin, endpoint, 16)
            points.extend(samples)
            colors.extend([color] * len(samples))

    @staticmethod
    def _draw_axes(axes, pose, length, linewidth):
        colors = ("red", "green", "blue")
        origin = pose[:3, 3]
        for axis, color in enumerate(colors):
            direction = pose[:3, axis] * length
            axes.quiver(*origin, *direction, color=color, linewidth=linewidth)

    @staticmethod
    def _densify_trajectory(points, spacing):
        if len(points) < 2:
            return points
        dense = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            distance = float(np.linalg.norm(end - start))
            sample_count = max(2, int(np.ceil(distance / spacing)) + 1)
            dense.extend(np.linspace(start, end, sample_count)[1:])
        return np.asarray(dense, dtype=np.float64)
