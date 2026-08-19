import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2

from preprocess.data_types.ObjectTypes import (
    ObjectFrameData,
    ObjectMaskData,
    ObjectTrackingResult,
)
from preprocess.object_tracking.CoTracker import CoTracker, CoTrackerConfig
from preprocess.object_tracking.DINOSAM import DINOSAM, DINOSAMConfig
from preprocess.object_tracking.KptsSelector import KptsSelector, KptsSelectorConfig
from preprocess.object_tracking.ObjectTriangulator import (
    ObjectTriangulator,
    ObjectTriangulatorConfig,
)
from utils.utils_math import time_it


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class ObjectProcessUnit:
    """单个物体识别处理单元。

    当前阶段只需要 unit 目录和唯一视频；不依赖 VIO，也不构造 Cam。
    """

    unit_dir: Path
    video_path: Path


class ObjectTrackingGenerator:
    """协调单个数据单元的 DINO-SAM 物体识别。

    这个类是 object_tracking 模块的对外入口：
        1. 找到 unit 下唯一视频。
        2. 按 stride/max_frames 流式抽帧。
        3. 复用一个 DINOSAM 实例处理多帧。
        4. 组织 mask、vis、JSON 和 report 的输出布局。
    """

    def __init__(
        self,
        unit_dir: str | Path,
        cfg,
        dinosam=None,
        cotracker=None,
        triangulator=None,
    ):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.cfg = cfg
        self.unit = self._build_unit(self.unit_dir)
        self.prompts = dict(cfg.prompts)
        if not self.prompts:
            raise ValueError("At least one --prompt key=value must be provided")

        self.output_dir = self.unit_dir / "preprocess" / "objects"
        self.all_data_dir = self.output_dir / "all_data"
        self.vis_dir = self.output_dir / "vis"
        self.result_path = self.output_dir / "dinosam_results.json"
        self.report_path = self.output_dir / "report.json"
        # 测试时可以注入 fake dinosam；真实运行时这里为空，由本类创建并负责 cleanup。
        self.dinosam = dinosam
        self._owns_dinosam = dinosam is None
        # 测试时可以注入 fake cotracker；真实运行时由本类按需创建。
        self.cotracker = cotracker
        # 测试时可以注入 fake triangulator；真实运行时由本类按需创建。
        self.triangulator = triangulator
        self.keypoints_dir = self.output_dir / "keypoints"
        self.keypoints_path = self.keypoints_dir / "kptsselector_results.json"
        self.tracks_dir = self.output_dir / "tracks"
        self.tracks_vis_dir = self.tracks_dir / "vis"
        self.tracks_path = self.tracks_dir / "cotracker_results.json"
        self.triangulation_dir = self.output_dir / "triangulation"
        self.triangulation_path = self.triangulation_dir / "object_3d_results.json"
        self.triangulation_qa_path = self.triangulation_dir / "object_3d_vis.png"
        self.triangulation_ply_path = self.triangulation_dir / "object_3d_vis.ply"

    @time_it
    def get_object_data(
        self,
        max_frames: int | None = 30,
        stride: int = 10,
    ) -> ObjectTrackingResult:
        """运行 DINO-SAM 抽帧识别并保存 mask、可视化和 JSON。

        这是离线观察入口，不追求一次跑完整视频；默认只抽少量帧，方便快速看 prompt
        是否写得合适。达到 max_frames 后会立即停止读视频。
        """
        if stride <= 0:
            raise ValueError("stride must be a positive integer")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_data_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        if self.dinosam is None:
            self.dinosam = DINOSAM(self.cfg.dinosam)

        # 这里保持流式读取，不把整段视频一次性放进内存。
        t_start = time.perf_counter()
        frames = []
        sampled_images = []
        decoded_frames = 0
        sampled_frames = 0
        fps = 0.0

        cap = cv2.VideoCapture(str(self.unit.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.unit.video_path}")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                raise RuntimeError("Video FPS must be positive")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx = decoded_frames
                decoded_frames += 1

                if frame_idx % stride != 0:
                    continue
                if max_frames is not None and sampled_frames >= max_frames:
                    continue

                # 当前 object 模块尚未接 VIO/camera.csv；时间戳先按视频 FPS 估算。
                timestamp_ns = int(round(frame_idx * 1_000_000_000 / fps))
                frame_data = self._process_frame(frame, frame_idx, timestamp_ns)
                frames.append(frame_data)
                sampled_images.append(frame.copy())
                sampled_frames += 1
                if max_frames is not None and sampled_frames >= max_frames:
                    break
        finally:
            cap.release()
            if self._owns_dinosam and self.dinosam is not None:
                self.dinosam.cleanup()

        keypoints_report = None
        if bool(getattr(self.cfg, "select_keypoints", False)):
            keypoints_report = self._select_reference_keypoints(frames)
        tracks_report = None
        if bool(getattr(self.cfg, "track_keypoints", False)):
            tracks_report = self._track_keypoints(frames, sampled_images)
        triangulation_report = None
        if bool(getattr(self.cfg, "triangulate_objects", False)):
            triangulation_report = self._triangulate_objects(frames, sampled_images)

        report = self._build_report(
            decoded_frames=decoded_frames,
            sampled_frames=sampled_frames,
            fps=fps,
            elapsed_seconds=time.perf_counter() - t_start,
            keypoints_report=keypoints_report,
            tracks_report=tracks_report,
            triangulation_report=triangulation_report,
        )
        result = ObjectTrackingResult(
            unit_dir=self.unit_dir,
            video_path=self.unit.video_path,
            output_dir=self.output_dir,
            frames=tuple(frames),
            report=report,
        )
        self._atomic_write_json(self.result_path, result.to_dict())
        self._atomic_write_json(self.report_path, report)
        return result

    def _process_frame(
        self,
        frame,
        frame_idx: int,
        timestamp_ns: int,
    ) -> ObjectFrameData:
        # 每个被采样的原始帧有一个独立目录，便于后续加入深度、光流或跟踪结果。
        frame_dir = self.all_data_dir / f"{frame_idx:05d}"
        vis_path = self.vis_dir / f"dinosam_{frame_idx:05d}.png"
        vis, _, raw_objects = self.dinosam.process_and_save(
            frame,
            self.prompts,
            frame_dir,
        )
        if vis is not None:
            cv2.imwrite(str(vis_path), vis)

        # DINOSAM 返回 numpy/path 结构；这里转成 data_types，统一 JSON 契约。
        objects = tuple(
            ObjectMaskData(
                key=item["key"],
                prompt=item["prompt"],
                confidence=item["confidence"],
                boxes=item["boxes"],
                confidences=item["confidences"],
                mask_path=item["mask_path"],
            )
            for item in raw_objects
        )
        return ObjectFrameData(
            frame_idx=frame_idx,
            timestamp_ns=timestamp_ns,
            objects=objects,
            combined_mask_path=frame_dir / "mask_arm_and_obj.png",
            vis_path=vis_path if vis is not None else None,
        )

    def _select_reference_keypoints(
        self,
        frames: list[ObjectFrameData],
    ) -> dict:
        """从参考帧的 obj* mask 中选择 CoTracker 查询点。"""
        ref_frame = int(self.cfg.ref_frame)
        frame = next((item for item in frames if item.frame_idx == ref_frame), None)
        if frame is None:
            raise ValueError(
                "ref_frame must be one of the sampled frame indices: "
                f"ref_frame={ref_frame}, sampled={[item.frame_idx for item in frames]}"
            )

        selector = KptsSelector(self.cfg.kpts_selector)
        self.keypoints_dir.mkdir(parents=True, exist_ok=True)

        image_bgr = self._read_video_frame(ref_frame)
        objects = {}
        for object_data in frame.objects:
            if not object_data.key.startswith("obj"):
                continue
            mask = cv2.imread(str(object_data.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Object mask not found: {object_data.mask_path}")

            points, vis = selector.select_from_mask(mask, image_bgr=image_bgr)
            if len(points) == 0:
                raise ValueError(f"No keypoints selected for object: {object_data.key}")
            vis_path = self.keypoints_dir / f"kptsselector_vis_{object_data.key}.png"
            cv2.imwrite(str(vis_path), vis)
            objects[object_data.key] = {
                "prompt": object_data.prompt,
                "points": points.tolist(),
                "count": int(len(points)),
                "mask_path": str(object_data.mask_path),
                "vis_path": str(vis_path),
            }

        if not objects:
            raise ValueError("No obj* masks found for keypoint selection")

        document = {
            "method": "AUTO_MASK_CONTOUR_EQUIDISTANT",
            "ref_frame": ref_frame,
            "objects": objects,
        }
        self._atomic_write_json(self.keypoints_path, document)
        return {
            "ref_frame": ref_frame,
            "objects": {
                key: {
                    "count": value["count"],
                    "vis_path": value["vis_path"],
                }
                for key, value in objects.items()
            },
            "path": str(self.keypoints_path),
        }

    def _track_keypoints(
        self,
        frames: list[ObjectFrameData],
        sampled_images: list,
    ) -> dict:
        """使用 CoTracker 跟踪参考帧 object keypoints。"""
        if not self.keypoints_path.is_file():
            raise FileNotFoundError(
                "Missing keypoints file. Run with --select-keypoints first: "
                f"{self.keypoints_path}"
            )
        with self.keypoints_path.open("r", encoding="utf-8") as stream:
            keypoints_document = json.load(stream)

        frame_indices = [frame.frame_idx for frame in frames]
        ref_frame = int(keypoints_document["ref_frame"])
        if ref_frame not in frame_indices:
            raise ValueError(
                "Keypoint ref_frame must be in the sampled tracking sequence: "
                f"ref_frame={ref_frame}, sampled={frame_indices}"
            )
        ref_sequence_index = frame_indices.index(ref_frame)

        keypoints_by_object = {
            key: value["points"]
            for key, value in keypoints_document.get("objects", {}).items()
        }
        if not keypoints_by_object:
            raise ValueError("No object keypoints found for CoTracker")

        cotracker = self.cotracker or CoTracker(self.cfg.cotracker)
        tracks_document, vis_frames = cotracker.track(
            sampled_images,
            frame_indices,
            keypoints_by_object,
            ref_sequence_index,
        )

        self.tracks_vis_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, vis in zip(frame_indices, vis_frames):
            cv2.imwrite(str(self.tracks_vis_dir / f"cotracker_{frame_idx:05d}.png"), vis)
        self._atomic_write_json(self.tracks_path, tracks_document)

        return {
            "ref_frame": ref_frame,
            "frames": len(frame_indices),
            "objects": {
                key: {
                    "points": len(value["points"]),
                }
                for key, value in keypoints_document.get("objects", {}).items()
            },
            "path": str(self.tracks_path),
            "vis_dir": str(self.tracks_vis_dir),
        }

    def _triangulate_objects(
        self,
        frames: list[ObjectFrameData],
        sampled_images: list,
    ) -> dict:
        """使用 VIO 位姿把 CoTracker 2D tracks 三角化成物体 3D 位姿。"""
        if not self.tracks_path.is_file():
            raise FileNotFoundError(
                "Missing CoTracker tracks. Run with --track-keypoints first: "
                f"{self.tracks_path}"
            )
        pose_path = self.unit_dir / "preprocess" / "vio" / "poses.json"
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing VIO poses: {pose_path}")

        with self.tracks_path.open("r", encoding="utf-8") as stream:
            tracks_document = json.load(stream)

        sampled_frame_indices = [frame.frame_idx for frame in frames]
        track_frame_indices = [int(item) for item in tracks_document["frames"]]
        if sampled_frame_indices != track_frame_indices:
            raise ValueError(
                "Current sampled frames must match CoTracker result frames: "
                f"sampled={sampled_frame_indices}, tracks={track_frame_indices}"
            )

        triangulator = self.triangulator or ObjectTriangulator(
            self.unit_dir,
            self.cfg.triangulator,
        )
        document, qa = triangulator.triangulate(tracks_document, sampled_images)
        triangulator.save_outputs(document, qa)

        return {
            "pose_method": document["pose_method"],
            "frames": len(document["frames"]),
            "objects": {
                key: {
                    "points": value["triangulated_points"],
                    "center_world": value["center_world"],
                }
                for key, value in document["objects"].items()
            },
            "path": str(self.triangulation_path),
            "qa": str(self.triangulation_qa_path),
            "ply": str(self.triangulation_ply_path),
        }

    def _read_video_frame(self, frame_idx: int):
        """按原始 frame_idx 从视频中读取一帧 BGR 图，用于关键点可视化背景。"""
        cap = cv2.VideoCapture(str(self.unit.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.unit.video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot read video frame: {frame_idx}")
            return frame
        finally:
            cap.release()

    def _build_report(
        self,
        decoded_frames: int,
        sampled_frames: int,
        fps: float,
        elapsed_seconds: float,
        keypoints_report: dict | None = None,
        tracks_report: dict | None = None,
        triangulation_report: dict | None = None,
    ) -> dict:
        # report 面向运行诊断；dinosam_results.json 面向下游数据消费。
        report = {
            "status": "completed",
            "unit_dir": str(self.unit_dir),
            "video_path": str(self.unit.video_path),
            "output_dir": str(self.output_dir),
            "decoded_frames": decoded_frames,
            "sampled_frames": sampled_frames,
            "fps": fps,
            "stride": self.cfg.stride,
            "max_frames": self.cfg.max_frames,
            "prompts": self.prompts,
            "elapsed_seconds": elapsed_seconds,
            "average_fps": sampled_frames / elapsed_seconds
            if elapsed_seconds > 0 and sampled_frames
            else 0.0,
            "outputs": {
                "results": str(self.result_path),
                "report": str(self.report_path),
                "vis_dir": str(self.vis_dir),
            },
        }
        if keypoints_report is not None:
            report["keypoints"] = keypoints_report
            report["outputs"]["keypoints"] = str(self.keypoints_path)
        if tracks_report is not None:
            report["tracks"] = tracks_report
            report["outputs"]["tracks"] = str(self.tracks_path)
            report["outputs"]["tracks_vis_dir"] = str(self.tracks_vis_dir)
        if triangulation_report is not None:
            report["triangulation"] = triangulation_report
            report["outputs"]["triangulation"] = str(self.triangulation_path)
            report["outputs"]["triangulation_qa"] = str(self.triangulation_qa_path)
            report["outputs"]["triangulation_ply"] = str(self.triangulation_ply_path)
        return report

    @staticmethod
    def _build_unit(unit_dir: Path) -> ObjectProcessUnit:
        # 和主 preprocess 约定保持一致：一个 unit 只能有一个直属视频，避免输出歧义。
        if not unit_dir.is_dir():
            raise FileNotFoundError(f"Unit directory does not exist: {unit_dir}")
        videos = sorted(
            path
            for path in unit_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise FileNotFoundError(f"No video file found under {unit_dir}")
        if len(videos) > 1:
            raise ValueError(f"Multiple video files found under {unit_dir}: {videos}")
        return ObjectProcessUnit(unit_dir=unit_dir, video_path=videos[0])

    @staticmethod
    def _atomic_write_json(path: Path, document: dict) -> None:
        # 原子替换避免写到一半中断时留下半截 JSON。
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        )
        temporary_path = Path(handle.name)
        try:
            with handle:
                json.dump(document, handle, indent=2)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _parse_prompts(items: list[str] | None) -> dict[str, str]:
    """解析可重复传入的 `--prompt key=value` 参数。"""
    prompts = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Prompt must be key=value: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Prompt must be key=value: {item}")
        prompts[key] = value
    return prompts


def _build_cfg(args) -> SimpleNamespace:
    """把 CLI 参数整理成 Generator 使用的轻量 cfg 对象。"""
    prompts = _parse_prompts(args.prompt)
    if not prompts:
        raise ValueError("At least one --prompt key=value must be provided")
    return SimpleNamespace(
        prompts=prompts,
        max_frames=args.max_frames,
        stride=args.stride,
        select_keypoints=args.select_keypoints,
        track_keypoints=args.track_keypoints,
        triangulate_objects=args.triangulate_objects,
        ref_frame=args.ref_frame,
        dinosam=DINOSAMConfig(
            dino_model_id=args.dino_model_id,
            sam2_repo_id=args.sam2_repo_id,
            sam2_checkpoint_name=args.sam2_checkpoint_name,
            sam2_config=args.sam2_config,
            box_threshold=args.box_threshold,
        ),
        kpts_selector=KptsSelectorConfig(
            kpts_n_bands=args.kpts_n_bands,
        ),
        cotracker=CoTrackerConfig(
            cotracker_res=args.cotracker_res,
            cotracker_chunk_size=args.cotracker_chunk_size,
            cotracker_viz_trail_len=args.cotracker_viz_trail_len,
        ),
        triangulator=ObjectTriangulatorConfig(
            pose_method=args.object_pose_method,
            step=args.triangulation_step,
            smooth_window=args.triangulation_smooth_window,
            smooth_polyorder=args.triangulation_smooth_polyorder,
            ba_f_scale=args.triangulation_ba_f_scale,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    # unit/prompt/max-frames/stride 是日常调试最常用的参数。
    parser.add_argument("--unit", type=str, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--select-keypoints", action="store_true")
    parser.add_argument("--track-keypoints", action="store_true")
    parser.add_argument("--triangulate-objects", action="store_true")
    parser.add_argument("--ref-frame", type=int, default=0)
    parser.add_argument("--kpts-n-bands", type=int, default=10)
    parser.add_argument("--cotracker-res", type=int, default=640)
    parser.add_argument("--cotracker-chunk-size", type=int, default=100)
    parser.add_argument("--cotracker-viz-trail-len", type=int, default=20)
    parser.add_argument("--object-pose-method", type=str, default="pca2")
    parser.add_argument("--triangulation-step", type=int, default=1)
    parser.add_argument("--triangulation-smooth-window", type=int, default=7)
    parser.add_argument("--triangulation-smooth-polyorder", type=int, default=2)
    parser.add_argument("--triangulation-ba-f-scale", type=float, default=3.0)
    # 下面这组是模型选择参数，默认走质量更高但更重的 SAM2 large。
    parser.add_argument(
        "--dino-model-id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
    )
    parser.add_argument(
        "--sam2-repo-id",
        type=str,
        default="facebook/sam2-hiera-large",
    )
    parser.add_argument(
        "--sam2-checkpoint-name",
        type=str,
        default="sam2_hiera_large.pt",
    )
    parser.add_argument("--sam2-config", type=str, default="sam2_hiera_l.yaml")
    args = parser.parse_args()

    cfg = _build_cfg(args)
    generator = ObjectTrackingGenerator(args.unit, cfg)
    result = generator.get_object_data(
        max_frames=cfg.max_frames,
        stride=cfg.stride,
    )
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
