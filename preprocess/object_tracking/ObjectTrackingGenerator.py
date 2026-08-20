import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from preprocess.data_types.ObjectTypes import ObjectFrameData, ObjectMaskData, ObjectTrackingResult
from preprocess.data_types.PhaseTypes import PhaseSequence
from preprocess.data_types.VIOTypes import VIOResult
from preprocess.object_tracking.CoTracker import CoTracker
from preprocess.object_tracking.DINOSAM import DINOSAM
from preprocess.object_tracking.KptsSelector import KptsSelector
from preprocess.object_tracking.ObjectTriangulator import ObjectTriangulator
from utils.utils_math import time_it

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class ObjectProcessUnit:
    """单个物体识别处理单元。"""

    unit_dir: Path
    video_path: Path


class ObjectTrackingGenerator:
    """按 HumanEgo indices 顺序协调 DINO-SAM 到三角化。"""

    def __init__(self, unit_dir, cfg, vio_result=None, phase_result=None, dinosam=None, cotracker=None, triangulator=None):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.cfg = cfg
        self.unit = self._build_unit(self.unit_dir)
        self.vio_result = vio_result
        self.phase_result = phase_result
        self.prompts = dict(OmegaConf.select(cfg, "prompts", default={}) or {})
        if not self.prompts:
            raise ValueError("Object prompts are required")
        self.output_dir = self.unit_dir / "preprocess" / "objects"
        self.all_data_dir = self.output_dir / "all_data"
        self.vis_dir = self.output_dir / "vis"
        self.dinosam_video_path = self.vis_dir / "dinosam_vis.mp4"
        self.cotracker_video_path = self.vis_dir / "cotracker_vis.mp4"
        self.result_path = self.output_dir / "dinosam_results.json"
        self.report_path = self.output_dir / "report.json"
        self.keypoints_dir = self.output_dir / "keypoints"
        self.keypoints_path = self.keypoints_dir / "kptsselector_results.json"
        self.tracks_dir = self.output_dir / "tracks"
        self.tracks_vis_dir = self.tracks_dir / "vis"
        self.tracks_path = self.tracks_dir / "cotracker_results.json"
        self.triangulation_dir = self.output_dir / "triangulation"
        self.triangulation_path = self.triangulation_dir / "object_3d_results.json"
        self.triangulation_qa_path = self.triangulation_dir / "object_3d_vis.png"
        self.triangulation_ply_path = self.triangulation_dir / "object_3d_vis.ply"
        self.dinosam = dinosam
        self.cotracker = cotracker
        self.triangulator = triangulator
        self._owns_dinosam = dinosam is None

    def cleanup(self):
        """释放该单元创建的对象模型。"""
        if self._owns_dinosam and self.dinosam is not None:
            self.dinosam.cleanup()
            self.dinosam = None

    @time_it
    def get_object_data(self, force=False):
        """运行 HumanEgo 对象阶段并保存结果。"""
        if self.vio_result is None or self.phase_result is None:
            raise ValueError("VIO and phase results are required")
        object_centric = self._build_object_centric_indices()
        raw_manipulation = self._raw_manipulation_frames()
        tracking_sequence = object_centric + raw_manipulation
        reference_position = int(self.cfg.indices.reference_index)
        if reference_position < 0:
            reference_position += len(object_centric)
        if not 0 <= reference_position < len(object_centric):
            raise ValueError("indices.reference_index is outside object-centric sequence")
        reference_frame = object_centric[reference_position]
        if len(object_centric) < 2:
            raise ValueError("Object-centric sequence must contain at least two frames")
        fingerprint = self._build_fingerprint(
            object_centric,
            raw_manipulation,
        )
        if bool(getattr(self.cfg, "reuse_existing", False)) and not force:
            cached = self._load_cached_result(fingerprint)
            if cached is not None:
                return cached
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_data_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        try:
            print("║ [Objects] Stage 1/4: DINO-SAM segmentation", flush=True)
            frame_data, images, fps = self._run_dinosam(tracking_sequence)
            print("║ [Objects] Stage 2/4: reference keypoint selection", flush=True)
            keypoint_report = self._select_reference_keypoints(
                frame_data,
                images,
                reference_frame,
            )
            print("║ [Objects] Stage 3/4: CoTracker keypoint tracking", flush=True)
            tracks_report, tracks_document = self._track_keypoints(
                images,
                tracking_sequence,
                reference_frame,
            )
            print("║ [Objects] Stage 4/4: 3D triangulation and PCA pose", flush=True)
            triangulation_report = self._triangulate_objects(
                tracks_document,
                images,
                object_centric,
            )
            report = {
                "status": "completed", "unit_dir": str(self.unit_dir),
                "video_path": str(self.unit.video_path), "prompts": self.prompts,
                "fps": fps, "object_centric_frames": object_centric,
                "raw_manipulation_frames": raw_manipulation,
                "tracking_frames": tracking_sequence,
                "input_fingerprint": fingerprint,
                "outputs": {
                    "results": self._relative_path(self.result_path),
                    "dinosam_video": self._relative_path(self.dinosam_video_path),
                    "keypoints": self._relative_path(self.keypoints_path),
                    "tracks": self._relative_path(self.tracks_path),
                    "cotracker_video": self._relative_path(self.cotracker_video_path),
                    "triangulation": self._relative_path(self.triangulation_path),
                },
                "reference_frame": reference_frame,
                "keypoints": keypoint_report, "tracks": tracks_report, "triangulation": triangulation_report,
            }
            result = ObjectTrackingResult(
                self.unit_dir,
                self.unit.video_path,
                self.output_dir,
                tuple(frame_data),
                report,
            )
            if bool(self.cfg.output.export_json):
                self._atomic_write_json(self.result_path, result.to_dict())
            self._atomic_write_json(self.report_path, report)
            return result
        finally:
            self.cleanup()

    def _build_fingerprint(self, object_centric, raw_manipulation):
        digest = hashlib.sha256()
        for path in (self.unit.video_path, self.unit_dir / "object_prompts.yaml"):
            if path.is_file():
                digest.update(path.name.encode("utf-8"))
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
        digest.update(json.dumps({
            "object_centric": object_centric,
            "raw_manipulation": raw_manipulation,
            "phase_summary": self.phase_result.summary,
        }).encode("utf-8"))
        digest.update(json.dumps(OmegaConf.to_container(self.cfg, resolve=True), sort_keys=True).encode("utf-8"))
        digest.update(json.dumps([
            {"frame_idx": frame.frame_idx, "timestamp_ns": frame.timestamp_ns, "c2w": frame.c2w.tolist()}
            for frame in self.vio_result.trajectory.frames
            if frame.frame_idx in set(object_centric + raw_manipulation)
        ], sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def _load_cached_result(self, fingerprint):
        required = (self.result_path, self.report_path, self.tracks_path, self.triangulation_path)
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            return None
        try:
            with self.report_path.open("r", encoding="utf-8") as stream:
                report = json.load(stream)
            if report.get("status") != "completed" or report.get("input_fingerprint") != fingerprint:
                return None
            with self.result_path.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
            frames = []
            for frame in document.get("frames", []):
                objects = tuple(
                    ObjectMaskData(
                        item["key"], item["prompt"], item["confidence"],
                        np.asarray(item["boxes"], dtype=float),
                        np.asarray(item["confidences"], dtype=float),
                        self.unit_dir / item["mask_path"],
                    ) for item in frame.get("objects", [])
                )
                frames.append(ObjectFrameData(
                    int(frame["frame_idx"]), int(frame["timestamp_ns"]), objects,
                    self.unit_dir / frame["combined_mask_path"],
                    self.unit_dir / frame["vis_path"] if frame.get("vis_path") else None,
                ))
            report = dict(report)
            report["cache_reused"] = True
            return ObjectTrackingResult(self.unit_dir, self.unit.video_path, self.output_dir, tuple(frames), report)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _build_object_centric_indices(self):
        phase_frames = {frame.frame_idx: frame for frame in self.phase_result.frames}
        raw = sorted(idx for idx, frame in phase_frames.items() if frame.mode in (0, 3, 4))
        if not raw:
            raise ValueError("Phase result contains no STOP/TRANSITION/FINISHED frames")
        first_raw = raw[0]
        all_indices = sorted(phase_frames)
        max_context = int(self.cfg.indices.object_centric_max_frames)
        min_context = int(self.cfg.indices.object_centric_min_frames)
        context = [idx for idx in all_indices if idx < first_raw][-max_context:]
        sequence = list(context)
        if len(context) < min_context:
            needed = min_context - len(context)
            sequence.extend(raw[:needed])
            if len(sequence) < min_context:
                raise ValueError("Not enough frames for HumanEgo object-centric sequence")
        return sequence

    def _raw_manipulation_frames(self):
        return [frame.frame_idx for frame in self.phase_result.frames if frame.mode in (0, 3, 4)]

    def _run_dinosam(self, frame_indices):
        if self.dinosam is None:
            self.dinosam = DINOSAM(self.cfg.dinosam)
        frames_by_idx, images_by_idx = {}, {}
        cap = cv2.VideoCapture(str(self.unit.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.unit.video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        visual_stream = self._open_visual_stream(
            self.dinosam_video_path,
            fps,
            bool(self.cfg.output.export_dinosam_vis or self.cfg.output.export_gif),
        )
        requested_indices = list(dict.fromkeys(int(idx) for idx in frame_indices))
        requested_set = set(requested_indices)
        prompt_count = sum(1 for prompt in self.prompts.values() if str(prompt).strip())
        prompt_progress = tqdm(
            total=len(requested_indices) * prompt_count,
            desc="DINO-SAM prompts",
            unit="prompt",
            dynamic_ncols=True,
        )
        frame_progress = tqdm(
            total=len(requested_indices),
            desc="DINO-SAM frames",
            unit="frame",
            dynamic_ncols=True,
        )
        try:
            decoded_idx = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                idx = decoded_idx
                decoded_idx += 1
                if idx not in requested_set:
                    continue
                vio_frame = self._vio_frame(idx)
                frame_dir = self.all_data_dir / f"{idx:05d}"
                vis, _, raw_objects = self.dinosam.process_and_save(
                    image,
                    self.prompts,
                    frame_dir,
                    progress=prompt_progress,
                )
                frame_progress.update(1)
                frame_progress.set_postfix(frame=idx)
                if vis is not None:
                    self._write_visual_frame(visual_stream, vis, len(frames_by_idx))
                objects = tuple(ObjectMaskData(item["key"], item["prompt"], item["confidence"], item["boxes"], item["confidences"], item["mask_path"]) for item in raw_objects)
                frames_by_idx[idx] = ObjectFrameData(
                    idx,
                    vio_frame.timestamp_ns,
                    objects,
                    frame_dir / "mask_arm_and_obj.png",
                    None,
                )
                images_by_idx[idx] = image.copy()
            missing = sorted(requested_set - frames_by_idx.keys())
            if missing:
                raise ValueError(
                    "Cannot sequentially read phase-selected video frame(s): "
                    f"{missing}"
                )
        finally:
            frame_progress.close()
            prompt_progress.close()
            cap.release()
            self._close_visual_stream(visual_stream)
            if self._owns_dinosam and self.dinosam is not None:
                self.dinosam.cleanup()
                self.dinosam = None
        frames = [frames_by_idx[idx] for idx in frame_indices]
        images = [images_by_idx[idx] for idx in frame_indices]
        return frames, images, fps

    def _vio_frame(self, frame_idx):
        for frame in self.vio_result.trajectory.frames:
            if frame.frame_idx == frame_idx:
                return frame
        raise ValueError(f"Missing VIO pose for frame {frame_idx}")

    def _select_reference_keypoints(self, frames, images, ref_frame_idx):
        frame_indices = [frame.frame_idx for frame in frames]
        if ref_frame_idx not in frame_indices:
            raise ValueError("Reference frame is outside the tracking sequence")
        ref_index = frame_indices.index(ref_frame_idx)
        frame = frames[ref_index]
        selector = KptsSelector(self.cfg.kpts_selector)
        self.keypoints_dir.mkdir(parents=True, exist_ok=True)
        objects = {}
        object_items = [item for item in frame.objects if item.key.startswith("obj")]
        for object_data in tqdm(
            object_items,
            desc="KptsSelector objects",
            unit="object",
            dynamic_ncols=True,
        ):
            mask = cv2.imread(str(object_data.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Object mask not found: {object_data.mask_path}")
            points, vis = selector.select_from_mask(mask, image_bgr=images[ref_index])
            if len(points) == 0:
                raise ValueError(f"No keypoints selected for object: {object_data.key}")
            vis_path = self.vis_dir / f"kptsselector_vis_{object_data.key}.png"
            if bool(self.cfg.output.export_keypoints_vis):
                cv2.imwrite(str(vis_path), vis)
            objects[object_data.key] = {"prompt": object_data.prompt, "points": points.tolist(), "count": len(points), "mask_path": str(object_data.mask_path), "vis_path": str(vis_path)}
        if not objects:
            raise ValueError("No obj* masks found for keypoint selection")
        document = {
            "method": "AUTO_MASK_CONTOUR_EQUIDISTANT",
            "ref_frame": frame.frame_idx,
            "objects": {
                key: {
                    **value,
                    "mask_path": self._relative_path(value["mask_path"]),
                    "vis_path": self._relative_path(value["vis_path"]),
                }
                for key, value in objects.items()
            },
        }
        self._atomic_write_json(self.keypoints_path, document)
        return {
            "ref_frame": frame.frame_idx,
            "objects": {key: {"count": value["count"]} for key, value in objects.items()},
            "path": self._relative_path(self.keypoints_path),
        }

    def _track_keypoints(self, images, frame_indices, ref_frame_idx):
        with self.keypoints_path.open("r", encoding="utf-8") as stream:
            keypoints = json.load(stream)
        ref_frame = int(keypoints["ref_frame"])
        if ref_frame != ref_frame_idx:
            raise ValueError("Keypoint reference frame does not match object-centric reference")
        ref_index = frame_indices.index(ref_frame)
        keypoints_by_object = {key: value["points"] for key, value in keypoints["objects"].items()}
        cotracker = self.cotracker or CoTracker(self.cfg.cotracker)
        tracks_document, vis_frames = cotracker.track(images, frame_indices, keypoints_by_object, ref_index)
        self.tracks_vis_dir.mkdir(parents=True, exist_ok=True)
        visual_stream = None
        if bool(self.cfg.output.export_tracks_vis or self.cfg.output.export_gif):
            visual_stream = self._open_visual_stream(
                self.cotracker_video_path,
                self._video_fps(),
                True,
            )
            try:
                for frame_idx, vis in tqdm(
                    zip(frame_indices, vis_frames),
                    total=len(frame_indices),
                    desc="CoTracker save visualization",
                    unit="frame",
                    dynamic_ncols=True,
                ):
                    self._write_visual_frame(
                        visual_stream,
                        vis,
                        frame_indices.index(frame_idx),
                    )
            finally:
                self._close_visual_stream(visual_stream)
        self._atomic_write_json(self.tracks_path, tracks_document)
        return (
            {
                "ref_frame": ref_frame,
                "frames": len(frame_indices),
                "path": self._relative_path(self.tracks_path),
            },
            tracks_document,
        )

    def _triangulate_objects(self, tracks_document, images, object_centric):
        tracking_frames = [int(item) for item in tracks_document["frames"]]
        selected_positions = [tracking_frames.index(idx) for idx in object_centric]
        selected_document = dict(tracks_document)
        selected_document["frames"] = list(object_centric)
        selected_document["objects"] = {}
        for key, value in tracks_document.get("objects", {}).items():
            selected_document["objects"][key] = {
                **value,
                "tracks": [value["tracks"][index] for index in selected_positions],
                "visibility": [value["visibility"][index] for index in selected_positions],
            }
        selected_images = [images[index] for index in selected_positions]
        triangulator = self.triangulator or ObjectTriangulator(self.unit_dir, self.cfg.triangulator)
        document, qa = triangulator.triangulate(
            selected_document,
            selected_images,
            vio_result=self.vio_result,
        )
        triangulator.save_outputs(document, qa)
        return {
            "pose_method": document["pose_method"],
            "frames": len(document["frames"]),
            "objects": {
                key: {"points": value["triangulated_points"]}
                for key, value in document["objects"].items()
            },
            "path": self._relative_path(self.triangulation_path),
        }

    @staticmethod
    def _build_unit(unit_dir):
        if not unit_dir.is_dir():
            raise FileNotFoundError(f"Unit directory does not exist: {unit_dir}")
        videos = sorted(path for path in unit_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            raise FileNotFoundError(f"No video file found under {unit_dir}")
        if len(videos) > 1:
            raise ValueError(f"Multiple video files found under {unit_dir}: {videos}")
        return ObjectProcessUnit(unit_dir, videos[0])

    def _relative_path(self, path):
        try:
            return str(Path(path).resolve().relative_to(self.unit_dir))
        except ValueError:
            return str(path)

    def _video_fps(self):
        capture = cv2.VideoCapture(str(self.unit.video_path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        return fps if fps > 0 else 30.0

    def _open_visual_stream(self, path, fps, enabled):
        if not enabled:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        return {
            "path": path,
            "fps": fps if fps > 0 else 30.0,
            "writer": None,
            "frames": [],
            "export_gif": bool(self.cfg.output.export_gif),
            "gif_ratio": int(self.cfg.output.gif_frame_ratio),
        }

    @staticmethod
    def _write_visual_frame(stream, frame, index):
        if stream is None:
            return
        # VideoWriter 的尺寸必须在第一帧到达后确定，因此延迟创建 writer。
        if stream["writer"] is None:
            h, w = frame.shape[:2]
            stream["writer"] = cv2.VideoWriter(
                str(stream["path"]),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(stream["fps"]),
                (w, h),
            )
            if not stream["writer"].isOpened():
                raise RuntimeError(f"Unable to open visualization writer: {stream['path']}")
        stream["writer"].write(frame)
        if stream["export_gif"] and index % stream["gif_ratio"] == 0:
            stream["frames"].append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _close_visual_stream(stream):
        if stream is None:
            return
        writer = stream["writer"]
        if writer is None:
            return
        writer.release()
        if not stream["path"].is_file() or stream["path"].stat().st_size == 0:
            raise RuntimeError(f"Visualization video was not created: {stream['path']}")
        if stream["export_gif"] and stream["frames"]:
            gif_path = stream["path"].with_suffix(".gif")
            imageio.mimsave(
                gif_path,
                stream["frames"],
                fps=float(stream["fps"]) / stream["gif_ratio"],
            )
            if not gif_path.is_file() or gif_path.stat().st_size == 0:
                raise RuntimeError(f"Visualization GIF was not created: {gif_path}")

    @staticmethod
    def _atomic_write_json(path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
        temporary_path = Path(handle.name)
        try:
            with handle:
                json.dump(document, handle, indent=2)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
