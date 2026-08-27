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
from preprocess.data_types.PhaseTypes import OPERATION_MODE, PhaseSequence
from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
    OPENCV_CAMERA_FRAME,
    VIOResult,
)
from preprocess.object_tracking.CoTracker import CoTracker
from preprocess.object_tracking.DINOSAM import DINOSAM
from preprocess.object_tracking.KptsSelector import KptsSelector
from preprocess.object_tracking.ObjectTriangulator import ObjectTriangulator
from preprocess.object_tracking.ObjectPosePropagator import ObjectPosePropagator
from preprocess.object_tracking.ObjectPoseQA import ObjectPoseQAExporter
from utils.utils_math import time_it
from utils.utils_artifact_store import FrameArtifactStore

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class ObjectProcessUnit:
    """单个物体识别处理单元。"""

    unit_dir: Path
    video_path: Path


class ObjectTrackingGenerator:
    """按 HumanEgo indices 顺序协调 DINO-SAM 到三角化。"""

    CACHE_VERSION = 7

    def __init__(
        self,
        unit_dir,
        cfg,
        vio_result=None,
        phase_result=None,
        dinosam=None,
        cotracker=None,
        triangulator=None,
        hands=None,
    ):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.cfg = cfg
        self.unit = self._build_unit(self.unit_dir)
        self.vio_result = vio_result
        self.phase_result = phase_result
        self.hands = hands
        self.store = FrameArtifactStore(self.unit_dir)
        self.prompts = dict(OmegaConf.select(cfg, "prompts", default={}) or {})
        if not self.prompts:
            raise ValueError("Object prompts are required")
        self.output_dir = self.store.temp_data_dir
        self.all_data_dir = self.store.temp_data_dir
        self.training_data_dir = self.store.all_data_dir
        self.vis_dir = self.store.vis_dir
        self.object_vis_dir = self.store.module_vis_dir("objects")
        self.dinosam_video_path = self.store.vis_dir / "dinosam_vis.mp4"
        self.cotracker_video_path = self.store.vis_dir / "cotracker_vis.mp4"
        self.result_path = self.store.temp_path("dinosam_results.json")
        self.report_path = self.store.module_vis_dir("objects") / "report.json"
        self.keypoints_dir = self.store.temp_data_dir
        self.keypoints_path = self.store.temp_path("kptsselector_results.json")
        self.tracks_dir = self.store.temp_data_dir
        self.tracks_vis_dir = self.object_vis_dir
        self.tracks_path = self.store.temp_path("cotracker_results.json")
        self.triangulation_dir = self.store.temp_data_dir
        self.triangulation_path = self.store.temp_path("object_3d_results.json")
        self.triangulation_qa_path = self.object_vis_dir / "camtriangulator_vis.png"
        self.triangulation_ply_path = self.store.temp_path("camtriangulator_vis.ply")
        self.pose_dir = self.store.temp_data_dir
        self.object_pose_path = self.store.temp_path("object_poses.json")
        self.object_centric_ply_path = self.store.temp_path("object_centric.ply")
        self.object_centric_png_path = self.object_vis_dir / "object_centric.png"
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
        if self.vio_result.trajectory.world_frame != ARIA_MPS_WORLD_FRAME:
            raise ValueError("Object tracking requires Aria MPS VIO poses")
        object_centric = self._build_object_centric_indices()
        raw_manipulation = self._raw_manipulation_frames()
        training_frames, finished_frames = self._training_frame_sets(raw_manipulation)
        self.training_frames = training_frames
        tracking_sequence = self._merge_tracking_frames(
            object_centric,
            sorted(training_frames),
        )
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
            sorted(training_frames),
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
            print("║ [Objects] Stage 4/4: 3D triangulation and object pose", flush=True)
            triangulation_report = self._triangulate_objects(
                tracks_document,
                images,
                object_centric,
            )
            triangulation_document = triangulation_report.pop("document")
            pose_document = self._propagate_object_poses(
                triangulation_document,
                sorted(training_frames),
            )
            qa_report = ObjectPoseQAExporter(
                self.pose_dir,
                png_path=self.object_centric_png_path,
                ply_path=self.object_centric_ply_path,
            ).export(
                triangulation_document,
                pose_document,
            )
            triangulation_report["object_poses"] = {
                "frames": pose_document["frame_count"],
                "dynamic_frames": pose_document["dynamic_frame_count"],
                "path": self._relative_path(self.object_pose_path),
            }
            triangulation_report["training_data"] = {
                "status": "pending",
                "frames": len(training_frames),
                "path": self._relative_path(self.training_data_dir),
            }
            triangulation_report["object_centric_qa"] = {
                **qa_report,
                "ply": self._relative_path(qa_report["ply"]),
                "png": self._relative_path(qa_report["png"]),
            }
            report = {
                "status": "completed", "unit_dir": str(self.unit_dir),
                "world_frame": ARIA_MPS_WORLD_FRAME,
                "world_origin": ARIA_MPS_WORLD_ORIGIN,
                "initial_heading": ARIA_MPS_INITIAL_HEADING,
                "video_path": str(self.unit.video_path), "prompts": self.prompts,
                "fps": fps, "object_centric_frames": object_centric,
                "raw_manipulation_frames": raw_manipulation,
                "training_frames": sorted(training_frames),
                "finished_frames": sorted(finished_frames),
                "tracking_frames": tracking_sequence,
                "input_fingerprint": fingerprint,
                "outputs": {
                    "results": self._relative_path(self.result_path),
                    "dinosam_video": self._relative_path(self.dinosam_video_path),
                    "keypoints": self._relative_path(self.keypoints_path),
                    "tracks": self._relative_path(self.tracks_path),
                    "cotracker_video": self._relative_path(self.cotracker_video_path),
                    "triangulation": self._relative_path(self.triangulation_path),
                    "object_poses": self._relative_path(self.object_pose_path),
                    "training_data": self._relative_path(self.training_data_dir),
                    "object_centric_ply": self._relative_path(
                        self.object_centric_ply_path
                    ),
                    "object_centric_png": self._relative_path(
                        self.object_centric_png_path
                    ),
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
        digest.update(
            f"glassego-object-cache-{self.CACHE_VERSION}".encode("utf-8")
        )
        digest.update(ARIA_MPS_WORLD_FRAME.encode("utf-8"))
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
            "hands": self._hands_fingerprint(),
        }).encode("utf-8"))
        digest.update(json.dumps(OmegaConf.to_container(self.cfg, resolve=True), sort_keys=True).encode("utf-8"))
        digest.update(json.dumps([
            {"frame_idx": frame.frame_idx, "timestamp_ns": frame.timestamp_ns, "c2w": frame.c2w.tolist()}
            for frame in self.vio_result.trajectory.frames
            if frame.frame_idx in set(object_centric + raw_manipulation)
        ], sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def _hands_fingerprint(self):
        if self.hands is None:
            return None
        values = []
        for item in self.hands.hands:
            sides = {}
            for side, hand in (("left", item.hand_l), ("right", item.hand_r)):
                if hand is None:
                    sides[side] = None
                    continue
                pose = hand.midpoint_pose_opt_world
                if pose is None:
                    pose = hand.midpoint_pose_raw_world
                sides[side] = {
                    "confidence": (
                        None
                        if hand.confidence is None
                        else float(hand.confidence)
                    ),
                    "grasp": float(hand.grasp_score),
                    "pose": np.asarray(pose).tolist() if pose is not None else None,
                }
            values.append({"idx": int(item.idx), "sides": sides})
        return values

    def _load_cached_result(self, fingerprint):
        required = (
            self.result_path,
            self.report_path,
            self.tracks_path,
            self.triangulation_path,
            self.object_pose_path,
            self.object_centric_ply_path,
            self.object_centric_png_path,
        )
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            return None
        try:
            with self.report_path.open("r", encoding="utf-8") as stream:
                report = json.load(stream)
            if report.get("status") != "completed" or report.get("input_fingerprint") != fingerprint:
                return None
            if report.get("world_frame") != ARIA_MPS_WORLD_FRAME:
                return None
            with self.result_path.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
            if (
                document.get("schema_version") != 2
                or document.get("world_frame") != ARIA_MPS_WORLD_FRAME
                or document.get("world_origin") != ARIA_MPS_WORLD_ORIGIN
                or document.get("initial_heading") != ARIA_MPS_INITIAL_HEADING
            ):
                return None
            with self.triangulation_path.open("r", encoding="utf-8") as stream:
                triangulation = json.load(stream)
            if (
                triangulation.get("schema_version") != 3
                or triangulation.get("world_frame") != ARIA_MPS_WORLD_FRAME
                or triangulation.get("world_origin") != ARIA_MPS_WORLD_ORIGIN
                or triangulation.get("initial_heading") != ARIA_MPS_INITIAL_HEADING
            ):
                return None
            with self.object_pose_path.open("r", encoding="utf-8") as stream:
                object_poses = json.load(stream)
            if (
                object_poses.get("schema_version") != 2
                or object_poses.get("world_frame") != ARIA_MPS_WORLD_FRAME
                or object_poses.get("world_origin") != ARIA_MPS_WORLD_ORIGIN
                or object_poses.get("initial_heading") != ARIA_MPS_INITIAL_HEADING
            ):
                return None
            if not object_poses.get("frames") or object_poses.get("anchor_key") is None:
                return None
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
        raw_manipulation = self._raw_manipulation_frames()
        if not raw_manipulation:
            raise ValueError("Phase result contains no operation frames")
        max_context = int(self.cfg.indices.object_centric_max_frames)
        min_context = int(self.cfg.indices.object_centric_min_frames)
        if max_context < min_context:
            raise ValueError("object_centric_max_frames must be >= object_centric_min_frames")
        all_frames = sorted({int(frame.frame_idx) for frame in self.phase_result.frames})
        first_manipulation = raw_manipulation[0]
        pre_operation = [
            frame_idx for frame_idx in all_frames if frame_idx < first_manipulation
        ][-max_context:]
        object_centric = list(pre_operation)
        if len(object_centric) < min_context:
            needed = min_context - len(object_centric)
            object_centric.extend(raw_manipulation[:needed])
        object_centric = list(dict.fromkeys(object_centric))
        if len(object_centric) < min_context:
            raise ValueError(
                "Not enough frames for object-centric context: "
                f"available={len(object_centric)}, minimum={min_context}"
            )
        return object_centric

    def _raw_manipulation_frames(self):
        return sorted({
            int(frame.frame_idx)
            for frame in self.phase_result.frames
            if frame.mode == OPERATION_MODE
        })

    def _training_frame_sets(self, operation_frames):
        """Return operation frames plus the configured terminal tail."""
        operation_frames = sorted({int(frame) for frame in operation_frames})
        if not operation_frames:
            raise ValueError("Cannot build training frames without operation frames")
        dataset_cfg = OmegaConf.select(self.cfg, "dataset_generation", default=None)
        tail_count = max(
            0,
            int(getattr(dataset_cfg, "finished_tail_frames", 5))
            if dataset_cfg is not None
            else 5,
        )
        last_operation = operation_frames[-1]
        following = sorted(
            int(frame.frame_idx)
            for frame in self.phase_result.frames
            if int(frame.frame_idx) > last_operation
        )
        finished_frames = set(following[:tail_count])
        return set(operation_frames) | finished_frames, finished_frames

    @staticmethod
    def _contiguous_runs(frame_indices):
        if not frame_indices:
            return []
        runs = []
        start = previous = int(frame_indices[0])
        for frame_idx in frame_indices[1:]:
            frame_idx = int(frame_idx)
            if frame_idx != previous + 1:
                runs.append(list(range(start, previous + 1)))
                start = frame_idx
            previous = frame_idx
        runs.append(list(range(start, previous + 1)))
        return runs

    @staticmethod
    def _merge_tracking_frames(object_centric, raw_manipulation):
        return list(dict.fromkeys(object_centric + raw_manipulation))

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
                frame_dir = self.store.frame_dir(
                    idx,
                    int(idx) in getattr(self, "training_frames", set()),
                )
                self.store.write_image(
                    idx,
                    "rgb.png",
                    image,
                    int(idx) in getattr(self, "training_frames", set()),
                )
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
            vis_path = self.object_vis_dir / f"kptsselector_vis_{object_data.key}.png"
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
        triangulator = self.triangulator or ObjectTriangulator(
            self.unit_dir,
            self.cfg.triangulator,
            output_dir=self.triangulation_dir,
        )
        document, qa = triangulator.triangulate(
            selected_document,
            selected_images,
            vio_result=self.vio_result,
        )
        triangulator.save_outputs(document, qa)
        if not cv2.imwrite(str(self.triangulation_qa_path), qa):
            raise IOError(f"Unable to write triangulation QA image: {self.triangulation_qa_path}")
        return {
            "pose_method": document["pose_method"],
            "frames": len(document["frames"]),
            "objects": {
                key: {"points": value["triangulated_points"]}
                for key, value in document["objects"].items()
            },
            "path": self._relative_path(self.triangulation_path),
            "document": document,
        }

    def _propagate_object_poses(self, triangulation_document, frame_indices):
        cfg = OmegaConf.select(self.cfg, "pose_propagation", default=None)
        document = ObjectPosePropagator(cfg).propagate(
            triangulation_document,
            list(frame_indices),
            self.vio_result,
            self.hands,
        )
        self._atomic_write_json(self.object_pose_path, document)
        return document

    def _write_training_data(
        self,
        pose_document,
        triangulation_document,
        frame_data,
        images,
        fps,
    ):
        frame_data_by_idx = {int(frame.frame_idx): frame for frame in frame_data}
        image_by_idx = {
            int(frame.frame_idx): image
            for frame, image in zip(frame_data, images)
        }
        vio_by_idx = {
            int(frame.frame_idx): frame
            for frame in self.vio_result.trajectory.frames
        }
        fx, fy, cx, cy = np.asarray(
            self.vio_result.calibration.intrinsics,
            dtype=np.float64,
        )
        camera_intrinsics = [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ]
        width, height = self.vio_result.calibration.resolution
        object_local_points = self._object_local_points(triangulation_document)
        for pose_frame in pose_document["frames"]:
            frame_idx = int(pose_frame["frame_idx"])
            detection = frame_data_by_idx[frame_idx]
            c2w = vio_by_idx[frame_idx].c2w.tolist()
            frame_dir = self.training_data_dir / f"{frame_idx:05d}"
            rgb_path = frame_dir / "rgb.png"
            frame_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(rgb_path), image_by_idx[frame_idx]):
                raise RuntimeError(f"Unable to write training RGB frame: {rgb_path}")
            hands = {
                side: {
                    "T_hand_to_world": value["T_hand_to_world"],
                    "grasp": value["grasp"],
                }
                for side, value in pose_frame["hands"].items()
                if value["T_hand_to_world"] is not None
            }
            object_keypoints = self._transform_object_points(
                object_local_points,
                pose_frame["objects"],
            )
            output = {
                "schema_version": 1,
                "world_frame": ARIA_MPS_WORLD_FRAME,
                "world_origin": ARIA_MPS_WORLD_ORIGIN,
                "initial_heading": ARIA_MPS_INITIAL_HEADING,
                "metadata": {
                    "idx": frame_idx,
                    "ts": int(pose_frame["timestamp_ns"]),
                    "timestamp_ns": int(pose_frame["timestamp_ns"]),
                    "w": int(width),
                    "h": int(height),
                    "fps": float(fps),
                    "k": camera_intrinsics,
                    "camera_frame": OPENCV_CAMERA_FRAME,
                    "c2w": c2w,
                    "camera_intrinsics": camera_intrinsics,
                    "anchor_key": pose_document["anchor_key"],
                    "is_finished": 0.0,
                    "world_transforms": {
                        "cam0": pose_document["cam0_c2w"],
                        "virtual_static_anchor": pose_document["anchor_to_world"],
                        "camera_to_world": c2w,
                        "anchor_to_world": pose_document["anchor_to_world"],
                        "world_to_anchor": pose_document["world_to_anchor"],
                    },
                },
                "obs": {
                    "rgb_path": str(rgb_path),
                    "mask_arm_path": str(
                        detection.combined_mask_path.parent / "mask_arm.png"
                    ),
                    "mask_obj_path": str(detection.combined_mask_path),
                    "source_video_path": str(self.unit.video_path),
                    "source_frame_idx": frame_idx,
                    "object_masks": {
                        item.key: str(item.mask_path)
                        for item in detection.objects
                        if item.key.startswith("obj")
                    },
                    "objects_kpts": object_keypoints,
                },
                "entities": {
                    "hands": hands,
                    "objects": pose_frame["objects"],
                },
            }
            path = frame_dir / "training_data.json"
            self._atomic_write_json(path, output)
        return {
            "frames": int(pose_document["frame_count"]),
            "path": self._relative_path(self.training_data_dir),
            "filename": "training_data.json",
        }

    @staticmethod
    def _object_local_points(triangulation_document):
        local_points = {}
        for key, value in triangulation_document["objects"].items():
            points_world = np.asarray(value["points_3d_world"], dtype=np.float64)
            object_to_world = np.asarray(
                value["object_to_world_matrix"],
                dtype=np.float64,
            )
            world_to_object = np.linalg.inv(object_to_world)
            local_points[key] = (
                world_to_object[:3, :3] @ points_world.T
            ).T + world_to_object[:3, 3]
        return local_points

    @staticmethod
    def _transform_object_points(local_points, frame_objects):
        result = {}
        for key, points in local_points.items():
            object_to_world = np.asarray(
                frame_objects[key]["T_obj_to_world"],
                dtype=np.float64,
            )
            points_world = (
                object_to_world[:3, :3] @ points.T
            ).T + object_to_world[:3, 3]
            result[key] = {"world": points_world.tolist()}
        return result

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
