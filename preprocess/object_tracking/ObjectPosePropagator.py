"""HumanEgo-style static initialization and hand-latched object poses."""

import numpy as np

from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
)


OBJECT_POSE_SCHEMA_VERSION = 2


class ObjectPosePropagator:
    """Propagate static object poses through grasped hand motion."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.min_hand_confidence = float(
            getattr(cfg, "min_hand_confidence", 0.5) if cfg is not None else 0.5
        )
        self.grasp_distance_threshold_m = float(
            getattr(cfg, "grasp_distance_threshold_m", 0.20)
            if cfg is not None
            else 0.20
        )
        if self.grasp_distance_threshold_m <= 0.0:
            raise ValueError("Object grasp distance threshold must be positive")
        self.lock_grasp_weight = float(
            getattr(cfg, "lock_grasp_weight", 0.60) if cfg is not None else 0.60
        )
        self.lock_confidence_weight = float(
            getattr(cfg, "lock_confidence_weight", 0.10)
            if cfg is not None
            else 0.10
        )
        self.lock_proximity_weight = float(
            getattr(cfg, "lock_proximity_weight", 0.30)
            if cfg is not None
            else 0.30
        )
        weight_sum = (
            self.lock_grasp_weight
            + self.lock_confidence_weight
            + self.lock_proximity_weight
        )
        if min(
            self.lock_grasp_weight,
            self.lock_confidence_weight,
            self.lock_proximity_weight,
        ) < 0.0 or weight_sum <= 0.0:
            raise ValueError("Object lock score weights must sum to a positive value")
        self.lock_grasp_weight /= weight_sum
        self.lock_confidence_weight /= weight_sum
        self.lock_proximity_weight /= weight_sum
        self.lock_enter_score = float(
            getattr(cfg, "lock_enter_score", 0.65) if cfg is not None else 0.65
        )
        self.lock_exit_score = float(
            getattr(cfg, "lock_exit_score", 0.45) if cfg is not None else 0.45
        )
        if self.lock_exit_score >= self.lock_enter_score:
            raise ValueError("Object lock exit score must be lower than enter score")
        self.lock_enter_frames = max(
            1,
            int(getattr(cfg, "lock_enter_frames", 1) if cfg is not None else 1),
        )
        self.lock_exit_frames = max(
            1,
            int(getattr(cfg, "lock_exit_frames", 2) if cfg is not None else 2),
        )
        self.disable_left_latching = bool(
            getattr(cfg, "disable_kinematic_latching_left", False)
            if cfg is not None
            else False
        )
        self.disable_right_latching = bool(
            getattr(cfg, "disable_kinematic_latching_right", False)
            if cfg is not None
            else False
        )

    def propagate(
        self,
        triangulation_document: dict,
        frame_indices: list[int],
        vio_result,
        hands=None,
    ) -> dict:
        self._validate_triangulation(triangulation_document)
        if not frame_indices:
            raise ValueError("Object pose propagation requires at least one frame")

        trajectory = vio_result.trajectory
        if trajectory.world_frame != ARIA_MPS_WORLD_FRAME:
            raise ValueError("Object pose propagation requires Aria MPS VIO poses")
        vio_frames = {frame.frame_idx: frame for frame in trajectory.frames}
        missing = sorted(set(frame_indices) - set(vio_frames))
        if missing:
            raise ValueError(f"Missing VIO frames for object poses: {missing[:5]}")

        hand_by_frame = self._hands_by_frame(hands, trajectory)
        object_keys = sorted(triangulation_document["objects"])
        anchor_key = object_keys[0]
        static_poses = {
            key: np.asarray(
                value["object_to_world_matrix"],
                dtype=np.float64,
            )
            for key, value in triangulation_document["objects"].items()
        }
        for key, pose in static_poses.items():
            self._validate_transform(pose, f"{key}.object_to_world_matrix")
        object_geometry = self._object_geometry(
            triangulation_document["objects"], static_poses
        )

        dynamic_poses = {key: pose.copy() for key, pose in static_poses.items()}
        anchor_to_world = static_poses[anchor_key]
        world_to_anchor = np.linalg.inv(anchor_to_world)
        moved_objects = set()
        hand_states = {
            "left": self._new_hand_state(),
            "right": self._new_hand_state(),
        }
        frames = []
        dynamic_frame_count = 0

        for frame_idx in frame_indices:
            hand_data = hand_by_frame.get(frame_idx)
            current_hands = {
                "left": self._hand_state(hand_data.hand_l if hand_data else None),
                "right": self._hand_state(hand_data.hand_r if hand_data else None),
            }
            for side in ("left", "right"):
                if not self._latching_enabled(side):
                    continue
                self._update_hand_latch(
                    hand_states[side],
                    current_hands[side],
                    dynamic_poses,
                    anchor_key,
                    object_geometry,
                    moved_objects,
                )

            objects = {}
            frame_is_dynamic = False
            for key, pose in dynamic_poses.items():
                state = self._object_state(hand_states, key)
                is_dynamic = state is not None
                frame_is_dynamic |= is_dynamic
                objects[key] = {
                    "T_obj_to_world": pose.tolist(),
                    "is_dynamic": is_dynamic,
                    "pose_source": (
                        "hand_latched"
                        if is_dynamic
                        else "hand_last_pose"
                        if key in moved_objects
                        else "static_initial"
                    ),
                    "latched_hand": state,
                    "lock_score": (
                        float(hand_states[state]["lock_score"])
                        if state is not None
                        else 0.0
                    ),
                    "T_obj_to_anchor": (
                        world_to_anchor @ pose
                    ).tolist(),
                }
            if frame_is_dynamic:
                dynamic_frame_count += 1

            frames.append(
                {
                    "frame_idx": int(frame_idx),
                    "timestamp_ns": int(vio_frames[frame_idx].timestamp_ns),
                    "objects": objects,
                    "hands": {
                        side: {
                            "present": current_hands[side]["pose"] is not None,
                            "confidence": current_hands[side]["confidence"],
                            "grasp": float(current_hands[side]["grasp"]),
                            "grasp_score": float(current_hands[side]["grasp"]),
                            "proximity_score": float(
                                current_hands[side]["proximity_score"]
                            ),
                            "lock_score": float(current_hands[side]["lock_score"]),
                            "candidate_object": current_hands[side]["candidate_object"],
                            "latched_object": hand_states[side]["object_key"],
                            "T_hand_to_world": (
                                current_hands[side]["pose"].tolist()
                                if current_hands[side]["pose"] is not None
                                else None
                            ),
                            "T_hand_to_anchor": (
                                (
                                    world_to_anchor
                                    @ current_hands[side]["pose"]
                                ).tolist()
                                if current_hands[side]["pose"] is not None
                                else None
                            ),
                        }
                        for side in ("left", "right")
                    },
                }
            )

        return {
            "schema_version": OBJECT_POSE_SCHEMA_VERSION,
            "world_frame": ARIA_MPS_WORLD_FRAME,
            "world_origin": ARIA_MPS_WORLD_ORIGIN,
            "initial_heading": ARIA_MPS_INITIAL_HEADING,
            "method": "humanego_static_init_hand_latch_propagation",
            "anchor_key": anchor_key,
            "cam0_c2w": triangulation_document["cam0_c2w"],
            "anchor_to_world": anchor_to_world.tolist(),
            "world_to_anchor": world_to_anchor.tolist(),
            "frame_count": len(frames),
            "dynamic_frame_count": dynamic_frame_count,
            "objects": object_keys,
            "frames": frames,
        }

    def _latching_enabled(self, side: str) -> bool:
        if side == "left":
            return not self.disable_left_latching
        if side == "right":
            return not self.disable_right_latching
        raise ValueError(f"Unknown hand side: {side}")

    @staticmethod
    def _new_hand_state() -> dict:
        return {
            "is_grasping": False,
            "object_key": None,
            "T_lock_h2obj": None,
            "enter_count": 0,
            "exit_count": 0,
            "lock_score": 0.0,
            "candidate_object": None,
        }

    def _update_hand_latch(
        self,
        state: dict,
        hand: dict,
        dynamic_poses: dict[str, np.ndarray],
        anchor_key: str,
        object_geometry: dict[str, dict],
        moved_objects: set[str],
    ) -> None:
        hand_pose = hand["pose"]
        candidate_key, candidate_score, proximity_score = self._best_object(
            hand,
            dynamic_poses,
            anchor_key,
            object_geometry,
        )
        hand["candidate_object"] = candidate_key
        hand["proximity_score"] = proximity_score
        hand["lock_score"] = candidate_score

        if state["is_grasping"]:
            locked_key = state["object_key"]
            hand["candidate_object"] = locked_key
            locked_score, locked_proximity = self._object_lock_score(
                hand,
                locked_key,
                dynamic_poses,
                object_geometry,
            )
            state["lock_score"] = locked_score
            hand["lock_score"] = locked_score
            hand["proximity_score"] = locked_proximity
            if locked_score < self.lock_exit_score:
                state["exit_count"] += 1
            else:
                state["exit_count"] = 0
            if state["exit_count"] >= self.lock_exit_frames:
                self._reset_hand_latch(state)
            elif hand_pose is not None:
                dynamic_poses[locked_key] = hand_pose @ state["T_lock_h2obj"]
                moved_objects.add(locked_key)
            return

        state["exit_count"] = 0
        if candidate_key is None or candidate_score < self.lock_enter_score:
            state["enter_count"] = 0
            return
        if state.get("candidate_object") == candidate_key:
            state["enter_count"] += 1
        else:
            state["candidate_object"] = candidate_key
            state["enter_count"] = 1
        if state["enter_count"] < self.lock_enter_frames or hand_pose is None:
            return

        state["is_grasping"] = True
        state["object_key"] = candidate_key
        state["lock_score"] = candidate_score
        state["T_lock_h2obj"] = (
            np.linalg.inv(hand_pose) @ dynamic_poses[candidate_key]
        )
        dynamic_poses[candidate_key] = hand_pose @ state["T_lock_h2obj"]
        moved_objects.add(candidate_key)

    @staticmethod
    def _reset_hand_latch(state: dict) -> None:
        state.update(
            {
                "is_grasping": False,
                "object_key": None,
                "T_lock_h2obj": None,
                "enter_count": 0,
                "exit_count": 0,
                "lock_score": 0.0,
                "candidate_object": None,
            }
        )

    def _best_object(
        self,
        hand: dict,
        dynamic_poses: dict[str, np.ndarray],
        anchor_key: str,
        object_geometry: dict[str, dict],
    ) -> tuple[str | None, float, float]:
        if hand["pose"] is None:
            return None, 0.0, 0.0
        candidates = []
        for key in dynamic_poses:
            if key == anchor_key:
                continue
            score, proximity = self._object_lock_score(
                hand, key, dynamic_poses, object_geometry
            )
            candidates.append((score, key, proximity))
        if not candidates:
            return None, 0.0, 0.0
        score, key, proximity = max(candidates, key=lambda item: item[0])
        return key, float(score), float(proximity)

    def _object_lock_score(
        self,
        hand: dict,
        object_key: str | None,
        dynamic_poses: dict[str, np.ndarray],
        object_geometry: dict[str, dict],
    ) -> tuple[float, float]:
        if object_key is None or hand["pose"] is None:
            return 0.0, 0.0
        geometry = object_geometry[object_key]
        center_local = np.r_[geometry["center_local"], 1.0]
        center_world = (dynamic_poses[object_key] @ center_local)[:3]
        distance = float(np.linalg.norm(hand["pose"][:3, 3] - center_world))
        surface_gap = max(0.0, distance - geometry["radius"])
        proximity = float(
            np.clip(
                1.0 - surface_gap / self.grasp_distance_threshold_m,
                0.0,
                1.0,
            )
        )
        confidence = float(np.clip(hand["confidence"] or 0.0, 0.0, 1.0))
        score = (
            self.lock_grasp_weight * hand["grasp"]
            + self.lock_confidence_weight * confidence
            + self.lock_proximity_weight * proximity
        )
        return float(np.clip(score, 0.0, 1.0)), proximity

    @staticmethod
    def _object_geometry(
        objects: dict, static_poses: dict[str, np.ndarray]
    ) -> dict[str, dict]:
        geometry = {}
        for key, value in objects.items():
            points = np.asarray(value.get("points_3d_world", []), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                points = np.empty((0, 3), dtype=np.float64)
            center_world = value.get("center_world")
            if center_world is None and len(points):
                center_world = points.mean(axis=0)
            if center_world is None:
                center_world = static_poses[key][:3, 3]
            center_world = np.asarray(center_world, dtype=np.float64).reshape(3)
            center_local = (
                np.linalg.inv(static_poses[key]) @ np.r_[center_world, 1.0]
            )[:3]
            if len(points):
                points_local = (
                    np.linalg.inv(static_poses[key])
                    @ np.c_[points, np.ones(len(points))].T
                ).T[:, :3]
                radius = float(np.max(np.linalg.norm(points_local - center_local, axis=1)))
            else:
                radius = 0.0
            geometry[key] = {"center_local": center_local, "radius": radius}
        return geometry

    @staticmethod
    def _object_state(hand_states: dict, object_key: str) -> str | None:
        for side in ("left", "right"):
            if hand_states[side]["object_key"] == object_key:
                return side
        return None

    def _hands_by_frame(self, hands, trajectory) -> dict:
        if hands is None:
            return {}
        if hands.world_frame != ARIA_MPS_WORLD_FRAME:
            raise ValueError("Object pose propagation requires Aria MPS hand poses")
        if len(hands.hands) != len(trajectory.frames):
            raise ValueError(
                "Hands and VIO trajectories must be aligned for object propagation"
            )
        return {int(item.idx): item for item in hands.hands}

    def _hand_state(self, hand) -> dict:
        if hand is None:
            return {
                "pose": None,
                "confidence": None,
                "grasp": 0.0,
                "proximity_score": 0.0,
                "lock_score": 0.0,
                "candidate_object": None,
            }
        confidence = hand.confidence
        grasp = float(hand.grasp_score)
        if confidence is None or float(confidence) < self.min_hand_confidence:
            return {
                "pose": None,
                "confidence": None if confidence is None else float(confidence),
                "grasp": grasp,
                "proximity_score": 0.0,
                "lock_score": 0.0,
                "candidate_object": None,
            }
        pose = hand.midpoint_pose_opt_world
        if pose is None:
            translation = hand.midpoint_translation_opt_world
            orientation = hand.midpoint_orientation_opt_world
            if translation is not None and orientation is not None:
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = np.asarray(orientation, dtype=np.float64).reshape(3, 3)
                pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        if pose is None:
            return {
                "pose": None,
                "confidence": float(confidence),
                "grasp": grasp,
                "proximity_score": 0.0,
                "lock_score": 0.0,
                "candidate_object": None,
            }
        pose = np.asarray(pose, dtype=np.float64)
        ObjectPosePropagator._validate_transform(pose, "hand midpoint pose")
        return {
            "pose": pose,
            "confidence": float(confidence),
            "grasp": grasp,
            "proximity_score": 0.0,
            "lock_score": 0.0,
            "candidate_object": None,
        }

    @staticmethod
    def _validate_triangulation(document: dict) -> None:
        if document.get("schema_version") != 3:
            raise ValueError("Unsupported triangulation schema")
        expected = {
            "world_frame": ARIA_MPS_WORLD_FRAME,
            "world_origin": ARIA_MPS_WORLD_ORIGIN,
            "initial_heading": ARIA_MPS_INITIAL_HEADING,
        }
        for key, value in expected.items():
            if document.get(key) != value:
                raise ValueError(f"Triangulation has invalid {key}")
        if not document.get("objects"):
            raise ValueError("Triangulation contains no objects")

    @staticmethod
    def _validate_transform(transform: np.ndarray, name: str) -> None:
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"{name} must be a finite 4x4 transform")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(f"{name} must be homogeneous")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError(f"{name} rotation is not orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"{name} rotation must be right-handed")
