"""HumanEgo-style static initialization and hand-latched object poses."""

import numpy as np

from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
)


OBJECT_POSE_SCHEMA_VERSION = 1


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
            "has_released": False,
            "object_key": None,
            "T_lock_h2obj": None,
        }

    def _update_hand_latch(
        self,
        state: dict,
        hand: dict,
        dynamic_poses: dict[str, np.ndarray],
        anchor_key: str,
        moved_objects: set[str],
    ) -> None:
        is_grasp = hand["grasp"] > 0.5
        hand_pose = hand["pose"]
        if not is_grasp:
            state["has_released"] = True

        if (
            is_grasp
            and not state["is_grasping"]
            and state["has_released"]
            and hand_pose is not None
        ):
            state["is_grasping"] = True
            object_key = self._nearest_dynamic_object(
                hand_pose[:3, 3],
                dynamic_poses,
                anchor_key,
            )
            if object_key is not None:
                state["object_key"] = object_key
                state["T_lock_h2obj"] = (
                    np.linalg.inv(hand_pose) @ dynamic_poses[object_key]
                )
            else:
                state["object_key"] = None
                state["T_lock_h2obj"] = None
        elif not is_grasp and state["is_grasping"]:
            state["is_grasping"] = False
            state["object_key"] = None
            state["T_lock_h2obj"] = None

        if (
            state["is_grasping"]
            and state["object_key"] is not None
            and state["T_lock_h2obj"] is not None
            and hand_pose is not None
        ):
            dynamic_poses[state["object_key"]] = hand_pose @ state["T_lock_h2obj"]
            moved_objects.add(state["object_key"])

    def _nearest_dynamic_object(
        self,
        hand_position: np.ndarray,
        dynamic_poses: dict[str, np.ndarray],
        anchor_key: str,
    ) -> str | None:
        best_key = None
        best_distance = float("inf")
        for key, pose in dynamic_poses.items():
            if key == anchor_key:
                continue
            distance = float(np.linalg.norm(hand_position - pose[:3, 3]))
            if distance < best_distance:
                best_key = key
                best_distance = distance
        return best_key if best_distance < self.grasp_distance_threshold_m else None

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
            return {"pose": None, "confidence": None, "grasp": 0}
        confidence = hand.confidence
        grasp = int(hand.grasp_state or 0)
        if confidence is None or float(confidence) < self.min_hand_confidence:
            return {
                "pose": None,
                "confidence": None if confidence is None else float(confidence),
                "grasp": grasp,
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
            }
        pose = np.asarray(pose, dtype=np.float64)
        ObjectPosePropagator._validate_transform(pose, "hand midpoint pose")
        return {
            "pose": pose,
            "confidence": float(confidence),
            "grasp": grasp,
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
