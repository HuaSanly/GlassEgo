
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
import imageio
from preprocess.data_types.CamTypes import Cam, CamData
from preprocess.data_types.VIOTypes import VIOResult

def build_cam_from_disk(
    video_path: str,
    focal_length_px: float | None = None,
    vio_result: VIOResult | None = None,
) -> Cam:
    """Read an MP4 into the Cam container expected by HaMeRHandsGenerator.

    VIO results provide calibrated intrinsics, aligned nanosecond timestamps,
    and camera poses. Without VIO, the existing focal-length and identity-pose
    fallback is retained. Images are always stored as RGB.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video dimensions: {width}x{height}")

    if vio_result is None:
        focal = float(focal_length_px or max(width, height))
        k = np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        c2d = np.eye(4, dtype=np.float32)
        distortion = np.zeros(8, dtype=np.float32)
        trajectory_frames = None
        first_ts = 0
        fov = 0.0
    else:
        calibration = vio_result.calibration
        if (width, height) != calibration.resolution:
            cap.release()
            raise ValueError(
                "Video resolution does not match VIO calibration: "
                f"video={width}x{height}, "
                f"calibration={calibration.resolution[0]}x"
                f"{calibration.resolution[1]}"
            )
        fx, fy, cx, cy = calibration.intrinsics
        k = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        c2d = calibration.T_imu_camera.astype(np.float32)
        distortion = calibration.distortion.astype(np.float32)
        trajectory_frames = vio_result.trajectory.frames
        first_ts = trajectory_frames[0].timestamp_ns
        fov = float(np.degrees(2.0 * np.arctan(height / (2.0 * fy))))

    cam = Cam(
        fps=fps,
        first_ts=first_ts,
        fov=fov,
        h=height,
        w=width,
        k=k,
        d=distortion,
        c2d=c2d,
        data_path=str(Path(video_path).resolve().parent),
    )

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if trajectory_frames is None:
                timestamp_ns = int(round(frame_idx * 1_000_000_000 / fps))
                c2w = np.eye(4, dtype=np.float32)
                d2w = np.eye(4, dtype=np.float32)
            else:
                if frame_idx >= len(trajectory_frames):
                    raise ValueError(
                        "Video contains more frames than the VIO trajectory"
                    )
                pose = trajectory_frames[frame_idx]
                if pose.frame_idx != frame_idx:
                    raise ValueError(
                        "VIO frame indices are not aligned with the video: "
                        f"expected={frame_idx}, actual={pose.frame_idx}"
                    )
                timestamp_ns = pose.timestamp_ns
                c2w = pose.c2w.astype(np.float32)
                d2w = (
                    pose.c2w @ vio_result.calibration.T_cam_imu
                ).astype(np.float32)
            frame = CamData(
                idx=frame_idx,
                ts=timestamp_ns,
                img=frame_rgb,
                fov=fov,
                h=height,
                w=width,
                k=k.copy(),
                d=distortion.copy(),
                c2w=c2w.copy(),
                c2d=c2d.copy(),
                d2w=d2w.copy(),
            )
            cam.cam.append(frame)
            cam.tss.append(timestamp_ns)
            frame_idx += 1
    finally:
        cap.release()

    if not cam.cam:
        raise ValueError(f"Video contains no readable frames: {video_path}")
    if trajectory_frames is not None and len(cam.cam) != len(trajectory_frames):
        raise ValueError(
            "Video frame count does not match the VIO trajectory: "
            f"video={len(cam.cam)}, trajectory={len(trajectory_frames)}"
        )

    print(
        f"[Camera] Loaded {len(cam.cam)} RGB frames from {video_path} "
        f"({width}x{height} @ {fps:.3f} FPS)"
    )
    return cam


def create_video_from_frames(
    frames,
    save_path,
    fps=10,
    export_gif=True,
    ratio=10,
    export_video=True,
):
    """Write BGR uint8 frames to MP4 and/or a sampled RGB GIF."""
    if not export_video and not export_gif:
        return
    if not frames:
        raise ValueError("frames must not be empty")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if export_gif and (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, np.integer))
        or ratio <= 0
    ):
        raise ValueError(f"ratio must be a positive integer, got {ratio}")

    first_frame = frames[0]
    if not isinstance(first_frame, np.ndarray) or first_frame.dtype != np.uint8:
        raise ValueError("frames must be uint8 numpy arrays")
    if first_frame.ndim == 2:
        h, w = first_frame.shape
        is_grayscale = True
    elif first_frame.ndim == 3 and first_frame.shape[2] == 3:
        h, w, _ = first_frame.shape
        is_grayscale = False
    else:
        raise ValueError(f"Unsupported frame shape: {first_frame.shape}")

    expected_shape = (h, w) if is_grayscale else (h, w, 3)
    for index, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError(f"Frame {index} must be a uint8 numpy array")
        if frame.shape != expected_shape:
            raise ValueError(
                f"Frame {index} has shape {frame.shape}; expected {expected_shape}"
            )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    size = (w, h)
    writer = None
    if export_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, float(fps), size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Unable to open VideoWriter: {save_path}")

    outputs = []
    if export_video:
        outputs.append("MP4")
    if export_gif:
        outputs.append("GIF")
    print(f"Generating {' and '.join(outputs)} (Resolution={w}x{h}, FPS={fps})...")

    gif_frames = []
    try:
        for index, frame in enumerate(tqdm(frames, desc="Write frames", mininterval=1.0)):
            frame_bgr = (
                cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if is_grayscale
                else frame
            )
            if writer is not None:
                writer.write(frame_bgr)
            if export_gif and index % ratio == 0:
                gif_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        if writer is not None:
            writer.release()

    if export_video:
        if not save_path.is_file() or save_path.stat().st_size == 0:
            raise RuntimeError(f"Video output was not created: {save_path}")
        print(f"Video Saved to: {save_path}")

    if export_gif:
        gif_path = save_path.with_suffix(".gif")
        try:
            imageio.mimsave(
                str(gif_path),
                gif_frames,
                fps=float(fps) / ratio,
                loop=0,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to save GIF: {gif_path}") from exc
        if not gif_path.is_file() or gif_path.stat().st_size == 0:
            raise RuntimeError(f"GIF output was not created: {gif_path}")
        print(f"GIF Saved to: {gif_path}")
