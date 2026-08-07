import os
import json
import numpy as np
import argparse
from pathlib import Path
from dataclasses import fields, is_dataclass
from data_types.CamTypes import Cam,CamData
from data_types.HandsTypes import Hands
from hand_tracking.HaMeRHandsGenerator import HaMeRHandsGenerator
import cv2
from tqdm import tqdm
import imageio
def _build_cam_from_preprocessed_frames(data_path: str) -> Cam:
    """
    从磁盘上的每帧 JSON 文件重建轻量级 AriaCam 物体。
    这允许独立运行 MediaPipeHands（无需完整的预处理管道）。
    返回带有 cam[] 的 AriaCam，每帧包含 img、k、c2w。
    """

    all_data_dir = os.path.join(data_path, "preprocess", "all_data")
    if not os.path.isdir(all_data_dir):
        raise FileNotFoundError(f"all_data directory not found: {all_data_dir}")

    frame_dirs = sorted([d for d in os.listdir(all_data_dir) if d.isdigit()])
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found in {all_data_dir}")

    cam = Cam()
    cam.mps_path = data_path

    for fn in frame_dirs:
        frame_dir = os.path.join(all_data_dir, fn)
        cam_json_path = os.path.join(frame_dir, "aria_cam_rgb.json")
        rgb_path = os.path.join(frame_dir, "rgb.png")

        if not os.path.exists(cam_json_path) or not os.path.exists(rgb_path):
            continue

        with open(cam_json_path, 'r') as f:
            cam_d = json.load(f)

        img = cv2.imread(rgb_path)  # 保留为 BGR 以符合 AriaCam 约定

        frame = CamData(
            idx=int(fn),
            ts=cam_d.get('ts', 0),
            img=img,
            h=img.shape[0] if img is not None else 0,
            w=img.shape[1] if img is not None else 0,
            k=np.array(cam_d['k'], dtype=np.float64) if cam_d.get('k') is not None else np.eye(3),
            c2w=np.array(cam_d['c2w'], dtype=np.float64) if cam_d.get('c2w') is not None else np.eye(4),
            d=np.zeros(8),       # 畸变系数（图像已校正）
            c2d=np.eye(4),       # 相机到设备（已纠正的身份）
            d2w=np.eye(4),       # 设备到世界占位符
        )

        cam.tss.append(frame.ts)
        cam.cam.append(frame)

    if cam.cam:
        cam.k = cam.cam[0].k
        cam.h = cam.cam[0].h
        cam.w = cam.cam[0].w
        cam.fps = 30  # 默认 Aria RGB FPS;如果 aria_cam_rgb.json 有 fps 字段则覆盖
        if len(cam.cam) > 0:
            first_cam_json = os.path.join(all_data_dir, frame_dirs[0], "aria_cam_rgb.json")
            if os.path.exists(first_cam_json):
                with open(first_cam_json, 'r') as f:
                    first_d = json.load(f)
                cam.fps = first_d.get('fps', 30)

    print(f"[MediaPipeHands] Loaded {len(cam.cam)} frames from disk")
    return cam


def build_cam_from_disk(
    video_path: str,
    focal_length_px: float | None = None,
) -> Cam:
    """Read an MP4 into the Cam container expected by HaMeRHandsGenerator.

    Since a normal video does not contain camera calibration or poses, this
    uses an approximate pinhole intrinsic matrix and identity camera poses.
    Images are stored as RGB and timestamps use nanoseconds.
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

    focal = float(focal_length_px or max(width, height))
    k = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2d = np.eye(4, dtype=np.float32)
    d2w = np.eye(4, dtype=np.float32)
    distortion = np.zeros(8, dtype=np.float32)

    cam = Cam(
        fps=fps,
        first_ts=0,
        h=height,
        w=width,
        k=k,
        d=distortion,
        c2d=c2d,
        mps_path=str(Path(video_path).resolve().parent),
    )

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ns = int(round(frame_idx * 1_000_000_000 / fps))
            frame = CamData(
                idx=frame_idx,
                ts=timestamp_ns,
                img=frame_rgb,
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

    print(
        f"[Camera] Loaded {len(cam.cam)} RGB frames from {video_path} "
        f"({width}x{height} @ {fps:.3f} FPS)"
    )
    return cam

def create_video_from_frames(frames, save_path, fps=10, export_gif=True, ratio=10):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not frames:
        print("Warning: frames is empty!")
        return

    first_frame = frames[0]
    if len(first_frame.shape) == 2:
        h, w = first_frame.shape
        is_grayscale = True
    else:
        h, w, _ = first_frame.shape
        is_grayscale = False

    size = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, size)

    if export_gif:
        gif_frames = []

    print(f"Generating Video and GIF (Resolution={w}x{h}, FPS={fps})...")

    i = 0
    for frame in tqdm(frames):
        i+=1
        if is_grayscale:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)

        if export_gif:
            if i%ratio == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                gif_frames.append(frame_rgb)

    out.release()
    print(f"Video Saved to: {save_path}")
    if export_gif and gif_frames:
        output_gif_path = save_path.replace('.mp4', '.gif')
        print(f"Gif Saved to: {output_gif_path}")
        try:
            imageio.mimsave(output_gif_path, gif_frames, fps=fps//ratio, loop=0)
        except Exception as e:
            print(f"Failed: {e}")

def _run_hamer_hands_legacy(data_path: str, cfg_path: str, cam=None,
                    export_video: bool = False, export_gif: bool = False) -> Hands:
    """
    Preprocess.py 集成的入口点。

    参数：
        data_path：根数据目录。
        cfg_path：AriaHands.yaml 配置的路径。
        cam：可选的预构建 Cam 物体。如果无，则从磁盘重建。
        export_video：是否导出可视化视频。
        export_gif：是否与视频一起导出GIF。
    """

    if cam is None:
        cam = build_cam_from_disk(data_path)

    gen = HaMeRHandsGenerator(data_path, cfg_path,  cam)
    aria_hands = gen.get_hands_data()

    # 使用特定于方法的文件名保存每帧 JSON
    aria_hands.save_hands_json(filename="hamer_hands.json")
    print(f"[HaMeR] Saved hamer_hands.json for {len(aria_hands)} frames")

    # ── 可视化视频（骨架+HUD叠加）──
    if export_video and len(cam.cam) > 0:
        import cv2
        from tqdm import tqdm
        print(f"[HaMeR] Generating visualization video …")
        vis_frames = []
        for idx in tqdm(range(len(cam.cam)), desc="HaMeR Vis"):
            cam_d = cam.cam[idx]
            img = cam_d.img
            if img is None:
                img_path = os.path.join(data_path, "preprocess", "all_data",
                                        f"{cam_d.idx:05d}", "rgb.png")
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
            if img is None:
                continue
            img = img.copy()

            if idx < len(aria_hands.hands):
                img = gen.draw_aria_hands_skeleton(
                    img, aria_hands.hands[idx],
                    cam_d.k, getattr(cam_d, 'd', np.zeros(8)), cam_d.c2w
                )
                img = gen.draw_aria_hands_panel(img, idx, aria_hands.hands[idx])

            vis_frames.append(img)

        if vis_frames:
            vis_dir = os.path.join(data_path, "preprocess", "vis")
            os.makedirs(vis_dir, exist_ok=True)
            save_path = os.path.join(vis_dir, "hamer_hands_vis.mp4")
            create_video_from_frames(vis_frames, save_path, cam.fps, export_gif)
            print(f"[HaMeR] Saved visualization → {save_path}")

    return aria_hands


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _json_safe(value):
    """Convert dataclasses and NumPy values into JSON-compatible objects."""
    if is_dataclass(value):
        return {
            item.name: _json_safe(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _draw_hand_overlay(frame_bgr, hand_data, label: str, color) -> None:
    """Draw HaMeR's 21 projected points and a compact status label."""
    if hand_data is None or hand_data.hand_keypoints_2d is None:
        return

    points = np.asarray(hand_data.hand_keypoints_2d)
    if points.shape[0] < 21:
        return

    for start, end in HAND_CONNECTIONS:
        x1, y1 = points[start].astype(int)
        x2, y2 = points[end].astype(int)
        cv2.line(frame_bgr, (x1, y1), (x2, y2), color, 2)

    for x, y in points[:21].astype(int):
        cv2.circle(frame_bgr, (x, y), 3, color, -1)

    wrist = points[0].astype(int)
    text = f"{label} conf={float(hand_data.confidence or 0.0):.2f}"
    text += f" grasp={int(hand_data.grasp_state)}"
    cv2.putText(
        frame_bgr,
        text,
        (int(wrist[0]), max(20, int(wrist[1]) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def run_hamer_hands(data_path: str, cfg_path: str, cam=None,
                    export_video: bool = False,
                    export_gif: bool = False) -> Hands:
    """Run the complete HaMeR hand pipeline for a video or prebuilt ``Cam``."""
    if cam is None:
        cam = build_cam_from_disk(data_path)

    generator = HaMeRHandsGenerator(data_path, cfg_path, cam)
    hands = generator.get_hands_data()

    source = Path(data_path)
    output_root = Path(__file__).resolve().parent / "localdata"
    output_root.mkdir(parents=True, exist_ok=True)
    output_stem = source.stem if source.stem else "hamer"
    json_path = output_root / f"{output_stem}_hamer_hands.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(hands), file, ensure_ascii=False, indent=2)
    print(f"[HaMeR] Saved {json_path} ({len(hands)} frames)")

    if export_video and cam.cam:
        vis_frames = []
        for idx, cam_data in enumerate(tqdm(cam.cam, desc="HaMeR Vis")):
            if cam_data.img is None:
                continue

            # build_cam_from_disk stores RGB; OpenCV drawing/writing uses BGR.
            frame_bgr = cv2.cvtColor(cam_data.img, cv2.COLOR_RGB2BGR)
            frame_hands = hands.hands[idx] if idx < len(hands.hands) else None
            if frame_hands is not None:
                _draw_hand_overlay(
                    frame_bgr,
                    frame_hands.hand_r,
                    "Right",
                    (0, 255, 0),
                )
                _draw_hand_overlay(
                    frame_bgr,
                    frame_hands.hand_l,
                    "Left",
                    (0, 140, 255),
                )

            vis_frames.append(frame_bgr)

        if vis_frames:
            save_path = output_root / f"{output_stem}_hamer_hands_vis.mp4"
            create_video_from_frames(
                vis_frames,
                str(save_path),
                fps=cam.fps,
                export_gif=export_gif,
            )
            print(f"[HaMeR] Saved visualization: {save_path}")

    return hands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaMeR Hand Mesh Recovery Tracking")
    parser.add_argument(
        "--video_path",
        "--data_path",
        dest="video_path",
        type=str,
        required=True,
        help="Input MP4 path",
    )
    parser.add_argument("--cfg_path", type=str, default="")
    parser.add_argument("--export_video", action="store_true")
    parser.add_argument("--export_gif", action="store_true")
    args = parser.parse_args()
    print(f"[HaMeR] video_path={args.video_path}")
    run_hamer_hands(args.video_path, args.cfg_path,
                    export_video=args.export_video, export_gif=args.export_gif)
