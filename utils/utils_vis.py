import cv2
import numpy as np
from typing import List, Tuple

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    HEADER = f"{BOLD}{BLUE}"
    OKBLUE = f"{BOLD}{CYAN}"
    OKGREEN = f"{BOLD}{GREEN}"
    WARNING = f"{BOLD}{YELLOW}"
    FAIL = f"{BOLD}{RED}"
    ENDC = END


C_CYAN = (255, 255, 0); C_GREEN = (0, 255, 0); C_RED = (0, 0, 255)
C_GOLD = (0, 215, 255); C_WHITE = (255, 255, 255); C_GRAY = (100, 100, 100)


def draw_glass_rect(img, pt1, pt2, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.rectangle(img, pt1, pt2, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def draw_arc_gauge(img, center, value, vmax, label):
    cx, cy = center
    r = 22
    v = float(np.clip(value / (vmax + 1e-6), 0.0, 1.0))
    cv2.ellipse(img, (cx, cy), (r, r), 0, 225, -45, (60, 60, 60), 3, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r, r), 0, 225, 225 + int(270 * v), (0, 200, 255), 3, cv2.LINE_AA)
    cv2.putText(img, label, (cx - 6, cy + 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


def draw_status_bar(img, pos, width, val, max_val, label, color):
    x, y = pos
    bar_w = int((val / max_val) * width) if max_val > 0 else 0
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + width, y + 5), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + bar_w, y + 5), color, -1)






def draw_text_with_shadow(img: np.ndarray, text: str, position: Tuple[int, int],
                          font_scale: float, color: Tuple[int, int, int],
                          thickness: int = 2) -> None:
    """
    使用深色阴影背景绘制文本，以便在复杂的相机流上获得更好的可读性。

    参数：
        img (np.ndarray): OpenCV 图像画布。
        text (str): 要显示的文本。
        position (Tuple[int, int]): 文本字符串 (x, y) 的左下角。
        font_scale (float)：字体比例因子。
        color (Tuple[int, int, int]): BGR 格式的主要文本颜色。
        thickness (int): 正文的粗细。
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    x, y = position

    # 绘制阴影（偏移+1和-1）
    cv2.putText(img, text, (x + 1, y + 1), font, font_scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x - 1, y - 1), font, font_scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)

    # 绘制实际的前景文本
    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def get_traj_colors(horizon: int) -> List[Tuple[int, int, int]]:
    """
    生成轨迹可视化的颜色渐变列表。
    优雅地淡出，代表时间步入未来。

    参数：
        horizon (int): 预测轨迹中的步数。

    返回：
        List[Tuple[int, int, int]]：BGR 颜色列表。
    """
    return[
        (int(bg), int(bg), int(r))
        for bg, r in zip(np.linspace(200, 0, horizon), np.linspace(255, 100, horizon))
    ]

def project_3d_to_2d(p_3d: np.ndarray, K_mat: np.ndarray) -> Tuple[int, int]:
    """
    将相机坐标系中的 3D 点投影到 2D 像素坐标。

    参数：
        p_3d (np.ndarray)：3D 点 [X、Y、Z]。
        K_mat (np.ndarray)：3x3 相机内参矩阵。

    返回：
        Tuple[int, int]：(u, v) 像素坐标。
    """
    if p_3d[2] < 1e-6:  # 防止被零除
        return (0, 0)

    u_proj = int(K_mat[0, 0] * p_3d[0] / p_3d[2] + K_mat[0, 2])
    v_proj = int(K_mat[1, 1] * p_3d[1] / p_3d[2] + K_mat[1, 2])

    return (u_proj, v_proj)