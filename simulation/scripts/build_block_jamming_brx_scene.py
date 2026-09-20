"""Build the BRX Block Jamming MuJoCo scene from the vendor URDF."""

from __future__ import annotations

import argparse
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_ROOT = Path("/home/huasan/BRX20260825")
DEFAULT_OUTPUT = SIMULATION_ROOT / "tasks" / "block_jamming_brx.xml"
# Keep the task in the right-arm workspace while leaving a clear gap between
# the red block and the front wall of the blue tray.
TARGET_BODY_POSITION = (0.64, 0.02, 0.555)
BLOCK_BODY_POSITION = (0.64, -0.165, 0.565)


def _element(parent: ET.Element, tag: str, **attributes: object) -> ET.Element:
    return ET.SubElement(
        parent,
        tag,
        {key: str(value) for key, value in attributes.items()},
    )


def _convert_urdf(robot_root: Path) -> ET.Element:
    urdf_path = robot_root / "urdf" / "BRX20260825.urdf"
    mesh_root = robot_root / "meshes"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"BRX URDF not found: {urdf_path}")
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"BRX mesh directory not found: {mesh_root}")

    urdf_text = urdf_path.read_text(encoding="utf-8").replace(
        "package://BRX20260825/meshes/",
        f"{mesh_root.resolve()}/",
    )
    with tempfile.TemporaryDirectory(prefix="glassego_brx_") as temp_dir:
        temp_root = Path(temp_dir)
        resolved_urdf = temp_root / "BRX20260825.urdf"
        converted_mjcf = temp_root / "BRX20260825.xml"
        resolved_urdf.write_text(urdf_text, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(resolved_urdf))
        mujoco.mj_saveLastXML(str(converted_mjcf), model)
        return ET.parse(converted_mjcf).getroot()


def _configure_robot(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("autolimits", "true")
    root.set("model", "glassego block jamming brx")

    option = _element(root, "option", timestep="0.002", integrator="implicitfast")
    option.set("gravity", "0 0 -9.81")
    visual = _element(root, "visual")
    _element(
        visual,
        "headlight",
        diffuse="0.62 0.62 0.62",
        ambient="0.32 0.32 0.32",
        specular="0.08 0.08 0.08",
    )
    _element(visual, "quality", shadowsize="4096")
    _element(root, "statistic", extent="1.6", center="0.65 0 0.55")

    for geom in root.findall(".//geom"):
        geom.set("group", "2")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Converted BRX MJCF has no worldbody")

    right_wrist = worldbody.find(".//body[@name='ArmR08_Link']")
    left_wrist = worldbody.find(".//body[@name='ArmL08_Link']")
    if right_wrist is None or left_wrist is None:
        raise ValueError("Converted BRX MJCF is missing wrist bodies")
    # TCP is the midpoint of the two finger collision boxes, not the gripper-base tip.
    _element(right_wrist, "site", name="right_ee_site", pos="0.151 0.0007 -0.0084", size="0.006", rgba="1 0.3 0.1 0.7")
    _element(left_wrist, "site", name="left_ee_site", pos="0.151 -0.0007 -0.0084", size="0.006", rgba="0.1 0.5 1 0.7")

    for body_name in ("JawBlock01_Link", "JawBlock02_Link", "JawBlock03_Link", "JawBlock04_Link"):
        body = worldbody.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"Converted BRX MJCF is missing {body_name}")
        _element(
            body,
            "geom",
            name=f"{body_name}_collision",
            type="box",
            pos="0.035 0 0.009",
            size="0.045 0.014 0.009",
            group="3",
            rgba="0 0 0 0",
            contype="1",
            conaffinity="1",
            condim="4",
            friction="1.8 0.02 0.001",
        )


def _add_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(1, asset)
    _element(asset, "texture", type="skybox", name="skybox", builtin="gradient", rgb1="0.82 0.86 0.9", rgb2="0.5 0.56 0.62", width="512", height="3072")
    _element(asset, "texture", type="2d", name="work_grid", builtin="checker", mark="edge", rgb1="0.94 0.94 0.91", rgb2="0.88 0.89 0.86", markrgb="0.55 0.57 0.56", width="600", height="600")
    _element(asset, "material", name="work_surface", texture="work_grid", texrepeat="12 11", texuniform="true", reflectance="0.02")
    _element(asset, "material", name="table_edge", rgba="0.72 0.74 0.72 1")
    _element(asset, "material", name="container_blue", rgba="0.18 0.54 0.82 0.72", reflectance="0.04")
    _element(asset, "material", name="block_red", rgba="0.86 0.03 0.02 1", reflectance="0.03")


def _add_trapezoid_container(worldbody: ET.Element) -> None:
    body = _element(
        worldbody,
        "body",
        name="target_container",
        pos=" ".join(f"{value:g}" for value in TARGET_BODY_POSITION),
    )
    _element(body, "geom", name="target_container_bounds", type="box", size="0.105 0.08 0.04", pos="0 0 0.034", rgba="0 0 0 0", contype="0", conaffinity="0")
    _element(body, "geom", name="target_container_floor", type="box", size="0.075 0.055 0.004", pos="0 0 0", material="container_blue", friction="1.2 0.02 0.001")

    wall_height = 0.07
    bottom_x, top_x = 0.075, 0.105
    bottom_y, top_y = 0.055, 0.08
    x_angle = math.atan2(top_x - bottom_x, wall_height)
    y_angle = math.atan2(top_y - bottom_y, wall_height)
    x_center, y_center = (bottom_x + top_x) / 2, (bottom_y + top_y) / 2
    wall_half_height = math.hypot(wall_height, top_x - bottom_x) / 2
    _element(body, "geom", name="target_wall_left", type="box", size=f"0.004 {y_center:.6f} {wall_half_height:.6f}", pos=f"{-x_center:.6f} 0 {wall_height / 2:.6f}", euler=f"0 {-x_angle:.8f} 0", material="container_blue")
    _element(body, "geom", name="target_wall_right", type="box", size=f"0.004 {y_center:.6f} {wall_half_height:.6f}", pos=f"{x_center:.6f} 0 {wall_height / 2:.6f}", euler=f"0 {x_angle:.8f} 0", material="container_blue")
    wall_half_height = math.hypot(wall_height, top_y - bottom_y) / 2
    _element(body, "geom", name="target_wall_front", type="box", size=f"{x_center:.6f} 0.004 {wall_half_height:.6f}", pos=f"0 {-y_center:.6f} {wall_height / 2:.6f}", euler=f"{y_angle:.8f} 0 0", material="container_blue")
    _element(body, "geom", name="target_wall_back", type="box", size=f"{x_center:.6f} 0.004 {wall_half_height:.6f}", pos=f"0 {y_center:.6f} {wall_height / 2:.6f}", euler=f"{-y_angle:.8f} 0 0", material="container_blue")
    _element(body, "site", name="target_site", pos="0 0 0.04", size="0.008", rgba="0 0 0 0")


def _add_task(worldbody: ET.Element) -> None:
    _element(worldbody, "geom", name="floor", type="plane", size="3 3 0.05", pos="0 0 -0.44", rgba="0.42 0.45 0.47 1", friction="1 0.01 0.001")
    _element(worldbody, "geom", name="task_table", type="box", size="0.62 0.56 0.025", pos="0.86 0 0.525", material="work_surface", friction="1.2 0.02 0.001")
    _element(worldbody, "geom", name="task_table_edge", type="box", size="0.625 0.565 0.012", pos="0.86 0 0.488", material="table_edge")
    head = worldbody.find(".//body[@name='Head03_Link']")
    if head is None:
        raise ValueError("Converted BRX MJCF is missing Head03_Link")
    # Relative to the middle eye: 50 mm up (-Y), 15 mm forward (-X).
    # Head +Z points image-right; pitch the optical axis down 55 degrees.
    _element(head, "camera", name="ego_rgbd", pos="-0.072512 -0.134789 -0.0187", xyaxes="0 0 1 -0.8191520443 -0.5735764364 0", fovy="60")
    _element(worldbody, "camera", name="task_overview", pos="1.45 -1.25 1.25", xyaxes="0.68 0.73 0 -0.45 0.42 0.79", fovy="48")
    _element(worldbody, "light", name="key_light", pos="0.65 -0.35 1.8", dir="0.15 0.15 -1", diffuse="0.8 0.78 0.72", directional="false")
    _element(worldbody, "light", name="fill_light", pos="0.2 0.8 1.2", dir="0.45 -0.35 -0.8", diffuse="0.35 0.42 0.5", directional="false")
    _add_trapezoid_container(worldbody)

    block = _element(
        worldbody,
        "body",
        name="block",
        pos=" ".join(f"{value:g}" for value in BLOCK_BODY_POSITION),
    )
    _element(block, "freejoint", name="block_freejoint")
    _element(block, "geom", name="block_geom", type="box", size="0.015 0.015 0.015", mass="0.02", material="block_red", condim="4", friction="1.8 0.03 0.001", solref="0.01 1")
    _element(block, "site", name="block_grasp_site", size="0.006", rgba="1 1 1 0.5")


def _add_control(root: ET.Element) -> None:
    equality = _element(root, "equality")
    _element(equality, "joint", joint1="JawBlock02_Joint", joint2="JawBlock01_Joint", polycoef="0 -1 0 0 0")
    _element(equality, "joint", joint1="JawBlock04_Joint", joint2="JawBlock03_Joint", polycoef="0 -1 0 0 0")

    actuator = _element(root, "actuator")
    joint_ranges = {
        "FoldingModularJoint02_Joint": (-1.31, 2.1),
        "FoldingModularJoint03_Joint": (-1.1338, 1.74),
        "Trunk_Joint": (-0.785, 2.1),
        "Head02_Joint": (-1.23, 1.57),
        "Head03_Joint": (-1.0, 1.0),
        "ArmR02_Joint": (-1.57, 1.57),
        "ArmR03_Joint": (-1.57, 0.52),
        "ArmR04_Joint": (-1.57, 1.57),
        "ArmR05_Joint": (-0.53, 1.57),
        "ArmR06_Joint": (-2.1, 2.1),
        "ArmR07_Joint": (-0.52, 0.52),
        "ArmR08_Joint": (-1.0, 1.0),
        "ArmL02_Joint": (-1.57, 1.2),
        "ArmL03_Joint": (-0.52, 1.57),
        "ArmL04_Joint": (-1.57, 1.57),
        "ArmL05_Joint": (-1.57, 0.53),
        "ArmL06_Joint": (-2.1, 2.1),
        "ArmL07_Joint": (-0.52, 0.52),
        "ArmL08_Joint": (-1.0, 1.0),
    }
    for joint_name, (lower, upper) in joint_ranges.items():
        _element(
            actuator,
            "position",
            name=joint_name,
            joint=joint_name,
            ctrlrange=f"{lower} {upper}",
            kp="120" if "Arm" in joint_name else "200",
            kv="12" if "Arm" in joint_name else "20",
        )
    _element(actuator, "position", name="right_gripper", joint="JawBlock01_Joint", ctrlrange="0 0.041", kp="800", kv="35")
    _element(actuator, "position", name="left_gripper", joint="JawBlock03_Joint", ctrlrange="0 0.041", kp="800", kv="35")


def _add_contact_exclusions(root: ET.Element) -> None:
    contact = _element(root, "contact")
    _element(contact, "exclude", body1="JawBlock01_Link", body2="JawBlock02_Link")
    _element(contact, "exclude", body1="JawBlock03_Link", body2="JawBlock04_Link")


def _add_oracle_grasp_constraint(root: ET.Element) -> None:
    equality = root.find("equality")
    if equality is None:
        equality = _element(root, "equality")
    # Disabled by default; the Oracle script activates it only after confirmed
    # finger contact.  This gives a deterministic task-level baseline without
    # changing normal policy episodes.
    _element(
        equality,
        "weld",
        name="oracle_grasp_weld",
        body1="block",
        body2="ArmR08_Link",
        active="false",
        solref="0.01 1",
        solimp="0.9 0.95 0.001",
    )


def build_scene(robot_root: Path, output_path: Path) -> Path:
    root = _convert_urdf(robot_root.resolve())
    _configure_robot(root)
    _add_assets(root)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Converted BRX MJCF has no worldbody")
    _add_task(worldbody)
    _add_control(root)
    _add_contact_exclusions(root)
    _add_oracle_grasp_constraint(root)
    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="unicode", xml_declaration=False)
    mujoco.MjModel.from_xml_path(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, default=DEFAULT_ROBOT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_path = build_scene(args.robot_root.expanduser(), args.output.expanduser().resolve())
    print(f"[ok] wrote BRX MuJoCo scene: {output_path}")


if __name__ == "__main__":
    main()
