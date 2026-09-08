"""Build a dual-WXAI Block Jamming scene from official Trossen assets."""

from __future__ import annotations

from pathlib import Path

import mujoco


TROSSEN_ASSET_ROOT = Path("/home/huasan/trossen_arm_mujoco/trossen_arm_mujoco/assets")
WXAI_FOLLOWER_XML = TROSSEN_ASSET_ROOT / "wxai" / "wxai_follower.xml"
OUTPUT_XML = Path(__file__).resolve().parents[1] / "tasks" / "block_jamming_wxai.xml"
ARM_FORWARD_QUAT = [0.7071067811865476, 0.0, 0.0, 0.7071067811865475]


def add_scene_assets(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="task_skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.82, 0.88, 0.96],
        rgb2=[1.0, 1.0, 1.0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="floor_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.72, 0.74, 0.77],
        rgb2=[0.62, 0.64, 0.67],
        markrgb=[0.9, 0.9, 0.9],
        width=300,
        height=300,
    )
    spec.add_material(
        name="floor_grid",
        textures=["floor_grid"],
        texrepeat=[4.0, 4.0],
        texuniform=True,
        reflectance=0.15,
    )
    spec.add_material(name="task_table_white", rgba=[0.96, 0.96, 0.93, 1.0])
    spec.add_material(name="block_blue", rgba=[0.05, 0.24, 0.85, 1.0])
    spec.add_material(name="target_yellow", rgba=[1.0, 0.86, 0.08, 1.0])


def add_robot_pair(spec: mujoco.MjSpec) -> None:
    left_mount = spec.worldbody.add_frame(
        name="left_wxai_mount",
        pos=[-0.17, -0.38, 0.06],
        quat=ARM_FORWARD_QUAT,
    )
    right_mount = spec.worldbody.add_frame(
        name="right_wxai_mount",
        pos=[0.17, -0.38, 0.06],
        quat=ARM_FORWARD_QUAT,
    )

    spec.attach(mujoco.MjSpec.from_file(str(WXAI_FOLLOWER_XML)), prefix="left_", frame=left_mount)
    spec.attach(mujoco.MjSpec.from_file(str(WXAI_FOLLOWER_XML)), prefix="right_", frame=right_mount)


def add_task_objects(spec: mujoco.MjSpec) -> None:
    world = spec.worldbody
    world.add_light(
        name="task_light",
        pos=[0.0, -0.25, 1.2],
        dir=[0.0, 0.2, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        material="floor_grid",
    )
    world.add_geom(
        name="task_table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.36, 0.32, 0.025],
        pos=[0.0, 0.0, 0.025],
        material="task_table_white",
        friction=[1.2, 0.02, 0.001],
    )

    target = world.add_body(name="target_box", pos=[0.14, 0.12, 0.065])
    target.add_geom(
        name="target_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.055, 0.055, 0.015],
        material="target_yellow",
        friction=[1.5, 0.02, 0.001],
    )
    target.add_site(
        name="target_site",
        pos=[0.0, 0.0, 0.017],
        size=[0.012],
        rgba=[1.0, 0.75, 0.0, 0.8],
    )

    block = world.add_body(name="block", pos=[-0.14, 0.12, 0.075])
    block.add_freejoint(name="block_freejoint")
    block.add_geom(
        name="block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.025],
        material="block_blue",
        mass=0.05,
        condim=4,
        friction=[2.0, 0.03, 0.001],
        solref=[0.01, 1.0],
    )
    block.add_site(
        name="block_grasp_site",
        pos=[0.0, 0.0, 0.0],
        size=[0.008],
        rgba=[1.0, 1.0, 1.0, 0.6],
    )

    world.add_camera(
        name="ego_rgbd",
        pos=[0.0, -0.7, 0.5],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 0.5, 0.87],
        fovy=72,
    )
    world.add_camera(
        name="task_overview",
        pos=[0.58, -0.72, 0.55],
        xyaxes=[0.78, 0.63, 0.0, -0.32, 0.39, 0.86],
        fovy=48,
    )


def build_scene() -> mujoco.MjSpec:
    if not WXAI_FOLLOWER_XML.exists():
        raise FileNotFoundError(f"Official WXAI follower XML not found: {WXAI_FOLLOWER_XML}")

    spec = mujoco.MjSpec()
    spec.modelname = "glassego block jamming wxai"
    spec.compiler.autolimits = True
    spec.compiler.meshdir = str(TROSSEN_ASSET_ROOT / "meshes")
    spec.compiler.texturedir = str(TROSSEN_ASSET_ROOT)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.stat.center = [0.0, -0.12, 0.18]
    spec.stat.extent = 0.8

    add_scene_assets(spec)
    add_robot_pair(spec)
    add_task_objects(spec)
    return spec


def main() -> None:
    spec = build_scene()
    OUTPUT_XML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_XML.write_text(spec.to_xml(), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))
    print(f"[ok] wrote {OUTPUT_XML}")
    print(f"[ok] model: bodies={model.nbody} joints={model.njnt} actuators={model.nu}")


if __name__ == "__main__":
    main()
