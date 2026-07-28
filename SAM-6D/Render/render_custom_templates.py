import blenderproc as bproc

import os
import argparse
import cv2
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--cad_path', help="The path of CAD model")
parser.add_argument('--output_dir', help="The path to save CAD templates")
parser.add_argument('--normalize', default=True, help="Whether to normalize CAD model or not")
parser.add_argument('--colorize', default=False, help="Whether to colorize CAD model or not")
parser.add_argument('--base_color', default=0.05, help="The base color used in CAD model")
args = parser.parse_args()

# set the cnos camera path
render_dir = os.path.dirname(os.path.abspath(__file__))
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')

bproc.init()

def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    return 1/(2*radius)


def purge_invalid_structs():
    """Drop BlenderProc wrappers whose Blender RNA was deleted by clean_up().

    Otherwise ``render_nocs()`` -> ``UndoAfterExecution`` -> ``get_instances()``
    crashes with: ReferenceError: StructRNA of type Material has been removed.
    """
    from blenderproc.python.types.StructUtility import Struct

    for inst in list(Struct.__refs__):
        try:
            _ = inst.blender_obj.name
        except ReferenceError:
            Struct.__refs__.discard(inst)


def prepare_cad_for_render(cad_path: str) -> str:
    """Write a colorless mesh copy so load_obj does not create per-face materials."""
    mesh = trimesh.load(cad_path, force="mesh")
    if hasattr(mesh, "visual"):
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    out_path = os.path.join(args.output_dir, "_cad_render_no_color.ply")
    os.makedirs(args.output_dir, exist_ok=True)
    mesh.export(out_path)
    return out_path


# load cnos camera pose
cam_poses = np.load(cnos_cam_fpath)

# calculating the scale of CAD model
if args.normalize:
    scale = get_norm_info(args.cad_path)
else:
    scale = 1

cad_path_for_render = prepare_cad_for_render(args.cad_path)
print(f"[render_custom_templates] using colorless CAD: {cad_path_for_render}")

for idx, cam_pose in enumerate(cam_poses):
    
    bproc.clean_up()
    purge_invalid_structs()

    # load object
    obj = bproc.loader.load_obj(cad_path_for_render)[0]
    obj.set_scale([scale, scale, scale])
    obj.set_cp("category_id", 1)

    # Always assign a fresh material (tless-style); avoids leftover face-color mats.
    base = float(args.base_color)
    if str(args.colorize).lower() in {"1", "true", "yes"}:
        color = [base, base, base, 0.0]
    else:
        color = [0.4, 0.4, 0.4, 0.0]
    material = bproc.material.create(f"obj_mat_{idx}")
    material.set_principled_shader_value("Base Color", color)
    mats = obj.get_materials()
    if mats:
        for mat_idx in range(len(mats)):
            obj.set_material(mat_idx, material)
    else:
        obj.set_material(0, material)

    # convert cnos camera poses to blender camera poses
    cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
    cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
    bproc.camera.add_camera_pose(cam_pose)
    
    # set light
    light_scale = 2.5
    light_energy = 1000
    light1 = bproc.types.Light()
    light1.set_type("POINT")
    light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
    light1.set_energy(light_energy)

    bproc.renderer.set_max_amount_of_samples(50)
    # render the whole pipeline
    data = bproc.renderer.render()
    # render nocs
    purge_invalid_structs()
    data.update(bproc.renderer.render_nocs())
    
    # check save folder
    save_fpath = os.path.join(args.output_dir, "templates")
    if not os.path.exists(save_fpath):
        os.makedirs(save_fpath)

    # save rgb image
    color_bgr_0 = data["colors"][0]
    color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
    cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

    # save mask
    mask_0 = data["nocs"][0][..., -1]
    cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)
    
    # save nocs
    xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
    np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))