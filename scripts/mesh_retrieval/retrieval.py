import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import argparse
import numpy as np
from retrieval.obj_retrieval import find_obj_candidates, pick_and_rescale_parts
import trimesh
import shutil

def _retrieve_part_meshes(info_dict, save_dir, gt_data_root):
    mesh_save_dir = os.path.join(save_dir, "plys")
    obj_save_dir = os.path.join(save_dir, "objs")
    os.makedirs(mesh_save_dir, exist_ok=True)
    os.makedirs(obj_save_dir, exist_ok=True)
    print(save_dir)
    if os.path.exists(os.path.join(save_dir, "object.ply")):
        return 
    HASHBOOK_PATH = "retrieval/retrieval_hash_no_handles.json"

    obj_candidates = find_obj_candidates(
        info_dict,
        gt_data_root,
        HASHBOOK_PATH,
        gt_file_name="object.json",
        num_states=5,
        metric_num_samples=4096,
        keep_top=3,
    )
    
    retrieved_mesh_specs = pick_and_rescale_parts(
        info_dict, obj_candidates, gt_data_root, gt_file_name="object.json"
    )

    scene = trimesh.Scene()
    for i, mesh_spec in enumerate(retrieved_mesh_specs):
        part_spec = info_dict["diffuse_tree"][i]
        part_meshes = []
        for file in mesh_spec["files"]:
            mesh = trimesh.load(os.path.join(mesh_spec["dir"], file), force="mesh")
            part_meshes.append(mesh)
        part_mesh = trimesh.util.concatenate(part_meshes)
        part_mesh.vertices -= part_mesh.bounding_box.centroid
        transformation = trimesh.transformations.compose_matrix(
            scale=mesh_spec["scale_factor"],
            angles=[0, 0, np.radians(90) if mesh_spec["z_rotate_90"] else 0],
            translate=part_spec["aabb"]["center"],
        )
        part_mesh.apply_transform(transformation)

        # Save as PLY
        ply_path = os.path.join(mesh_save_dir, f"part_{i}.ply")
        part_mesh.export(ply_path)

        # Save as OBJ
        obj_path = os.path.join(obj_save_dir, f"part_{i}.obj")
        part_mesh.export(obj_path)

        # Update info_dict
        info_dict["diffuse_tree"][i]["plys"] = [f"plys/part_{i}.ply"]
        info_dict["diffuse_tree"][i]["objs"] = [f"objs/part_{i}.obj"]

        scene.add_geometry(part_mesh)

    scene.export(os.path.join(save_dir, "object.ply"))
    del mesh, scene
    return info_dict

def main(args):
    with open(os.path.join(args.src_dir, args.json_name), "r") as f:
        info_dict = json.load(f)

    if 'meta' not in info_dict.keys():
        info_dict['meta'] = {
            'obj_cat': 'StroageFurniture'
        }

    updated_json = _retrieve_part_meshes(info_dict, args.src_dir, args.gt_data_root)

    if updated_json is not None:
        with open(os.path.join(args.src_dir, args.json_name), "w") as f:
            json.dump(updated_json, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='path to the directory containing object.json')
    parser.add_argument('--json_name', type=str, default='object.json', help='name of the json file')
    parser.add_argument('--gt_data_root', type=str, default='./', help='path to the ground truth data')
    args = parser.parse_args()
    main(args)
