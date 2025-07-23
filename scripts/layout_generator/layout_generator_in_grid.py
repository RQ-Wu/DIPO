import json
import numpy as np
import random
# from my_utils.render import prepare_meshes, draw_boxes_axiss_anim
from PIL import Image
import os
import argparse

def sample_base(grid_x, grid_y, base_size, is_wide=True):
    """
    Generates a base with random size. return base information and grid list
    
    Args:
        grid_x (int): Number of columns in the grid.
        grid_y (int): Number of row in the grid.
        is_wide (bool): If True, the layout is wide; otherwise, it's narrow.
    
    Returns:
        Dict: A dictionaries representing of the base.
        List: A coordinate list of the grid.
    """
    # Generate a base with random size (float)
    base_width = base_size[0]
    base_height = base_size[1]
    base_depth = base_size[2]
    base = {
        "id": 0,
        "parent": -1,
        "name": "base",
        "children": [],
        "joint":{
            "type": "fixed",
            "range": [0, 0],
            "axis": {
                "origin": [0, 0, 0],
                "direction": [0, 0, 0]
            }
        },
        "aabb":{
            "center": [0, 0, 0],
            "size": [base_width, base_height, base_depth]
        }
    }

    # Generate a grid list
    grid_x_list = np.linspace(-base_width * 0.98 / 2, base_width * 0.98 / 2, grid_x + 1)
    grid_y_list = np.linspace(base_height * 0.98 / 2, -base_height * 0.98 / 2, grid_y + 1)
    
    # grid_coords is a array (grid_x, grid_y)
    grid_coords = np.array(np.meshgrid(grid_x_list, grid_y_list)).T

    return base, grid_coords

def generate_part_in_grid(base, grid_coords, x1, x2, y1, y2, name, joint, handles, list_len, angle, drawer_depth_ratio, prismatic_ratio):
    x1_coord, y1_coord = grid_coords[x1, y1]
    x2_coord, y2_coord = grid_coords[x2, y2]
    part_width = np.abs(x2_coord - x1_coord)
    part_height = np.abs(y2_coord - y1_coord)
    center_x = (x2_coord + x1_coord) / 2
    center_y = (y2_coord + y1_coord) / 2

    if name == "drawer":
        part_depth = drawer_depth_ratio * base["aabb"]["size"][2]
    elif name == "door":
        part_depth = np.random.uniform(0.01, 0.1)
    else:
        return None

    center_z = base["aabb"]["size"][2] / 2 - part_depth / 2
    
    # generate joint
    if joint["type"] == "prismatic":
        joint_axis = [0, 0, 1]
        joint_origin = [center_x, center_y, center_z]
        joint_range = [0, prismatic_ratio * part_depth]
    elif joint["type"] == "revolute":
        joint_origin = [0, 0, 0]
        joint_range = [0, angle]
        if joint["hinge"] == "left":
            joint_origin[0] = center_x - part_width / 2
            joint_origin[1] = center_y - part_height / 2
            joint_origin[2] = center_z - part_depth / 2
            joint_range[1] = -joint_range[1]
            joint_axis = [0, 1, 0]
        elif joint["hinge"] == "right":
            joint_origin[0] = center_x + part_width / 2
            joint_origin[1] = center_y - part_height / 2
            joint_origin[2] = center_z - part_depth / 2
            joint_axis = [0, 1, 0]
        elif joint["hinge"] == "top":
            joint_origin[0] = center_x - part_width / 2
            joint_origin[1] = center_y + part_height / 2
            joint_origin[2] = center_z - part_depth / 2
            joint_range[1] = -joint_range[1]
            joint_axis = [1, 0, 0]
        elif joint["hinge"] == "bottom":
            joint_origin[0] = center_x - part_width / 2
            joint_origin[1] = center_y - part_height / 2
            joint_origin[2] = center_z - part_depth / 2
            joint_axis = [1, 0, 0]
        else:
            return None
    else:
        return None

    # generate part
    part_id = list_len
    list_len += 1
    part = {
        "id": part_id,
        "parent": base['id'],
        "name": name,
        "children": [],
        "joint": {
            "type": joint["type"],
            "range": joint_range,
            "axis": {
                "origin": joint_origin,
                "direction": joint_axis
            }
        },
        "aabb": {
            "center": [center_x, center_y, center_z],
            "size": [part_width, part_height, part_depth]
        }
    }
    base['children'].append(part_id)

    # add handles / knobs
    handle_json_list = []
    if handles:
        for handle in handles:
            handle_x = handle['x'] * (x2_coord - x1_coord) + x1_coord
            handle_y = handle['y'] * (y2_coord - y1_coord) + y1_coord
            handle_z = handle['size'][2] / 2 + center_z + part_depth / 2
            handle_id = list_len
            list_len += 1
            handle_json = {
                "id": handle_id,
                "parent": part_id,
                "name": handle["name"],
                "children": [],
                "joint": {
                    "type": "fixed",
                    "range": [0, 0],
                    "axis": {
                        "origin": [0, 0, 0],
                        "direction": [0, 0, 0]
                    }
                },
                "aabb": {
                    "center": [handle_x, handle_y, handle_z],
                    "size": handle["size"]
                }
            }
            part['children'].append(handle_id)
            handle_json_list.append(handle_json)
    
    return base, part, handle_json_list, list_len

def generate_layout(info):
    base, grid_coords = sample_base(info['grid_x'], info['grid_y'], info['base_size'], info['is_wide'])
    articulate_tree = [base]
    list_len = 1
    angle = np.random.choice([90.0, 90.0, 90.0, 135.0, 180.0])
    drawer_depth_ratio = 0.9
    prismatic_ratio = 5 / 9
    for part in info['part']:
        x1 = part['x1']
        x2 = part['x2']
        y1 = part['y1']
        y2 = part['y2']
        name = part['name']
        joint = part['joint']
        handles = part.get('handles', [])
        
        base, part, handle_json_list, list_len = generate_part_in_grid(base, grid_coords, x1, x2, y1, y2, name, joint, handles, list_len, angle, drawer_depth_ratio, prismatic_ratio)
        
        if part:
            articulate_tree.append(part)
            articulate_tree.extend(handle_json_list)

    articulate_tree[0] = base
    # save as json
    # with open('articulate_tree.json', 'w') as f:
    #     json.dump({'diffuse_tree': articulate_tree}, f, indent=4)
    # json_data = json.load(open('articulate_tree.json', 'r'))
    # vis_meshes = prepare_meshes(json_data)
    # vis_img = Image.fromarray(draw_boxes_axiss_anim(
    #             vis_meshes["bbox_0"], 
    #             vis_meshes["bbox_1"], 
    #             vis_meshes["axiss"], 
    #             mode="graph", 
    #             resolution=256
    #         ))
    # vis_img.save(os.path.join("vis_img.png"))

    return articulate_tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the part connectivity graph from an image")
    parser.add_argument("--save_path", type=str, required=True, help="path to the save the response")
    args = parser.parse_args() 

    root_path = args.save_path
    for class_name in os.listdir(root_path):
        for model_id in os.listdir(os.path.join(root_path, class_name)):
            json_name = os.path.join(root_path, class_name, model_id, 'gpt_pred.json')
            print(json_name)
            with open(json_name, 'r') as f:
                info = json.load(f)['layout']
                try:
                    articulate_tree = generate_layout(info)
                except Exception as e:
                    continue
                json_info = {
                    "meta": {
                        "obj_cat": class_name
                    },
                    "diffuse_tree": articulate_tree
                }
            with open(json_name.replace('gpt_pred', 'object'), 'w') as f:
                json.dump(json_info, f, indent=4)

