import blenderproc as bproc

import math
import os
import bpy
import json
import random
import imageio
import argparse
import numpy as np
import mathutils    

context = bpy.context
scene = context.scene
render = scene.render

bproc.init()
bproc.renderer.set_output_format(file_format="PNG", enable_transparency=True)


render.engine = 'BLENDER_EEVEE_NEXT'  # Ensure this engine is available
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

    
def _add_lighting():
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 1.3
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100
    
    bpy.data.worlds["World"].use_nodes = True
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0.2

def _setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 3, 0)
    bpy.data.cameras["Camera"].lens_unit = "FOV"
    bpy.data.cameras["Camera"].angle = 40 * np.pi / 180

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty


def _sample_camera_loc(phi=None, theta=None, r=3.0):
    '''
    phi: inclination angle
    theta: azimuth angle
    r: radius
    '''
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z], dtype=np.float32)
        
def _load_objs(tree, data_root):
    for node in tree:
        for obj_path in node['objs']:
            objs = bproc.loader.load_obj(f'{data_root}/{obj_path}')

def _merge_meshes(objects):
    """
    合并多个Blender对象的网格数据。
    :param objects: 要合并的对象列表
    :return: 合并后的新对象
    """
    # 进入编辑模式
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    merged_object = bpy.context.active_object
    return merged_object
 
def _load_and_merge_objs(tree, data_root):
    """ 
    加载 .obj 文件，并合并：
    1. 同一 node 内的所有 3D 模型文件
    2. 具有 fixed 连接的子节点合并到父节点（base 除外）
    """

    # 记录所有节点的已加载 3D 对象
    node_objs = {}

    # 遍历节点，加载 .obj 文件
    for node in tree:
        obj_list = []  # 用于存储当前节点的 3D 模型对象
        center = None
        for obj_path in node['objs']:
            obj = bproc.loader.load_obj(f'{data_root}/{obj_path}')
            bbox = obj[0].get_bound_box()
            if center is None:
                center = (bbox[0] + bbox[1] + bbox[2] + bbox[3] + bbox[4] + bbox[5] + bbox[6] + bbox[7]) / 8
            else:
                center += ((bbox[0] + bbox[1] + bbox[2] + bbox[3] + bbox[4] + bbox[5] + bbox[6] + bbox[7]) / 8)
            for o in obj:
                obj_list.append(o.blender_obj)

        # 如果当前节点包含多个对象，则进行合并
        merged_obj = _merge_meshes(obj_list)
        merged_obj.name = str(node['id'])  # 以节点 id 作为合并后的对象名称
        center = center / len(obj_list)
        node_objs[str(node['id'])] = {
            'obj': merged_obj,
            'center': center
        }

    # 遍历节点，将 fixed 连接的子节点合并到父节点
    for node in tree:
        parent_id = node['parent']
        if parent_id != -1 and node['joint']['type'] == 'fixed':  # 非 base 且 joint 为 fixed
            child_obj = node_objs[str(node['id'])]
            parent_obj = node_objs.get(str(parent_id), None)

            iter_count = 0
            while parent_obj is None:
                iter_count += 1
                for node2 in tree:
                    if node2['id'] == parent_id:
                        parent_id = node2['parent']
                        break
                parent_obj = node_objs.get(str(parent_id), None)
                if iter_count > 10:
                    break

            if child_obj and parent_obj:
                # 合并子节点到父节点
                merged_obj = _merge_meshes([parent_obj['obj'], child_obj['obj']])
                merged_obj.name = str(parent_id)  # 以父节点 id 作为合并后的对象名称
                # 更新父节点的对象
                node_objs[parent_id] =  {
                    'obj': merged_obj,
                    'center': (parent_obj['center'] + child_obj['center']) / 2
                }  
                node_objs[str(node['id'])] = None
    # import ipdb
    # ipdb.set_trace()
    return node_objs

def _set_scene(n_imgs=2):
    _setup_camera()
    _add_lighting()
    
    # backface culling
    for material in bpy.data.materials:
        material.use_backface_culling = True
    
    phi_seg = np.linspace(np.pi/3, np.pi/2, n_imgs+1)
    theta_seg = np.linspace(-5*np.pi/6, -np.pi/6, n_imgs+1)
    
    phis = [np.random.uniform(phi_seg[i], phi_seg[i+1]) for i in range(n_imgs)] # angle relative to z-axis
    thetas = [np.random.uniform(theta_seg[i], theta_seg[i+1]) for i in range(n_imgs)] # front view: -np.pi/2
    random.shuffle(phis)
    random.shuffle(thetas)
    
    # sample camera locations
    for i in range(n_imgs):
        r = np.random.uniform(4,4.5)
        location = _sample_camera_loc(phis[i], thetas[i], r)
        rotation_matrix = bproc.camera.rotation_from_forward_vec([0, 0, 0] - location)
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
    
def _render_scene():
    data = bproc.renderer.render()
    data.update(
        bproc.renderer.render_segmap(map_by=["instance"])
    )
    
    return data


def _write_imgs(data_root, data):
    save_img_dir = f'{data_root}/imgs'
    os.makedirs(save_img_dir, exist_ok=True)

    # rendered images
    rgbs = data['colors']
    # save images
    for i, rgb in enumerate(rgbs):
        fname = str(i+19).zfill(2)
        imageio.imwrite(f'{save_img_dir}/{fname}.png', rgb)

def _write_mp4(src_dir, data):
    for view in range(data.shape[0]):
    # 保存渲染结果到指定目录
        save_img_dir = os.path.join(src_dir, 'imgs')
        os.makedirs(save_img_dir, exist_ok=True)

        i = 0
        save_path = os.path.join(save_img_dir, f'animation_{view}.mp4')
        imageio.mimwrite(save_path, data[view], fps=10, quality=8, format='mp4')

def compute_translation(a, b, c, x, y, z, alpha):
    # 归一化旋转轴方向
    axis = np.array([a, b, c], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)  

    # 构造反对称矩阵 K
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # 计算旋转矩阵 R (Rodrigues' rotation formula)
    I = np.eye(3)
    R = I + math.sin(alpha) * K + (1 - math.cos(alpha)) * (K @ K)

    # 计算原始偏移量
    T = np.array([x, y, z])

    # 计算旋转后的偏移量
    T_prime = R @ T

    # 计算需要的平移量 ΔP
    delta_P = T_prime - T
    return delta_P

def degrees_to_radians(degrees):
    """Convert an angle from degrees to radians."""
    return degrees * math.pi / 180.0

def _apply_transformations_step(node_objs, tree, step, frame=8):
    """
    根据 joint 限制，对合并后的对象进行随机变换（旋转或平移）。
    """
    ids = []
    for node in tree:
        node_id = node['id']
        obj_dict = node_objs.get(str(node_id))
        # import ipdb; ipdb.set_trace()
        try:
            obj = bpy.data.objects[str(node_id)]
        except:
            continue
        center = obj_dict['center']

        if obj is None:
            continue  # 跳过无效对象

        joint = node.get('joint', {})
        joint_type = joint.get('type', None)

        if joint_type == 'revolute':
            # Revolute Joint: 在指定范围内随机旋转
            angle_range = joint.get('range', [0, 0])
            # angle = (angle_range[1] - angle_range[0]) / frame
            if step == 0:
                angle = degrees_to_radians(angle_range[0])
            else:
                angle = degrees_to_radians((angle_range[1] - angle_range[0]) / (frame - 1))

            direction = np.array(joint['axis']['direction'])
            new_direction = np.array([direction[0], -direction[2], direction[1]])
            direction = new_direction / np.linalg.norm(new_direction)  # 归一化平移方向
            direction = direction / np.linalg.norm(direction)  # 归一化旋转轴
            origin = np.array(joint['axis']['origin'])
            new_direction = np.array([origin[0], -origin[2], origin[1]])
            origin = mathutils.Vector(new_direction)
            
            # 将局部坐标绕joint.origin进行旋转
            rotation_quat = mathutils.Quaternion(direction, angle)
            loc_diff = compute_translation(direction[0], direction[1], direction[2], origin[0], origin[1], origin[2], angle)
            obj.location -= mathutils.Vector(loc_diff)
            obj.rotation_euler = (rotation_quat @ obj.rotation_euler.to_quaternion()).to_euler()

        if joint_type == 'prismatic':
            ids.append(node_id)
            # Prismatic Joint: 在指定范围内随机平移
            move_range = joint.get('range', [0, 0])
            if step == 0:
                offset = move_range[0]
            else:
                offset = (move_range[1] - move_range[0]) / (frame - 1)
            
            direction = np.array(joint['axis']['direction'])
            new_direction = np.array([direction[0], -direction[2], direction[1]])
            direction = new_direction / np.linalg.norm(new_direction)  # 归一化平移方向

            # 平移对象
            # import ipdb
            # ipdb.set_trace()
            obj.location += mathutils.Vector(direction * offset)

    return node_objs

def _apply_random_transformations(node_objs, tree):
    """
    根据 joint 限制，对合并后的对象进行随机变换（旋转或平移）。
    """
    for node in tree:
        node_id = node['id']
        obj = node_objs.get(node_id)

        if obj is None:
            continue  # 跳过无效对象

        joint = node.get('joint', {})
        joint_type = joint.get('type', None)

        if joint_type == 'revolute':
            # Revolute Joint: 在指定范围内随机旋转
            angle_range = joint.get('range', [0, 0])
            random_angle = np.radians(random.uniform(angle_range[0], angle_range[1]))

            direction = np.array(joint['axis']['direction'])
            new_direction = np.array([0, 0, 0])
            new_direction[0] = direction[0]
            new_direction[1] = -direction[2]
            new_direction[2] = direction[1]
            direction = new_direction / np.linalg.norm(new_direction)  # 归一化平移方向
            direction = direction / np.linalg.norm(direction)  # 归一化旋转轴
            origin = mathutils.Vector(np.array(joint['axis']['origin']))

            # 旋转矩阵
            world_position = obj.location
            
            # 将物体位置从世界坐标系转换为相对于joint.origin的局部坐标系
            local_position = world_position - origin
            
            # 将局部坐标绕joint.origin进行旋转
            rotation_quat = mathutils.Quaternion(direction, random_angle)
            local_position_rotated = rotation_quat @ origin
            obj.location = local_position_rotated
            obj.rotation_euler = (rotation_quat @ obj.rotation_euler.to_quaternion()).to_euler()

        if joint_type == 'prismatic':
            # Prismatic Joint: 在指定范围内随机平移
            move_range = joint.get('range', [0, 0])
            random_offset = random.uniform((move_range[0] + move_range[1]) / 2, move_range[1])
            
            direction = np.array(joint['axis']['direction'])
            new_direction = np.array([0, 0, 0])
            new_direction[0] = direction[0]
            new_direction[1] = -direction[2]
            new_direction[2] = direction[1]
            direction = new_direction / np.linalg.norm(new_direction)  # 归一化平移方向

            # 平移对象
            obj.location += mathutils.Vector(direction * random_offset)

    return node_objs

def render_imgs(model_path, n_imgs=20, save_path='.', json_name='object.json', incremental=False):
    # load json file
    with open(os.path.join(model_path, json_name), 'r') as f:
        src = json.load(f)
    # load textured objs into blender (w/ semantic+instance ids)
    node_objs = _load_and_merge_objs(src['diffuse_tree'], f'{model_path}')
    _set_scene(n_imgs)
    raw_data_list = np.random.rand(n_imgs, 2, 512, 512, 4)
    for step in range(2):
        render.engine = 'BLENDER_EEVEE_NEXT'  # Ensure this engine is available
        render.resolution_x = 512
        render.resolution_y = 512
        render.resolution_percentage = 100
        _apply_transformations_step(node_objs, src['diffuse_tree'], step, frame=2)
    # render images
        raw_data = _render_scene()
        for view in range(n_imgs):
            raw_data_list[view, step] = raw_data['colors'][view]
    
    # _write_mp4('.', raw_data_list)
    _write_mp4(save_path, raw_data_list)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data subdirectory')
    parser.add_argument('--n_imgs', type=int, default=20, help='number of images to render for each model')
    parser.add_argument('--incremental', action='store_true', help='whether to render images incrementally')
    parser.add_argument('--save_dir', type=str, default='.', help='directory to save rendered images')
    parser.add_argument('--json_name', type=str, default='object.json', help='name of the json file to load')
    args = parser.parse_args()
     
    # try:
    render_imgs(args.data, args.n_imgs, args.data, args.json_name, args.incremental)
    # except Exception as e:
    #     with open('render_err.log', 'a') as f:
    #         f.write(f'{args.data}\n')
    #         f.write(f'{e}\n')
