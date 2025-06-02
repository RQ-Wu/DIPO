import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import torch
import argparse
import numpy as np
from PIL import Image, ImageOps
import imageio
# import ipdb
# ipdb.set_trace()
from my_utils.plot import viz_graph
from my_utils.misc import load_config
import torchvision.transforms as T
from diffusers import DDPMScheduler
from models.denoiser import Denoiser
from my_utils.render import rescale_axis
from my_utils.refs import joint_ref, sem_ref
from my_utils.render import prepare_meshes, draw_boxes_axiss_anim
from dataset.utils import make_white_background, load_input_from, convert_data_range, parse_tree
import models
import torch.nn.functional as F
from io import BytesIO
import base64
from scripts.graph_pred.api import predict_graph_twomode
import subprocess

def run_retrieve(src_dir, json_name, data_root):
    fn_call = ['python', 'scripts/mesh_retrieval/retrieve.py', '--src_dir', src_dir, '--json_name', json_name, '--gt_data_root', data_root]
    try:
        subprocess.run(fn_call, check=True,  stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f'Error from run_retrieve: {src_dir}')
        print(f'Error: {e}')

def make_white_background(src_img):
    '''Make the white background for the input RGBA image.'''
    src_img.load() 
    background = Image.new("RGB", src_img.size, (255, 255, 255))
    background.paste(src_img, mask=src_img.split()[3]) # 3 is the alpha channel
    return background

def pad_to_square(img, fill=0):
    """Pad image to square with given fill value (default: 0 = black)."""
    width, height = img.size
    if width == height:
        return img
    max_side = max(width, height)
    delta_w = max_side - width
    delta_h = max_side - height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=fill)

def load_img(img_path):
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    with Image.open(img_path) as img:
        if img.mode == 'RGBA':
            img = make_white_background(img)
        img = img.convert('RGB')  # Ensure it's 3-channel for normalization
        img = pad_to_square(img, fill=0)
        img = transform(img)
    img_batch = img.unsqueeze(0).cuda()

    return img_batch


def load_frame_with_imageio(frame):
    """
    将单帧图像处理为符合 DINO 模型输入的格式。
    """
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img = Image.fromarray(frame)  # 转为 PIL 图像
    if img.mode == 'RGBA':
        img = make_white_background(img)
    img = transform(img)  # 应用预处理
    return img.unsqueeze(0).cuda()  # 增加 batch 维度

def read_video_as_batch_with_imageio(video_path):
    """
    使用 imageio 读取视频并将所有帧处理为 batch 格式 (B, C, H, W)。
    """
    reader = imageio.get_reader(video_path)
    batch_frames = []

    try:
        for frame in reader:
            # 加载帧并处理为 (1, C, H, W)
            processed_frame = load_frame_with_imageio(frame)
            batch_frames.append(processed_frame)

        reader.close()
        if batch_frames:
            return torch.cat(batch_frames, dim=0).cuda()  # 在 batch 维度堆叠，并转移到 GPU
        else:
            print("视频没有有效帧")
            return None
    except Exception as e:
        print(f"处理视频时出错: {e}")
        return None

def extract_dino_feature(img_path_1, img_path_2):
    print('Extracting DINO feature...')
    feat_1 = load_img(img_path_1)
    feat_2 = load_img(img_path_2)
    frames = torch.cat([feat_1, feat_2], dim=0)
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda()
    with torch.no_grad():
        feat = dinov2_vitb14_reg.forward_features(frames)["x_norm_patchtokens"]
    # release the GPU memory of the model
    feat_input = torch.cat([feat[0], feat[-1]], dim=0).unsqueeze(0)
    torch.cuda.empty_cache()
    return feat_input

def set_scheduler(n_steps=100):
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear', prediction_type='epsilon')
    scheduler.set_timesteps(n_steps)
    return scheduler

def prepare_model_input(data, cond, feat, n_samples):
    # attention masks
    attr_mask = torch.from_numpy(cond['attr_mask']).unsqueeze(0).repeat(n_samples, 1, 1)
    key_pad_mask = torch.from_numpy(cond['key_pad_mask'])
    graph_mask = torch.from_numpy(cond['adj_mask'])
    # input image feature
    f = feat.repeat(n_samples, 1, 1)
    # input noise
    B, C = data.shape
    noise = torch.randn([n_samples, B, C], dtype=torch.float32)
    # dummy image feature (used for guided diffusion)
    dummy_feat = torch.from_numpy(np.load('systems/dino_dummy.npy').astype(np.float32))
    dummy_feat = dummy_feat.unsqueeze(0).repeat(n_samples, 1, 1)
    # dummy object category
    cat = torch.zeros(1, dtype=torch.long).repeat(n_samples)
    return {
        "noise": noise.cuda(),
        "attr_mask": attr_mask.cuda(),
        "key_pad_mask": key_pad_mask.cuda(),
        "graph_mask": graph_mask.cuda(),
        "dummy_f": dummy_feat.cuda(),
        'cat': cat.cuda(),
        'f': f.cuda(),  
    }

def prepare_model_input_nocond(feat, n_samples):
    # attention masks
    cond_example = np.zeros((32*5, 32*5), dtype=bool)
    attr_mask = np.eye(32, 32, dtype=bool)
    attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
    attr_mask = torch.from_numpy(attr_mask).unsqueeze(0).repeat(n_samples, 1, 1)
    key_pad_mask = torch.from_numpy(cond_example).unsqueeze(0).repeat(n_samples, 1, 1)
    graph_mask = torch.from_numpy(cond_example).unsqueeze(0).repeat(n_samples, 1, 1)
    # input image feature
    f = feat.repeat(n_samples, 1, 1)
    # input noise
    data = np.zeros((32*5, 6), dtype=bool)
    noise = torch.randn(data.shape, dtype=torch.float32).repeat(n_samples, 1, 1)
    # dummy image feature (used for guided diffusion)
    dummy_feat = torch.from_numpy(np.load('systems/dino_dummy.npy').astype(np.float32))
    dummy_feat = dummy_feat.unsqueeze(0).repeat(n_samples, 1, 1)
    # dummy object category
    cat = torch.zeros(1, dtype=torch.long).repeat(n_samples)
    return {
        "noise": noise.cuda(),
        "attr_mask": attr_mask.cuda(),
        "key_pad_mask": key_pad_mask.cuda(),
        "graph_mask": graph_mask.cuda(),
        "dummy_f": dummy_feat.cuda(),
        'cat': cat.cuda(),
        'f': f.cuda(),  
    }

def save_graph(pred_graph, save_dir):
    print(f'Saving the predicted graph to {save_dir}/pred_graph.json')
    # save the response
    with open(os.path.join(save_dir, "pred_graph.json"), "w") as f:
        json.dump(pred_graph, f, indent=4)
    # Visualize the graph
    # img_graph = Image.fromarray(viz_graph(pred_graph))
    # img_graph.save(os.path.join(save_dir, "pred_graph.png"))

def forward(model, scheduler, inputs, omega=0.5):
    print('Running inference...')
    noisy_x = inputs['noise']
    for t in scheduler.timesteps:
        timesteps = torch.tensor([t], device=inputs['noise'].device)
        outputs_cond = model(
            x=noisy_x,
            cat=inputs['cat'],
            timesteps=timesteps,
            feat=inputs['f'], 
            key_pad_mask=inputs['key_pad_mask'],
            graph_mask=inputs['graph_mask'],
            attr_mask=inputs['attr_mask'],
            label_free=True,
        ) # take condtional image as input
        if omega != 0:
            outputs_free = model(
                x=noisy_x,
                cat=inputs['cat'],
                timesteps=timesteps,
                feat=inputs['dummy_f'], 
                key_pad_mask=inputs['key_pad_mask'],
                graph_mask=inputs['graph_mask'],
                attr_mask=inputs['attr_mask'],
                label_free=True,
            ) # take the dummy DINO features for the condition-free mode
            noise_pred = (1 + omega) * outputs_cond['noise_pred'] - omega * outputs_free['noise_pred']
        else:
            noise_pred = outputs_cond['noise_pred']
        noisy_x = scheduler.step(noise_pred, t, noisy_x).prev_sample
    return noisy_x

def _convert_json(x, c):
    out = {"meta": {}, "diffuse_tree": []}
    n_nodes = c["n_nodes"]
    par = c["parents"].tolist()
    adj = c["adj"]
    np.fill_diagonal(adj, 0) # remove self-loop for the root node
    if "obj_cat" in c:
        out["meta"]["obj_cat"] = c["obj_cat"]

    # convert the data to original range
    data = convert_data_range(x)
    # parse the tree
    out["diffuse_tree"] = parse_tree(data, n_nodes, par, adj)
    return out

def post_process(output, cond, save_root, gt_data_root, visualize=False):
    print('Post-processing...')
    N = output.shape[0]
    for i in range(N):
        cond_n = {}
        cond_n['n_nodes'] = cond['n_nodes'][i] 
        cond_n['parents'] = cond['parents'][i]
        cond_n['adj'] = cond['adj'][i]
        cond_n['obj_cat'] = cond['cat']
        # convert the raw model output to the json format
        out_json = _convert_json(output, cond_n)
        save_dir = os.path.join(save_root, str(i))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "object.json"), "w") as f:
            json.dump(out_json, f, indent=4)
        

        # retrieve part meshes (call python script)
        # print(f"Retrieving part meshes for the object {i}...")
        # os.system(f"python scripts/mesh_retrieval/retrieve.py --src_dir {save_dir} --json_name object.json --gt_data_root {gt_data_root}")
        
        if visualize:
            print(f"Visualizing the object {i}...")

            # visualize the object in two states with parts represented in bbox
            vis_meshes = prepare_meshes(out_json)
            vis_img = Image.fromarray(draw_boxes_axiss_anim(
                vis_meshes["bbox_0"], 
                vis_meshes["bbox_1"], 
                vis_meshes["axiss"], 
                mode="graph", 
                resolution=256
            ))

            # save the image
            vis_img.save(os.path.join(save_dir, "vis_img.png"))


    

def load_model(ckpt_path, config):
    print('Loading model from checkpoint...')
    model = models.make(config.name, config)
    state_dict = torch.load(ckpt_path)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model.cuda()

def convert_pred_graph(pred_graph):
    cond = {}
    B, K = pred_graph.shape[:2]
    adj = np.zeros((B, K, K), dtype=np.float32)
    padding = np.zeros((B, 5 * K, 5* K), dtype=bool)
    parents = np.zeros((B, K), dtype=np.int32)
    n_nodes = np.zeros((B,), dtype=np.int32)
    for b in range(B):
        node_len = 0
        for k in range(K):
            if pred_graph[b, k] == k and k > 0:
                node_len = k
                break
            node = pred_graph[b, k]
            adj[b, k, node] = 1
            adj[b, node, k] = 1
            parents[b, k] = node
        adj[b, node_len:] = 1
        padding[b, :, :5 * node_len] = 1
        parents[b, 0] = -1
        n_nodes[b] = node_len
    adj_mask = adj.astype(bool).repeat(5, axis=1).repeat(5, axis=2)
    attr_mask = np.eye(32, 32, dtype=bool)
    attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)

    cond['adj_mask'] = adj_mask
    cond['attr_mask'] = attr_mask
    cond['key_pad_mask'] = padding

    cond['adj'] = adj
    cond['parents'] = parents
    cond['n_nodes'] = n_nodes
    cond['cat'] = 'StorageFurniture'

    data = np.zeros((32*5, 6), dtype=bool)

    return data, cond

def bfs_tree_simple(tree_list):
    order = [0] * len(tree_list)
    queue = []
    current_node_idx = 0
    for node_idx, node in enumerate(tree_list):
        if node['parent'] == -1:
            queue.append(node['id'])
            order[current_node_idx] = node_idx
            current_node_idx += 1
            break
    while len(queue) > 0:
        current_node = queue.pop(0)
        for node_idx, node in enumerate(tree_list):
            if node['parent'] == current_node:
                queue.append(node['id'])
                order[current_node_idx] = node_idx
                current_node_idx += 1

    return order

def get_graph_from_gpt(img_path_1, img_path_2):
    first_img = Image.open(img_path_1)
    first_img_data = first_img.resize((1024, 1024))
    buffer = BytesIO()
    first_img_data.save(buffer, format="PNG")
    buffer.seek(0)
    # encode the image as base64
    first_encoded_image = base64.b64encode(buffer.read()).decode("utf-8")


    second_img = Image.open(img_path_2)
    second_img_data = second_img.resize((1024, 1024))
    buffer = BytesIO()
    second_img_data.save(buffer, format="PNG")
    buffer.seek(0)
    # encode the image as base64
    second_encoded_image = base64.b64encode(buffer.read()).decode("utf-8")

    pred_gpt = predict_graph_twomode('', first_img_data=first_encoded_image, second_img_data=second_encoded_image)
    print(pred_gpt)
    pred_graph = pred_gpt['diffuse_tree']
    # order = bfs_tree_simple(pred_graph)
    # pred_graph = [pred_graph[i] for i in order]
    
    
    # generate array [0, 1, 2, ..., 31] for init
    graph_array = np.array([i for i in range(32)])
    for node_idx, node in enumerate(pred_graph):
        if node['parent'] == -1:
            graph_array[node_idx] = node_idx
        else:
            graph_array[node_idx] = node['parent']

    # new axis for batch
    graph_array = np.expand_dims(graph_array, axis=0)
    
    return torch.from_numpy(graph_array).cuda().repeat(3, 1)

def run_demo(args):
    # extract DINOV2 feature from the input image
    feat = extract_dino_feature(args.img_path_1, args.img_path_2)
    scheduler = set_scheduler(args.n_denoise_steps)
    # load the checkpoint of the model
    model = load_model(args.ckpt_path, args.config.system.model)

    # inference
    with torch.no_grad():
        pred_graph = get_graph_from_gpt(args.img_path_1, args.img_path_2)
        print(pred_graph)
        data, cond = convert_pred_graph(pred_graph)
        inputs = prepare_model_input(data, cond, feat, n_samples=args.n_samples)
        output = forward(model, scheduler, inputs, omega=args.omega).cpu().numpy()

        # post-process
        post_process(output, cond, args.save_dir, args.gt_data_root, visualize=True)

    # retrieve
    # for sample in os.listdir(args.save_dir):
    #     sample_dir = os.path.join(args.save_dir, sample)
    #     run_retrieve(sample_dir, 'object.json', '../singapo')

if __name__ == '__main__':
    '''
    Script for running the inference on an example image input.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path_1", type=str, default='examples/1.png', help="path to the input image")
    parser.add_argument("--img_path_2", type=str, default='examples/1_open_1.png', help="path to the input image")
    parser.add_argument("--ckpt_path", type=str, default='exps/singapo/final/ckpts/last.ckpt', help="path to the checkpoint of the model")
    parser.add_argument("--config_path", type=str, default='exps/singapo/final/config/parsed.yaml', help="path to the config file")
    parser.add_argument("--use_example_graph", action="store_true", default=False, help="if you don't have the openai key yet, turn on to use the example graph for inference")
    parser.add_argument("--save_dir", type=str, default='results', help="path to save the output")
    parser.add_argument("--gt_data_root", type=str, default='./', help="the root directory of the original data, used for part mesh retrieval")
    parser.add_argument("--n_samples", type=int, default=3, help="number of samples to generate given the input")
    parser.add_argument("--omega", type=float, default=0.5, help="the weight of the condition-free mode in the inference")
    parser.add_argument("--n_denoise_steps", type=int, default=100, help="number of denoising steps")
    args = parser.parse_args()

    assert os.path.exists(args.img_path_1), "The input image does not exist"
    # assert os.path.exists(args.ckpt_path), "The checkpoint does not exist"
    assert os.path.exists(args.config_path), "The config file does not exist"
    os.makedirs(args.save_dir, exist_ok=True)

    config = load_config(args.config_path)
    args.config = config

    run_demo(args)