import os
import subprocess
import argparse
from tqdm.contrib.concurrent import process_map
from functools import partial

def run_retrieve(src_dir, json_name, data_root):
    fn_call = ['python', 'scripts/mesh_retrieval/retrieve.py', '--src_dir', src_dir, '--json_name', json_name, '--gt_data_root', data_root]
    try:
        subprocess.run(fn_call, check=True,  stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f'Error from run_retrieve: {src_dir}')
        print(f'Error: {e}')
    return ' '.join(fn_call)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='path to the directory containing object.json')
    parser.add_argument('--gt_data_root', type=str, default='./', help='path to the ground truth data')
    
    args = parser.parse_args()

    root_path = args.src_dir
    for class_name in os.listdir(root_path):
        for model_id in os.listdir(os.path.join(root_path, class_name)):
            json_path = os.path.join(root_path, class_name, model_id, 'object.json')
            object_path = os.path.join(root_path, class_name, model_id, 'object.ply')
            if os.path.exists(json_path):
                if not os.path.exists(object_path):
                    print(json_path)
                    src_dir = os.path.join(root_path, class_name, model_id)
                    run_retrieve(src_dir, 'object.json', args.gt_data_root)