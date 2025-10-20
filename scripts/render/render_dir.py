
import os
import json
import argparse
import subprocess
from tqdm.contrib.concurrent import process_map


def _prepare_data_list(fpath):
    data_list = []
    with open(fpath, 'r') as f:
        data = json.load(f)
    # load the data list under each key
    for k in data.keys():
        data_list += data[k]


def render_data(data):
    # run the blenderproc script to render the input images (default: 20 images per object)
    fn_call = ['blenderproc', 'run', 'scripts/preprocess/render_script.py', '--n_imgs', '20', '--data']

    command = fn_call + [data]
    try:
        print(f'processing {data}')
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f'Error from render_script: {data}')
        print(f'Error: {e}')

if __name__ == '__main__':
    arser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='path to the directory containing object.json')
    args = parser.parse_args()
    
    cats = os.listdir(args.src_dir)
    for class_name in cats:
        class_path = os.path.join(root_path, class_name)
        for model_name in os.listdir(class_path):
            model_path = os.path.join(class_path, model_name)
            if os.path.exists(os.path.join(model_path, 'object.json')) and not os.path.exists(os.path.join(model_path, 'imgs')):
                render_data(model_path)




