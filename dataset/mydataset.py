import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import json
import numpy as np
from PIL import Image
import torchvision.transforms as T
from dataset.base_dataset import BaseDataset
import random
from tqdm import tqdm
import imageio
import torch

def make_white_background(src_img):
    '''Make the white background for the input RGBA image.'''
    src_img.load() 
    background = Image.new("RGB", src_img.size, (255, 255, 255))
    background.paste(src_img, mask=src_img.split()[3]) # 3 is the alpha channel
    return background

class MyDataset(BaseDataset):

    """
    Dataset for training and testing on the PartNet-Mobility and ACD datasets (with our preprocessing).
    The GT graph is given.
    """

    def __init__(self, hparams, model_ids, mode="train", json_name="object.json"):
        self.hparams = hparams
        self.json_name = json_name
        self.model_ids = self._filter_models(model_ids)
        self.mode = mode
        self.map_cat = False
        self.get_acd_mapping()

        self.no_GT = (
            True if self.hparams.get("test_no_GT", False) and self.hparams.get("test_pred_G", False)
            else False
        )
        self.pred_G = (
            False
            if mode in ["train", "val"]
            else self.hparams.get("test_pred_G", False)
        )

        if mode == 'test':
            if "acd" in hparams.test_which:
                self.map_cat = True
        
        self.files = self._cache_data()
        print(f"[INFO] {mode} dataset: {len(self)} data samples loaded.")

    def _cache_data_train(self):
        json_data_root = self.hparams.json_root
        data_root = self.hparams.root
        # number of views per model and in total
        n_views_per_model = self.hparams.n_views_per_model
        n_views = n_views_per_model * len(self.model_ids)
        # json files for each model
        json_files = []
        # mapping to the index of the corresponding model in json_files
        model_mappings = []
        # space for dinov2 patch features
        feats = np.empty((n_views, 512, 768), dtype=np.float16)
        # space for object masks on image patches
        obj_masks = np.empty((n_views, 256), dtype=bool)
        # input images (not required in training)
        imgs = None
        # load data for non-aug views
        i = 0  # index for views
        for j, model_id in enumerate(self.model_ids):
            print(model_id)
            # if j % 10 == 0 and torch.distributed.get_rank() == 0:
            #     print(f"\rLoading training data: {j}/{len(self.model_ids)}")
            # 3D data
            with open(os.path.join(json_data_root, model_id, self.json_name), "r") as f:
                json_file = json.load(f)
            json_files.append(json_file)
            filenames = os.listdir(os.path.join(data_root, model_id, 'features'))
            filenames = [f for f in filenames if 'high_res' not in f]
            filenames = filenames[:self.hparams.n_views_per_model]
            for filename in filenames:
                view_feat = np.load(os.path.join(data_root, model_id, 'features', filename))
                first_frame_feat = view_feat[0]
                if self.hparams.frame_mode == 'last_frame':
                    second_frame_feat = view_feat[-2]
                elif self.hparams.frame_mode == 'random_state_frame':
                    second_frame_feat = view_feat[-1]
                else:
                    raise NotImplementedError("Please provide correct frame mode: last_frame | random_state_frame")
                feats[i : i + 1, :256, :] = first_frame_feat.astype(np.float16)
                feats[i : i + 1, 256:, :] = second_frame_feat.astype(np.float16)
                i = i + 1
            model_mappings += [j] * n_views_per_model
            # object masks for all views
            # all_obj_masks = np.load(
            #     os.path.join(json_data_root, model_id, "features/patch_obj_masks.npy")
            # )  # (20, Np)
            # obj_masks[i : i + n_views_per_model] = all_obj_masks[:n_views_per_model]
        return {
            "len": n_views,
            "gt_files": json_files,
            "features": feats,
            "obj_masks": None,
            "model_mappings": model_mappings,
            "imgs": imgs,
        }

    def _cache_data_non_train(self):
        # number of views per model and in total
        n_views_per_model = 2
        n_views = n_views_per_model * len(self.model_ids)
        # json files for each model
        gt_files = []
        pred_files = []  # for predicted graphs
        # mapping to the index of the corresponding model in json_files
        model_mappings = []
        # space for dinov2 patch features
        feats = np.empty((n_views, 512, 768), dtype=np.float16)
        # space for input images
        first_imgs = np.empty((n_views, 128, 128, 3), dtype=np.uint8)
        second_imgs = np.empty((n_views, 128, 128, 3), dtype=np.uint8)
        # transformation for input images
        transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.Resize(128, interpolation=T.InterpolationMode.BICUBIC),
            ]
        )

        i = 0  # index for views
        desc = f'Loading {self.mode} data'
        for j, model_id in tqdm(enumerate(self.model_ids), total=len(self.model_ids), desc=desc):
            with open(os.path.join(self.hparams.json_root, model_id, self.json_name), "r") as f:
                json_file = json.load(f)
            gt_files.append(json_file)
            # filename_dir = os.path.join(self.hparams.root, model_id, 'features')
            for filename in ['18.npy', '19.npy']:
                view_feat = np.load(os.path.join(self.hparams.root, model_id, 'features', filename))
                first_frame_feat = view_feat[0]
                if self.hparams.frame_mode == 'last_frame':
                    second_frame_feat = view_feat[-2]
                elif self.hparams.frame_mode == 'random_state_frame':
                    second_frame_feat = view_feat[-1]
                else:
                    raise NotImplementedError("Please provide correct frame mode: last_frame | random_state_frame")
                feats[i : i + 1, :256, :] = first_frame_feat.astype(np.float16)
                feats[i : i + 1, 256:, :] = second_frame_feat.astype(np.float16)

                video_path = os.path.join(self.hparams.root, model_id, 'imgs', 'animation_' + filename.replace('.npy', '.mp4'))
                reader = imageio.get_reader(video_path)
                frames = []
                for frame in reader:
                    frames.append(frame)
                reader.close()

                first_img = Image.fromarray(frames[0])
                if first_img.mode == 'RGBA':
                    first_img = make_white_background(first_img)


                first_img = np.asarray(transform(first_img), dtype=np.int8)
                first_imgs[i] = first_img

                if self.hparams.frame_mode == 'last_frame':
                    second_img = Image.fromarray(frames[-1])
                elif self.hparams.frame_mode == 'random_state_frame':
                    second_img_path = video_path.replace('animation', 'random').replace('.mp4', '.png')
                    second_img = Image.open(second_img_path)
                if second_img.mode == 'RGBA':
                    second_img = make_white_background(second_img)
                second_img = np.asarray(transform(second_img), dtype=np.int8)
                second_imgs[i] = second_img

                i = i + 1
            # mapping to json file
            model_mappings += [j] * n_views_per_model

        return {
            "len": n_views,
            "gt_files": gt_files,
            "pred_files": pred_files,
            "features": feats,
            "model_mappings": model_mappings,
            "imgs": [first_imgs, second_imgs],
        }

    def _cache_data(self):
        """
        Function to cache data from disk.
        """
        if self.mode == "train":
            return self._cache_data_train()
        else:
            return self._cache_data_non_train()

    def _get_item_train_val(self, index):
        model_i = self.files["model_mappings"][index]
        gt_file = self.files["gt_files"][model_i]
        data, cond = self._prepare_input_GT(
            file=gt_file, model_id=self.model_ids[model_i]
        )
        if self.mode == "val":
            # input image for visualization
            img_first = self.files["imgs"][0][index]
            img_last = self.files["imgs"][1][index]
            cond["img"] = np.concatenate([img_first, img_last], axis=1)
        # else:
        #     # object masks on patches
        #     # obj_mask = self.files["obj_masks"][index][None, ...].repeat(self.hparams.K * 5, axis=0)
        #     cond["img_obj_mask"] = [None]
        return data, cond

    def _get_item_test(self, index):
        model_i = self.files["model_mappings"][index]

        gt_file = None if self.no_GT else self.files["gt_files"][model_i] 

        if self.hparams.get('G_dir', None) is None:
            data, cond = self._prepare_input_GT(file=gt_file, model_id=self.model_ids[model_i])
        else:
            if index % 2 == 0:
                filename = '18.json'
            else:
                filename = '19.json'
            pred_file_path = os.path.join(self.hparams.G_dir, self.model_ids[model_i], filename)
            with open(pred_file_path, "r") as f:
                pred_file = json.load(f)
            data, cond = self._prepare_input(model_id=self.model_ids[model_i], pred_file=pred_file, gt_file=gt_file)
        # input image for visualization
        img_first = self.files["imgs"][0][index]
        img_last = self.files["imgs"][1][index]
        cond["img"] = np.concatenate([img_first, img_last], axis=1)
        return data, cond

    def __getitem__(self, index):
        # input image features
        feat = self.files["features"][index]

        # prepare input, GT data and other axillary info
        if self.mode == "test":
            data, cond = self._get_item_test(index)
        else:
            data, cond = self._get_item_train_val(index)

        return data, cond, feat

    def __len__(self):
        return self.files["len"]

if __name__ == '__main__':
    from types import SimpleNamespace

    class EnhancedNamespace(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    hparams = {
        "name": "dm_singapo",
        "json_root": "/home/users/ruiqi.wu/singapo/",   # root directory of the dataset
        "batch_size": 20,  # batch size for training
        "num_workers": 8,  # number of workers for data loading
        "K": 32,    # maximum number of nodes (parts) in the graph (object)
        "split_file": "/home/users/ruiqi.wu/singapo/data/data_split.json",
        "n_views_per_model": 5,
        "root": "/home/users/ruiqi.wu/manipulate_3d_generate/data/blender_version",
        "frame_mode": "last_frame"
    }
    hparams = EnhancedNamespace(**hparams)
    with open(hparams.split_file , "r") as f:
        splits = json.load(f)

        train_ids = splits["train"]
        val_ids = [i for i in train_ids if "augmented" not in i]

    val_ids = [val_id for val_id in val_ids if os.path.exists(os.path.join(hparams.root, val_id, "features"))]

    dataset = MyDataset(hparams, model_ids=val_ids[:20], mode="valid")
    for i in range(20):
        data, cond, feat = dataset.__getitem__(i)
    import ipdb
    ipdb.set_trace()