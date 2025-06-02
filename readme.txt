# Environment Setting
> pip install -r requirements.txt

> torch==2.3.1+cu118 torchvision==0.18.1+cu118

(Please set GPT API with your own keys).
### Note
If 'segmentation fault' is reported, please enter the comman 'unset LD_LIBRARY_PATH'


# run demo
python demo_img.py \
    --config_path configs/config.yaml \
    --ckpt_path dipo.ckpt \
    --img_path_1 path/to/resting/state/image \
    --img_path_2 path/to/articulated/state/image \
    --save_dir results

