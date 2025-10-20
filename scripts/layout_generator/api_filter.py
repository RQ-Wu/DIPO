import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import re
import json
import base64
import argparse
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI
from scripts.layout_generator.prompt_filter import message_filter
from scripts.layout_generator.layout_generator_in_grid import generate_layout
import cv2
import random

# Initialize the OpenAI client
client = AzureOpenAI(
    azure_endpoint="your endpoint,
    api_key="your key",
    api_version="your version",
)

def encode_video_frames(video_path: str, center_crop=False):
    """Extract the first and last frame from a video and encode them as base64."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, first_frame = cap.read()
    if not success:
        raise ValueError("Failed to read the first frame.")

    # Get the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    success, last_frame = cap.read()
    if not success:
        raise ValueError("Failed to read the last frame.")

    cap.release()

    def process_frame(frame):
        # Convert BGR (OpenCV format) to RGB (PIL format)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize and center crop
        if center_crop:
            image = image.resize((256, 256))
            width, height = image.size
            left = (width - 224) / 2
            top = (height - 224) / 2
            right = (width + 224) / 2
            bottom = (height + 224) / 2
            image = image.crop((left, top, right, bottom))
        else:
            image = image.resize((224, 224))

        # Encode to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        return encoded

    return process_frame(first_frame), process_frame(last_frame)

def filter_object(video_path):
    new_message = message_filter.copy()
    image_data, image_open_data = encode_video_frames(video_path)
    user_dict = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_open_data}"},
            }
        ]
    }
    new_message.append(user_dict)
    completion = client.chat.completions.create(
        model="yfb-gpt-4o", 
        messages=new_message,
        response_format={"type": "text"},
        temperature=1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print('processing the response...')

    content = completion.choices[0].message.content
    print(content)

    filter_flag = False
    if 'yes' in content or 'Yes' in content or 'YES' in content:
        filter_flag = True

    return filter_flag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the part connectivity graph from an image")
    parser.add_argument("--save_path", type=str, required=True, help="path to the save the response") 
    args = parser.parse_args()

    data_root = args.save_path
    for class_name in os.listdir(data_root):
        class_path = os.path.join(data_root, class_name)
        for model_id in os.listdir(class_path):
            model_path = os.path.join(class_path, model_id)
            if os.path.exists(os.path.join(model_path, 'imgs')):
                video_path = os.path.join(model_path, 'imgs', 'animation_10.mp4')
                filter_flag = filter_object(video_path)
                print(f'{model_path}: {filter_flag}')

