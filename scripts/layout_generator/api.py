import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import re
import json
import base64
import argparse
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI
from scripts.layout_generator.prompt import messaages
from scripts.layout_generator.prompt_description import description_messaages
from scripts.layout_generator.layout_generator_in_grid import generate_layout
import cv2
import random

# Initialize the OpenAI client
client = AzureOpenAI(
    azure_endpoint="your endpoint,
    api_key="your key",
    api_version="your version",
)

def predict_layout(description):
    new_message = messaages.copy()
    user_dict = {
        "role": "user",
        "content": description
    }
    new_message.append(user_dict)
    print(f'generate the layout of {description}')
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

    content = completion.choices[0].message.content
    print(content)
    src = (re.search(r"```python\n(.*?)\n```", content, re.DOTALL).group(1)).split('=')[-1]
    
    return eval(src)

def generate_description(command):
    new_message = description_messaages.copy()
    user_dict = {
        "role": "user",
        "content": command
    }
    new_message.append(user_dict)
    print(f'generate the description of {command}')
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
    content = completion.choices[0].message.content
    print(content)

    return content



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the part connectivity graph from an image")
    parser.add_argument("--save_path", type=str, required=True, help="path to the save the response")
    parser.add_argument("--obj_num", type=int, required=True, help="quantitative of generated objects")   
    args = parser.parse_args()

    for i in range(0, args.obj_num):
        loop_flag = True
        while loop_flag:
            try:
                difficulty = random.choice(['mid', 'complex'])
                class_name = random.choice(['Storage Furniture', 'Table'])    

                command = difficulty + ', ' + class_name
                description = generate_description(command)
                layout = predict_layout(description)

                object_dict = {
                    'description': description,
                    'layout': layout
                }
                
                data_path = os.path.join(args.save_path, class_name.replace(' ', ''), str(i))
                os.makedirs(data_path, exist_ok=True)

                with open(os.path.join(data_path, 'gpt_pred.json'), 'w', encoding='utf-8') as f:
                    json.dump(object_dict, f, indent=4)
                print(layout)
            except Exception as e:
                print(e)
            else:
                loop_flag = False
