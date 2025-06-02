import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import re
import json
import base64
import argparse
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI
from scripts.graph_pred.prompt_workflow_new import messages
# Initialize the OpenAI client
client = AzureOpenAI(
    azure_endpoint="your_endpoint",
    api_key="your_key",
    api_version="your_version",
)

def encode_image(image_path: str, center_crop=False):
    """Resize and encode the image as base64"""
    # load the image
    image = Image.open(image_path)

    # resize the image to 224x224
    if center_crop: # (resize to 256x256 and then center crop to 224x224)
        image = image.resize((256, 256))
        width, height = image.size
        left = (width - 224) / 2
        top = (height - 224) / 2
        right = (width + 224) / 2
        bottom = (height + 224) / 2
        image = image.crop((left, top, right, bottom))
    else:
        image = image.resize((224, 224))

    # conver the image to bytes
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    # encode the image as base64
    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded_image

def display_image(image_data):
    """Display the image from the base64 encoded image data"""
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img.show()
    img.close()


def convert_format(src):
    '''Convert the JSON format from the response to a tree format'''
    def _sort_nodes(tree):
        num_nodes = len(tree)
        sorted_tree = [dict() for _ in range(num_nodes)]
        for node in tree:
            sorted_tree[node["id"]] = node
        return sorted_tree

    def _traverse(node, parent_id, current_id):
        for key, value in node.items():
            node_id = current_id[0]
            current_id[0] += 1

            # Create the node
            tree_node = {
                "id": node_id,
                "parent": parent_id,
                "name": key,
                "children": [],
            }

            # Traverse children if they exist
            if isinstance(value, list):
                for child in value:
                    child_id = _traverse(child, node_id, current_id)
                    tree_node["children"].append(child_id)

            # Add this node to the tree
            tree.append(tree_node)
            return node_id

    tree = []
    current_id = [0]
    _traverse(src, -1, current_id)
    diffuse_tree = _sort_nodes(tree)
    return diffuse_tree

def predict_graph_twomode(image_path, first_img_data=None, second_img_data=None, debug=False, center_crop=False):
    '''Predict the part connectivity graph from the image'''
    # Encode the image
    if first_img_data is None or second_img_data is None:
        first_img_data = encode_image(image_path, center_crop)
        second_img_data = encode_image(image_path.replace('close', 'open'), center_crop)
    # if debug:
    #     display_image(image_data) # for double checking the image
    #     breakpoint()
    new_message = messages.copy()
    new_message.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{first_img_data}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{second_img_data}"},
                    }
                ],
            },
    )
    # Get the completion from the model
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

    # Extract the response
    content = completion.choices[0].message.content

    src = json.loads(re.search(r"```json\n(.*?)\n```", content, re.DOTALL).group(1))
    print(src)
    # Convert the JSON format to tree format
    diffuse_tree = convert_format(src)

    return {"diffuse_tree": diffuse_tree, "original_response": content}

def save_response(save_path, response):
    '''Save the response to a json file'''
    with open(save_path, "w") as file:
        json.dump(response, file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the part connectivity graph from an image")
    parser.add_argument("--img_path", type=str, required=True, help="path to the image")
    parser.add_argument("--save_path", type=str, required=True, help="path to the save the response")
    parser.add_argument("--center_crop", action="store_true", help="whether to center crop the image to 224x224, otherwise resize to 224x224")   
    args = parser.parse_args()

    try:
        response = predict_graph(args.img_path, args.center_crop)
        save_response(args.save_path, response)
        response = predict_graph_twomode(args.img_path, args.center_crop)
        save_response(args.save_path[:-5] + 'twomode.json', response)
    except Exception as e:
        with open('openai_err.log', 'a') as f:
            f.write('---------------------------\n')
            f.write(f'{args.img_path}\n')
            f.write(f'{e}\n')
