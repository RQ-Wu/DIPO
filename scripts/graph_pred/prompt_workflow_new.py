import os
from PIL import Image
from io import BytesIO
import base64

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
        image = image.resize((512, 512))

    # conver the image to bytes
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    # encode the image as base64
    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded_image

system_prompt = """
You are an expert in the recognition, structural parsing, and physical‑feasibility validation of articulated objects from image inputs.

You will be provided with two rendered images of the same object:
1. A closed‑state image (all movable parts in their fully closed positions)
2. An open‑state image (all movable parts in their fully opened positions)

Your task is to analyze the object's articulated structure and generate a connectivity graph describing the part relationships.

You must follow this workflow:

1. Part Detection:
   - Detect candidate parts with their coarse position in the **closed-state image**, with optional assistance from the **open-state image** to resolve ambiguous or occluded parts.
   - Allowed part types: ['base', 'door', 'drawer', 'handle', 'knob', 'tray']
   - Ignore small decorative things directly attached to the base.
   - There must be exactly one "base"; "tray" is only allowed if the object is a microwave (but "tray" is not the component must be shown in microwave).

2. Step-by-Step Reasoning:
   1. Part Listing: List all detected parts and their counts (do not infer attachment yet) 
   2. Validation: Enforce structural rules:
      - Exactly one base
      - Each door or drawer may have at most two handles or knobs
      - Every handle/knob must attach to a door or drawer
      - Trays only appear in microwaves
   3. Attachment Inference: For each non-base part, infer its parent (e.g., "drawer_1 (attached to base)").
      - Use the open-state image to assist in verifying attachments if the closed-state image is ambiguous.
   4. Connectivity Graph Construction: Output a JSON tree of part relationships. The base is the root.

Example output:
```json
{
  "base": [
    { "door": [ { "handle": [] } ] },
    { "drawer": [ { "handle": [] } ] }
  ]
}
```

Final Output: You **MUST** output a single JSON tree representing the part connectivity of the object. Use the open-state image to improve the accuracy and completeness of your structural interpretation, but base your parsing primarily on the closed-state image.
Note: You **MUST** give the final output in JSON format.
"""

answer_sys = """
Great — please upload the two images of the same object:

One in the closed state (all movable parts fully closed)

One in the open state (all movable parts fully open)

Once I receive both, I’ll proceed with:

Part detection on the closed image

Structured reasoning and validation

Attachment inference

Final part connectivity graph in JSON format

Let me know when you're ready to continue!
"""

answer_1 = """
Great! Let’s go through the articulated object parsing workflow based on the provided **closed-state** and **open-state** images.

---

### **1. Part Detection**

From the **closed-state image** (first image), with verification from the **open-state image** (second image):

- **Base**: 1 base structure (entire wooden cabinet)
- **Doors**: 4 doors  
  - 2 on the upper section (left & right)  
  - 2 on the lower section (left & right)
- **Drawers**: 4 drawers  
  - 3 in a row at the middle  
  - 1 larger drawer centered below the middle row
- **Handles**: 8 handles  
  - 1 per drawer  
  - 1 per door

---

### **2. Step-by-Step Reasoning**

#### 2.1 **Part Listing**

- `base`: 1  
- `door`: 4  
- `drawer`: 4  
- `handle`: 8

#### 2.2 **Validation**

- ✅ Exactly one base  
- ✅ Each door has 1 handle → 4 handles for 4 doors  
- ✅ Each drawer has 1 handle → 4 handles for 4 drawers  
- ✅ No trays → ✓ (not a microwave)  
- ✅ Every handle is attached to a door or drawer (verified using open-state image)

All rules are satisfied.

#### 2.3 **Attachment Inference**

From closed/open-state comparison:

- Each **door** is directly attached to the **base**
- Each **drawer** is mounted to slide out of the **base**
- Each **handle** is on its respective **door** or **drawer**

---

### **3. Connectivity Graph Construction**

Final hierarchical structure:

```json
{
  "base": [
    { "door_1": [ { "handle_1": [] } ] },
    { "door_2": [ { "handle_2": [] } ] },
    { "door_3": [ { "handle_3": [] } ] },
    { "door_4": [ { "handle_4": [] } ] },
    { "drawer_1": [ { "handle_5": [] } ] },
    { "drawer_2": [ { "handle_6": [] } ] },
    { "drawer_3": [ { "handle_7": [] } ] },
    { "drawer_4": [ { "handle_8": [] } ] }
  ]
}
```
"""

answer_2 = """
Thanks for the new image pair! Let's analyze the articulated structure of the **washing machine** based on the provided closed-state and open-state images.

---

### **1. Part Detection**

From the **closed-state image**, validated using the **open-state image**:

- **Base**: 1 main body (the washing machine structure)
- **Door**: 1 circular front-loading door
- **Handle**: Notice that there is no handle attached to the dooor
- **Knob**: 1 control knob on the top of the machine

---

### **2. Step-by-Step Reasoning**

#### 2.1 **Part Listing**

- `base`: 1  
- `door`: 1
- `knob`: 1

#### 2.2 **Validation**

- ✅ Exactly one base
- ✅ No drawers, trays → Valid  
- ✅ Trays not present (correct: not a microwave)  

→ **All structural constraints satisfied**

#### 2.3 **Attachment Inference**

- The **door** is front-mounted and clearly opens via a **hinge** on the left side (from viewer's perspective)
- The **knob** is directly mounted on the **base** (top section)
---

### **3. Connectivity Graph Construction**

```json
{
  "base": [
    { "door": []},
    { "knob": []}
  ]
}
```
"""

answer_3 = """
Thanks! Let’s analyze the articulated structure of this oven using your provided closed and open state images.

---

### ✅ Articulated Object Parsing (Oven)

#### **1. Part Detection**

From the **closed-state** image (with **open-state** verification):

- **base**: 1 (the main oven body)
- **door**: 1 (the front oven door)
- **handle**: 1 (long horizontal handle on the door)

> 🔍 *The 6 small control knobs on top are **ignored** as instructed.*

---

#### **2. Step-by-Step Reasoning**

**2.1 Part Listing:**

- `base`: 1  
- `door`: 1  
- `handle`: 1

**2.2 Validation:**

- ✅ Exactly one base  
- ✅ Door has 1 handle  
- ✅ Handle is attached to the door  
- ✅ No trays (✓ valid for an oven)  
- ✅ No knobs or extra handles beyond what's allowed  

✔ All constraints are satisfied.

**2.3 Attachment Inference:**

- The **door** is bottom-hinged and connected to the **base**
- The **handle** is directly mounted on the door

---

#### **3. Final Connectivity Graph**

```json
{
  "base": [
    { "door": [ { "handle": [] } ] }
  ]
}
```
"""

answer_4 = """
Thanks! Let's now analyze the **wardrobe** based on the closed and open-state images you've provided.

---

### ✅ Articulated Object Parsing – Wardrobe

---

### **1. Part Detection**

- **Base**: 1 (entire wardrobe structure)
- **Doors**: 4 vertical panels (two left, two right)
- **Drawers**: 4 horizontal drawers at the bottom
- **Handles**: 8 in total  
  - 4 for doors (1 each)  
  - 4 for drawers (1 each)

---

### **2. Step-by-Step Reasoning**

#### **2.1 Part Listing**

- `base`: 1  
- `door`: 4  
- `drawer`: 4  
- `handle`: 8

#### **2.2 Validation**

- ✅ Exactly one base  
- ✅ Each door has 1 handle → valid  
- ✅ Each drawer has 1 handle → valid  
- ✅ No trays → ✓ valid (not a microwave)  
- ✅ Every handle is attached to a door or drawer (confirmed via open-state image)

✔ All structural rules are satisfied.

---

#### **2.3 Attachment Inference**

- Each **door** is hinged to the **base**
- Each **drawer** slides out of the **base**
- Each **handle** is mounted on one **door** or **drawer**

---

### **3. Final Connectivity Graph**

```json
{
  "base": [
    { "door_1": [ { "handle_1": [] } ] },
    { "door_2": [ { "handle_2": [] } ] },
    { "door_3": [ { "handle_3": [] } ] },
    { "door_4": [ { "handle_4": [] } ] },
    { "drawer_1": [ { "handle_5": [] } ] },
    { "drawer_2": [ { "handle_6": [] } ] },
    { "drawer_3": [ { "handle_7": [] } ] },
    { "drawer_4": [ { "handle_8": [] } ] }
  ]
}
```
"""

answer_all = [answer_sys, answer_1, answer_2, answer_3, answer_4]
messages = [
    {"role": "user", "content": system_prompt},
    {"role": "assistant", "content": answer_sys}
]

for i in range(4):
    root_path = './scripts/imgs_reference/'
    close_path = os.path.join(root_path, f'close{i + 1}.png')
    open_path = os.path.join(root_path, f'open{i + 1}.png')
    close_img = encode_image(close_path, center_crop=False)
    open_img = encode_image(open_path, center_crop=False)
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{close_img}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{open_img}"},
                }
            ],
        }
    )
    messages.append({"role": "assistant", "content": answer_all[i + 1]})