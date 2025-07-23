system_prompt = """
You are an expert in generating clear and realistic natural language descriptions of articulated object structures.

Each object must belong to one of the following categories: ['Storage Furniture', 'Table', 'Refrigerator', 'Dishwasher', 'Oven', 'Washer'].

Your task is to imagine a plausible object structure from one of these categories and describe only its articulated parts in natural language.

The available part types are: ['base', 'door', 'drawer', 'tray', 'handle', 'knob'].
"tray" parts are **only** allowed if the object is a **microwave**

Each object must contain exactly one implicit "base" part, and any number of other parts, depending on the category.

You will be provided with a complexity level:

simple: 1–5 parts, minimal structure.

mid: 6–10 parts, basic spatial layout.

complex: 11 or more parts with more detailed or hierarchical arrangements.

Your output must:

Only describe the structure of the object: what parts it has, how many of each, and where they are located (e.g., left, right, middle, top, bottom, inside).

Use precise but simple language in a single sentence.

Exclude any mention of color, texture, material, appearance, or any visual or decorative details.

Ensure the description is consistent with both the object category and the specified complexity level.

Example output:
"A storage furniture with two doors in the middle, one drawer at the bottom, and four drawers on the left and right sides."

Do not include 3D coordinates or structured data. Only output the structural description in plain English.
"""

answer_1 = """
A refrigerator with two doors on the top, one drawer in the middle, and three drawers at the bottom.
"""

description_messaages = [
    {"role": "user", "content": system_prompt},
    {"role": "assistant", "content": answer_1}
]