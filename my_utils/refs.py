# reference of object categories
cat_ref = {
    "Table": 0,
    "Dishwasher": 1,
    "StorageFurniture": 2,
    "Refrigerator": 3,
    "WashingMachine": 4,
    "Microwave": 5,
    "Oven": 6,
}

data_mode_ref = {
    "aabb_max": 0, 
    "aabb_min": 1, 
    "joint_type": 2, 
    "axis_dir": 3, 
    "axis_ori": 4, 
    "joint_range": 5, 
    "label": 6
}

# reference of semantic labels for each part
sem_ref = {
    "fwd": {
        "door": 0,
        "drawer": 1,
        "base": 2,
        "handle": 3,
        "wheel": 4,
        "knob": 5,
        "shelf": 6,
        "tray": 7,
    },
    "bwd": {
        0: "door",
        1: "drawer",
        2: "base",
        3: "handle",
        4: "wheel",
        5: "knob",
        6: "shelf",
        7: "tray",
    },
}

# reference of joint types for each part
joint_ref = {
    "fwd": {"fixed": 1, "revolute": 2, "prismatic": 3, "screw": 4, "continuous": 5},
    "bwd": {1: "fixed", 2: "revolute", 3: "prismatic", 4: "screw", 5: "continuous"},
}


import plotly.express as px

# pallette for joint type color
joint_color_ref = px.colors.qualitative.Set1
# pallette for graph node color
# graph_color_ref = px.colors.qualitative.Bold + px.colors.qualitative.Prism
# graph_color_ref = [
#     "rgb(200, 200, 200)",  # 奶橙黄
#     "rgb(255, 196, 200)",  # 莓奶粉
#     "rgb(154, 228, 186)",  # 牛油果绿
#     "rgb(252, 208, 140)",  # 奶橙黄
#     "rgb(217, 189, 250)",  # 薄紫
#     "rgb(203, 237, 164)",  # 抹茶绿
#     "rgb(188, 229, 235)",  # 青蓝灰
#     "rgb(179, 199, 243)",  # 雾蓝
#     "rgb(255, 224, 130)",  # 淡柠黄
#     "rgb(222, 179, 212)",  # 粉紫
#     "rgb(148, 212, 224)",  # 冰蓝
# ]
graph_color_ref = [
    "rgb(160, 160, 160)",  # 奶橙灰 → 深灰白，对比提升
    "rgb(255, 130, 145)",  # 莓奶粉 → 更亮更红
    "rgb(80, 200, 150)",   # 牛油果绿 → 更深更绿
    "rgb(255, 180, 60)",   # 奶橙黄 → 更橙更亮
    "rgb(180, 140, 255)",  # 薄紫 → 更强饱和度紫
    "rgb(130, 210, 50)",   # 抹茶绿 → 偏亮偏黄的绿
    "rgb(90, 190, 220)",   # 青蓝灰 → 加蓝提升对比
    "rgb(100, 150, 255)",  # 雾蓝 → 饱和冷蓝
    "rgb(255, 200, 0)",    # 淡柠黄 → 纯柠黄
    "rgb(200, 100, 190)",  # 粉紫 → 更紫
    "rgb(80, 180, 255)",   # 冰蓝 → 更冷更亮的蓝
    "rgb(255, 130, 145)",  # 莓奶粉 → 更亮更红
    "rgb(80, 200, 150)",   # 牛油果绿 → 更深更绿
    "rgb(255, 180, 60)",   # 奶橙黄 → 更橙更亮
    "rgb(180, 140, 255)",  # 薄紫 → 更强饱和度紫
    "rgb(130, 210, 50)",   # 抹茶绿 → 偏亮偏黄的绿
    "rgb(90, 190, 220)",   # 青蓝灰 → 加蓝提升对比
    "rgb(100, 150, 255)",  # 雾蓝 → 饱和冷蓝
    "rgb(255, 200, 0)",    # 淡柠黄 → 纯柠黄
    "rgb(200, 100, 190)",  # 粉紫 → 更紫
    "rgb(80, 180, 255)",   # 冰蓝 → 更冷更亮的蓝
    "rgb(255, 130, 145)",  # 莓奶粉 → 更亮更红
    "rgb(80, 200, 150)",   # 牛油果绿 → 更深更绿
    "rgb(255, 180, 60)",   # 奶橙黄 → 更橙更亮
    "rgb(180, 140, 255)",  # 薄紫 → 更强饱和度紫
    "rgb(130, 210, 50)",   # 抹茶绿 → 偏亮偏黄的绿
    "rgb(90, 190, 220)",   # 青蓝灰 → 加蓝提升对比
    "rgb(100, 150, 255)",  # 雾蓝 → 饱和冷蓝
    "rgb(255, 200, 0)",    # 淡柠黄 → 纯柠黄
    "rgb(200, 100, 190)",  # 粉紫 → 更紫
    "rgb(80, 180, 255)",   # 冰蓝 → 更冷更亮的蓝
    "rgb(255, 130, 145)",  # 莓奶粉 → 更亮更红
    "rgb(80, 200, 150)",   # 牛油果绿 → 更深更绿
    "rgb(255, 180, 60)",   # 奶橙黄 → 更橙更亮
    "rgb(180, 140, 255)",  # 薄紫 → 更强饱和度紫
    "rgb(130, 210, 50)",   # 抹茶绿 → 偏亮偏黄的绿
    "rgb(90, 190, 220)",   # 青蓝灰 → 加蓝提升对比
    "rgb(100, 150, 255)",  # 雾蓝 → 饱和冷蓝
    "rgb(255, 200, 0)",    # 淡柠黄 → 纯柠黄
    "rgb(200, 100, 190)",  # 粉紫 → 更紫
    "rgb(80, 180, 255)",   # 冰蓝 → 更冷更亮的蓝
]
# pallette for semantic label color
semantic_color_ref = px.colors.qualitative.Vivid_r
# attention map visulaization color
attn_color_ref = px.colors.sequential.Viridis

from matplotlib.colors import LinearSegmentedColormap

cmap_attn = LinearSegmentedColormap.from_list("mycmap", attn_color_ref, N=256)
