import numpy as np
from PIL import Image, ImageDraw

# 1. 讀取推論結果與圖片
heatmap = np.load("/output/nef_sim_output.npy")[0]  # 形狀為 (17, 64, 48)
img = Image.open("/input/test.jpg").convert('RGB').resize((192, 256))
draw = ImageDraw.Draw(img)

# 2. 定義 COCO 17 個點的連線對 (Index 從 0 開始)
# 格式: (點A, 點B)
SKELETON_PAIRS = [
    (5, 6), (5, 11), (6, 12), (11, 12),        # 軀幹 (肩膀與臀部)
    (5, 7), (7, 9),                            # 左手 (肩-肘-腕)
    (6, 8), (8, 10),                           # 右手 (肩-肘-腕)
    (11, 13), (13, 15),                        # 左腳 (臀-膝-踝)
    (12, 14), (14, 16),                        # 右腳 (臀-膝-踝)
    (0, 1), (0, 2), (1, 3), (2, 4)             # 頭部 (鼻-眼-耳)
]

# 3. 解析 17 個點的座標
joints = []
for i in range(17):
    joint_map = heatmap[i]
    y, x = np.unravel_index(np.argmax(joint_map), joint_map.shape)
    # 還原解析度 (48x64 -> 192x256)
    joints.append((x * 4, y * 4))

print("正在繪製骨架連線...")

# 4. 畫出連線
for pair in SKELETON_PAIRS:
    p1 = joints[pair[0]]
    p2 = joints[pair[1]]
    # 畫線，寬度設為 2，顏色設為青色 (Cyan)
    draw.line([p1, p2], fill=(0, 255, 255), width=2)

# 5. 畫出關節點 (紅色)
for joint in joints:
    r = 3
    draw.ellipse((joint[0]-r, joint[1]-r, joint[0]+r, joint[1]+r), fill='red')

# 6. 存檔
img.save("/output/final_pose_result.jpg")
print("✅ 完美！請查看 final_pose_result.jpg")