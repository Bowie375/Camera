# 数据文件结构
```
data
  - objects  # 物体级数据，有单投影左右相机结构光图片、32张alacarte条纹图投影结构光采集图片、严格对齐rgb图
    - object_name1
      - alacarte_32_camL  # 多投影结构光数据，投影32张alacarte条纹投影采集的左相机图片  
      - rgb  # 严格对齐的rgb图片（无投影）
      - single_shot  # 用8种投影图案拍摄的单投影结构光图片  
      - meta.npy
    - object_name2 ...
      - ...
  - scenes  # 场景级数据，手持拍摄，仅有但投影左右相机结构光图片、不严格对齐的rgb图.  
    - scene_name1
      - single_shot
      - rgb
      - meta.npy
    - scene_name2
      - ...
  - patterns  # 使用的投影图案
    - alacarte_32   # 32张alacarte条纹图，用于 alacarte_32_camL文件夹  
      - alacarte_32_id1.png
      - alacarte_32_id2.png
      - ...
    - single_shot_id1.png  # 单帧结构光投影1
    - single_shot_id2.png
    - ...
```
所有图片命名为：`cam_{L/R}_{pattern_id}.png`，`L/R` 表示左相机还是右相机，pattern_id是这张图片所使用的投影图案的id，`0008`表示无投影。`single_shot`下的投影图案在`patterns/`下通过`pattern_id`找到，`alacarte_32_camL`下的投影图案在 `patterns/alacarte_32` 下通过 `pattern_id` 找到。  

已知的相机信息在meta.npy中，可以用 `data/read_meta.py` 读取。  