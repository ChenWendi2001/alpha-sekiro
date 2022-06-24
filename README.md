# alpha-sekiro
Learn to play Sekiro with reinforcement learning.

### 注意！
本仓库是开发仓库，对任何文件的修改/增删请在`README.md`中给出对应的说明。

### Requirements
安装时请注意版本。
```bash
pip install pywin32==227
pip install pydirectinput

pip install tensorboard
pip install tensorboardX

pip install -U openmim
mim install mmcv-full
pip install mmpose
pip install mmdet
```

### 安装预训练姿态网络
- 在根目录执行
```bash
mim download mmpose --config topdown_heatmap_vipnas_mbv3_coco_256x192 --dest ./pretrained_model
```
- 下载以下文件到`pretrained_model`
```bash
http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```

### 源代码文件说明
#### src/utils
- 该文件夹包含模拟键盘与鼠标的输入的代码
- 该文件夹包含读取游戏界面中血量和耐力值的代码