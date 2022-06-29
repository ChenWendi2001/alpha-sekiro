# alpha-sekiro
Learn to play Sekiro with reinforcement learning.

### 注意！
本仓库是开发仓库，对任何文件的修改/增删请在`README.md`中给出对应的说明。

### Requirements
安装时请注意版本。
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install pywin32==227
pip install pydirectinput

pip install tensorboard
pip install tensorboardX

pip install -U openmim
mim install mmcv-full
pip install mmpose
pip install mmdet

pip install pymem
```

### 安装姿态识别网络(Optional)
- 在根目录执行
```bash
mim download mmpose --config topdown_heatmap_vipnas_mbv3_coco_256x192 --dest ./pose_model
```
- 下载以下文件到`pose_model`
```bash
http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```
