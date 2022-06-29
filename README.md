# alpha-sekiro
Learn to play Sekiro with reinforcement learning.

### 简介
本项目基于《只狼》游戏设计了一个鲁棒的训练环境，并基于多种视觉模型通过DQN强化学习的方式训练了一个较强的只狼AI，其可以击败弦一郎一阶段。具体效果可以查看此[b站视频](https://www.bilibili.com/video/BV1aS4y1p78e)。

### Requirements
安装时请注意版本。
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install pywin32==227
pip install pydirectinput

pip install tensorboard
pip install tensorboardX

pip install pymem

# 以下以来为可选依赖，如果不使用姿态视觉模型，无需安装
pip install -U openmim
mim install mmcv-full
pip install mmpose
pip install mmdet
```

### 安装姿态识别网络(可选）
- 在根目录执行
```bash
mim download mmpose --config topdown_heatmap_vipnas_mbv3_coco_256x192 --dest ./pose_model
```
- 下载以下文件到`pose_model`
```bash
http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```


### Getting Started
- 确认《只狼》游戏为`1.0.5`，安装`mod engine`通过[坚果云](https://www.jianguoyun.com/p/DX_Eu1AQ76KXCRjzocgEIAA)下载指定的`mod`并安装。

- 使用键盘进行游戏，并修改游戏按键：

   - 添加“J”键为“攻击”

   - 添加“K”键为“防御”

   - 添加“U、I、O、P”键为“上下左右视角移动”

- 调整游戏内设置

   - 使用窗口化进行游戏，修改分辨率为`1280x720`，将”质量设定“调整为中
   
   - 关闭”血腥效果“以及”字幕显示“，设置”亮度调整“为10

- 选择关卡："再战稀世强者"，”苇名弦一郎“，等待关卡载入结束后，按”esc“进行暂停

- 进入`src`文件夹，执行以下命令，若使用姿态模型，还需增加`--use_pose_detection`参数。
  ```bash
  python train.py --model_type cnn
  ```

- 按`T`开启训练
