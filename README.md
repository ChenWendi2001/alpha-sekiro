# alpha-sekiro
Learn to play Sekiro with reinforcement learning.

### 注意！
本仓库是开发仓库，对任何文件的修改/增删请在`README.md`中给出对应的说明。

### Requirements
安装时请注意版本。
```bash
pip install pywin32==227
```

### 源代码文件说明
#### src/utils/control.py
- 此文件负责模拟键盘与鼠标的输入
- 可以按需求添加按键组合以实现连招或是退出游戏等操作

#### src/utils/screenshot.py
- 此文件可以获取指定窗口的截图