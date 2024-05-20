# 香橙派5芯片字符检测
  香橙派5（rk3588）部署yolov8 + easyocr + pyqt5的芯片字符识别系统（Chip character recognition system deploying YOLOv8 + EasyOCR + PyQt5 on Orange Pi 5 (RK3588)）
<!-- PROJECT SHIELDS -->
<p align="center">
  <a href="https://github.com/ABigCyan/Chip-character-recognition">
    <img src="images/ui.png" alt="Logo" >
  </a>
  
<!-- PROJECT LOGO -->
  
## 目录

- [环境配置](#环境配置)
  - [电脑主机环境配置](#电脑主机环境配置)
  - [香橙派5环境配置](#香橙派5环境配置)
- [文件目录说明](#文件目录说明)
- [部署](#部署)
- [使用到的框架](#使用到的框架)
- [版本控制](#版本控制)
- [鸣谢](#鸣谢)

### 环境配置
  分别是x86pc和开发板的环境配置

#### 电脑主机环境配置
1)yolov8环境配置

用mimiconda创建虚拟环境(可以参考这个b站视频非常详尽：【【手把手带你实战Ultralytics】02-环境安装与配置】 https://www.bilibili.com/video/BV1vH4y1a72o/?share_source=copy_web&vd_source=d41740ad2b14d1c71d883e3bad08d3fd）

  pip安装pytorch（pytorch官网：https://pytorch.org/）
  ```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
  打开yolov8目录pip进行环境配置
```
pip install -e .
```
2)rknn-toolkit配置

电脑安装wsl虚拟机，获取https://github.com/rockchip-linux/rknn-toolkit

#### 香橙派5环境配置

1）conda环境配置和rknn-toolkit配置

和pc端一致，pytorch和miniconda注意要进行arm64版本的安装

2）pyqt5配置

pip安装会报错，猜测是无法安装arm版本的pyqt5，换成conda进行安装

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PyQt5
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PyQt5-tools 
```

由于conda的pyqt5会与pip安装的cv2冲突，可以把cv2从从pip移除，使用conda进行安装

```
pip unstall opencv-python
conda install opencv
```

如果下不动可以考虑换源（https://blog.csdn.net/innovationy/article/details/121510666）


### 文件目录说明
eg:

```
filetree 
├── LICENSE.txt
├── README.md
├── /account/
├── /bbs/
├── /docs/
│  ├── /rules/
│  │  ├── backend.txt
│  │  └── frontend.txt
├── manage.py
├── /oa/
├── /static/
├── /templates/
├── useless.md
└── /util/

```

