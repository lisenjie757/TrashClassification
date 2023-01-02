# 技术文档

## 环境

- 硬件平台：Raspberry Pi 4B（8GB RAM） 
- 操作系统：Raspberry Pi OS（64-bit）
- Python = 3.9.1

## 安装

安装必要的依赖

```bash
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt
```
## 使用

图片推理

```bash
python main.py --image 'test_data/banana_skin.png'
```

视频推理

```bash
python main.py --image 0  # Camera Id
```

## API调用

综合判断获取识别图片和结果
```py
from main import output

img, score, label = output()
```

使用本地模型获取识别图片和结果变量
```py
from main import parser, run

args = parser()
img, results = run(args)
```

使用阿里云端模型获取识别图片和结果变量
```py
import cv2
from main import Sample
from main import parser

args = parser()
img = cv2.imread(args.image)

cv2.imwrite('./tmp.jpg', img)
response = Sample.main(args)
```