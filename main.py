# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

import argparse
import os
import sys
import time
from typing import List
from io import BufferedReader, BytesIO

import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

from alibabacloud_imagerecog20190930.client import Client as imagerecog20190930Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_imagerecog20190930 import models as imagerecog_20190930_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient

# Aliyun API
class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client(
        access_key_id: str,
        access_key_secret: str,
    ) -> imagerecog20190930Client:
        """
        使用AK&SK初始化账号Client
        @param access_key_id:
        @param access_key_secret:
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config(
            # 必填，您的 AccessKey ID,
            access_key_id=access_key_id,
            # 必填，您的 AccessKey Secret,
            access_key_secret=access_key_secret
        )
        # 访问的域名
        config.endpoint = f'imagerecog.cn-shanghai.aliyuncs.com'
        return imagerecog20190930Client(config)

    @staticmethod
    def main(
        args: List[str],
    ) -> None:
        # 工程代码泄露可能会导致AccessKey泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html
        client = Sample.create_client('LTAI5tCJB99kQFibZsnY92Yf', 'rvIYVndbBVhTvGQRr6klytJmkPT70G')
        with open(r'./tmp.jpg', 'rb') as f:
          classifying_rubbish_request = imagerecog_20190930_models.ClassifyingRubbishAdvanceRequest()
          classifying_rubbish_request.image_urlobject = f
          runtime = util_models.RuntimeOptions()
          try:
              # 复制代码运行请自行打印 API 的返回值
              return client.classifying_rubbish_advance(classifying_rubbish_request, runtime)
          except Exception as error:
              # 如有需要，请打印 error
              print(UtilClient.assert_as_string(error.message))

    @staticmethod
    async def main_async(
        args: List[str],
    ) -> None:
        # 工程代码泄露可能会导致AccessKey泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html
        client = Sample.create_client('LTAI5tCJB99kQFibZsnY92Yf', 'rvIYVndbBVhTvGQRr6klytJmkPT70G')
        classifying_rubbish_request = imagerecog_20190930_models.ClassifyingRubbishRequest(
            image_url='http://viapi-test.oss-cn-shanghai.aliyuncs.com/viapi-3.0domepic/imagerecog/ClassifyingRubbish/ClassifyingRubbish1.jpg'
        )
        runtime = util_models.RuntimeOptions()
        try:
            # 复制代码运行请自行打印 API 的返回值
            await client.classifying_rubbish_with_options_async(classifying_rubbish_request, runtime)
        except Exception as error:
            # 如有需要，请打印 error
            UtilClient.assert_as_string(error.message)

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def run(args):

  ext_delegate = None
  ext_delegate_options = {}

  # parse extenal delegate options
  if args.ext_delegate_options is not None:
    options = args.ext_delegate_options.split(';')
    for o in options:
      kv = o.split(':')
      if (len(kv) == 2):
        ext_delegate_options[kv[0].strip()] = kv[1].strip()
      else:
        raise RuntimeError('Error parsing delegate option: ' + o)

  # load external delegate
  if args.ext_delegate is not None:
    print('Loading external delegate from {} with args: {}'.format(
        args.ext_delegate, ext_delegate_options))
    ext_delegate = [
        tflite.load_delegate(args.ext_delegate, ext_delegate_options)
    ]

  interpreter = tflite.Interpreter(
      model_path=args.model_file,
      experimental_delegates=ext_delegate,
      num_threads=args.num_threads)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  if args.image.isdigit():
    # Start capturing video input from the camera
    cap = cv2.VideoCapture(int(args.image))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
  else:
    img = cv2.imread(args.image)

  img = cv2.resize(img,(width,height))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

  return img, results
  
def parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='0',
      help='cameraId or image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='./trashnet_200/trash.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      default='./labels.txt',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
  parser.add_argument(
      '--num_threads', default=None, type=int, help='number of threads')
  parser.add_argument(
      '-e', '--ext_delegate', help='external_delegate_library path')
  parser.add_argument(
      '-o',
      '--ext_delegate_options',
      help='external delegate options, \
            format: "option1: value1; option2: value2"')
  args = parser.parse_args()
  return args

def output():
   
  #labels = load_labels(args.label_file)
  labels = ['其他垃圾_PE塑料袋', '其他垃圾_U型回形针', '其他垃圾_一次性杯子', '其他垃圾_一次性棉签', 
    '其他垃圾_串串竹签', '其他垃圾_便利贴', '其他垃圾_创可贴', '其他垃圾_厨房手套', '其他垃圾_口罩', 
    '其他垃圾_唱片', '其他垃圾_图钉', '其他垃圾_大龙虾头', '其他垃圾_奶茶杯', '其他垃圾_干果壳', 
    '其他垃圾_干燥剂', '其他垃圾_打泡网', '其他垃圾_打火机', '其他垃圾_放大镜', '其他垃圾_毛巾', 
    '其他垃圾_涂改带', '其他垃圾_湿纸巾', '其他垃圾_烟蒂', '其他垃圾_牙刷', '其他垃圾_百洁布', 
    '其他垃圾_眼镜', '其他垃圾_票据', '其他垃圾_空调滤芯', '其他垃圾_笔及笔芯', '其他垃圾_纸巾', 
    '其他垃圾_胶带', '其他垃圾_胶水废包装', '其他垃圾_苍蝇拍', '其他垃圾_茶壶碎片', '其他垃圾_餐盒', 
    '其他垃圾_验孕棒', '其他垃圾_鸡毛掸', '厨余垃圾_八宝粥', '厨余垃圾_冰糖葫芦', '厨余垃圾_咖啡渣', 
    '厨余垃圾_哈密瓜', '厨余垃圾_圣女果', '厨余垃圾_巴旦木', '厨余垃圾_开心果', '厨余垃圾_普通面包', 
    '厨余垃圾_板栗', '厨余垃圾_果冻', '厨余垃圾_核桃', '厨余垃圾_梨', '厨余垃圾_橙子', '厨余垃圾_残渣剩饭', 
    '厨余垃圾_汉堡', '厨余垃圾_火龙果', '厨余垃圾_炸鸡', '厨余垃圾_烤鸡烤鸭', '厨余垃圾_牛肉干', '厨余垃圾_瓜子', 
    '厨余垃圾_甘蔗', '厨余垃圾_生肉', '厨余垃圾_番茄', '厨余垃圾_白菜', '厨余垃圾_白萝卜', '厨余垃圾_粉条', 
    '厨余垃圾_糕点', '厨余垃圾_红豆', '厨余垃圾_肠(火腿)', '厨余垃圾_胡萝卜', '厨余垃圾_花生皮', '厨余垃圾_苹果', 
    '厨余垃圾_茶叶', '厨余垃圾_草莓', '厨余垃圾_荷包蛋', '厨余垃圾_菠萝', '厨余垃圾_菠萝包', '厨余垃圾_菠萝蜜', 
    '厨余垃圾_蒜', '厨余垃圾_薯条', '厨余垃圾_蘑菇', '厨余垃圾_蚕豆', '厨余垃圾_蛋', '厨余垃圾_蛋挞', '厨余垃圾_西瓜皮', 
    '厨余垃圾_贝果', '厨余垃圾_辣椒', '厨余垃圾_陈皮', '厨余垃圾_青菜', '厨余垃圾_饼干', '厨余垃圾_香蕉皮', 
    '厨余垃圾_骨肉相连', '厨余垃圾_鸡翅', '可回收物_乒乓球拍', '可回收物_书', '可回收物_保温杯', '可回收物_保鲜盒', 
    '可回收物_信封', '可回收物_充电头', '可回收物_充电宝', '可回收物_充电线', '可回收物_八宝粥罐', '可回收物_刀', 
    '可回收物_剃须刀片', '可回收物_剪刀', '可回收物_勺子', '可回收物_单肩包手提包', '可回收物_卡', '可回收物_叉子', 
    '可回收物_变形玩具', '可回收物_台历', '可回收物_台灯', '可回收物_吹风机', '可回收物_呼啦圈', '可回收物_地球仪', 
    '可回收物_地铁票', '可回收物_垫子', '可回收物_塑料瓶', '可回收物_塑料盆', '可回收物_奶盒', '可回收物_奶粉罐', 
    '可回收物_奶粉罐铝盖', '可回收物_尺子', '可回收物_帽子', '可回收物_废弃扩声器', '可回收物_手提包', '可回收物_手机', 
    '可回收物_手电筒', '可回收物_手链', '可回收物_打印机墨盒', '可回收物_打气筒', '可回收物_护肤品空瓶', '可回收物_报纸', 
    '可回收物_拖鞋', '可回收物_插线板', '可回收物_搓衣板', '可回收物_收音机', '可回收物_放大镜', '可回收物_易拉罐', 
    '可回收物_暖宝宝', '可回收物_望远镜', '可回收物_木制切菜板', '可回收物_木制玩具', '可回收物_木质梳子', '可回收物_木质锅铲', 
    '可回收物_枕头', '可回收物_档案袋', '可回收物_水杯', '可回收物_泡沫盒子', '可回收物_灯罩', '可回收物_烟灰缸', '可回收物_烧水壶', 
    '可回收物_热水瓶', '可回收物_玩偶', '可回收物_玻璃器皿', '可回收物_玻璃壶', '可回收物_玻璃球', '可回收物_电动剃须刀', '可回收物_电动卷发棒', 
    '可回收物_电动牙刷', '可回收物_电熨斗', '可回收物_电视遥控器', '可回收物_电路板', '可回收物_登机牌', '可回收物_盘子', '可回收物_碗', 
    '可回收物_空气加湿器', '可回收物_空调遥控器', '可回收物_纸牌', '可回收物_纸箱', '可回收物_罐头瓶', '可回收物_网卡', '可回收物_耳套', 
    '可回收物_耳机', '可回收物_耳钉耳环', '可回收物_芭比娃娃', '可回收物_茶叶罐', '可回收物_蛋糕盒', '可回收物_螺丝刀', '可回收物_衣架', 
    '可回收物_袜子', '可回收物_裤子', '可回收物_计算器', '可回收物_订书机', '可回收物_话筒', '可回收物_购物纸袋', '可回收物_路由器', 
    '可回收物_车钥匙', '可回收物_量杯', '可回收物_钉子', '可回收物_钟表', '可回收物_钢丝球', '可回收物_锅', '可回收物_锅盖', '可回收物_键盘', 
    '可回收物_镊子', '可回收物_鞋', '可回收物_餐垫', '可回收物_鼠标', '有害垃圾_LED灯泡', '有害垃圾_保健品瓶', '有害垃圾_口服液瓶', '有害垃圾_指甲油', 
    '有害垃圾_杀虫剂', '有害垃圾_温度计', '有害垃圾_滴眼液瓶', '有害垃圾_玻璃灯管', '有害垃圾_电池', '有害垃圾_电池板', '有害垃圾_碘伏空瓶', '有害垃圾_红花油', 
    '有害垃圾_纽扣电池', '有害垃圾_胶水', '有害垃圾_药品包装', '有害垃圾_药片', '有害垃圾_药膏', '有害垃圾_蓄电池', '有害垃圾_血压计']

  args = parser()
  img, results = run(args)
  top_k = results.argsort()[-5:][::-1]
  if results[top_k[0]] > 0.6:
    return img, results[top_k[0]], labels[top_k[0]]
  elif results[top_k[0]] < 0:
    return img, 0, '0'
  else:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./tmp.jpg', img)
    response = Sample.main(args)
    category = response.body.data.elements[0].category
    rubbish = response.body.data.elements[0].rubbish
    category_score = response.body.data.elements[0].category_score
    rubbish_score = response.body.data.elements[0].rubbish_score
    if category_score > 0.5:
      if rubbish_score > 0.5:
        return img, (category_score+rubbish_score)/2, category+'_'+rubbish
      else:
        return img, category_score, category+'_'+rubbish
    else:
      return img, 0, '0'

if __name__ == '__main__':
  while True:
    img1, score1, result1 = output()
    print('{:08.6f}: {}'.format(float(score1), result1))
    if cv2.waitKey(20)& 0xFF == ord('q'):
      break
