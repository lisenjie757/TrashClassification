# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import time
import hashlib
import hmac
import random
import json

from main import output

# 这个就是我们在阿里云注册产品和设备时的三元组啦
# 把我们自己对应的三元组填进去即可
options = {
  'productKey': 'a1jrjL3dvSp',
  'deviceName': 'GarbageImage',
  'deviceSecret': 'beb4362997f8b636a1cb0b3cd3b6485b',
  'regionId': 'cn-shanghai'
}

HOST = options['productKey'] + '.iot-as-mqtt.'+options['regionId']+'.aliyuncs.com'
PORT = 1883
PUB_TOPIC = "/sys/" + options['productKey'] + "/" + options['deviceName'] + "/thing/event/property/post";

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    # print("Connected with result code "+str(rc))
    pass

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(str(msg.payload))

def hmacsha1(key, msg):
    return hmac.new(key.encode(), msg.encode(), hashlib.sha1).hexdigest()

def getAliyunIoTClient():
    timestamp = str(int(time.time()))
    CLIENT_ID = "paho.py|securemode=3,signmethod=hmacsha1,timestamp="+timestamp+"|"
    CONTENT_STR_FORMAT = "clientIdpaho.pydeviceName"+options['deviceName']+"productKey"+options['productKey']+"timestamp"+timestamp
    # set username/password.
    USER_NAME = options['deviceName']+"&"+options['productKey']
    PWD = hmacsha1(options['deviceSecret'],CONTENT_STR_FORMAT)
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=False)
    client.username_pw_set(USER_NAME, PWD)
    return client


if __name__ == '__main__':
    client = getAliyunIoTClient()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(HOST, 1883, 300)
    client.loop_start()

    while True:
        # 为了更方便判断是那种垃圾，所以上传两个属性，即识别结果与识别结果的索引，图片先不需要上传
        # 1：可回收物
        # 2：有害垃圾
        # 3. 厨余垃圾
        # 4. 其他垃圾
        img, score, result = output()
        result_index = 0
        # 对结果进行正则化解析，提取到“可回收物”
        result_category = result.split("_")
        if result_category[0] in ["可回收物","可回收垃圾"]:
            result_index = 1
        elif result_category[0] in ["有害垃圾"]:
            result_index = 2
        elif result_category[0] in ["厨余垃圾","湿垃圾"]:
            result_index = 3
        elif result_category[0] in ["其他垃圾","干垃圾"]:
            result_index = 4
            
        payload_json = {
            'id': int(time.time()),
            'params': {
                "GarbageIndex": result_index,
                "GarbageResult": result
            },
            'method': "thing.event.property.post"
        }
        print('send data to iot server: ' + str(payload_json))
        client.publish(PUB_TOPIC, payload=str(payload_json), qos=0)
        time.sleep(1)

    client.loop_stop()
