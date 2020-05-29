import requests
import json
import numpy as np


def get_token():
    # 请求api地址
    url = "http://192.168.1.201:8080/login"
    # 请求参数
    body ={"account": "admin","password": "admin"}
    headers = {"Content-Type": "application/json;charset=utf-8"}
    # 执行请求
    response = requests.post(url, headers=headers, data=json.dumps(body))
    return json.loads(response.text)['data']['token']

def get_real_value(point):
    url="http://192.168.1.201:8080/openPlant/getMultiplePoint"
    headers = {"Content-Type": "application/json;charset=utf-8","token":get_token()}
    para={"pointCodes":point}
    # 执行请求
    response = requests.get(url, headers=headers, params=para)
    return json.loads(response.text)["data"][str(point)]["value"]


def get_history_value(point,st,et,interval):
    url="http://192.168.1.201:8080/openPlant/getMultiplePointHistorys"
    headers = {"Content-Type": "application/json;charset=utf-8","token":get_token()}
    para={"pointCodes":point,"startTime":st,"endTime":et,"interval":interval}
    point_array=point.split(",")
    # 执行请求
    response = requests.get(url, headers=headers, params=para)
    value_list=json.loads(response.text)["data"]
    value_array=[]
    for point_name in point_array:
        value_line=[]
        value_line_list=value_list[point_name]
        for item in value_line_list:
            value_line.append(item["value"])
        value_array.append(value_line)
    return np.array(value_array)

value=get_real_value("W3.ZK_UNIT1.50LAC30CT011")
history_value=get_history_value("W3.ZK_UNIT1.50LAC30CT011","2019-7-12 11:40:00","2019-7-12 11:55:00",5000)
print(value)