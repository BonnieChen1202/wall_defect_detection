#coding:utf-8
from ultralytics import YOLO
import cv2
#coding:utf-8
from ultralytics import YOLO
import cv2
import Config  # 新增：引入配置文件，统一类别
# 所需加载的模型目录
path = 'models/best.pt'
# 需要检测的图片地址
img_path = "TestFiles/dc9370592f74e1459f9249a7539da11f.jpg"

# 加载预训练模型
model = YOLO(path, task='detect')

# 检测图片
results = model(img_path, conf=0.35, iou=0.75)
# 打印外墙缺陷类别（替换原焊缝类别）
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    print(f"检测到楼宇外墙缺陷：{Config.CH_names[cls_id]}, 置信度：{box.conf[0]:.2f}")

res = results[0].plot()
cv2.imshow("楼宇外墙缺陷检测", res)  # 窗口标题修改
cv2.waitKey(0)
cv2.destroyAllWindows()