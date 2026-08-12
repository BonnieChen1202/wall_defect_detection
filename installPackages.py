import os

pkgs = ['ultralytics==8.2.59','PyQt5==5.15.2','pyqt5-tools==5.15.2.3.1','dill==0.3.8','numpy==1.23.0','huggingface_hub==0.23.4']

for each in pkgs:
    cmd_line = f"pip install {each} -i https://pypi.tuna.tsinghua.edu.cn/simple"
    os.system(cmd_line)
