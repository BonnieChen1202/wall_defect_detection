#coding:utf-8

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = '/Users/chenjia/Downloads/Attention_Run/weights/best.pt'

# 数据集类别与名称
names = {
    0: 'crack',          # 墙面裂缝
    1: 'peeling',       # 墙面起皮
    2: 'spalling',        # 砖墙脱落

}
# 数据集类别中文
CH_names = ['墙面裂缝','墙面起皮','砖墙脱落']

# csv文件保存路径
csv_save_path = 'save_data/save_detect_data.csv'


# ======================
# 缺陷自动保存文件夹（新增）
# ======================
auto_save_path = "缺陷自动保存"