# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                             QMessageBox, QHeaderView, QTableWidgetItem,
                             QAbstractItemView, QDialog, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton)
import sys
import os
from ultralytics import YOLO
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal,QCoreApplication
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import numpy as np
import torch
import csv
import warnings
import mss
import time
import os

warnings.filterwarnings('ignore')


# ==========================================
# 👉 软著必备：登录权限认证界面
# ==========================================
class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('智巡未来 - 墙体缺陷检测系统')
        self.resize(350, 200)
        # 去掉右上角的问号，保留关闭按钮
        self.setWindowFlags(Qt.WindowCloseButtonHint)

        # 1. 账号输入部分
        user_layout = QHBoxLayout()
        self.user_label = QLabel('管理员账号:')
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("默认: admin")
        user_layout.addWidget(self.user_label)
        user_layout.addWidget(self.user_input)

        # 2. 密码输入部分
        pwd_layout = QHBoxLayout()
        self.pwd_label = QLabel('系统密码:')
        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("默认: 123456")
        self.pwd_input.setEchoMode(QLineEdit.Password)  # 密码掩码显示为黑点
        pwd_layout.addWidget(self.pwd_label)
        pwd_layout.addWidget(self.pwd_input)

        # 3. 登录按钮
        self.login_btn = QPushButton('安全登录')
        self.login_btn.setMinimumHeight(40)
        self.login_btn.setStyleSheet("background-color: #1f77b4; color: white; font-weight: bold; font-size: 14px;")
        self.login_btn.clicked.connect(self.check_login)

        # 4. 总体垂直布局
        main_layout = QVBoxLayout()
        main_layout.addSpacing(20)
        main_layout.addLayout(user_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(pwd_layout)
        main_layout.addSpacing(20)
        main_layout.addWidget(self.login_btn)

        self.setLayout(main_layout)

    def check_login(self):
        """校验账号密码"""
        username = self.user_input.text().strip()
        password = self.pwd_input.text().strip()

        # 软著申请直接写死账号密码即可
        if username == 'admin' and password == '123456':
            self.accept()  # 验证成功，关闭对话框并继续
        else:
            QMessageBox.warning(self, '认证失败', '账号或密码错误，请重新输入！')
            self.pwd_input.clear()


# ==========================================
class MainWindow(QMainWindow):
    def __init__(self, parent=None):

        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow() # 加载画好的界面
        self.ui.setupUi(self)

        # ==========================================
        # 👉 新增：强制解锁窗口大小限制
        self.setMinimumSize(800, 600)  # 设置窗口能缩到的最小尺寸
        self.setMaximumSize(16777215, 16777215)  # 打破最大尺寸限制（Qt的默认无限大）
        # ==========================================

        self.initMain() # 初始化参数、模型、界面样式
        self.signalconnect()   # 绑定按钮点击事件

        self.sct = mss.mss()
        # 这里定义你要截取的 QuickTime 窗口在电脑屏幕上的坐标和大小
        # ⚠️ 注意：你需要根据实际情况微调这四个数字！
        self.monitor = {"top": 30, "left": 0, "width": 600, "height": 300}

        self.is_live_open = False  # 记录实时截屏是否开启
        self.capture_mode = 'none'  # 标记当前是读视频、摄像头还是读屏幕

        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)
        # 默认检测参数
        self.conf = 0.40   #设置检测置信度值
        self.iou = 0.45     #设置检测IOU值

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img) # 打开图片
        self.ui.comboBox.activated.connect(self.combox_change) # 打开视频
        self.ui.VideoBtn.clicked.connect(self.vedio_show) # 打开视频
        self.ui.CapBtn.clicked.connect(self.camera_show)# 打开摄像头(实时
        self.ui.SaveBtn.clicked.connect(self.save_detect_result)# 保存结果
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)
        self.ui.doubleSpinBox.valueChanged.connect(self.conf_value_change)
        self.ui.doubleSpinBox_2.valueChanged.connect(self.iou_value_change)
        self.ui.tableWidget.cellClicked.connect(self.on_cell_clicked)

    def live_show(self):
        self.is_live_open = not self.is_live_open

        if self.is_live_open:
            # 如果开启了其他模式（如视频或摄像头），先关掉它们
            if self.is_camera_open:
                self.camera_show()
            if self.cap and self.cap.isOpened():
                self.video_stop()
                self.ui.VideoBtn.setText('打开视频')

            self.ui.CapBtn.setText('关闭实时')
            self.capture_mode = 'screen'  # 设置为截屏模式

            # 清空表格和下拉框
            self.ui.tableWidget.setRowCount(0)
            self.ui.tableWidget.clearContents()
            self.ui.comboBox.clear()
            self.ui.comboBox.setDisabled(True)

            # 开启定时器，开始高速刷新
            self.timer_camera.start(30)  # 30毫秒刷新一次，约等于 30 帧/秒
            try:
                self.timer_camera.timeout.disconnect()
            except:
                pass
            self.timer_camera.timeout.connect(self.open_frame)

        else:
            self.ui.CapBtn.setText('实时检测')
            self.timer_camera.stop()
            self.capture_mode = 'none'
            self.ui.label_show.clear()
        #点哪个按钮，就执行哪个函数，这是 PyQt 的核心机制

    def initMain(self):
        self.show_width = 770 # 图片显示区域宽度
        self.show_height = 480# 图片显示区域高度

        # 新增：创建楼宇外墙检测专属保存文件夹
        if not os.path.exists(Config.save_path):
            os.makedirs(Config.save_path)

        self.org_path = None
        self.is_camera_open = False
        self.cap = None
        self.device = 0 if torch.cuda.is_available() else 'cpu'# 自动用GPU/CPU

        # 加载检测yolo模型
        self.model = YOLO(Config.model_path, task='detect') # 预先推理一次，加快后续检测速度
        self.model(np.zeros((48, 48, 3)).astype(np.uint8), device=self.device)    # 表格样式、CSV文件初始化...

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 更新视频图像
        self.timer_camera = QTimer()

        # 更新检测信息表格
        # self.timer_info = QTimer()
        # 保存视频
        self.timer_save_video = QTimer()

        # 表格样式设置
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(40)# 表格行高固定，每一行高度 = 40 像素，不会自动拉伸
        self.ui.tableWidget.setColumnWidth(0, 80)  # 设置列宽
        self.ui.tableWidget.setColumnWidth(1, 200)
        self.ui.tableWidget.setColumnWidth(2, 80)
        self.ui.tableWidget.setColumnWidth(3, 150)
        self.ui.tableWidget.setColumnWidth(4, 90)
        self.ui.tableWidget.setColumnWidth(5, 230)
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 表格铺满
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        # self.ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替

        # 检查csv文件是否存在，如果不存在则创建
        self.csv_header = ['文件路径', '目标编号', '类别', '置信度', '坐标位置']
        if not os.path.exists(Config.csv_save_path):
            with open(Config.csv_save_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_header)
                writer.writeheader()

        # ==========================
        # 实时检测去重冷却机制
        # key: cls_id（类别编号）
        # value: 上次保存该类别缺陷的时间戳
        # 同一类别在 save_cooldown 秒内只保存一次，避免30ms一帧狂刷磁盘
        # ==========================
        self.last_save_time = {}   # {cls_id: timestamp}
        self.save_cooldown = 5     # 冷却时间（秒），可按需调整

    def conf_value_change(self):
        # 改变置信度值
        cur_conf = round(self.ui.doubleSpinBox.value(), 2)
        self.conf = cur_conf

    def iou_value_change(self):
        # 改变iou值
        cur_iou = round(self.ui.doubleSpinBox_2.value(), 2)
        self.iou = cur_iou

    def open_img(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.VideoBtn.setText('打开视频')
            self.cap = None

        # 弹出的窗口名称：'打开图片'
        # 默认打开的目录：'./'
        # 只能打开*.jpg *.jpeg *.png *.bmp结尾的图片文件
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return

        self.ui.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        # 目标检测
        t1 = time.time()
        self.results = self.model(self.org_path, conf=self.conf, iou=self.iou)[0]#用 YOLO 模型检测
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)
           # 解析检测结果
        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]
        self.id_list = [i for i in range(len(self.location_list))]

        if self.ui.show_labels_and_conf.isChecked():
            now_img = self.results.plot()#把检测结果画到图上
        else:
            now_img = self.results.plot(labels=False, conf=False)

        self.draw_img = now_img    # 绘制完检测框

        # ==========================
        # 自动保存缺陷图片 (优化版：单图去重，取最高置信度)
        # ==========================
        if hasattr(self, "results") and self.results is not None:
            boxes = self.results.boxes
            if len(boxes) > 0:  # 确保检测到了至少一个缺陷
                # YOLO 默认按置信度降序排列，boxes[0] 就是全图置信度最高的那个缺陷
                best_box = boxes[0]

                cls_id = int(best_box.cls)
                class_name = Config.CH_names[cls_id]
                conf = float(best_box.conf)

                # 整个画面只保存一次，用全场"最严重"（置信度最高）的缺陷来命名文件
                self.auto_save_defect(now_img.copy(), class_name, conf)


        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img,(self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.CH_names[id]+ '_'+ str(index) for index,id in enumerate(self.cls_list)]
        choose_list = choose_list + target_names

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)

        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
        #   默认显示第一个目标框坐标
        #   设置坐标位置值
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, self.id_list, path=self.org_path)


    def detact_batch_imgs(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.VideoBtn.setText('打开视频')
            self.cap = None
        directory = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")  # 起始路径
        if not directory:
            return
        self.ui.comboBox.setDisabled(False)
        self.org_path = directory
        img_suffix = ['jpg','png','jpeg','bmp']
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory,file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                img_path = full_path
                self.org_img = tools.img_cvread(img_path)
                # 目标检测
                t1 = time.time()
                self.results = self.model(img_path,conf=self.conf, iou=self.iou)[0]
                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.ui.time_lb.setText(take_time_str)

                location_list = self.results.boxes.xyxy.tolist()
                self.location_list = [list(map(int, e)) for e in location_list]
                cls_list = self.results.boxes.cls.tolist()
                self.cls_list = [int(i) for i in cls_list]
                self.conf_list = self.results.boxes.conf.tolist()
                self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
                self.id_list = [i for i in range(len(self.location_list))]

                if self.ui.show_labels_and_conf.isChecked():
                    now_img = self.results.plot()
                else:
                    now_img = self.results.plot(labels=False,conf=False)

                self.draw_img = now_img

                # ==========================
                # 👉 新增：批量检测时的自动保存缺陷图片 (带框、去重)
                # ==========================
                if hasattr(self, "results") and self.results is not None:
                    boxes = self.results.boxes
                    if len(boxes) > 0:  # 确保检测到了至少一个缺陷
                        # YOLO 默认按置信度降序排列，boxes[0] 就是全图置信度最高的那个缺陷
                        best_box = boxes[0]

                        cls_id = int(best_box.cls)
                        class_name = Config.CH_names[cls_id]
                        conf = float(best_box.conf)

                        # 整个画面只保存一次，用全场"最严重"（置信度最高）的缺陷来命名文件
                        self.auto_save_defect(now_img.copy(), class_name, conf)


                # 获取缩放后的图片尺寸
                self.img_width, self.img_height = self.get_resize_size(now_img)
                resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)

                # 目标数目
                target_nums = len(self.cls_list)
                self.ui.label_nums.setText(str(target_nums))

                # 设置目标选择下拉框
                choose_list = ['全部']
                target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
                choose_list = choose_list + target_names

                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(choose_list)

                if target_nums >= 1:
                    self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                    self.ui.label_conf.setText(str(self.conf_list[0]))
                    #   默认显示第一个目标框坐标
                    #   设置坐标位置值
                    self.ui.label_xmin.setText(str(self.location_list[0][0]))
                    self.ui.label_ymin.setText(str(self.location_list[0][1]))
                    self.ui.label_xmax.setText(str(self.location_list[0][2]))
                    self.ui.label_ymax.setText(str(self.location_list[0][3]))
                else:
                    self.ui.type_lb.setText('')
                    self.ui.label_conf.setText('')
                    self.ui.label_xmin.setText('')
                    self.ui.label_ymin.setText('')
                    self.ui.label_xmax.setText('')
                    self.ui.label_ymax.setText('')

                self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, self.id_list, path=img_path)

                self.ui.tableWidget.scrollToBottom()
                QApplication.processEvents()  #刷新页面

    def draw_rect_and_tabel(self, results, img):
        now_img = img.copy()
        location_list = results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        for loacation, type_id, conf in zip(self.location_list, self.cls_list, self.conf_list):
            type_id = int(type_id)
            color = self.colors(int(type_id), True)
            now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], self.fontC, color)

        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))
        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)
        return now_img

    def combox_change(self):
        com_text = self.ui.comboBox.currentText()
        if com_text == '全部':
            cur_box = self.location_list
            if self.ui.show_labels_and_conf.isChecked():
                cur_img = self.results.plot()
            else:
                cur_img = self.results.plot(labels=False, conf=False)
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
        else:
            index = int(com_text.split('_')[-1])
            cur_box = [self.location_list[index]]
            if self.ui.show_labels_and_conf.isChecked():
                cur_img = self.results[index].plot()
            else:
                cur_img = self.results[index].plot(labels=False, conf=False)
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[index]])
            self.ui.label_conf.setText(str(self.conf_list[index]))

        # 设置坐标位置值
        self.ui.label_xmin.setText(str(cur_box[0][0]))
        self.ui.label_ymin.setText(str(cur_box[0][1]))
        self.ui.label_xmax.setText(str(cur_box[0][2]))
        self.ui.label_ymax.setText(str(cur_box[0][3]))

        resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.clear()
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)


    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Image files (*.avi *.mp4 *.wmv *.mkv)")
        if not file_path:
            return None
        self.org_path = file_path
        return file_path

    def video_start(self):
        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        # 清空下拉框
        self.ui.comboBox.clear()

        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def tabel_info_show(self, locations, clses, confs, target_ids, path=None):
        path = path
        if self.is_camera_open:
            path = 'Camera'

        for location, cls, conf, target_id in zip(locations, clses, confs, target_ids):
            row_count = self.ui.tableWidget.rowCount()  # 返回当前行数(尾部)
            self.ui.tableWidget.insertRow(row_count)  # 尾部插入一行
            item_id = QTableWidgetItem(str(row_count+1))  # 序号
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中
            item_path = QTableWidgetItem(str(path))  # 路径
            # item_path.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_cls = QTableWidgetItem(str(Config.CH_names[cls]))
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_location = QTableWidgetItem(str(location)) # 目标框位置
            # item_location.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_target_id = QTableWidgetItem(str(target_id))
            item_target_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_target_id)
            self.ui.tableWidget.setItem(row_count, 3, item_cls)
            self.ui.tableWidget.setItem(row_count, 4, item_conf)
            self.ui.tableWidget.setItem(row_count, 5, item_location)
        self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()
        # self.timer_info.stop()

    def open_frame(self):
        ret = False
        now_img = None

        # ==========================
        # 👉 核心修改：数据源分流
        # ==========================
        if getattr(self, 'capture_mode', 'none') == 'screen':
            # 1. 实时截屏模式
            sct_img = self.sct.grab(self.monitor)
            now_img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            ret = True
        elif self.cap is not None:
            # 2. 原有的读取视频模式
            ret, now_img = self.cap.read()

        if ret:
            t1 = time.time()
            results = self.model(now_img, conf=self.conf, iou=self.iou)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.ui.time_lb.setText(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
            self.id_list = [i for i in range(len(self.location_list))]

            if self.ui.show_labels_and_conf.isChecked():
                now_img = results.plot()
            else:
                now_img = results.plot(labels=False, conf=False)

            # ==========================
            # 👉 自动保存（实时截屏 + 视频模式，带冷却去重）
            #
            # 策略：对画面中置信度最高的那个缺陷（boxes[0]）判断：
            #   - 若该类别距上次保存已超过 save_cooldown 秒 → 保存，并更新时间戳
            #   - 否则跳过，不保存
            # 同一个持续出现的缺陷最多每 N 秒保存一张，不会重复刷盘。
            # ==========================
            if getattr(self, 'capture_mode', 'none') == 'screen' or self.cap is not None:
                boxes = results.boxes
                if len(boxes) > 0:
                    best_box = boxes[0]
                    cls_id = int(best_box.cls)
                    class_name = Config.CH_names[cls_id]
                    conf_val = float(best_box.conf)

                    now_time = time.time()
                    last_time = self.last_save_time.get(cls_id, 0)

                    if now_time - last_time >= self.save_cooldown:
                        # 更新该类别的冷却时间戳，再保存
                        self.last_save_time[cls_id] = now_time
                        self.auto_save_defect(now_img.copy(), class_name, conf_val)

            # 获取缩放后的图片尺寸
            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            # 目标数目
            target_nums = len(self.cls_list)
            self.ui.label_nums.setText(str(target_nums))

            # 设置目标选择下拉框
            choose_list = ['全部']
            target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
            choose_list = choose_list + target_names

            self.ui.comboBox.clear()
            self.ui.comboBox.addItems(choose_list)

            if target_nums >= 1:
                self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                self.ui.label_conf.setText(str(self.conf_list[0]))
                self.ui.label_xmin.setText(str(self.location_list[0][0]))
                self.ui.label_ymin.setText(str(self.location_list[0][1]))
                self.ui.label_xmax.setText(str(self.location_list[0][2]))
                self.ui.label_ymax.setText(str(self.location_list[0][3]))
            else:
                self.ui.type_lb.setText('')
                self.ui.label_conf.setText('')
                self.ui.label_xmin.setText('')
                self.ui.label_ymin.setText('')
                self.ui.label_xmax.setText('')
                self.ui.label_ymax.setText('')

            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, self.id_list, path=self.org_path)

        else:
            if self.cap:
                self.cap.release()
            self.timer_camera.stop()
            self.ui.VideoBtn.setText('打开视频')
            self.ui.CapBtn.setText('实时检测')

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            if self.cap and self.cap.isOpened():
                self.cap.release()
                cv2.destroyAllWindows()

        if self.cap and self.cap.isOpened():
            # 关闭视频
            self.ui.VideoBtn.setText('打开视频')
            self.ui.label_show.setText('')
            self.cap.release()
            cv2.destroyAllWindows()
            self.ui.label_show.clear()
            return

        video_path = self.get_video_path()
        if not video_path:
            return None

        self.ui.VideoBtn.setText('关闭视频')
        self.cap = cv2.VideoCapture(video_path)
        self.last_save_time = {}  # 每次重新打开视频，清空冷却记录
        self.video_start()
        self.ui.comboBox.setDisabled(True)

    def camera_show(self):
        """
        这里原先是打开摄像头的逻辑，改成了'实时截屏'的开关
        """
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.VideoBtn.setText('打开视频')
            self.ui.CapBtn.setText('关闭实时')  # 按钮文字变成关闭
            self.capture_mode = 'screen'  # 👉 告诉 open_frame，现在要读屏幕了！

            # 如果之前打开了视频，先释放掉
            if self.cap and self.cap.isOpened():
                self.cap.release()
                cv2.destroyAllWindows()
            self.ui.comboBox.setDisabled(True)

            # 重置冷却记录，避免上次实时检测的时间戳干扰本次
            self.last_save_time = {}

            # 启动定时器，每 30 毫秒截屏并检测一次
            self.timer_camera.start(30)
            try:
                self.timer_camera.timeout.disconnect()
            except:
                pass
            self.timer_camera.timeout.connect(self.open_frame)
        else:
            self.ui.CapBtn.setText('实时检测')
            self.capture_mode = 'none'
            self.timer_camera.stop()
            self.ui.label_show.clear()

    def get_resize_size(self, img):
        # 直接填满 label_show 整个框
        self.img_width = self.ui.label_show.width()
        self.img_height = self.ui.label_show.height()
        return self.img_width, self.img_height

    # ==========================
    # 安全版：自动保存缺陷图片
    # ==========================
    def auto_save_defect(self, img, class_name, conf):
        try:
            save_dir = "缺陷自动保存"
            os.makedirs(save_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            conf_str = f"{conf:.2f}"

            filename = f"{class_name}_{timestamp}_{conf_str}.png"
            save_path = os.path.join(save_dir, filename)

            # 👉 终极绝杀：放弃 cv2.imwrite，改用 imencode 二进制存入，完美解决中文路径存不进去的问题！
            cv2.imencode('.png', img)[1].tofile(save_path)
            print(f"成功保存带框缺陷图：{filename}")  # 在控制台打印一句提示，确保真的执行了

        except Exception as e:
            print("保存失败：", e)


    def save_detect_result(self):
        """
        保存检测结果
        """
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        if self.cap:
            res = QMessageBox.information(self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',QMessageBox.Yes | QMessageBox.No ,  QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                self.ui.VideoBtn.setText('打开视频')
                com_text = self.ui.comboBox.currentText()
                self.btn2Thread_object = btn2Thread(self.org_path, self.model, com_text,self.conf,self.iou, self.ui.show_labels_and_conf.isChecked())
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
            else:
                return
        else:
            # 创建csv文件
            if not os.path.exists(Config.csv_save_path):
                with open(Config.csv_save_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_header)
                    writer.writeheader()
            if os.path.isfile(self.org_path):
                fileName = os.path.basename(self.org_path)
                name , end_name= fileName.rsplit(".",1)
                save_name = name + '_detect_result.' + end_name
                save_img_path = os.path.join(Config.save_path, save_name)
                # 保存图片
                cv2.imwrite(save_img_path, self.draw_img)
                # 保存csv文件
                self.save_to_csv(self.org_path, self.id_list, self.cls_list, self.conf_list, self.location_list)
                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(save_img_path))
            else:
                img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
                for file_name in os.listdir(self.org_path):
                    full_path = os.path.join(self.org_path, file_name)
                    if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                        name, end_name = file_name.rsplit(".",1)
                        save_name = name + '_detect_result.' + end_name
                        save_img_path = os.path.join(Config.save_path, save_name)
                        self.results = self.model(full_path,conf=self.conf, iou=self.iou)[0]
                        if self.ui.show_labels_and_conf.isChecked():
                            now_img = self.results.plot()
                        else:
                            now_img = self.results.plot(labels=False,conf=False)
                        # 保存图片
                        cv2.imwrite(save_img_path, now_img)

                        # 保存csv文件
                        location_list = self.results.boxes.xyxy.tolist()
                        self.location_list = [list(map(int, e)) for e in location_list]
                        cls_list = self.results.boxes.cls.tolist()
                        self.cls_list = [int(i) for i in cls_list]
                        self.conf_list = self.results.boxes.conf.tolist()
                        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
                        self.id_list = [i for i in range(len(self.location_list))]
                        self.save_to_csv(full_path, self.id_list, self.cls_list, self.conf_list, self.location_list)

                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(Config.save_path))


    def update_process_bar(self,cur_num, total):
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
            return
        if self.progress_bar.isVisible() is False:
            # 点击取消保存时，终止进程
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total *100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()

    def on_cell_clicked(self, row, column):
        """
        鼠标点击表格触发，界面显示当前行内容
        """
        if self.cap:
            # 视频或摄像头不支持表格选择
            return
        img_path = self.ui.tableWidget.item(row, 1).text()
        target_id = int(self.ui.tableWidget.item(row, 2).text())
        now_type = self.ui.tableWidget.item(row, 3).text()
        conf_value = self.ui.tableWidget.item(row, 4).text()
        location_value = eval(self.ui.tableWidget.item(row, 5).text())

        self.ui.type_lb.setText(now_type)
        self.ui.label_conf.setText(str(conf_value))
        self.ui.label_xmin.setText(str(location_value[0]))
        self.ui.label_ymin.setText(str(location_value[1]))
        self.ui.label_xmax.setText(str(location_value[2]))
        self.ui.label_ymax.setText(str(location_value[3]))

        cur_commbox_text = now_type + '_' + str(target_id)

        now_img = tools.img_cvread(img_path)
        # 目标检测
        t1 = time.time()
        self.results = self.model(now_img, conf=self.conf, iou=self.iou)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
        choose_list = choose_list + target_names
        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)
        self.ui.comboBox.setCurrentText(cur_commbox_text)

        # 绘制窗口图片
        if self.ui.show_labels_and_conf.isChecked():
            now_img = self.results[target_id].plot()
        else:
            now_img = self.results[target_id].plot(labels=False, conf=False)
        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))

    def save_to_csv(self, file_path, res_id_list, cls_list, confidence_list, location_list):
        """
        保存检测结果为csv文件格式
        """
        # 写入数据
        with open(Config.csv_save_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_header)
            res_list = []
            for recognition_id, cls, confidence, location in zip(res_id_list, cls_list, confidence_list, location_list):
                data = {
                    '文件路径': file_path,
                    '目标编号': recognition_id,
                    '类别': Config.CH_names[cls],
                    '置信度': confidence,
                    '坐标位置': location
                }
                res_list.append(data)
            writer.writerows(res_list)


class btn2Thread(QThread):
    """
    进行检测后的视频保存
    """
    # 声明一个信号
    update_ui_signal = pyqtSignal(int,int)

    def __init__(self, path, model, com_text,conf,iou,show_label_and_conf):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        self.conf = conf
        self.iou = iou
        self.show_label_and_conf = show_label_and_conf
        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        self.is_running = True  # 标志位，表示线程是否正在运行

    def run(self):
        # VideoCapture方法是cv2库提供的读取视频方法
        cap = cv2.VideoCapture(self.org_path)
        # 设置需要保存视频的格式"xvid"
        # 该参数是MPEG-4编码类型，文件名后缀为.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 设置视频大小
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # VideoWriter方法是cv2库提供的保存视频方法
        # 按照设置的格式来out输出
        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))
        cur_num = 0

        # 确定视频打开并循环读取
        while (cap.isOpened() and self.is_running):
            cur_num += 1
            print('当前第{}帧，总帧数{}'.format(cur_num, total))
            ret, frame = cap.read()
            if ret == True:
                # 检测
                results = self.model(frame,conf=self.conf,iou=self.iou)[0]
                if self.show_label_and_conf:
                    frame = results.plot()
                else:
                    frame = results.plot(labels=False, conf=False)
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
        # 释放资源
        cap.release()
        out.release()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 👉 1. 优先拉起登录界面
    login_window = LoginDialog()

    # 👉 2. 拦截判断：如果登录成功（密码正确）
    if login_window.exec_() == QDialog.Accepted:
        # 3. 登录成功后，才实例化并显示真正的主界面
        win = MainWindow()
        win.show()
        sys.exit(app.exec_())
    else:
        # 4. 如果点击右上角叉号关闭了登录框，则直接退出程序
        sys.exit(0)