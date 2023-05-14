import functools

import os
import cv2
import sys

from PIL import Image
import torchvision.transforms as transforms
# fix conflicts between qt5 and cv2
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from model.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from util.tensor_util import pad_divide_by
from model.aggregate import aggregate_wbg
from dataset.range_transform import im_normalization
from util.jandf import db_eval_iou, db_eval_boundary
from numpy import mean
import pandas as pd

import numpy as np
import torch
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget, QApplication, QTextBrowser, QProgressBar, QFormLayout, QFrame, QComboBox, QCheckBox,
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, QSpinBox, QFileDialog,
                             QPlainTextEdit, QVBoxLayout, QSizePolicy, QButtonGroup, QSlider, QShortcut, QRadioButton)

from PyQt5.QtGui import QPixmap, QFont, QKeySequence, QImage, QTextCursor, QIcon
from PyQt5.QtCore import Qt, QTimer
from davisinteractive.utils.scribbles import scribbles2mask
from model.network import XMem


class App(QWidget):
    def __init__(self, image_paths, images, pad_gui, s2m, processor_gui, mask_path=None):
        #images：T*H*W
        super().__init__()
        self.pad = pad_gui
        self.images = images
        # formLayout = QFormLayout()
        self.ann_mask_path = None if mask_path is None else sorted([os.path.join(mask_path,i) for i in os.listdir(mask_path)])
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        hbox2 = QHBoxLayout()
        vbox2 = QVBoxLayout()
        # self.initialized = False
        self.device = torch.device("cuda")
        self.processor = processor_gui
        self.s2m = s2m
        self.objectids = 1
        self.k = 1
        self.num_frames = images.shape[0]#总帧数
        self.interact_image_idx = 0
        self.interactived_idx = []
        self.jandf_val = []
        self.j_val = []
        self.f_val = []
        # self.height = 20
        # self.width = 20
        self.imagespath = sorted(self.get_imagespath(image_paths))#更新self的h w


        self.setWindowTitle('GUI Demo')
        self.current_frame_num = 0#当前传播帧
        # self.mask = np.full_like(images, -1, dtype=np.int)
        self.current_h, self.current_w = images[0].shape[1:]#480p的h和w
        self.current_h, self.current_w = self.current_h - self.pad[2] - self.pad[3], self.current_w-self.pad[0] - self.pad[1]

        self.video_mask = None
        # self.video_mask = np.zeros((self.num_frames,self.current_h, self.current_w))#存放视频的mask np类型 0-1
        # self.images = images.to(self.device)
        # self.current_h, self.current_w = self.get_new_hw(self.height,self.width)
        # self.processor =

        self.play_video_flag = False

        self.current_mask = np.full((self.current_h, self.current_w), -1, dtype=np.int)#创建填充画布
        self.current_mask_qpix = Image.fromarray(np.uint8(self.current_mask)).toqpixmap()

        self.setGeometry(800, 100, self.current_w + 400, self.current_h*2+100)
        self.main_canvas = QLabel()
        self.main_canvas.setFixedSize(self.current_w, self.current_h)
        self.main_canvas.setPixmap(self.current_mask_qpix)#放入图片
        self.main_canvas.setMinimumSize(100, 100)

        self.show_mask_label = QLabel()
        self.show_mask_label.setFixedSize(self.current_w, self.current_h)
        self.show_mask_label.setPixmap(self.current_mask_qpix)  # 放入图片
        self.show_mask_label.setMinimumSize(100, 100)

        self.button1 = QPushButton('显示图片')
        self.button1.clicked.connect(self.openimageone)
        print(self.button1.size())
        # self.button1.setFixedSize()
        self.button2 = QPushButton('清理涂鸦')
        self.button2.clicked.connect(self.clear_current_mask)
        # self.button2.setFixedSize()
        self.button3 = QPushButton('播放视频')
        self.button3.clicked.connect(self.play_video)
        # self.button3.setFixedSize()
        self.button4 = QPushButton('展示掩码')
        self.button4.clicked.connect(self.showmask)
        # self.button4.setFixedSize()
        self.button5 = QPushButton('传播掩码')
        self.button5.clicked.connect(self.make_video_mask)
        self.button6 = QPushButton('保存结果')
        self.button6.clicked.connect(self.save_result)

        # self.button5 = QPushButton('重载视频')
        # self.button5.clicked.connect()
        self.s = QSlider(Qt.Horizontal)  # 水平方向
        self.s.setMinimum(0)  # 设置最小值
        self.s.setMaximum(self.num_frames-1)  # 设置最大值
        self.s.setSingleStep(1)  # 设置步长值
        #查看值value()	获取滑动条控件的值
        # self.s.setValue(30)  # 设置当前值
        self.s.setTickPosition(QSlider.TicksBelow)  # 设置刻度位置，在下方
        self.s.setTickInterval(1)  # 设置刻度间隔
        self.s.valueChanged.connect(self.valueChange)
        self.s.setMaximumWidth(self.current_w)

        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_finish)

        self.textBrowser = QTextBrowser()

        self.res_label = QLabel()
        self.res_label.setAlignment(Qt.AlignCenter)
        self.res_label.setText(f"{self.current_frame_num}/{self.num_frames-1}")
        self.res_label.setFont(QFont("Roman times", 20, QFont.Bold))
        # self.res_label.setFixedSize(200, self.current_h)

        # self.progressBar = QProgressBar()
        # self.progressBar.setMinimum(1)  # 设置进度条最小值
        # self.progressBar.setMaximum(self.num_frames)

        # self.result_info = QTextEdit()
        hbox.addWidget(self.button1)
        hbox.addWidget(self.button2)
        hbox.addWidget(self.button3)
        hbox.addWidget(self.button4)
        hbox.addWidget(self.button5)
        hbox.addWidget(self.button6)
        hbox.addStretch(0)
        # hbox.setMaximumWidth(self.current_w)
        # hbox.addWidget(self.button6)
        vbox.addLayout(hbox)
        vbox.addWidget(self.main_canvas)
        vbox.addWidget(self.show_mask_label)
        vbox.addWidget(self.s)
        # vbox.addWidget(self.progressBar)
        # formLayout.addWidget(hbox)
        # formLayout.(self.main_canvas)
        vbox2.addWidget(self.res_label)
        vbox2.addWidget(self.textBrowser)
        hbox2.addLayout(vbox)
        hbox2.addLayout(vbox2)
        self.setLayout(hbox2)

        self.show()

    def valueChange(self):#拖动事件绑定
        self.current_frame_num = self.s.value()
        self.res_label.setText(f"{self.current_frame_num}/{self.num_frames-1}")  # 标签展示
        from util.jandf import db_eval_iou, db_eval_boundary
        # self.timer.stop()
        self.openimage(self.imagespath[self.current_frame_num])#主画布显示图像
        if self.video_mask is not None:
            self.current_mask = np.uint8(self.video_mask[self.current_frame_num])
            self.textBrowser.append(f"第{self.current_frame_num}帧：j_val:{self.j_val[self.current_frame_num]} f_val:{self.f_val[self.current_frame_num]}\n")
            # self.res_label
            # self.res_label.setText(f"{self.current_frame_num}/{self.num_frames - 1}")
            print(self.current_frame_num)
            self.show_mask_label.setPixmap(
                Image.fromarray(np.uint8(self.video_mask[self.current_frame_num])*255).toqpixmap()) #mask画布显示掩码

    def save_result(self):
        list2 = zip(self.j_val,self.f_val)
        names = ['Jccard', 'F-score']
        test = pd.DataFrame(columns=names, data=list2)
        basename= os.path.basename(os.path.dirname(self.imagespath[0]))
        test.to_csv(f'./saves/{basename}_res.csv')
        print(test)


    # def stop_video(self):
    #     self.play_video_flag = False

    def cut_pad(self,pad_mask):
        if self.pad[2] + self.pad[3] > 0:
            pad_mask = pad_mask[:, self.pad[2]:-self.pad[3], :]
        if self.pad[0] + self.pad[1] > 0:
            pad_mask = pad_mask[:, :, self.pad[0]:-self.pad[1]]
        return pad_mask

    def make_video_mask(self):
        self.j_val = []
        self.f_val = []
        # scribble = {}
        # scribble['mask'] = self.mask_proba
        # scribble['idx'] = self.interactived_idx[-1]
        print(f'交互帧{self.interactived_idx[-1]}')
        out_masks = self.processor.interact(self.mask_proba, self.interactived_idx[-1])

        self.current_mask = np.full((self.current_h, self.current_w), -1, dtype=np.int)
        del self.mask_dim,self.mask_proba
        self.video_mask = self.cut_pad(out_masks)

        #计算j&f
        print(self.video_mask.shape)
        for i in range(self.video_mask.shape[0]):
            label_mask = np.uint8(self.video_mask[i]) * 255  # np,255
            ann_mask = cv2.imread(self.ann_mask_path[i], 0)
            j_val = db_eval_iou(label_mask, ann_mask)
            f_val = db_eval_boundary(label_mask, ann_mask)
            self.j_val.append(j_val)
            self.f_val.append(f_val)
            self.jandf_val.append((j_val + f_val) / 2)
        # print(self.video_mask.size())
        # print(np.unique(np.uint8(self.video_mask[self.current_frame_num])))
        self.textBrowser.append('interact finish!\n')

    def showmask(self):

        img = Image.fromarray(self.current_mask.astype('uint8')*255)

        # 保存图像到文件
        img.save(r'C:\Users\chen\Desktop\test23\my_image.png')

        mask,_ = pad_divide_by(torch.tensor(self.current_mask),16,self.current_mask.shape[-2:])#480 864
        # save_images_from_nparray(self.cut_pad(np.uint8(mask.unsqueeze(0).cpu().numpy()) * 255),
        #                          r'C:\Users\chen\Desktop\gui_res\scribble')
        self.scribbeltomask_gui(mask.numpy())
        self.show_mask_label.setPixmap(Image.fromarray(np.uint8(self.mask_dim.cpu().numpy()) * 255).toqpixmap())
        # self.show_mask_label.setPixmap(Image.fromarray(np.uint8(self.current_mask.cpu().numpy())*255).toqpixmap())
        # save_images_from_nparray(np.uint8(self.mask_dim.unsqueeze(0).cpu().numpy()) * 255,
        #                          r'C:\Users\chen\Desktop\gui_res\mask')
        self.textBrowser.insertPlainText('show down\n')

    # def showvideomask(self):
    #     self.show_mask_label.setPixmap(Image.fromarray(np.uint8(self.video_mask[self.current_frame_num].numpy())*255).toqpixmap())

    def get_imagespath(self,image_paths):
        file_name = os.listdir(image_paths)
        file_path_list = []
        for i in file_name:
            file_path_list.append(os.path.join(image_paths,i))
        # example_img = cv2.imread(file_path_list[0])
        # self.height = example_img.shape[0]
        # self.width = example_img.shape[1]
        # self.num_frames = len(file_path_list)
        return file_path_list

    def timer_finish(self):
        self.s.setValue(self.current_frame_num)  # 设置当前值
        self.openimage(self.imagespath[self.current_frame_num])
        # print(self.current_frame_num)
        # self.show_mask_label.setPixmap(
        #     Image.fromarray(np.uint8(self.mask_dim.squeeze(0).cpu().numpy()) * 255).toqpixmap())
        if self.video_mask is not None:
            self.show_mask_label.setPixmap(
                Image.fromarray(np.uint8(self.video_mask[self.current_frame_num]) * 255).toqpixmap())
        # self.progressBar.setValue(self.current_frame_num)
        if self.play_video_flag:
            if self.current_frame_num != (self.num_frames-1):
                self.current_frame_num += 1
            else:
                self.play_video_flag = False
                self.button3.setText('播放视频')
                self.timer.stop()
                # for i in self.jandf_val:
                #     self.textBrowser.insertPlainText(f'[{i}]\n')
                print(len(self.jandf_val))
                self.textBrowser.append(f'min J&F 第{self.jandf_val.index(min(self.jandf_val))}帧:{min(self.jandf_val)}\n')
                self.textBrowser.append(f'平均Jaccard:{mean(self.j_val)}')
                self.textBrowser.append(f'平均F-score:{mean(self.f_val)}')

    def play_video(self):
        if not self.play_video_flag:
            self.button3.setText('停止视频')
            self.play_video_flag = True
            self.timer.start(50)
        else:
            self.button3.setText('播放视频')
            self.play_video_flag = False
            self.timer.stop()


    def openimage(self,path):
        pix = QPixmap(path)
        pix = pix.scaled(self.current_w, self.current_h, Qt.KeepAspectRatio)
        self.main_canvas.setPixmap(pix)  # 放入图片
        self.update()

    def openimageone(self):
        self.current_frame_num = 0
        pix = QPixmap(self.imagespath[0])
        pix = pix.scaled(self.current_w, self.current_h, Qt.KeepAspectRatio)
        self.main_canvas.setPixmap(pix)  # 放入图片
        self.update()


    def clear_current_mask(self):
        self.current_mask = np.full((self.current_h, self.current_w), -1, dtype=np.int)
        print(torch.cuda.max_memory_allocated())
        # self.video_mask = np.zeros((self.num_frames, self.current_h, self.current_w))
        # self.interactived_idx = []

    # 鼠标点击事件
    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()


    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        self.flag = False
        self.interact_image_idx = self.current_frame_num
        # print(0)
        # plotnp(self.current_mask)

    # 鼠标移动事件
    # 由于加了if self.flag，所以等于只有flag = True,即鼠标点击并移动时，才会继续执行下面的操作
    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()-self.main_canvas.x()
            self.y1 = event.y()-self.main_canvas.y()
            print((self.x1, self.y1))
            if (self.x1 < self.current_w) and (self.y1 < self.current_h):
                self.current_mask[(self.y1, self.x1)] = self.objectids
            self.update()

    # def get_new_hw(self, h, w):#
    #     if w <= 960:
    #         return h, w
    #     scale = w/h
    #     new_h = 480
    #     new_w = int(scale * new_h)
    #     return new_h, new_w

    def get_image_np(self, idx):
        # self.inter_img_np = cv2.imread(self.imagespath[idx])
        # self.inter_img_np = cv2.resize(self.inter_img_np, (self.current_w,self.current_h), interpolation=cv2.INTER_CUBIC)
        # # self.inter_img_np = np.expand_dims(self.inter_img_np.swapaxes(0, 2).swapaxes(1, 2), axis=0)
        # trans1 = transforms.Compose([
        #     transforms.ToTensor(),
        #     im_normalization,
        # ])
        # self.inter_img_np = trans1(self.inter_img_np).unsqueeze(0).to(self.device)
        self.inter_img_np = self.images[idx].unsqueeze(0).to(self.device)

    def scribbeltomask_gui(self, np_mask):
        self.interactived_idx.append(self.current_frame_num)
        kernel = np.ones((3, 3), np.uint8)
        mask = torch.zeros((self.k, 1, self.current_h + self.pad[2] + self.pad[3], self.current_w+self.pad[0] + self.pad[1]), dtype=torch.float32, device=self.device)#16倍数大小
        for ki in range(1, self.k + 1):
            p_srb = (np_mask == ki).astype(np.uint8)
            p_srb = cv2.dilate(p_srb, kernel).astype(np.bool)

            n_srb = ((np_mask != ki) * (np_mask != -1)).astype(np.uint8)
            n_srb = cv2.dilate(n_srb, kernel).astype(np.bool)

            Rs = torch.from_numpy(np.stack([p_srb, n_srb], 0)).unsqueeze(0).float().to(self.device)


            # Use hard mask because we train S2M with such
            self.get_image_np(self.interact_image_idx)
            pad_current_mask, _ = pad_divide_by(torch.tensor(self.current_mask),16,self.current_mask.shape[-2:])
            pad_current_mask = pad_current_mask.numpy()
            a = torch.tensor((pad_current_mask==ki).astype(int)).float().unsqueeze(0).unsqueeze(0).to(self.device)
            inputs = torch.cat([self.inter_img_np, a, Rs], 1)
            mask[ki - 1] = torch.sigmoid(self.s2m(inputs))
        # noforeground_mask = aggregate_wbg(mask, keep_bg=False, hard=True)
        mask = aggregate_wbg(mask, keep_bg=True, hard=True)
        mask_dim =mask.squeeze(1)
        mask_dim = self.cut_pad(mask_dim)
        self.mask_dim = torch.argmax(mask_dim, dim=0)#h w
        self.mask_proba, _ = pad_divide_by(mask, 16, mask.shape[-2:])
        # self.mask_proba = mask
        self.interactived_idx.append(self.interact_image_idx)

def plotnp(img_np):
    mat = np.unique(img_np)
    img_a = np.where(img_np > mat[0], 255, 0)
    plt.imshow(img_a,cmap='gray')
    plt.show()

def save_images_from_nparray(np_array, folder_path):
    """
    保存numpy数组为n张单通道图像
    :param np_array: numpy数组，维度为(n, h, w)
    :param folder_path: 保存图像的文件夹路径
    """
    for i in range(np_array.shape[0]):
        img = np_array[i, :, :]
        img = img.astype(np.uint8)
        img_path = f"{folder_path}/{i:04d}.png"
        cv2.imwrite(img_path, img)

from argparse import ArgumentParser
from model.fusion_net import FusionNet
from dataset.gui_read_images import get_tensor_images
from inference_core import InferenceCore


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_objects', default=1)
    parser.add_argument('--fusion_model', default='saves/fusion.pth')
    parser.add_argument('--s2m_model', default='saves/s2m.pth')
    parser.add_argument('--davis', default='../DAVIS')
    parser.add_argument('--output', default='../output')
    parser.add_argument('--save_mask', action='store_true')

    parser.add_argument('--model', default='./saves/XMem.pth')

    parser.add_argument('--save_all', action='store_true',
                        help='Save all frames. Useful only in YouTubeVOS/long-time video', )

    parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')

    # Long-term memory options
    parser.add_argument('--disable_long_term', default=True, action='store_true')
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements',
                        help='LT_max in paper, increase if objects disappear for a long time',
                        type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int,
                        default=-1)

    # Multi-scale options
    parser.add_argument('--save_scores', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--size', default=480, type=int,
                        help='Resize the shorter side to this size. -1 to use original resolution. ')

    args = parser.parse_args()
    config = vars(args)
    config['enable_long_term'] = False
    config['enable_long_term_count_usage'] = False

    torch.autograd.set_grad_enabled(False)

    # images = {}
    # num_objects = {}
    # Loads all the images

    xmem = XMem(config, args.model).cuda().eval()
    if args.model is not None:
        model_weights = torch.load(args.model)
        xmem.load_weights(model_weights, init_as_zero_if_needed=True)

    s2m_model_path = r'saves/s2m.pth'
    s2m_saved = torch.load(s2m_model_path)
    s2m_model = S2M().cuda().eval()
    s2m_model.load_state_dict(s2m_saved)

    fusion_saved = torch.load(args.fusion_model)
    fusion_model = FusionNet().cuda().eval()
    fusion_model.load_state_dict(fusion_saved)

    path = r'F:\data_VATUV_ALL\data480p_50\JPEGImages\480p\truck_004'
    # path = r'F:\data_VATUV_ALL\data480p_50\JPEGImages/480p/bike_005'
    # path = r'/data/cjq/data480p_50/JPEGImages/480p/bike_005/'
    images = get_tensor_images(path)
    images, pad = pad_divide_by(images, 16, images.shape[-2:])
    print(images.shape)
    processor = InferenceCore(config, xmem, fusion_model, images.unsqueeze(0), config['num_objects'], mem_profile=0, device='cuda:0')
    # processor = DAVISProcessor(config, xmem, fusion_model, s2m_model, images.unsqueeze(0), config['num_objects'])
    mask_path = r'F:\data_VATUV_ALL\data480p_50\Annotations\480p\truck_004'
    # mask_path = r'/data/cjq/data480p_50/Annotations/480p/bike_005'
    # mask_path = r'F:\data_VATUV_ALL\data480p_50\Annotations/480p/bike_005'
    app = QApplication(sys.argv)
    ex = App(path, images, pad, s2m_model, processor, mask_path)
    sys.exit(app.exec_())




