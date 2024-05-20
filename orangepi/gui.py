import easyocr

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHeaderView,  QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QImage, QPainterPath, QPainter, QColor, QBitmap, QColor, QEnterEvent
from PyQt5.QtCore import QTimer, Qt, QRectF, QSize, QRect
import cv2
from untitled import Ui_MainWindow  

from test import GenerateMeshgrid, IOU, NMS, postprocess

import numpy as np
from rknnlite.api import RKNNLite

import os
import urllib
import traceback
import time
from math import exp

import csv
from datetime import datetime

reader = easyocr.Reader(['en'])

output_dir = './output'

RKNN_MODEL = 'yolo.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = True

CLASSES = ['chip']

meshgrid = []

class_num = len(CLASSES)
headNum = 3
strides = [8, 16, 32]
mapSize = [[80, 80], [40, 40], [20, 20]]
nmsThresh = 0.45
objectThresh = 0.35

input_imgH = 640
input_imgW = 640


class VideoPlayer(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.draggable = False
        self.offset = None
        
        self.rknn = RKNNLite()
        self.rknn_model = 'yolo.rknn'  
        self.load_rknn_model()  
        
        self.tableWidget.setColumnCount(6)
        
        self.tableWidget.setFrameStyle(0)
        self.tableWidget.setGridStyle(Qt.NoPen)
        
        self.tableWidget.setHorizontalHeaderLabels(['ID', 'Text', 'Confidence', 'Time', 'Save Path', 'Duration'])
        self.tableWidget.setVerticalHeaderLabels(['1', '2', '3', '4', '5', '6'])

        
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.tableWidget.horizontalHeader().setStyleSheet("QHeaderView { font: bold; }")
        self.tableWidget.verticalHeader().setStyleSheet("QHeaderView { font: bold; }")

        
        self.cap = cv2.VideoCapture(11)
        self.captured_frame_flag = False
        self.is_camera_open = False 
        self.openButton.clicked.connect(self.toggle_camera)

        self.escButton.clicked.connect(self.clickButtonCloseWindow)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  
        
               
        self.is_paused = False
        self.captured_frame = None
        self.pushButton.clicked.connect(self.toggle_video)
        self.detectButton.clicked.connect(self.display_detected_frame)
        
        self.image_path = None
        self.imageButton.clicked.connect(self.open_image_dialog)
        
        self.exportButton.clicked.connect(self.exportTable)
        
         
    def clickButtonCloseWindow(self):
        self.close()
    

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.draggable = True
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if self.draggable:
            self.move(event.globalPos() - self.offset)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.draggable = False
		
   
    def load_rknn_model(self):
        
        print('--> Load RKNN model')
        ret = self.rknn.load_rknn(self.rknn_model)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
      
        print('--> Init runtime environment')
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')
        
        
    def open_image_dialog(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpeg *.jpg *.bmp);;All Files (*)", options=options)
        if self.image_path: 
            image = cv2.imread(self.image_path)
            if image is not None:
                self.captured_frame = image
                self.captured_frame_flag = True
                h, w, c = image.shape
                q_image = QImage(image.data, w, h, c * w, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image)
                rounded_pixmap = self.changeImage(None, 70, pixmap)
                self.videolabel.setPixmap(rounded_pixmap) 
    
    def toggle_camera(self):
        if not self.is_camera_open:
            
            ret = self.cap.open(11)  
            if ret:
                self.is_camera_open = True
                self.openButton.setText('CLOSE')  
                self.timer.start(30)  
            else:
                print("Error: Unable to open camera")
        else:
            
            self.cap.release()  
            self.is_camera_open = False
            self.openButton.setText('OPEN')  
            self.timer.stop()  
            self.videolabel.clear()
            self.detectlabel.clear()
            self.captured_frame_flag = False
        
        
    def update_frame(self):
        if self.is_camera_open:
            if not self.is_paused:
                self.videolabel.setStyleSheet("border-radius: 70px;")
                ret, frame = self.cap.read()  
                if ret:
                   
                    h, w, c = frame.shape
                    q_image = QImage(frame.data, w, h, c * w, QImage.Format_BGR888)
                    
                    
                    rounded_pixmap = self.changeImage(q_image, 70, None)
                    
                    self.videolabel.setPixmap(rounded_pixmap)
                else:
                    print("Error: Failed to capture frame")
            else:
                pass
        else:
            pass
            #self.videolabel.clear()
            
    def changeImage(self, pixmap, radius, qmap):
        if qmap:
            pixmap = qmap
        else:
            pixmap = QPixmap.fromImage(pixmap)
        size = QSize(640, 480) 
        pixmap = pixmap.scaled(size, Qt.KeepAspectRatio)
        
       
        size = QSize(pixmap.size())
        mask = QBitmap(size)
        
        
        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(mask.rect(), Qt.white)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRoundedRect(mask.rect(), radius, radius)
        painter.end()
        
    
        pixmap.setMask(mask)
        
        return pixmap
        
    def toggle_video(self):
        self.is_paused = not self.is_paused 
        if self.is_paused:
            self.captured_frame = self.cap.read()[1]
            self.captured_frame_flag = True
            print("Video paused")
        else:
            print("Video resumed")
            
    def display_detected_frame(self):
        if self.captured_frame_flag:  
            
            current_datetime = datetime.now()
            date_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
            GenerateMeshgrid()
            orig_img = self.captured_frame
            img_h, img_w = orig_img.shape[:2]

            origimg = cv2.resize(orig_img, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
            origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)

            img = np.expand_dims(origimg, 0)

            outputs = self.rknn.inference(inputs=[img])

            out = []
            for i in range(len(outputs)):
                out.append(outputs[i])

            predbox = postprocess(out[0], img_h, img_w)

            
            for i in range(len(predbox)):
                start_time = time.time() * 1000
                
                xmin = int(predbox[i].xmin)
                ymin = int(predbox[i].ymin)
                xmax = int(predbox[i].xmax)
                ymax = int(predbox[i].ymax)
                classId = predbox[i].classId
                score = predbox[i].score
                
                cropped_image = orig_img[ymin:ymax, xmin:xmax]

                resized_image = cv2.resize(cropped_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

                smoothed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

                gray_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)

                #_, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


                text_otsu = reader.readtext(gray_image)
                
                end_time = time.time() * 1000
                duration = end_time - start_time
                
                self.tableWidget.insertRow(i)
                self.tableWidget.setItem(i, 0, QTableWidgetItem(f"{i}"))
                self.tableWidget.setItem(i, 1, QTableWidgetItem(f"{[text[1] for text in text_otsu]}"))
                self.tableWidget.setItem(i, 2, QTableWidgetItem(f"{score}"))
                self.tableWidget.setItem(i, 3, QTableWidgetItem(date_str))
                self.tableWidget.setItem(i, 4, QTableWidgetItem(f"{output_dir}"))
                self.tableWidget.setItem(i, 5, QTableWidgetItem(f'{duration:.2f}'))
                
                self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
                
                
            for i in range(len(predbox)):
                xmin = int(predbox[i].xmin)
                ymin = int(predbox[i].ymin)
                xmax = int(predbox[i].xmax)
                ymax = int(predbox[i].ymax)
                classId = predbox[i].classId
                score = predbox[i].score


                cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                ptext = (xmin, ymin - 20)  
                title = CLASSES[classId] + ":%.2f" % (score)
                cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            
            h, w, c = orig_img.shape
            q_image = QImage(orig_img.data, w, h, c * w, QImage.Format_BGR888)
                
            rounded_pixmap = self.changeImage(q_image, 70, None)

            self.detectlabel.setPixmap(rounded_pixmap)
            
    def exportTable(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Export Table", "",
                                                  "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            if not fileName.endswith('.csv'):
                fileName += '.csv'
    
            try:
                with open(fileName, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    
                    headers = [self.tableWidget.horizontalHeaderItem(i).text() for i in range(self.tableWidget.columnCount())]
                    writer.writerow(headers)
                    
                    for row in range(self.tableWidget.rowCount()):
                        row_data = []
                        for col in range(self.tableWidget.columnCount()):
                            item = self.tableWidget.item(row, col)
                            if item is not None:
                                row_data.append(item.text())
                            else:
                                row_data.append('')
                        writer.writerow(row_data)
                    
                    QMessageBox.information(self, 'Success', 'Table exported successfully.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'An error occurred: {e}')
     
        
    def closeEvent(self, event):
        self.cap.release()  
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoPlayer()
    window.show()
    sys.exit(app.exec_())
    
    
