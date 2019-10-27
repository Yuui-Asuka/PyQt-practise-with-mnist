import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from network import Worker
from PIL import ImageGrab, Image
import tensorflow as tf
import numpy as np
import scipy.misc

class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.epoch = None
        self.hidden_cells = None
        self.learning_rate = None
        self.layer_num = None
        self.optimizer = ''
        self.activation_function = ''
        self.loss_function = ''
        self.nomorlization = None
        self.batch_size = None
        self.keep_prob = None
        self.bias = None
        self.stddev = None 
        self.batch_norm = None
        self.decay = None

    def initUI(self):
        self.statusBar().showMessage('ready')
        self.setGeometry(500,100,1200,800)
        self.setWindowTitle('手写数字识别')
        self.hidden_cells_button = MyButton.hiddenCellsButton(self)
        self.epoch_button = MyButton.epochButton(self)
        self.learning_rate_button = MyButton.learningRateButton(self)
        self.train_start_button = MyButton.startLearning(self)
        self.layer_num_button = MyButton.layerNum(self)
        self.figure_visualization_button = MyButton.figureVisualization(self)
        self.keep_prob_button = MyButton.dropoutRate(self)
        self.batch_size_button = MyButton.batchSize(self)
        self.bias_button = MyButton.bias(self)
        self.stddev_button = MyButton.stddev(self)
        self.decay_button = MyButton.decay(self)
        self.reco_button = MyButton.recognise(self)
        self.figure_visualization_button.setEnabled(False)
       # self.stop_button = MyButton.stopTraining(self)
        self.lcd1 = QLCDNumber(self)
        self.lcd1.move(400,50)
        self.lcd2 = QLCDNumber(self)
        self.lcd2.move(400,100)
        self.lcd3 = QLCDNumber(self)
        self.lcd3.setNumDigits(6)
        self.lcd3.move(400,150)
        self.lcd4 = QLCDNumber(self)
        self.lcd4.move(400,200)
        self.lcd5 = QLCDNumber(self)
        self.lcd5.move(950,50)
        self.lcd6 = QLCDNumber(self)
        self.lcd6.move(950,100)
        self.lcd7 = QLCDNumber(self)
        self.lcd7.move(950,150)
        self.lcd8 = QLCDNumber(self)
        self.lcd8.move(950,200)
        self.lcd9 = QLCDNumber(self)
        self.lcd9.move(950,250)
        self.font = QFont("Roman times",10,QFont.Bold)
        self.label = QLabel('结果显示区：',self)
        self.label.setFont(self.font)
        self.label.move(700,370)
        self.label1 = QLabel(self)
        self.label1.resize(350,150)
        self.label1.move(700,400)
        pe = QPalette()
        pe.setColor(QPalette.WindowText,Qt.blue)     
        self.label1.setAutoFillBackground(True)
        self.decay_button.setEnabled(False)
        self.label1.setPalette(pe)                
        self.label1.setFont(self.font)         
        self.label2 = QLabel('请选择优化器',self)
        self.label2.setFont(self.font)
        self.label2.move(50,370)
        
        self.label3 = QLabel('请选择激活函数',self)
        self.label3.setFont(self.font)
        self.label3.move(50,470)

        self.label3 = QLabel('请选择输出单元',self)
        self.label3.setFont(self.font)
        self.label3.move(50,570)

        self.label3 = QLabel('请选择正则化方式',self)
        self.label3.setFont(self.font)
        self.label3.resize(200,20)
        self.label3.move(50,670)

        self.label4 = QLabel('是否进行梯度截断?',self)
        self.label4.setFont(self.font)
        self.label4.resize(200,20)
        self.label4.move(400,370)
        
        self.label5 = QLabel('是否使用批规范化？',self)
        self.label5.setFont(self.font)
        self.label5.resize(200,20)
        self.label5.move(400,470)

        self.combo = QComboBox(self)
        self.combo.setFont(QFont("Arial",10))
        self.combo.addItems(["SGD","RMS","Adagrad","Adadelta","Adam"])        
        self.combo.resize(200,40)
        self.combo.move(50,400)

        self.combo2 = QComboBox(self)
        self.combo2.setFont(QFont('Arial',10))
        self.combo2.addItems(["relu","leaky","sigmoid","tanh","swish"])
        self.combo2.resize(200,40)
        self.combo2.move(50,500)

        self.combo3 = QComboBox(self)
        self.combo3.setFont(QFont('Arial',10))
        self.combo3.addItems(["softmax","softplus","softsign"])
        self.combo3.resize(200,40)
        self.combo3.move(50,600)

        self.combo4 = QComboBox(self)
        self.combo4.setFont(QFont('Arial',10))
        self.combo4.addItems(['不使用','L1正则化','L2正则化'])
        self.combo4.resize(200,40)
        self.combo4.move(50,700)

        self.combo5 = QComboBox(self)
        self.combo5.setFont(QFont('Arial',10))
        self.combo5.addItems(['不截断','截断为5','截断为10'])
        self.combo5.resize(200,40)
        self.combo5.move(400,400)

        self.combo6 = QComboBox(self)
        self.combo6.setFont(QFont('Arial',10))
        self.combo6.addItem('不使用')
        self.combo6.addItem('使用')
        self.combo6.resize(200,40)
        self.combo6.move(400,500)
        self.combo6.currentIndexChanged.connect(self.batchNomor)
        self.show()

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black,2)
        painter.setPen(pen)
        painter.drawRect(QRect(self.label1.geometry().x(),self.label1.geometry().y(),350,150))
        painter.end()

    def batchNomor(self,i):
        if i == 1:
            self.decay_button.setEnabled(True)
        elif i == 0:
            self.decay_button.setEnabled(False)
    #def comboxActive(self,text):
    #    self.optimizer = text

    def buttonClicked_1(self,minimum,maximum):
        self.slider_1 = MySlider(minimum,maximum)
        self.slider_1.show()
        self.slider_1.sliderIntValue.connect(self.lcd1.display)        

    def buttonClicked_2(self,minimum,maximum):
        self.slider_2 = MySlider(minimum,maximum)
        self.slider_2.show()
        self.slider_2.sliderIntValue.connect(self.lcd2.display)

    def echo(self, value):        
        self.learning_rate = value    
        reply = QMessageBox.information(self, "学习率设置为：",   "得到：{}\n".format(value), QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.lcd3.display(value)
        else:
            pass

    def buttonClicked_3(self,event):
        value, ok = QInputDialog.getDouble(self, "输入框标题", "这是提示信息\n\n请输入小数:", 0.001, 0, 0.5, 4)
        if ok:
            self.echo(value)
        else:
            pass

    def buttonClicked_5(self,minimum,maximum):
        self.slider_3 = MySlider(minimum,maximum)
        self.slider_3.show()
        self.slider_3.sliderIntValue.connect(self.lcd4.display)

    def echo_8(self,value):
        self.keep_prob = value
        reply = QMessageBox.information(self, "dropout设置为：",   "得到：{}\n".format(value), QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.lcd5.display(value)
        else:
            pass
        
    def buttonClicked_8(self):
        value, ok = QInputDialog.getDouble(self, "输入框标题", "这是提示信息\n\n请输入小数:", 1.0, 0.1, 1.0, 1)
        if ok:
            self.echo_8(value)
        else:
            pass

    def echo_10(self,value):
        self.bias = value
        reply = QMessageBox.information(self, "偏置设置为：",   "得到：{}\n".format(value), QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.lcd7.display(value)
        else:
            pass
        
    def buttonClicked_10(self):
        value, ok = QInputDialog.getDouble(self, "输入框标题", "这是提示信息\n\n请输入小数:", 0.1, 0, 0.5, 3)
        if ok:
            self.echo_10(value)
        else:
            pass

    def echo_11(self,value):
        self.stddev = value
        reply = QMessageBox.information(self, "标准差设置为：",   "得到：{}\n".format(value), QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.lcd8.display(value)
        else:
            pass
        
    def buttonClicked_11(self):
        value, ok = QInputDialog.getDouble(self, "输入框标题", "这是提示信息\n\n请输入小数:", 0.1, 0, 0.5, 3)
        if ok:
            self.echo_11(value)
        else:
            pass
 
    def echo_12(self,value):
        self.decay = value
        reply = QMessageBox.information(self, "衰减系数设置为：",   "得到：{}\n".format(value), QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.lcd9.display(value)
        else:
            pass

    def buttonClicked_12(self):
        value, ok = QInputDialog.getDouble(self, "输入框标题", "这是提示信息\n\n请输入小数:", 0.95, 0.5, 1.0, 4)
        if ok:
            self.echo_12(value)
        else:
            pass

    def buttonClicked_9(self,minimum,maximum):
        self.slider_4 = MySlider(minimum,maximum)
        self.slider_4.show()
        self.slider_4.sliderIntValue.connect(self.lcd6.display)      

    def buttonClicked_4(self):
        self.epoch = int(self.lcd2.value())
        self.hidden_cells = int(self.lcd1.value())
        self.layer_num = int(self.lcd4.value())
        self.optimizer = self.combo.currentText()
        self.activation_function = self.combo2.currentText()
        self.loss_function = self.combo3.currentText()
        self.nomorlization = self.combo4.currentIndex()
        self.clip = self.combo5.currentIndex()
        self.batch_norm = self.combo6.currentIndex()
        self.batch_size = int(self.lcd6.value())
        self.train_start = Worker(self.hidden_cells,self.epoch,self.learning_rate,self.layer_num,self.optimizer,self.activation_function,
                                  self.loss_function,self.nomorlization,self.batch_size,self.keep_prob,self.bias,self.stddev,self.clip,self.batch_norm,self.decay)
        self.train_start.start()
        self.train_start_button.setEnabled(False)
        self.hidden_cells_button.setEnabled(False)
        self.epoch_button.setEnabled(False)
        self.learning_rate_button.setEnabled(False)
        self.layer_num_button.setEnabled(False)
        self.keep_prob_button.setEnabled(False)
        self.batch_size_button.setEnabled(False)
        self.bias_button.setEnabled(False)
        self.stddev_button.setEnabled(False)
        self.decay_button.setEnabled(False)
        self.train_start.breakSignal.connect(self.label1.setText)
        self.train_start.stopSignal.connect(self.buttonClicked_6)
        
    def buttonClicked_6(self):
        self.train_start.close_thread()
        self.train_start_button.setEnabled(True)
        self.hidden_cells_button.setEnabled(True)
        self.epoch_button.setEnabled(True)
        self.learning_rate_button.setEnabled(True)
        self.layer_num_button.setEnabled(True)
        self.figure_visualization_button.setEnabled(True)
        self.keep_prob_button.setEnabled(True)
        self.batch_size_button.setEnabled(True)
        self.bias_button.setEnabled(True)
        self.stddev_button.setEnabled(True)
        self.decay_button.setEnabled(True)

    def buttonClicked_7(self):
        self.figure = Myfigure()
        self.figure.show()

    def buttonClicked_13(self):
        self.recognise_window = MyMnistWindow()
        self.recognise_window.show()
        
class MyButton:

    @staticmethod
    def hiddenCellsButton(win):
        btn = QPushButton('每一层单元数量',win)
        btn.setToolTip('点击调节每一层神经单元数量')
        btn.resize(200,40)
        btn.move(50,50)
        btn.clicked.connect(lambda:win.buttonClicked_1(1,500))
        return btn
        
    @staticmethod
    def epochButton(win):
        btn = QPushButton('迭代周期',win)
        btn.setToolTip('点击调节迭代周期数量')
        btn.resize(200,40)
        btn.move(50,100)
        btn.clicked.connect(lambda:win.buttonClicked_2(1,50))
        return btn

    @staticmethod
    def learningRateButton(win):
        btn = QPushButton('初始学习率',win)
        btn.setToolTip('点击设置初始学习率')
        btn.resize(200,40)
        btn.move(50,150)
        btn.clicked.connect(win.buttonClicked_3)
        return btn

    @staticmethod
    def startLearning(win):
        btn = QPushButton('开始训练！',win)
        btn.resize(200,40)
        btn.move(50,250)
        btn.clicked.connect(win.buttonClicked_4)
        return btn

    @staticmethod
    def layerNum(win):
        btn = QPushButton('隐藏层数量',win)
        btn.resize(200,40)
        btn.move(50,200)
        btn.clicked.connect(lambda:win.buttonClicked_5(1,50))
        return btn

    @staticmethod
    def figureVisualization(win):
        btn = QPushButton('查看统计图',win)
        btn.resize(200,40)
        btn.move(400,600)
        btn.clicked.connect(win.buttonClicked_7)
        return btn

    @staticmethod
    def dropoutRate(win):
        btn = QPushButton('神经元激活比例',win)
        btn.resize(200,40)
        btn.move(600,50)
        btn.clicked.connect(win.buttonClicked_8)
        return btn

    @staticmethod
    def batchSize(win):
        btn = QPushButton('批次大小',win)
        btn.resize(200,40)
        btn.move(600,100)
        btn.clicked.connect(lambda:win.buttonClicked_9(10,1000))
        return btn

    @staticmethod
    def bias(win):
        btn = QPushButton('偏置',win)
        btn.resize(200,40)
        btn.move(600,150)
        btn.clicked.connect(win.buttonClicked_10)
        return btn

    @staticmethod
    def stddev(win):
        btn = QPushButton('标准差',win)
        btn.resize(200,40)
        btn.move(600,200)
        btn.clicked.connect(win.buttonClicked_11)
        return btn

    @staticmethod
    def decay(win):
        btn = QPushButton('加权指数平均值衰减系数',win)
        btn.resize(200,40)
        btn.move(600,250)
        btn.clicked.connect(win.buttonClicked_12)
        return btn

    @staticmethod
    def recognise(win):
        btn = QPushButton('进入识别',win)
        btn.resize(200,40)
        btn.move(400,700)
        btn.clicked.connect(win.buttonClicked_13)
        return btn


class MySlider(QWidget):

    sliderIntValue = pyqtSignal(int)
    sliderFloatValue = pyqtSignal(float)

    def __init__(self,minimum,maximum):
        super().__init__()
        self.initHslider(minimum,maximum)
        self.value = None

    def initHslider(self,minimum,maximum):
        lcd = QLCDNumber(self)
        self.sld = QSlider(Qt.Horizontal,self)
        self.sld.setMinimum(minimum)
        self.sld.setMaximum(maximum)
        btn = QPushButton("确定")
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(lcd,1,0)
        grid.addWidget(self.sld,2,0)
        grid.addWidget(btn,3,0)
        self.setLayout(grid)
        self.sld.valueChanged.connect(lcd.display)
        btn.clicked.connect(self.close)
        btn.clicked.connect(self.sendValue)      

    def sendValue(self):
        if isinstance (self.sld.value(),int):
            self.sliderIntValue.emit(self.sld.value())

        elif isinstance(self.sld.value(),float):
            self.sliderFloatValue.emit(self.sld.value())

    #def getValue(self):
    #    return self.sld.value()
        

class Myfigure(QWidget):
    def __init__(self):
        super().__init__()
        self.initFigure()

    def initFigure(self):
        
        self.label = QLabel(self)
        self.label.setFixedSize(800,800)
        self.label.move(160, 160)
        
        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(200,200,200,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )
        self.label.setScaledContents (True) 
        myfig = QPixmap('myfig.png')
        myfig.scaled(self.label.width(), self.label.height())
        self.label.setPixmap(myfig)
   

class MyMnistWindow(QWidget):

    def __init__(self):
        super(MyMnistWindow, self).__init__()
        self.resize(424, 500)  
        self.setWindowFlags(Qt.FramelessWindowHint) 
        self.setMouseTracking(False)
        qr = self.frameGeometry()
        self.cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(self.cp)
        self.move(qr.topLeft())

        self.pos_xy = []  
        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 2, 420, 420)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('识别结果：', self)
        self.label_result_name.setGeometry(10, 440, 80, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(90, 440, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(140, 440, 50, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(200, 440, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("关闭", self)
        self.btn_close.setGeometry(260, 440, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)
        self.show()


    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 28, Qt.SolidLine)
        painter.setPen(pen)
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        pos_tmp = (event.pos().x(), event.pos().y())        
        self.pos_xy.append(pos_tmp)
        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

    def btn_recognize_on_clicked(self):
        from win32api import GetSystemMetrics
        resolution_ratio = (GetSystemMetrics(0),GetSystemMetrics(1))
        ratio_xy = ((resolution_ratio[0] - 420)/2 ,(resolution_ratio[1] -420)/2-40,(resolution_ratio[0] + 420)/2,(resolution_ratio[1] + 420)/2-50)
        im = ImageGrab.grab(ratio_xy)
        im = im.resize((28, 28), Image.ANTIALIAS)  
        recognize_result = self.recognize_img(im)  
        self.label_result.setText(str(recognize_result))  
        self.update()

    def recognize_img(self, img):      
        img = img.convert('L')  
        img = np.array(img)
        img = np.max(img) - img
        minmax = np.where(img<40,0,img)
        minmax = minmax/(np.max(minmax) - np.min(minmax))
        scipy.misc.imsave('sss.png',minmax)
        self.sess = tf.InteractiveSession()
        print(minmax)
        minmax = minmax.reshape(1,784)
        init = tf.global_variables_initializer()
        saver = tf.train.import_meta_graph('mnist\mnist.ckpt.meta')        
        self.sess.run(init)
        saver.restore(self.sess, 'mnist/mnist.ckpt') 
        graph = tf.get_default_graph() 
        x = graph.get_tensor_by_name("input:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        y_conv = graph.get_tensor_by_name("prediction:0")
        prediction = tf.argmax(y_conv, 1)
        predint = prediction.eval(feed_dict={x: minmax, keep_prob: 1.0, is_training: False}, session=self.sess)  
        return predint[0]

    def btn_clear_on_clicked(self):
        self.pos_xy.clear()
        self.label_result.setText('')
        self.sess.close()
        self.update()

    def btn_close_on_clicked(self):
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())
