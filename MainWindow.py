from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from Ui_MainWindow import Ui_MainWindow
import cv2
import sys
import os
import numpy as np
import json
from pathlib import Path

from model import Model_CNN_Classfication

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent=parent)
        self.setupUi(self)
        
        # Model
        self.Model_Leaf_Disease = Model_CNN_Classfication()

        # File path
        self.filePath = ""

        # Event handles
        self.btChooseImg.clicked.connect(self.chooseImage)
        self.btTraining.clicked.connect(self.trainingModel)
        self.btPredict.clicked.connect(self.predictImage)

        # Load json file
        path = '../TF_CNN_Leaf Disease/input/cassava-leaf-disease-classification/'
        with open(os.path.join(path + 'label_num_to_disease_map.json')) as f:
            self.data = json.loads(f.read())

    
    def alert(self, title, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.exec_()

    def chooseImage(self):
        #Show file dialog
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            image, _ = QFileDialog.getOpenFileName(None, "Choose Image", "","Image Files (*.jpg *.jpeg *png);;All Files (*)", options = options)
            if image is not None:
                #print(image)
                self.filePath = image
                Img = cv2.imread(image)
                Img = cv2.resize(Img, (640, 480))
                rgbImg = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImg.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImg.data, w, h, bytesPerLine, QImage.Format_RGB888)
                pImg = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.lbImg.setPixmap(QtGui.QPixmap(pImg))

                fileName = Path(image).resolve().stem
                fileTail =os.path.splitext(image)[1]
                imgFile = fileName + fileTail
                self.lbFileName.setText(imgFile)
        except:
            pass


    def trainingModel(self):
        if(self.txtEpochs.text() == "" or self.txtBatchSize.text() == ""):
            self.alert(title="Warning", message="Have not imported epochs and batch size!!")
        else:
            self.btChooseImg.setEnabled(False)
            self.btPredict.setEnabled(False)
            self.alert(title="Notification", message="Model is training, please follow terminal!!")
            self.Model_Leaf_Disease.create_Model()
            self.Model_Leaf_Disease.training_Model(BatchSize=int(self.txtBatchSize.text()), Epochs=int(self.txtEpochs.text()))
            self.Model_Leaf_Disease.save_model()
            self.alert(title="Notification", message="Model training successfully!!")
            self.btChooseImg.setEnabled(True)
            self.btPredict.setEnabled(True)

    def predictImage(self):
        if(self.filePath == ""):
            self.alert(title="Warning", message="No photo selected!!")
        else:
            print(self.filePath)
            Load_Model = self.Model_Leaf_Disease.load_model()
            ID_LB = self.Model_Leaf_Disease.predict(filePath=self.filePath)
            print(ID_LB)
            self.lbDiseaseName.setText(self.data['{}'.format(ID_LB)])    



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())