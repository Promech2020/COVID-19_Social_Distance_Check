# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SocialDistanceCheck.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QLineEdit, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit, QMessageBox
from social_distance_check import check_social_distance

class Ui_MainWindow(QWidget):

    #Variables to send
    file_path = ""
    minimum_dist = ""
    time_to_wait_before = ""
    time_to_wait_between = "" 
    output_frame_size = ""
    audio_path = ""
    webcam_center_target_distance = ""
    online_video_link = ""

    def setupUi(self, MainWindow):
        self.window = MainWindow
        MainWindow.setObjectName("SocialDistanceCheck")
        MainWindow.resize(600, 350)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(600, 350))
        MainWindow.setMaximumSize(QtCore.QSize(600, 350))
        MainWindow.setDocumentMode(False)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(600, 350))
        self.centralwidget.setMaximumSize(QtCore.QSize(600, 350))
        self.centralwidget.setBaseSize(QtCore.QSize(600, 350))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setContentsMargins(15, 15, 15, 15)
        self.gridLayout.setHorizontalSpacing(35)
        self.gridLayout.setVerticalSpacing(10)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout.addLayout(self.horizontalLayout, 0, 1, 1, 1)

        #Input Video file Label
        self.input_video_file_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.input_video_file_label.setFont(font)
        self.input_video_file_label.setObjectName("input_video_file_label")
        self.gridLayout.addWidget(self.input_video_file_label, 0, 0, 1, 1)

        #Webcam button
        self.select_webcam_button = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.select_webcam_button.sizePolicy().hasHeightForWidth())
        self.select_webcam_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.select_webcam_button.setFont(font)
        self.select_webcam_button.setObjectName("select_webcam_button")
        self.horizontalLayout.addWidget(self.select_webcam_button)
        self.select_webcam_button.clicked.connect(self.webcam_clicked)

        #Online Video Button
        self.online_video_button = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.online_video_button.sizePolicy().hasHeightForWidth())
        self.online_video_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.online_video_button.setFont(font)
        self.online_video_button.setObjectName("online_video_button")
        self.horizontalLayout.addWidget(self.online_video_button)
        self.online_video_button.clicked.connect(self.online_clicked)

        #Browse Video Button
        self.browse_video_file_button = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browse_video_file_button.sizePolicy().hasHeightForWidth())
        self.browse_video_file_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.browse_video_file_button.setFont(font)
        self.browse_video_file_button.setObjectName("browse_video_file_button")
        self.horizontalLayout.addWidget(self.browse_video_file_button)
        self.browse_video_file_button.clicked.connect(self.browse_files)


        #Minimum distance label
        self.minimum_distance_input_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.minimum_distance_input_label.setFont(font)
        self.minimum_distance_input_label.setObjectName("minimum_distance_input_label")
        self.gridLayout.addWidget(self.minimum_distance_input_label, 1, 0, 1, 1)

        
        #Minimum distance between people
        self.minimum_distance_combo_box = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.minimum_distance_combo_box.sizePolicy().hasHeightForWidth())
        self.minimum_distance_combo_box.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.minimum_distance_combo_box.setFont(font)
        self.minimum_distance_combo_box.setStatusTip("")
        self.minimum_distance_combo_box.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.minimum_distance_combo_box.setAutoFillBackground(False)
        self.minimum_distance_combo_box.setEditable(False)
        self.minimum_distance_combo_box.setInsertPolicy(QtWidgets.QComboBox.InsertBeforeCurrent)
        self.minimum_distance_combo_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.minimum_distance_combo_box.setMinimumContentsLength(0)
        self.minimum_distance_combo_box.setIconSize(QtCore.QSize(16, 16))
        self.minimum_distance_combo_box.setFrame(True)
        self.minimum_distance_combo_box.setObjectName("minimum_distance_combo_box")
        self.minimum_distance_combo_box.addItem("")
        self.minimum_distance_combo_box.addItem("")
        self.minimum_distance_combo_box.addItem("")
        self.minimum_distance_combo_box.addItem("")
        self.minimum_distance_combo_box.addItem("")
        self.minimum_distance_combo_box.addItem("")
        self.gridLayout.addWidget(self.minimum_distance_combo_box, 1, 1, 1, 1)

        
        #Time to wait before starting warning label
        self.time_to_wait_to_start_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.time_to_wait_to_start_label.setFont(font)
        self.time_to_wait_to_start_label.setObjectName("time_to_wait_to_start_label")
        self.gridLayout.addWidget(self.time_to_wait_to_start_label, 2, 0, 1, 1)

        
        
        #Time to wait before starting warning
        self.time_to_wait_to_start_combo_box = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.time_to_wait_to_start_combo_box.sizePolicy().hasHeightForWidth())
        self.time_to_wait_to_start_combo_box.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.time_to_wait_to_start_combo_box.setFont(font)
        self.time_to_wait_to_start_combo_box.setObjectName("time_to_wait_to_start_combo_box")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.time_to_wait_to_start_combo_box.addItem("")
        self.gridLayout.addWidget(self.time_to_wait_to_start_combo_box, 2, 1, 1, 1)

        
        #Time to wait between warning label
        self.time_to_wait_in_between_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.time_to_wait_in_between_label.setFont(font)
        self.time_to_wait_in_between_label.setObjectName("time_to_wait_in_between_label")
        self.gridLayout.addWidget(self.time_to_wait_in_between_label, 3, 0, 1, 1)

        #Time to wait between warnings
        
        self.time_to_wait_in_between = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.time_to_wait_in_between.sizePolicy().hasHeightForWidth())
        self.time_to_wait_in_between.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.time_to_wait_in_between.setFont(font)
        self.time_to_wait_in_between.setObjectName("time_to_wait_in_between")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.time_to_wait_in_between.addItem("")
        self.gridLayout.addWidget(self.time_to_wait_in_between, 3, 1, 1, 1)

        
        #Output Frame size label
        self.output_frame_size_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.output_frame_size_label.setFont(font)
        self.output_frame_size_label.setObjectName("output_frame_size_label")
        self.gridLayout.addWidget(self.output_frame_size_label, 4, 0, 1, 1)

        #Output frame size combobox
        self.frame_size_combobox = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_size_combobox.sizePolicy().hasHeightForWidth())
        self.frame_size_combobox.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_size_combobox.setFont(font)
        self.frame_size_combobox.setObjectName("frame_size_combobox")
        self.frame_size_combobox.addItem("")
        self.frame_size_combobox.addItem("")
        self.frame_size_combobox.addItem("")
        self.gridLayout.addWidget(self.frame_size_combobox, 4, 1, 1, 1)

        #Alert filename label
        self.alert_filename_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.alert_filename_label.setFont(font)
        self.alert_filename_label.setObjectName("alert_filename_label")
        self.gridLayout.addWidget(self.alert_filename_label, 5, 0, 1, 1)

        #Alert file browser button
        self.alert_file_browser_button = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.alert_file_browser_button.sizePolicy().hasHeightForWidth())
        self.alert_file_browser_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.alert_file_browser_button.setFont(font)
        self.alert_file_browser_button.setObjectName("alert_file_browser_button")
        self.gridLayout.addWidget(self.alert_file_browser_button, 5, 1, 1, 1)
        self.alert_file_browser_button.clicked.connect(self.browse_alert_file)



        #Proceed button
        self.proceed_to_detection_button = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.proceed_to_detection_button.sizePolicy().hasHeightForWidth())
        self.proceed_to_detection_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.proceed_to_detection_button.setFont(font)
        self.proceed_to_detection_button.setObjectName("proceed_to_detection_button")
        self.gridLayout.addWidget(self.proceed_to_detection_button, 6, 1, 1, 1)
        self.proceed_to_detection_button.clicked.connect(self.proceed_processing)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.time_to_wait_to_start_combo_box.setCurrentIndex(5)
        self.minimum_distance_combo_box.setCurrentIndex(3)
        self.frame_size_combobox.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def webcam_clicked(self):
        self.file_path  = "WebCam"
        self.webcam_target_distance = QDialog()
        # add a vertical layout to dialog
        self.vlayout1 = QVBoxLayout()
        # add a horizontal layout to take user inputs.
        self.hlayout1_1 = QHBoxLayout()
        # add a horizontal layout for buttons.
        self.hlayout1_2 = QHBoxLayout()

        self.webcam_target_distance.setLayout(self.vlayout1)
        self.vlayout1.addLayout(self.hlayout1_1)
        self.vlayout1.addLayout(self.hlayout1_2)
        # add the items to layout instead of dialog
        self.webcam_target_distance_label = QLabel("Distance between webcam and target:", self.webcam_target_distance)
        self.hlayout1_1.addWidget(self.webcam_target_distance_label)

        self.webcam_target_distance_user_input = QLineEdit(self.webcam_target_distance)
        self.hlayout1_1.addWidget(self.webcam_target_distance_user_input)

        self.webcam_target_distance_unit = QLabel("Meters", self.webcam_target_distance)
        self.hlayout1_1.addWidget(self.webcam_target_distance_unit)

        self.okButton1 = QPushButton("OK", self.webcam_target_distance)
        self.hlayout1_2.addWidget(self.okButton1)
        self.cancelButton1 = QPushButton("Cancel", self.webcam_target_distance)
        self.hlayout1_2.addWidget(self.cancelButton1)
        
        self.webcam_target_distance.setWindowTitle("Get Distance")
        self.webcam_target_distance.show()

        self.okButton1.clicked.connect(self.ok_pressed1)
        self.cancelButton1.clicked.connect(self.cancel_pressed1)

    def ok_pressed1(self):
        self.distance = self.webcam_target_distance_user_input.text()
        if self.distance == "":
            self.msg = QMessageBox()
            self.msg.setWindowTitle("No Distance")
            self.msg.setText("Please provide us some data to work on.")
            self.msg.setIcon(QMessageBox.Critical)
            self.msg.setStandardButtons(QMessageBox.Ok)
            self.msg.setDefaultButton(QMessageBox.Ok)
            self.msg.buttonClicked.connect(self.close)
            x = self.msg.exec_()
        else:
            self.webcam_center_target_distance = self.distance
            self.webcam_target_distance.close()

    def cancel_pressed1(self):
        self.webcam_target_distance.close()

    def online_clicked(self):
        self.get_online_data = QDialog()
        # add a vertical layout to dialog
        self.vlayout2 = QVBoxLayout()

        # add a horizontal layout for buttons.
        self.hlayout2_1 = QHBoxLayout()

        self.get_online_data.setLayout(self.vlayout2)
        # add the items to layout instead of dialog
        self.online_data_link_label = QLabel("Please provide us some link to the online video:", self.get_online_data)
        self.vlayout2.addWidget(self.online_data_link_label)

        self.online_data_link_user_input = QLineEdit(self.get_online_data)
        self.vlayout2.addWidget(self.online_data_link_user_input)

        self.vlayout2.addLayout(self.hlayout2_1)
        self.okButton2 = QPushButton("OK", self.get_online_data)
        self.hlayout2_1.addWidget(self.okButton2)
        self.cancelButton2 = QPushButton("Cancel", self.get_online_data)
        self.hlayout2_1.addWidget(self.cancelButton2)
        
        self.get_online_data.setWindowTitle("Get video link")
        self.get_online_data.show()

        self.okButton2.clicked.connect(self.ok_pressed2)
        self.cancelButton2.clicked.connect(self.cancel_pressed2)


    def ok_pressed2(self):
        self.link = self.online_data_link_user_input.text()
        if self.link == "":
            self.msg0 = QMessageBox()
            self.msg0.setWindowTitle("No Link Provided")
            self.msg0.setText("Please provide us some link to work on.")
            self.msg0.setIcon(QMessageBox.Critical)
            self.msg0.setStandardButtons(QMessageBox.Ok)
            self.msg0.setDefaultButton(QMessageBox.Ok)
            self.msg0.buttonClicked.connect(self.close)
            x = self.msg0.exec_()
        else:
            self.online_video_link = self.link
            self.get_online_data.close()

    def cancel_pressed2(self):
        self.get_online_data.close()


    def browse_files(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','c:/', "Video files (*.mp4 *.avi)")
        videoPath = fname[0]
        self.file_path = videoPath

    def browse_alert_file(self):
        a_name = QFileDialog.getOpenFileName(self, 'Open file','c:/', "Audio files (*.mp3)")
        audioPath = a_name[0]
        self.audio_path = audioPath

    def proceed_processing(self):
        self.minimum_dist = self.minimum_distance_combo_box.currentText()
        self.time_to_wait_before = self.time_to_wait_to_start_combo_box.currentText()
        self.time_to_wait_between = self.time_to_wait_in_between.currentText()
        
        if self.file_path == "":
            self.show_pop_up_1()
        else:
            self.show_pop_up_2()

    def show_pop_up_1(self):
        self.msg1 = QMessageBox()
        self.msg1.setWindowTitle("No Video Path")
        self.msg1.setText("You have neither selected webcam, nor given any video file path. We will proceed with our default video.")
        self.msg1.setIcon(QMessageBox.Warning)
        self.msg1.setStandardButtons(QMessageBox.Ok|QMessageBox.Cancel)
        self.msg1.setDefaultButton(QMessageBox.Cancel)
        self.msg1.buttonClicked.connect(self.pop_up_button1)
        x1 = self.msg1.exec_()

    def pop_up_button1(self, i):
        if i.text() == "OK":
            self.msg1.close()
            MainWindow.close()
            check_social_distance(self.file_path, self.minimum_dist, self.time_to_wait_before, self.time_to_wait_between)
            

    def show_pop_up_2(self):
        self.msg2 = QMessageBox()
        self.msg2.setWindowTitle("Confirmation")
        self.msg2.setText(f"Your have selected:\n \
                    {self.file_path} as video input.\n \
                    {self.minimum_dist} as minimum distance to maintain between people.\n \
                    {self.time_to_wait_before} as time to wait before starting to play warning. \n \
                    {self.time_to_wait_between} as time to wait between playing warnings.")
        self.msg2.setIcon(QMessageBox.Information)
        self.msg2.setStandardButtons(QMessageBox.Ok|QMessageBox.Cancel)
        self.msg2.setDefaultButton(QMessageBox.Cancel)
        self.msg2.buttonClicked.connect(self.pop_up_button2)
        x2 = self.msg2.exec_()

    def pop_up_button2(self, j):
        if j.text() == "OK":
            self.msg2.close()
            MainWindow.close()
            check_social_distance(self.file_path, self.minimum_dist, self.time_to_wait_before, self.time_to_wait_between)
            
            

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.time_to_wait_to_start_combo_box.setItemText(0, _translate("MainWindow", "5 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(1, _translate("MainWindow", "10 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(2, _translate("MainWindow", "15 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(3, _translate("MainWindow", "20 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(4, _translate("MainWindow", "25 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(5, _translate("MainWindow", "30 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(6, _translate("MainWindow", "35 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(7, _translate("MainWindow", "40 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(8, _translate("MainWindow", "45 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(9, _translate("MainWindow", "50 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(10, _translate("MainWindow", "55 Seconds"))
        self.time_to_wait_to_start_combo_box.setItemText(11, _translate("MainWindow", "60 Seconds"))
        self.minimum_distance_combo_box.setItemText(0, _translate("MainWindow", "0.5 Meters"))
        self.minimum_distance_combo_box.setItemText(1, _translate("MainWindow", "1 Meters"))
        self.minimum_distance_combo_box.setItemText(2, _translate("MainWindow", "1.5 Meters"))
        self.minimum_distance_combo_box.setItemText(3, _translate("MainWindow", "2 Meters"))
        self.minimum_distance_combo_box.setItemText(4, _translate("MainWindow", "2.5 Meters"))
        self.minimum_distance_combo_box.setItemText(5, _translate("MainWindow", "3 Meters"))
        self.output_frame_size_label.setText(_translate("MainWindow", "Output Video Frame Size"))
        self.input_video_file_label.setText(_translate("MainWindow", "Input video file:"))
        self.proceed_to_detection_button.setText(_translate("MainWindow", "Proceed"))
        self.minimum_distance_input_label.setText(_translate("MainWindow", "Minimum distance to maintain between people:"))
        self.time_to_wait_in_between.setItemText(0, _translate("MainWindow", "5 Seconds"))
        self.time_to_wait_in_between.setItemText(1, _translate("MainWindow", "10 Seconds"))
        self.time_to_wait_in_between.setItemText(2, _translate("MainWindow", "15 Seconds"))
        self.time_to_wait_in_between.setItemText(3, _translate("MainWindow", "20 Seconds"))
        self.time_to_wait_in_between.setItemText(4, _translate("MainWindow", "25 Seconds"))
        self.time_to_wait_in_between.setItemText(5, _translate("MainWindow", "30 Seconds"))
        self.time_to_wait_in_between.setItemText(6, _translate("MainWindow", "35 Seconds"))
        self.time_to_wait_in_between.setItemText(7, _translate("MainWindow", "40 Seconds"))
        self.time_to_wait_in_between.setItemText(8, _translate("MainWindow", "45 Seconds"))
        self.time_to_wait_in_between.setItemText(9, _translate("MainWindow", "50 Seconds"))
        self.time_to_wait_in_between.setItemText(10, _translate("MainWindow", "55 Seconds"))
        self.time_to_wait_in_between.setItemText(11, _translate("MainWindow", "60 Seconds"))
        self.select_webcam_button.setText(_translate("MainWindow", "WebCam"))
        self.online_video_button.setText(_translate("MainWindow", "Online Video"))
        self.browse_video_file_button.setText(_translate("MainWindow", "Browse"))
        self.time_to_wait_in_between_label.setText(_translate("MainWindow", "Time to wait between playing warnings:"))
        self.time_to_wait_to_start_label.setText(_translate("MainWindow", "Time to wait before playing warning:"))
        self.alert_filename_label.setText(_translate("MainWindow", "Input alert mp3 file:"))
        self.frame_size_combobox.setItemText(0, _translate("MainWindow", "720 x 480"))
        self.frame_size_combobox.setItemText(1, _translate("MainWindow", "1280 x 720"))
        self.frame_size_combobox.setItemText(2, _translate("MainWindow", "1920 x 1080"))
        self.alert_file_browser_button.setText(_translate("MainWindow", "Browse mp3 file"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

