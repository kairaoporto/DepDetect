import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QProgressBar, QApplication
from PyQt5.QtGui import QFontMetrics, QFontDatabase
from PyQt5.QtCore import QTimer
import sys

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

#Import dataset
df = pd.read_csv('depressionData.csv')

#Scrub dataset
#Remove unwanted variables
del df ['Timestamp']

#One-hot encoding on non-numeric variables
df = pd.get_dummies(df, columns=['Age', 'Feeling sad', 'Irritable towards people',
                                 'Trouble sleeping at night', 'Problems concentrating or making decision',
                                'loss of appetite', 'Feeling of guilt', 'Problems of bonding with people',
                                'Suicide attempt','Depressed'])

del df ['Depressed_No']

X = df.drop('Depressed_Yes',axis=1)
y = df['Depressed_Yes']

#Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=10, shuffle=True)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,       
    min_child_weight=1,
    gamma=0.1,
    subsample=1.0,
    colsample_bytree=0.5,
    reg_alpha=0,
    reg_lambda=1,
    objective='binary:logistic',
    scale_pos_weight= 1, 
)

#Fit algorithm to training data
model.fit(X_train,y_train)
 
#Evaluate results
model_predict = model.predict(X_test)

def setup_window(window, name):
    window.setObjectName(name)
    window.setEnabled(True)
    window.setFixedSize(800, 600)
    window.centralwidget = QtWidgets.QWidget(window)
    window.centralwidget.setObjectName("centralwidget")

def set_background(window):
    window.background = QtWidgets.QLabel(window.centralwidget)
    window.background.setGeometry(QtCore.QRect(0, 0, 800, 600))
    window.background.setText("")
    window.background.setPixmap(QtGui.QPixmap("background.png"))
    window.background.setScaledContents(False)
    window.background.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
    window.background.setObjectName("background")

def setup_line(window):
    window.line = QtWidgets.QFrame(window.centralwidget)
    window.line.setGeometry(QtCore.QRect(40, 120, 711, 16))
    window.line.setMidLineWidth(4)
    window.line.setFrameShape(QtWidgets.QFrame.HLine)
    window.line.setFrameShadow(QtWidgets.QFrame.Sunken)
    window.line.setObjectName("line")

def setup_first_ui(window):
    setup_window(window, "FirstWindow")
    set_background(window)

    window.design1 = QtWidgets.QLabel(window.centralwidget)
    window.design1.setGeometry(QtCore.QRect(200, 70, 421, 331))
    window.design1.setAutoFillBackground(False)
    window.design1.setFrameShape(QtWidgets.QFrame.NoFrame)
    window.design1.setText("")
    window.design1.setPixmap(QtGui.QPixmap("depdetect1.png"))
    window.design1.setScaledContents(True)
    window.design1.setAlignment(QtCore.Qt.AlignCenter)
    window.design1.setWordWrap(False)
    window.design1.setObjectName("design1")
    
    window.titlename = QtWidgets.QLabel(window.centralwidget)
    window.titlename.setGeometry(QtCore.QRect(325, 380, 211, 61))
    
    font_path = "BrixtonSansBld.ttf"  
    QtGui.QFontDatabase.addApplicationFont(font_path)
    
    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size = 43  
    scaled_font_size = int(base_font_size * dpi_scaling_factor)

    font = QtGui.QFont("Brixton Sans TC")
    font.setPixelSize(scaled_font_size)
    font.setBold(True)
    font.setWeight(75)
    window.titlename.setFont(font)
    window.titlename.setAutoFillBackground(False)
    window.titlename.setObjectName("titlename")
    window.titlename.setStyleSheet("color: #692B28;")

    window.setCentralWidget(window.centralwidget)
    
    window.opacity_effect_design1 = QtWidgets.QGraphicsOpacityEffect()
    window.design1.setGraphicsEffect(window.opacity_effect_design1)

    window.opacity_effect_titlename = QtWidgets.QGraphicsOpacityEffect()
    window.titlename.setGraphicsEffect(window.opacity_effect_titlename)

    window.retranslateUi()

def setup_second_ui(window):
    setup_window(window, "SecondWindow")
    set_background(window)

    font_path = "BrixtonSansBld.ttf"  
    QFontDatabase.addApplicationFont(font_path)
    
    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size = 43  
    scaled_font_size = int(base_font_size * dpi_scaling_factor)

    font = QtGui.QFont("Brixton Sans TC")
    font.setPixelSize(scaled_font_size)

    window.titlename = QtWidgets.QLabel(window.centralwidget)
    window.titlename.setGeometry(QtCore.QRect(270, 240, 301, 61))
    window.titlename.setFont(font)
    window.titlename.setStyleSheet("color: #692B28;")
    window.titlename.setScaledContents(False)

    window.titlename_2 = QtWidgets.QLabel(window.centralwidget)
    window.titlename_2.setGeometry(QtCore.QRect(210, 300, 411, 61))
    window.titlename_2.setFont(font)
    window.titlename_2.setStyleSheet("color: #692B28;")
    window.titlename_2.setScaledContents(False)

    window.setCentralWidget(window.centralwidget)

    window.opacity_effect_titlename = QtWidgets.QGraphicsOpacityEffect()
    window.titlename.setGraphicsEffect(window.opacity_effect_titlename)
    window.opacity_effect_titlename_2 = QtWidgets.QGraphicsOpacityEffect()
    window.titlename_2.setGraphicsEffect(window.opacity_effect_titlename_2)
    window.retranslateUi()

def setup_third_ui(window):
    setup_window(window, "ThirdWindow")
    set_background(window)
    setup_line(window)
    
    font_path = "BrixtonSansBld.ttf"  
    QFontDatabase.addApplicationFont(font_path)
    
    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 43
    base_font_size_2 = 20

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)

    font = QtGui.QFont("Brixton Sans TC")
    font.setPixelSize(scaled_font_size_1)

    font_path1 = "Belleza-Regular.ttf"  
    QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")
    font1.setPixelSize(scaled_font_size_2)

    window.getting_started = QtWidgets.QLabel(window.centralwidget)
    window.getting_started.setGeometry(QtCore.QRect(40, 70, 281, 51))
    window.getting_started.setFont(font)
    window.getting_started.setStyleSheet("color: #692B28;")  
    window.getting_started.setObjectName("getting_started")

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 130, 711, 91))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
    "border-radius: 15px;\n"
    "color: white;\n"  
    "qproperty-alignment: AlignCenter;\n"  
    "border: 2px solid transparent;\n"  
    "border-color: transparent;\n"
    "padding: 5px;\n"
    "}"
    "QPushButton:hover {\n"
    "background-color: #AEDBE6;\n"  
    "}")
    window.con_button.setObjectName("con_button")

    window.name = QtWidgets.QLabel(window.centralwidget)
    window.name.setGeometry(QtCore.QRect(40, 190, 711, 61))
    window.name.setFont(font1)
    window.name.setStyleSheet("color: #692B28;")
    window.name.setObjectName("name")
    window.wr_name = QtWidgets.QLineEdit(window.centralwidget)
    window.wr_name.setGeometry(QtCore.QRect(150, 200, 231, 41))
    window.wr_name.setFont(font1)
    window.wr_name.setStyleSheet("color: #692B28;")
    window.wr_name.setText("")
    window.wr_name.setFrame(False)
    window.wr_name.setObjectName("wr_name")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.opacity_effect_getting_started = QtWidgets.QGraphicsOpacityEffect()
    window.getting_started.setGraphicsEffect(window.opacity_effect_getting_started)
    window.retranslateUi()

    def go_to_fourth_ui():
        name = window.wr_name.text().split(' ')[0]  
        if name.strip() == '':
            error_dialog = QtWidgets.QMessageBox()
            error_dialog.setIcon(QtWidgets.QMessageBox.Warning)
            error_dialog.setWindowTitle("DepDetect")
            error_dialog.setText("Please enter your name to continue.")

            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
        
            error_dialog.setIconPixmap(icon.pixmap(48, 48))
            error_dialog.setFont(font1)
        
            error_dialog.setStyleSheet("""QMessageBox {
                                      background-color: #FFF5F3;
                                      }
                                      QMessageBox QLabel {
                                      color: #692B28;  # Change text color here
                                      }""")
        
            error_dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
            error_dialog.exec_()
           
        else:
            window.name = name  
            setup_fourth_ui(window, name)

    window.con_button.clicked.connect(go_to_fourth_ui)
    window.wr_name.returnPressed.connect(go_to_fourth_ui)

def setup_fourth_ui(window, name):
    setup_window(window, "FourthWindow")
    set_background(window)
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"  
    QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 43
    base_font_size_2 = 20
    base_font_size_3 = 15

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)

    window.back_button = QtWidgets.QPushButton(window.centralwidget)
    window.back_button.setGeometry(QtCore.QRect(-20, 20, 93, 28))
    font1.setPixelSize(scaled_font_size_3)
    window.back_button.setFont(font1)
    window.back_button.setStyleSheet("background: #9DB9CC;\n"
                                     "border-radius: 10px;\n"
                                     "color: white;\n"  
                                     "qproperty-alignment: AlignCenter;\n"  
                                     "border: 2px solid transparent;\n"  
                                     "border-color: transparent;\n"
                                     "padding: 5px;\n"
                                     "}"
                                     "QPushButton:hover {\n"
                                     "background-color: #AEDBE6;\n"  
                                     "}")
    window.back_button.setObjectName("back_button")

    font_path = "BrixtonSansBld.ttf"  
    QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")
    font.setPixelSize(scaled_font_size_1)

    window.hello_name = QtWidgets.QLabel(window.centralwidget)
    window.hello_name.setGeometry(QtCore.QRect(40, 70, 281, 51))
    window.hello_name.setFont(font)
    window.hello_name.setStyleSheet("color: #692B28;") 
    window.hello_name.setObjectName("hello_name")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 130, 711, 91))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
                                    "border-radius: 15px;\n"
                                    "color: white;\n" 
                                    "qproperty-alignment: AlignCenter;\n"  
                                    "border: 2px solid transparent;\n"  
                                    "border-color: transparent;\n"
                                    "padding: 5px;\n"
                                    "}"
                                    "QPushButton:hover {\n"
                                    "background-color: #AEDBE6;\n"  
                                    "}")
    window.con_button.setObjectName("con_button")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi(name)
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_third_ui():
        setup_third_ui(window)
        window.opacity_effect_getting_started.setOpacity(1.0) 

    window.back_button.clicked.connect(go_to_third_ui)

    def go_to_fifth_ui():
        setup_fifth_ui(window)

    window.con_button.clicked.connect(go_to_fifth_ui)

def setup_fifth_ui(window):
    setup_window(window, "FifthWindow")
    set_background(window)
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"  
    QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"  
    QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 41
    base_font_size_2 = 20
    base_font_size_3 = 15

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)

    window.back_button = QtWidgets.QPushButton(window.centralwidget)
    window.back_button.setGeometry(QtCore.QRect(-20, 20, 93, 28))
    font1.setPixelSize(scaled_font_size_3)
    window.back_button.setFont(font1)
    window.back_button.setStyleSheet("background: #9DB9CC;\n"
                                     "border-radius: 10px;\n"
                                     "color: white;\n"  
                                     "qproperty-alignment: AlignCenter;\n"  
                                     "border: 2px solid transparent;\n"  
                                     "border-color: transparent;\n"
                                     "padding: 5px;\n"
                                     "}"
                                     "QPushButton:hover {\n"
                                     "background-color: #AEDBE6;\n"  
                                     "}")
    window.back_button.setObjectName("back_button")

    window.understanding_better = QtWidgets.QLabel(window.centralwidget)
    window.understanding_better.setGeometry(QtCore.QRect(40, 70, 461, 51))
    font.setPixelSize(scaled_font_size_1)
    window.understanding_better.setFont(font)
    window.understanding_better.setStyleSheet("color: #692B28;")  
    window.understanding_better.setObjectName("understanding_better")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 130, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
                                    "border-radius: 15px;\n"
                                    "color: white;\n"  
                                    "qproperty-alignment: AlignCenter;\n"  
                                    "border: 2px solid transparent;\n" 
                                    "border-color: transparent;\n"
                                    "padding: 5px;\n"
                                    "}"
                                    "QPushButton:hover {\n"
                                    "background-color: #AEDBE6;\n"  
                                    "}")
    window.con_button.setObjectName("con_button")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_fourth_ui():
        setup_fourth_ui(window, window.name)

    window.back_button.clicked.connect(go_to_fourth_ui)

    def go_to_sixth_ui():
        setup_sixth_ui(window, 0)

    window.con_button.clicked.connect(go_to_sixth_ui)


new_data = [
    0,  # Age 25-30
    0,  # Age 30-35
    0,  # Age 35-40
    0,  # Age 40-45
    0,  # Age 45-50
    0,  # Feeling sad No
    0,  # Feeling sad Sometimes
    0,  # Feeling sad Yes
    0,  # Irritable towards people No
    0,  # Irritable towards people Sometimes
    0,  # Irritable towards people Yes
    0,  # Trouble sleeping at night No
    0,  # Trouble sleeping at night Two or more days a week
    0,  # Trouble sleeping at night Yes
    0,  # Problems concentrating or making decisions No
    0,  # Problems concentrating or making decisions Often
    0,  # Problems concentrating or making decisions Yes
    0,  # Loss of appetite No
    0,  # Loss of appetite Not at all
    0,  # Loss of appetite Yes
    0,  # Feeling of guilt Maybe
    0,  # Feeling of guilt No
    0,  # Feeling of guilt Yes
    0,  # Problems of bonding with people No
    0,  # Problems of bonding with people Sometimes
    0,  # Problems of bonding with people Yes
    0,  # Suicide attempt No
    0,  # Suicide attempt Not interested to say
    0,  # Suicide attempt Yes
]

def update_user_input(choice, category_index):
    start_index = category_index * 5
    new_data[start_index:start_index + 5] = [0] * 5
    new_data[start_index + choice - 1] = 1


def update_user_input1(choice, category_index):
    start_index = 5 + (category_index * 3)
    new_data[start_index:start_index + 3] = [0] * 3
    new_data[start_index + choice - 1] = 1

def setup_sixth_ui(window, current_category_index):
    setup_window(window, "SixthWindow")
    set_background(window)     
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"  
    QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"  
    QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 30
    base_font_size_2 = 20
    base_font_size_3 = 19
    base_font_size_4 = 15

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)
    scaled_font_size_4 = int(base_font_size_4 * dpi_scaling_factor)

    window.back_button = QtWidgets.QPushButton(window.centralwidget)
    window.back_button.setGeometry(QtCore.QRect(-20, 20, 93, 28))
    font1.setPixelSize(scaled_font_size_4)
    window.back_button.setFont(font1)
    window.back_button.setStyleSheet("background: #9DB9CC;\n"
         "border-radius: 10px;\n"
         "color: white;\n"  
         "qproperty-alignment: AlignCenter;\n" 
         "border: 2px solid transparent;\n"  
         "border-color: transparent;\n"
         "padding: 5px;\n"
         "}"
         "QPushButton:hover {\n"
         "background-color: #AEDBE6;\n"
         "}")
    window.back_button.setObjectName("back_button")

    window.question = QtWidgets.QLabel(window.centralwidget)
    window.question.setGeometry(QtCore.QRect(40, 70, 511, 51))
    font.setPixelSize(scaled_font_size_1)
    window.question.setFont(font)
    window.question.setStyleSheet("color: #692B28;")
    window.question.setObjectName("question")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 90, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
    "border-radius: 15px;\n"
    "color: white;\n" 
    "qproperty-alignment: AlignCenter;\n" 
    "border: 2px solid transparent;\n"  
    "border-color: transparent;\n"
    "padding: 5px;\n"
    "}"
    "QPushButton:hover {\n"
    "background-color: #AEDBE6;\n"  
    "}")
    window.con_button.setObjectName("con_button")

    window.user_choice = QtWidgets.QLineEdit(window.centralwidget)
    window.user_choice.setGeometry(QtCore.QRect(270, 280, 231, 41))
    window.user_choice.setFont(font1)
    window.user_choice.setStyleSheet("color: #692B28;")
    window.user_choice.setText("")
    window.user_choice.setFrame(False)
    window.user_choice.setObjectName("user_choice")

    window.choice = QtWidgets.QLabel(window.centralwidget)
    window.choice.setGeometry(QtCore.QRect(50, 280, 221, 61))
    window.choice.setFont(font1)
    window.choice.setStyleSheet("color: #122D2E;")
    window.choice.setObjectName("choice")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_fifth_ui():
        setup_fifth_ui(window)

    window.back_button.clicked.connect(go_to_fifth_ui)

    def go_to_seventh_ui():
        user_input = window.user_choice.text()
        if not user_input.isdigit() or not 1 <= int(user_input) <= 5:
            error_dialog = QMessageBox(QMessageBox.Icon.Warning, "Invalid input", "Please enter a number between 1 and 5.")
            error_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
            error_dialog.setIconPixmap(icon.pixmap(48, 48))

            font_path1 = "Belleza-Regular.ttf"
            QtGui.QFontDatabase.addApplicationFont(font_path1)
            font1 = QtGui.QFont("Belleza")
            font1.setPixelSize(scaled_font_size_3)
            error_dialog.setFont(font1)

            error_dialog.exec()
            return

        confirm_dialog = QMessageBox(QMessageBox.Icon.Question, 'Confirm', 
                                "Are you sure about your answer?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
        icon = QtGui.QIcon('Depdetect.png')
        confirm_dialog.setWindowIcon(icon)
        confirm_dialog.setIconPixmap(icon.pixmap(48, 48))

        font_path1 = "Belleza-Regular.ttf"
        QtGui.QFontDatabase.addApplicationFont(font_path1)
        font1 = QtGui.QFont("Belleza")
        font1.setPixelSize(scaled_font_size_3)
        confirm_dialog.setFont(font1)

        reply = confirm_dialog.exec()

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Age Input: {user_input}")
            update_user_input(int(user_input), 0)
            print("Updated user input: ", new_data)
            setup_seventh_ui(window, 0)

    window.user_choice.setFocus()
    window.con_button.clicked.connect(go_to_seventh_ui)
    window.user_choice.returnPressed.connect(go_to_seventh_ui)

def setup_seventh_ui(window, current_category_index):
    setup_window(window, "SeventhWindow")
    set_background(window)     
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"  
    QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"  
    QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 30
    base_font_size_2 = 20
    base_font_size_3 = 19

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)

    window.question2 = QtWidgets.QLabel(window.centralwidget)
    window.question2.setGeometry(QtCore.QRect(40, 55, 711, 91))
    font.setPixelSize(scaled_font_size_1)
    window.question2.setFont(font)
    window.question2.setStyleSheet("color: #692B28;")
    window.question2.setObjectName("question2")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 60, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
    "border-radius: 15px;\n"
    "color: white;\n" 
    "qproperty-alignment: AlignCenter;\n" 
    "border: 2px solid transparent;\n"  
    "border-color: transparent;\n"
    "padding: 5px;\n"
    "}"
    "QPushButton:hover {\n"
    "background-color: #AEDBE6;\n"  
    "}")
    window.con_button.setObjectName("con_button")

    window.user_choice = QtWidgets.QLineEdit(window.centralwidget)
    window.user_choice.setGeometry(QtCore.QRect(258, 230, 231, 41))
    window.user_choice.setFont(font1)
    window.user_choice.setStyleSheet("color: #692B28;")
    window.user_choice.setText("")
    window.user_choice.setFrame(False)
    window.user_choice.setObjectName("user_choice")

    window.choice = QtWidgets.QLabel(window.centralwidget)
    window.choice.setGeometry(QtCore.QRect(44, 232, 221, 61))
    window.choice.setFont(font1)
    window.choice.setStyleSheet("color: #122D2E;")
    window.choice.setObjectName("choice")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_eighth_ui():
        user_input = window.user_choice.text()
        if not user_input.isdigit() or not 1 <= int(user_input) <= 3:
            error_dialog = QMessageBox(QMessageBox.Icon.Warning, "Invalid input", "Please enter a number between 1 and 3.")
            error_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
            error_dialog.setIconPixmap(icon.pixmap(48, 48))

            font_path1 = "Belleza-Regular.ttf"
            QtGui.QFontDatabase.addApplicationFont(font_path1)
            font1 = QtGui.QFont("Belleza")
            font1.setPixelSize(scaled_font_size_3)
            error_dialog.setFont(font1)

            error_dialog.exec()
            return

        confirm_dialog = QMessageBox(QMessageBox.Icon.Question, 'Confirm', 
                                "Are you sure about your answer?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
        icon = QtGui.QIcon('Depdetect.png')
        confirm_dialog.setWindowIcon(icon)
        confirm_dialog.setIconPixmap(icon.pixmap(48, 48))

        font_path1 = "Belleza-Regular.ttf"
        QtGui.QFontDatabase.addApplicationFont(font_path1)
        font1 = QtGui.QFont("Belleza")
        font1.setPixelSize(scaled_font_size_3)
        confirm_dialog.setFont(font1)

        reply = confirm_dialog.exec()

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Feeling Sad Input: {user_input}")
            update_user_input1(int(user_input), 0)
            print("Updated user input: ", new_data)
            setup_eighth_ui(window, 1)

    window.user_choice.setFocus()
    window.con_button.clicked.connect(go_to_eighth_ui)
    window.user_choice.returnPressed.connect(go_to_eighth_ui)

def setup_eighth_ui(window, current_category_index):
    setup_window(window, "EighthWindow")
    set_background(window)
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 27
    base_font_size_2 = 20
    base_font_size_3 = 19

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)

    window.question3 = QtWidgets.QLabel(window.centralwidget)
    window.question3.setGeometry(QtCore.QRect(40, 58, 711, 91))
    font.setPixelSize(scaled_font_size_1)
    window.question3.setFont(font)
    window.question3.setStyleSheet("color: #692B28;")
    window.question3.setObjectName("question3")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 60, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
                                    "border-radius: 15px;\n"
                                    "color: white;\n"  
                                    "qproperty-alignment: AlignCenter;\n" 
                                    "border: 2px solid transparent;\n" 
                                    "border-color: transparent;\n"
                                    "padding: 5px;\n"
                                    "}"
                                    "QPushButton:hover {\n"
                                    "background-color: #AEDBE6;\n"  
                                    "}")
    window.con_button.setObjectName("con_button")

    window.user_choice = QtWidgets.QLineEdit(window.centralwidget)
    window.user_choice.setGeometry(QtCore.QRect(258, 230, 231, 41))
    window.user_choice.setFont(font1)
    window.user_choice.setStyleSheet("color: #692B28;")
    window.user_choice.setText("")
    window.user_choice.setFrame(False)
    window.user_choice.setObjectName("user_choice")

    window.choice = QtWidgets.QLabel(window.centralwidget)
    window.choice.setGeometry(QtCore.QRect(44, 232, 221, 61))
    window.choice.setFont(font1)
    window.choice.setStyleSheet("color: #122D2E;")
    window.choice.setObjectName("choice")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_ninth_ui():
        user_input = window.user_choice.text()
        if not user_input.isdigit() or not 1 <= int(user_input) <= 3:
            error_dialog = QMessageBox(QMessageBox.Icon.Warning, "Invalid input", "Please enter a number between 1 and 3.")
            error_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
            error_dialog.setIconPixmap(icon.pixmap(48, 48))

            font_path1 = "Belleza-Regular.ttf"
            QtGui.QFontDatabase.addApplicationFont(font_path1)
            font1 = QtGui.QFont("Belleza")
            font1.setPixelSize(scaled_font_size_3)
            error_dialog.setFont(font1)

            error_dialog.exec()
            return

        confirm_dialog = QMessageBox(QMessageBox.Icon.Question, 'Confirm', 
                                "Are you sure about your answer?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
        icon = QtGui.QIcon('Depdetect.png')
        confirm_dialog.setWindowIcon(icon)
        confirm_dialog.setIconPixmap(icon.pixmap(48, 48))

        font_path1 = "Belleza-Regular.ttf"
        QtGui.QFontDatabase.addApplicationFont(font_path1)
        font1 = QtGui.QFont("Belleza")
        font1.setPixelSize(scaled_font_size_3)
        confirm_dialog.setFont(font1)

        reply = confirm_dialog.exec()

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Irritable towards people Input: {user_input}")
            update_user_input1(int(user_input), 1)
            print("Updated user input: ", new_data)
            setup_ninth_ui(window, 2)

    window.user_choice.setFocus()
    window.con_button.clicked.connect(go_to_ninth_ui)
    window.user_choice.returnPressed.connect(go_to_ninth_ui)

def setup_ninth_ui(window, current_category_index):
    setup_window(window, "NinthWindow")
    set_background(window)
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 27
    base_font_size_2 = 20
    base_font_size_3 = 19

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)

    window.question4 = QtWidgets.QLabel(window.centralwidget)
    window.question4.setGeometry(QtCore.QRect(40, 58, 711, 91))
    font.setPixelSize(scaled_font_size_1)
    window.question4.setFont(font)
    window.question4.setStyleSheet("color: #692B28;")
    window.question4.setObjectName("question3")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 60, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
                                    "border-radius: 15px;\n"
                                    "color: white;\n"  
                                    "qproperty-alignment: AlignCenter;\n" 
                                    "border: 2px solid transparent;\n" 
                                    "border-color: transparent;\n"
                                    "padding: 5px;\n"
                                    "}"
                                    "QPushButton:hover {\n"
                                    "background-color: #AEDBE6;\n"  
                                    "}")
    window.con_button.setObjectName("con_button")

    window.user_choice = QtWidgets.QLineEdit(window.centralwidget)
    window.user_choice.setGeometry(QtCore.QRect(258, 230, 231, 41))
    window.user_choice.setFont(font1)
    window.user_choice.setStyleSheet("color: #692B28;")
    window.user_choice.setText("")
    window.user_choice.setFrame(False)
    window.user_choice.setObjectName("user_choice")

    window.choice = QtWidgets.QLabel(window.centralwidget)
    window.choice.setGeometry(QtCore.QRect(44, 232, 221, 61))
    window.choice.setFont(font1)
    window.choice.setStyleSheet("color: #122D2E;")
    window.choice.setObjectName("choice")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_tenth_ui():
        user_input = window.user_choice.text()
        if not user_input.isdigit() or not 1 <= int(user_input) <= 3:
            error_dialog = QMessageBox(QMessageBox.Icon.Warning, "Invalid input", "Please enter a number between 1 and 3.")
            error_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
            error_dialog.setIconPixmap(icon.pixmap(48, 48))

            font_path1 = "Belleza-Regular.ttf"
            QtGui.QFontDatabase.addApplicationFont(font_path1)
            font1 = QtGui.QFont("Belleza")
            font1.setPixelSize(scaled_font_size_3)
            error_dialog.setFont(font1)

            error_dialog.exec()
            return

        confirm_dialog = QMessageBox(QMessageBox.Icon.Question, 'Confirm', 
                                "Are you sure about your answer?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
        icon = QtGui.QIcon('Depdetect.png')
        confirm_dialog.setWindowIcon(icon)
        confirm_dialog.setIconPixmap(icon.pixmap(48, 48))

        font_path1 = "Belleza-Regular.ttf"
        QtGui.QFontDatabase.addApplicationFont(font_path1)
        font1 = QtGui.QFont("Belleza")
        font1.setPixelSize(scaled_font_size_3)
        confirm_dialog.setFont(font1)

        reply = confirm_dialog.exec()

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Trouble sleeping at night Input: {user_input}")
            update_user_input1(int(user_input), 2)
            print("Updated user input: ", new_data)
            setup_tenth_ui(window, 3)

    window.user_choice.setFocus()
    window.con_button.clicked.connect(go_to_tenth_ui)
    window.user_choice.returnPressed.connect(go_to_tenth_ui)


def setup_tenth_ui(window, current_category_index):
    setup_window(window, "TenthWindow")
    set_background(window)
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 27
    base_font_size_2 = 20
    base_font_size_3 = 19

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)

    window.question5 = QtWidgets.QLabel(window.centralwidget)
    window.question5.setGeometry(QtCore.QRect(40, 58, 711, 91))
    font.setPixelSize(scaled_font_size_1)
    window.question5.setFont(font)
    window.question5.setStyleSheet("color: #692B28;")
    window.question5.setObjectName("question3")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 60, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
                                    "border-radius: 15px;\n"
                                    "color: white;\n"  
                                    "qproperty-alignment: AlignCenter;\n" 
                                    "border: 2px solid transparent;\n" 
                                    "border-color: transparent;\n"
                                    "padding: 5px;\n"
                                    "}"
                                    "QPushButton:hover {\n"
                                    "background-color: #AEDBE6;\n"  
                                    "}")
    window.con_button.setObjectName("con_button")

    window.user_choice = QtWidgets.QLineEdit(window.centralwidget)
    window.user_choice.setGeometry(QtCore.QRect(258, 230, 231, 41))
    window.user_choice.setFont(font1)
    window.user_choice.setStyleSheet("color: #692B28;")
    window.user_choice.setText("")
    window.user_choice.setFrame(False)
    window.user_choice.setObjectName("user_choice")

    window.choice = QtWidgets.QLabel(window.centralwidget)
    window.choice.setGeometry(QtCore.QRect(44, 232, 221, 61))
    window.choice.setFont(font1)
    window.choice.setStyleSheet("color: #122D2E;")
    window.choice.setObjectName("choice")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_eleventh_ui():
        user_input = window.user_choice.text()
        if not user_input.isdigit() or not 1 <= int(user_input) <= 3:
            error_dialog = QMessageBox(QMessageBox.Icon.Warning, "Invalid input", "Please enter a number between 1 and 3.")
            error_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
            error_dialog.setIconPixmap(icon.pixmap(48, 48))

            font_path1 = "Belleza-Regular.ttf"
            QtGui.QFontDatabase.addApplicationFont(font_path1)
            font1 = QtGui.QFont("Belleza")
            font1.setPixelSize(scaled_font_size_3)
            error_dialog.setFont(font1)

            error_dialog.exec()
            return

        confirm_dialog = QMessageBox(QMessageBox.Icon.Question, 'Confirm', 
                                "Are you sure about your answer?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
        icon = QtGui.QIcon('Depdetect.png')
        confirm_dialog.setWindowIcon(icon)
        confirm_dialog.setIconPixmap(icon.pixmap(48, 48))

        font_path1 = "Belleza-Regular.ttf"
        QtGui.QFontDatabase.addApplicationFont(font_path1)
        font1 = QtGui.QFont("Belleza")
        font1.setPixelSize(scaled_font_size_3)
        confirm_dialog.setFont(font1)

        reply = confirm_dialog.exec()

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Problems concentrating or making decision Input: {user_input}")
            update_user_input1(int(user_input), 3)
            print("Updated user input: ", new_data)
            setup_eleventh_ui(window, 4)

    window.user_choice.setFocus()
    window.con_button.clicked.connect(go_to_eleventh_ui)
    window.user_choice.returnPressed.connect(go_to_eleventh_ui)

def setup_eleventh_ui(window, current_category_index):
    setup_window(window, "EleventhWindow")
    set_background(window)
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 27
    base_font_size_2 = 20
    base_font_size_3 = 19

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)
    
    window.question6 = QtWidgets.QLabel(window.centralwidget)
    window.question6.setGeometry(QtCore.QRect(40, 58, 711, 91))
    font.setPixelSize(scaled_font_size_1)
    window.question6.setFont(font)
    window.question6.setStyleSheet("color: #692B28;")
    window.question6.setObjectName("question3")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 60, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
                                    "border-radius: 15px;\n"
                                    "color: white;\n"  
                                    "qproperty-alignment: AlignCenter;\n" 
                                    "border: 2px solid transparent;\n" 
                                    "border-color: transparent;\n"
                                    "padding: 5px;\n"
                                    "}"
                                    "QPushButton:hover {\n"
                                    "background-color: #AEDBE6;\n"  
                                    "}")
    window.con_button.setObjectName("con_button")

    window.user_choice = QtWidgets.QLineEdit(window.centralwidget)
    window.user_choice.setGeometry(QtCore.QRect(258, 230, 231, 41))
    window.user_choice.setFont(font1)
    window.user_choice.setStyleSheet("color: #692B28;")
    window.user_choice.setText("")
    window.user_choice.setFrame(False)
    window.user_choice.setObjectName("user_choice")

    window.choice = QtWidgets.QLabel(window.centralwidget)
    window.choice.setGeometry(QtCore.QRect(44, 232, 221, 61))
    window.choice.setFont(font1)
    window.choice.setStyleSheet("color: #122D2E;")
    window.choice.setObjectName("choice")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_twelfth_ui():
        user_input = window.user_choice.text()
        if not user_input.isdigit() or not 1 <= int(user_input) <= 3:
            error_dialog = QMessageBox(QMessageBox.Icon.Warning, "Invalid input", "Please enter a number between 1 and 3.")
            error_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
            error_dialog.setIconPixmap(icon.pixmap(48, 48))

            font_path1 = "Belleza-Regular.ttf"
            QtGui.QFontDatabase.addApplicationFont(font_path1)
            font1 = QtGui.QFont("Belleza")
            font1.setPixelSize(scaled_font_size_3)
            error_dialog.setFont(font1)

            error_dialog.exec()
            return

        confirm_dialog = QMessageBox(QMessageBox.Icon.Question, 'Confirm', 
                                "Are you sure about your answer?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
        icon = QtGui.QIcon('Depdetect.png')
        confirm_dialog.setWindowIcon(icon)
        confirm_dialog.setIconPixmap(icon.pixmap(48, 48))

        font_path1 = "Belleza-Regular.ttf"
        QtGui.QFontDatabase.addApplicationFont(font_path1)
        font1 = QtGui.QFont("Belleza")
        font1.setPixelSize(scaled_font_size_3)
        confirm_dialog.setFont(font1)

        reply = confirm_dialog.exec()

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Loss of appetite: {user_input}")
            update_user_input1(int(user_input), 4)
            print("Updated user input: ", new_data)
            setup_twelfth_ui(window, 5)

    window.user_choice.setFocus()
    window.con_button.clicked.connect(go_to_twelfth_ui)
    window.user_choice.returnPressed.connect(go_to_twelfth_ui)

def setup_twelfth_ui(window, current_category_index):
    setup_window(window, "TwelfthWindow")
    set_background(window)
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 27
    base_font_size_2 = 20
    base_font_size_3 = 19

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)

    window.question7 = QtWidgets.QLabel(window.centralwidget)
    window.question7.setGeometry(QtCore.QRect(40, 58, 711, 91))
    font.setPixelSize(scaled_font_size_1)
    window.question7.setFont(font)
    window.question7.setStyleSheet("color: #692B28;")
    window.question7.setObjectName("question3")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 60, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
                                    "border-radius: 15px;\n"
                                    "color: white;\n"  
                                    "qproperty-alignment: AlignCenter;\n" 
                                    "border: 2px solid transparent;\n" 
                                    "border-color: transparent;\n"
                                    "padding: 5px;\n"
                                    "}"
                                    "QPushButton:hover {\n"
                                    "background-color: #AEDBE6;\n"  
                                    "}")
    window.con_button.setObjectName("con_button")

    window.user_choice = QtWidgets.QLineEdit(window.centralwidget)
    window.user_choice.setGeometry(QtCore.QRect(258, 230, 231, 41))
    window.user_choice.setFont(font1)
    window.user_choice.setStyleSheet("color: #692B28;")
    window.user_choice.setText("")
    window.user_choice.setFrame(False)
    window.user_choice.setObjectName("user_choice")

    window.choice = QtWidgets.QLabel(window.centralwidget)
    window.choice.setGeometry(QtCore.QRect(44, 232, 221, 61))
    window.choice.setFont(font1)
    window.choice.setStyleSheet("color: #122D2E;")
    window.choice.setObjectName("choice")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_thirteenth_ui():
        user_input = window.user_choice.text()
        if not user_input.isdigit() or not 1 <= int(user_input) <= 3:
            error_dialog = QMessageBox(QMessageBox.Icon.Warning, "Invalid input", "Please enter a number between 1 and 3.")
            error_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
            error_dialog.setIconPixmap(icon.pixmap(48, 48))

            font_path1 = "Belleza-Regular.ttf"
            QtGui.QFontDatabase.addApplicationFont(font_path1)
            font1 = QtGui.QFont("Belleza")
            font1.setPixelSize(scaled_font_size_3)
            error_dialog.setFont(font1)

            error_dialog.exec()
            return

        confirm_dialog = QMessageBox(QMessageBox.Icon.Question, 'Confirm', 
                                "Are you sure about your answer?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
        icon = QtGui.QIcon('Depdetect.png')
        confirm_dialog.setWindowIcon(icon)
        confirm_dialog.setIconPixmap(icon.pixmap(48, 48))

        font_path1 = "Belleza-Regular.ttf"
        QtGui.QFontDatabase.addApplicationFont(font_path1)
        font1 = QtGui.QFont("Belleza")
        font1.setPixelSize(scaled_font_size_3)
        confirm_dialog.setFont(font1)

        reply = confirm_dialog.exec()

        if reply == QMessageBox.StandardButton.Yes:
            print(f" Feelings of guilt Input: {user_input}")
            update_user_input1(int(user_input), 5)
            print("Updated user input: ", new_data)
            setup_thirteenth_ui(window, 6)

    window.user_choice.setFocus()
    window.con_button.clicked.connect(go_to_thirteenth_ui)
    window.user_choice.returnPressed.connect(go_to_thirteenth_ui)

def setup_thirteenth_ui(window, current_category_index):
    setup_window(window, "ThirteenthWindow")
    set_background(window)
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 27
    base_font_size_2 = 20
    base_font_size_3 = 19

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)

    window.question8 = QtWidgets.QLabel(window.centralwidget)
    window.question8.setGeometry(QtCore.QRect(40, 58, 711, 91))
    font.setPixelSize(scaled_font_size_1)
    window.question8.setFont(font)
    window.question8.setStyleSheet("color: #692B28;")
    window.question8.setObjectName("question3")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 60, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
                                    "border-radius: 15px;\n"
                                    "color: white;\n"  
                                    "qproperty-alignment: AlignCenter;\n" 
                                    "border: 2px solid transparent;\n" 
                                    "border-color: transparent;\n"
                                    "padding: 5px;\n"
                                    "}"
                                    "QPushButton:hover {\n"
                                    "background-color: #AEDBE6;\n"  
                                    "}")
    window.con_button.setObjectName("con_button")

    window.user_choice = QtWidgets.QLineEdit(window.centralwidget)
    window.user_choice.setGeometry(QtCore.QRect(258, 230, 231, 41))
    window.user_choice.setFont(font1)
    window.user_choice.setStyleSheet("color: #692B28;")
    window.user_choice.setText("")
    window.user_choice.setFrame(False)
    window.user_choice.setObjectName("user_choice")

    window.choice = QtWidgets.QLabel(window.centralwidget)
    window.choice.setGeometry(QtCore.QRect(44, 232, 221, 61))
    window.choice.setFont(font1)
    window.choice.setStyleSheet("color: #122D2E;")
    window.choice.setObjectName("choice")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_fourteenth_ui():
        user_input = window.user_choice.text()
        if not user_input.isdigit() or not 1 <= int(user_input) <= 3:
            error_dialog = QMessageBox(QMessageBox.Icon.Warning, "Invalid input", "Please enter a number between 1 and 3.")
            error_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
            error_dialog.setIconPixmap(icon.pixmap(48, 48))

            font_path1 = "Belleza-Regular.ttf"
            QtGui.QFontDatabase.addApplicationFont(font_path1)
            font1 = QtGui.QFont("Belleza")
            font1.setPixelSize(scaled_font_size_3)
            error_dialog.setFont(font1)

            error_dialog.exec()
            return

        confirm_dialog = QMessageBox(QMessageBox.Icon.Question, 'Confirm', 
                                "Are you sure about your answer?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
        icon = QtGui.QIcon('Depdetect.png')
        confirm_dialog.setWindowIcon(icon)
        confirm_dialog.setIconPixmap(icon.pixmap(48, 48))

        font_path1 = "Belleza-Regular.ttf"
        QtGui.QFontDatabase.addApplicationFont(font_path1)
        font1 = QtGui.QFont("Belleza")
        font1.setPixelSize(scaled_font_size_3)
        confirm_dialog.setFont(font1)

        reply = confirm_dialog.exec()

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Bonding with people Input: {user_input}")
            update_user_input1(int(user_input), 6)
            print("Updated user input: ", new_data)
            setup_fourteenth_ui(window, 7)

    window.user_choice.setFocus()
    window.con_button.clicked.connect(go_to_fourteenth_ui)
    window.user_choice.returnPressed.connect(go_to_fourteenth_ui)


def setup_fourteenth_ui(window, current_category_index):
    setup_window(window, "FourteenthWindow")
    set_background(window)
    setup_line(window)

    font_path1 = "Belleza-Regular.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path1)
    font1 = QtGui.QFont("Belleza")

    font_path = "BrixtonSansBld.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 27
    base_font_size_2 = 20
    base_font_size_3 = 19

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
    scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
    scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)

    window.question9 = QtWidgets.QLabel(window.centralwidget)
    window.question9.setGeometry(QtCore.QRect(40, 58, 711, 91))
    font.setPixelSize(scaled_font_size_1)
    window.question9.setFont(font)
    window.question9.setStyleSheet("color: #692B28;")
    window.question9.setObjectName("question3")

    font1.setPixelSize(scaled_font_size_2)

    window.info1 = QtWidgets.QLabel(window.centralwidget)
    window.info1.setGeometry(QtCore.QRect(40, 60, 741, 261))
    window.info1.setFont(font1)
    window.info1.setStyleSheet("color: #122D2E;")
    window.info1.setObjectName("info1")

    window.con_button = QtWidgets.QPushButton(window.centralwidget)
    window.con_button.setGeometry(QtCore.QRect(480, 390, 261, 41))
    window.con_button.setFont(font1)
    window.con_button.setStyleSheet("background: #9DB9CC;\n"
                                    "border-radius: 15px;\n"
                                    "color: white;\n"  
                                    "qproperty-alignment: AlignCenter;\n" 
                                    "border: 2px solid transparent;\n" 
                                    "border-color: transparent;\n"
                                    "padding: 5px;\n"
                                    "}"
                                    "QPushButton:hover {\n"
                                    "background-color: #AEDBE6;\n"  
                                    "}")
    window.con_button.setObjectName("con_button")

    window.user_choice = QtWidgets.QLineEdit(window.centralwidget)
    window.user_choice.setGeometry(QtCore.QRect(258, 230, 231, 41))
    window.user_choice.setFont(font1)
    window.user_choice.setStyleSheet("color: #692B28;")
    window.user_choice.setText("")
    window.user_choice.setFrame(False)
    window.user_choice.setObjectName("user_choice")

    window.choice = QtWidgets.QLabel(window.centralwidget)
    window.choice.setGeometry(QtCore.QRect(44, 232, 221, 61))
    window.choice.setFont(font1)
    window.choice.setStyleSheet("color: #122D2E;")
    window.choice.setObjectName("choice")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)

    def go_to_loading_screen():
        user_input = window.user_choice.text()
        if not user_input.isdigit() or not 1 <= int(user_input) <= 3:
            error_dialog = QMessageBox(QMessageBox.Icon.Warning, "Invalid input", "Please enter a number between 1 and 3.")
            error_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
            icon = QtGui.QIcon('Depdetect.png')
            error_dialog.setWindowIcon(icon)
            error_dialog.setIconPixmap(icon.pixmap(48, 48))

            font_path1 = "Belleza-Regular.ttf"
            QtGui.QFontDatabase.addApplicationFont(font_path1)
            font1 = QtGui.QFont("Belleza")
            font1.setPixelSize(scaled_font_size_3)
            error_dialog.setFont(font1)

            error_dialog.exec()
            return

        confirm_dialog = QMessageBox(QMessageBox.Icon.Question, 'Confirm', 
                                "Are you sure about your answer?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet("""QMessageBox {
                                  background-color: #FFF5F3;
                                  }
                                  QMessageBox QLabel {
                                  color: #692B28;  # Change text color here
                                  }""")
        icon = QtGui.QIcon('Depdetect.png')
        confirm_dialog.setWindowIcon(icon)
        confirm_dialog.setIconPixmap(icon.pixmap(48, 48))

        font_path1 = "Belleza-Regular.ttf"
        QtGui.QFontDatabase.addApplicationFont(font_path1)
        font1 = QtGui.QFont("Belleza")
        font1.setPixelSize(scaled_font_size_3)
        confirm_dialog.setFont(font1)

        reply = confirm_dialog.exec()

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Suicide attempt Input: {user_input}")
            update_user_input1(int(user_input), 7)
            print("Updated user input: ", new_data)
            loading_data(window)

    window.user_choice.setFocus()
    window.con_button.clicked.connect(go_to_loading_screen)
    window.user_choice.returnPressed.connect(go_to_loading_screen)

progress_timer = None

def loading_data(window):
    setup_window(window, "LoadingWindow") 
    set_background(window)

    font_path = "BrixtonSansBld.ttf"
    QtGui.QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Brixton Sans TC")

    app = QApplication.instance()
    dpi_scaling_factor = app.devicePixelRatio()

    base_font_size_1 = 41

    scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)

    window.loading_label = QtWidgets.QLabel(window.centralwidget)
    window.loading_label.setGeometry(QtCore.QRect(280, 200, 711, 91))
    font.setPixelSize(scaled_font_size_1)
    window.loading_label.setFont(font)
    window.loading_label.setStyleSheet("color: #692B28;")
    window.loading_label.setObjectName("loading_label")
    window.loading_label.setText("Analyzing data...")

    window.progress_bar = QtWidgets.QProgressBar(window.centralwidget)
    window.progress_bar.setGeometry(QtCore.QRect(225, 275, 400, 25))
    window.progress_bar.setMinimum(0)
    window.progress_bar.setMaximum(100)
    window.progress_bar.setValue(0)
    window.progress_bar.setStyleSheet("QProgressBar { color: #122D2E; background-color: #FFFFFF; border: 1px solid #9DB9CC; border-radius: 10px; }"
                                      "QProgressBar::chunk { background-color: #9DB9CC; }")

    window.setCentralWidget(window.centralwidget)
    window.menubar = QtWidgets.QMenuBar(window)
    window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
    window.menubar.setObjectName("menubar")
    window.setMenuBar(window.menubar)
    window.statusbar = QtWidgets.QStatusBar(window)
    window.statusbar.setObjectName("statusbar")
    window.setStatusBar(window.statusbar)

    window.retranslateUi()
    QtCore.QMetaObject.connectSlotsByName(window)
    
    global progress_timer
    progress_timer = QTimer()
    progress_timer.timeout.connect(lambda: update_progress(window))
    progress_timer.start(25)  

def update_progress(window):
    current_value = window.progress_bar.value()
    if current_value >= 100:
        progress_timer.stop()
        go_to_results(window)
    else:
        window.progress_bar.setValue(current_value + 1)
                
def go_to_results(window):
        print( 'Confusion Matrix:\n')
        print(confusion_matrix(y_test, model_predict))

        print( '\nClassification Report:\n')
        print(classification_report(y_test, model_predict))
            
        data_prediction = model.predict([new_data])
            
        setup_results_ui(window)
        window.confusion_matrix_result.setText("Confusion Matrix:\n" + str(confusion_matrix(y_test, model_predict)))
        window.classification_report_result.setText("\t\t\t\tClassification Report:\n" + str(classification_report(y_test, model_predict)))
            

        output_text = "Is the person currently experiencing depression? (1 - Yes, 0 - No) : " + str(data_prediction[0])
        window.output.setText(output_text)


        if data_prediction[0] == 0:
            message = "Based on the information you provided, it's unlikely that you're experiencing depression.\nHowever, if you feel distressed, please seek help from a professional."
        else:
            message = "Based on the information you provided, there's a chance you could be experiencing symptoms of\ndepression. It's important to consult with a mental health professional to get the most accurate\ndiagnosis and best care possible."

        window.message_label.setText(message)

def setup_results_ui(window):
     setup_window(window, "ResultsWindow")
     set_background(window)     
     setup_line(window)

     font_path2 = "MonospaceBold.ttf"  
     QtGui.QFontDatabase.addApplicationFont(font_path2)
     font2 = QtGui.QFont("Monospace")
    
     font_path1 = "Belleza-Regular.ttf"  
     QtGui.QFontDatabase.addApplicationFont(font_path1)
     font1 = QtGui.QFont("Belleza")

     font_path = "BrixtonSansBld.ttf"  
     QtGui.QFontDatabase.addApplicationFont(font_path)
     font = QtGui.QFont("Brixton Sans TC")

     app = QApplication.instance()
     dpi_scaling_factor = app.devicePixelRatio()

     base_font_size_1 = 39
     base_font_size_2 = 17
     base_font_size_3 = 18
     base_font_size_4 = 12

     scaled_font_size_1 = int(base_font_size_1 * dpi_scaling_factor)
     scaled_font_size_2 = int(base_font_size_2 * dpi_scaling_factor)
     scaled_font_size_3 = int(base_font_size_3 * dpi_scaling_factor)
     scaled_font_size_4 = int(base_font_size_4 * dpi_scaling_factor)

     window.try_again_button = QtWidgets.QPushButton(window.centralwidget)
     window.try_again_button.setGeometry(QtCore.QRect(-20, 20, 93, 28))
     font1.setPixelSize(scaled_font_size_4)
     window.try_again_button.setFont(font1)
     window.try_again_button.setStyleSheet("background: #9DB9CC;\n"
                                           "border-radius: 10px;\n"
                                           "color: white;\n"  
                                           "qproperty-alignment: AlignCenter;\n"  
                                           "border: 2px solid transparent;\n"  
                                           "border-color: transparent;\n"
                                           "padding: 5px;\n"
                                           "}"
                                           "QPushButton:hover {\n"
                                           "background-color: #AEDBE6;\n"  
                                           "}")
     window.try_again_button.setObjectName("try_again_button")
     
     window.results = QtWidgets.QLabel(window.centralwidget)
     window.results.setGeometry(QtCore.QRect(40, 56, 711, 91))
     font.setPixelSize(scaled_font_size_1)
     window.results.setFont(font)
     window.results.setStyleSheet("color: #692B28;")
     window.results.setObjectName("question3")

     window.confusion_matrix_result = QtWidgets.QLabel(window.centralwidget)
     window.confusion_matrix_result.setGeometry(QtCore.QRect(40, 57, 741, 261))
     font2.setPixelSize(scaled_font_size_2)
     window.confusion_matrix_result.setFont(font2)
     window.confusion_matrix_result.setStyleSheet("color: #122D2E;")
     window.confusion_matrix_result.setObjectName("confusion_matrix_result")

     window.classification_report_result = QtWidgets.QLabel(window.centralwidget)
     window.classification_report_result.setGeometry(QtCore.QRect(220, 150, 760, 240))
     font2.setPixelSize(scaled_font_size_2)
     window.classification_report_result.setFont(font2)
     window.classification_report_result.setStyleSheet("color: #122D2E;")
     window.classification_report_result.setObjectName("classification_report_result")
  
     window.line2 = QtWidgets.QFrame(window.centralwidget)
     window.line2.setGeometry(QtCore.QRect(40, 370, 711, 16))
     window.line2.setMidLineWidth(4)
     window.line2.setFrameShape(QtWidgets.QFrame.HLine)
     window.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
     window.line2.setObjectName("line2")
    
     window.output = QtWidgets.QLabel(window.centralwidget)
     window.output.setGeometry(QtCore.QRect(40, 265, 741, 261))
     font1.setPixelSize(scaled_font_size_3)
     window.output.setFont(font1)
     window.output.setStyleSheet("color: #122D2E;")
     window.output.setObjectName("output")

     window.message_label = QtWidgets.QLabel(window.centralwidget)
     window.message_label.setGeometry(QtCore.QRect(42, 395, 700, 100))  
     font1.setPixelSize(scaled_font_size_3)  
     window.message_label.setFont(font1)
     window.message_label.setStyleSheet("color: #122D2E;")
     window.message_label.setObjectName("message_label")
    
     window.setCentralWidget(window.centralwidget)
     window.menubar = QtWidgets.QMenuBar(window)
     window.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
     window.menubar.setObjectName("menubar")
     window.setMenuBar(window.menubar)
     window.statusbar = QtWidgets.QStatusBar(window)
     window.statusbar.setObjectName("statusbar")
     window.setStatusBar(window.statusbar)

     window.retranslateUi()
     QtCore.QMetaObject.connectSlotsByName(window)

     def go_to_third_ui():
        setup_third_ui(window)
        window.opacity_effect_getting_started.setOpacity(1.0)  

     window.try_again_button.clicked.connect(lambda: go_to_third_ui())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        setup_first_ui(self)
        self.setWindowIcon(QtGui.QIcon('Depdetect.png'))
        self.setWindowTitle("DepDetect")

        self.timer1 = QtCore.QTimer()
        self.timer1.timeout.connect(self.on_fade_animation_finished)
        self.timer1.setSingleShot(True)

        self.timer2 = QtCore.QTimer()
        self.timer2.timeout.connect(self.on_second_fade_animation_finished)
        self.timer2.setSingleShot(True)

        self.start_fade_animation_1()

    def start_fade_animation_1(self):
        self.fade_animation_design1 = QtCore.QPropertyAnimation(
            self.opacity_effect_design1, b"opacity")
        self.fade_animation_design1.setDuration(2000)
        self.fade_animation_design1.setStartValue(0.0)
        self.fade_animation_design1.setEndValue(1.0)

        self.fade_animation_titlename = QtCore.QPropertyAnimation(
            self.opacity_effect_titlename, b"opacity")
        self.fade_animation_titlename.setDuration(2000)
        self.fade_animation_titlename.setStartValue(0.0)
        self.fade_animation_titlename.setEndValue(1.0)

        self.fade_animation_design1.start()
        self.fade_animation_titlename.start()

        self.fade_animation_titlename.finished.connect(self.start_transition_timer_1)

    def start_transition_timer_1(self):
        self.timer1.start(1000)  

    def on_fade_animation_finished(self):
        setup_second_ui(self)
        self.start_fade_animation_2()

    def start_fade_animation_2(self):
        self.fade_animation_titlename_2 = QtCore.QPropertyAnimation(
            self.opacity_effect_titlename, b"opacity")
        self.fade_animation_titlename_2.setDuration(1000)
        self.fade_animation_titlename_2.setStartValue(0.0)
        self.fade_animation_titlename_2.setEndValue(1.0)

        self.fade_animation_titlename_3 = QtCore.QPropertyAnimation(
            self.opacity_effect_titlename_2, b"opacity")
        self.fade_animation_titlename_3.setDuration(1000)
        self.fade_animation_titlename_3.setStartValue(0.0)
        self.fade_animation_titlename_3.setEndValue(1.0)

        self.fade_animation_titlename_2.start()
        self.fade_animation_titlename_3.start()

        self.fade_animation_titlename_3.finished.connect(self.start_transition_timer_2)

    def start_transition_timer_2(self):
        self.timer2.start(1000)  

    def on_second_fade_animation_finished(self):
        setup_third_ui(self)
        self.start_fade_animation_3()

    def start_fade_animation_3(self):
        self.fade_animation_getting_started = QtCore.QPropertyAnimation(
            self.opacity_effect_getting_started, b"opacity")
        self.fade_animation_getting_started.setDuration(1000)
        self.fade_animation_getting_started.setStartValue(0.0)
        self.fade_animation_getting_started.setEndValue(1.0)

        self.fade_animation_getting_started.start()

    def retranslateUi(self, name=""):
        _translate = QtCore.QCoreApplication.translate
        if self.objectName() == "FirstWindow":
            self.setWindowTitle(_translate("FirstWindow", "DepDetect"))
            self.titlename.setText(_translate("FirstWindow", "DepDetect"))
            
        elif self.objectName() == "SecondWindow":
            self.setWindowTitle(_translate("SecondWindow", "DepDetect"))
            self.titlename.setText(_translate("SecondWindow", "Your Personal"))
            self.titlename_2.setText(_translate("SecondWindow", "Depression Predictor"))
            
        elif self.objectName() == "ThirdWindow":
            self.setWindowTitle(_translate("ThirdWindow", "Depdetect"))
            self.getting_started.setText(_translate("ThirdWindow", "Getting Started"))
            self.info1.setText(_translate("ThirdWindow", "DepDetect is designed specifically for adults aged 25-50. Let\'s begin by getting to know\n"
                "you a little better. Please write your name below.\n"
                ""))
            self.con_button.setText(_translate("ThirdWindow", "Continue"))
            self.name.setText(_translate("ThirdWindow", "Your Name:"))
            
        elif self.objectName() == "FourthWindow":
            self.setWindowTitle(_translate("FourthWindow", "DepDetect"))
            self.back_button.setText(_translate("FourthWindow", "  back"))
            self.hello_name.setText(_translate("FourthWindow", f"Hello, {name        }"))
            self.info1.setText(_translate("FourthWindow", "At DepDetect, we\'re here to help provide insights"
                " into your behavioral and emotional\n well-being.\n"
                ""))
            self.con_button.setText(_translate("FourthWindow", "Continue"))
            
        elif self.objectName() == "FifthWindow":
            self.setWindowTitle(_translate("FifthWindow", "DepDetect"))
            self.back_button.setText(_translate("FifthWindow", "  back"))
            self.understanding_better.setText(_translate("FifthWindow", "Understanding You Better"))
            self.info1.setText(_translate("FifthWindow", "Before we begin, we need some information. Please answer the following questions as\n"
                "honestly as possible. Remember, your responses will remain completely anonymous and\n"
                "confidential.\n"
                "\n"
                "Once you\'ve answered, our machine learning model will analyze your responses to predict\n"
                "your risk of depression. Please remember that this is a predictive tool and not a definitive\n"
                "diagnosis. For any mental health concerns, always consult with a qualified healthcare\n"
                "professional.\n"
                ""))
            self.con_button.setText(_translate("FifthWindow", "Continue"))
            
        elif self.objectName() == "SixthWindow":
            self.setWindowTitle(_translate("SixthWindow", "DepDetect"))
            self.back_button.setText(_translate("SixthWindow", "  back"))
            self.question.setText(_translate("SixthWindow", "Please enter your age group: "))
            self.info1.setText(_translate("SixthWindow", " 1. Age 25-30\n"
                " 2. Age 30-35\n"
                " 3. Age 35-40\n"
                " 4. Age 40-45\n"
                " 5. Age 45-50\n"
                ""))
            self.con_button.setText(_translate("SixthWindow", "Continue"))
            self.choice.setText(_translate("SixthWindow", "Enter your choice (1-5):\n"
                ""))

        elif self.objectName() == "SeventhWindow":
            self.setWindowTitle(_translate("SeventhWindow", "DepDetect"))
            self.question2.setText(_translate("SeventhWindow", "Have you been feeling down or sad lately?"))
            self.info1.setText(_translate("SeventhWindow", " 1. No\n"
                " 2. Sometimes\n"
                " 3. Yes"))
            self.con_button.setText(_translate("SeventhWindow", "Continue"))
            self.choice.setText(_translate("SeventhWindow", "Enter your choice (1-3):\n"
                ""))
            
        elif self.objectName() == "EighthWindow":
            self.setWindowTitle(_translate("EighthWindow", "DepDetect"))
            self.question3.setText(_translate("EighthWindow", "Do you find yourself feeling irritable towards other people?"))
            self.info1.setText(_translate("EighthWindow", " 1. No\n"
                " 2. Sometimes\n"
                " 3. Yes"))
            self.con_button.setText(_translate("EighthWindow", "Continue"))
            self.choice.setText(_translate("EighthWindow", "Enter your choice (1-3):\n"
                ""))
            
        elif self.objectName() == "NinthWindow":
            self.setWindowTitle(_translate("NinthWindow", "DepDetect"))
            self.question4.setText(_translate("NinthWindow", "Have you been experiencing trouble sleeping at night?"))
            self.info1.setText(_translate("NinthWindow", " 1. No\n"
                " 2. Two or more days a week\n"
                " 3. Yes"))
            self.con_button.setText(_translate("NinthWindow", "Continue"))
            self.choice.setText(_translate("NinthWindow", "Enter your choice (1-3):\n"
                ""))
            
        elif self.objectName() == "TenthWindow":
            self.setWindowTitle(_translate("TenthWindow", "DepDetect"))
            self.question5.setText(_translate("TenthWindow", "Do you have problems concentrating or making decisions?"))
            self.info1.setText(_translate("TenthWindow", " 1. No\n"
                " 2. Often\n"
                " 3. Yes"))
            self.con_button.setText(_translate("TenthWindow", "Continue"))
            self.choice.setText(_translate("TenthWindow", "Enter your choice (1-3):\n"
                ""))
            
        elif self.objectName() == "EleventhWindow":
            self.setWindowTitle(_translate("EleventhWindow", "DepDetect"))
            self.question6.setText(_translate("EleventhWindow", "Have you been experiencing a loss of appetite?"))
            self.info1.setText(_translate("EleventhWindow", " 1. No\n"
                " 2. Not at all\n"
                " 3. Yes"))
            self.con_button.setText(_translate("EleventhWindow", "Continue"))
            self.choice.setText(_translate("EleventhWindow", "Enter your choice (1-3):\n"
                ""))

        elif self.objectName() == "TwelfthWindow":
            self.setWindowTitle(_translate("TwelfthWindow", "DepDetect"))
            self.question7.setText(_translate("TwelfthWindow", "Do you often have feelings of guilt?"))
            self.info1.setText(_translate("TwelfthWindow", " 1. Maybe\n"
                " 2. No\n"
                " 3. Yes"))
            self.con_button.setText(_translate("TwelfthWindow", "Continue"))
            self.choice.setText(_translate("TwelfthWindow", "Enter your choice (1-3):\n"
                ""))

        elif self.objectName() == "ThirteenthWindow":
            self.setWindowTitle(_translate("ThirteenthWindow", "DepDetect"))
            self.question8.setText(_translate("ThirteenthWindow", "Do you have problems bonding with people?"))
            self.info1.setText(_translate("ThirteenthWindow", " 1. No\n"
                " 2. Sometimes\n"
                " 3. Yes"))
            self.con_button.setText(_translate("ThirteenthWindow", "Continue"))
            self.choice.setText(_translate("ThirteenthWindow", "Enter your choice (1-3):\n"
                ""))

        elif self.objectName() == "FourteenthWindow":
            self.setWindowTitle(_translate("FourteenthWindow", "DepDetect"))
            self.question9.setText(_translate("FourteenthWindow", "Have you ever attempted suicide?"))
            self.info1.setText(_translate("FourteenthWindow", " 1. No\n"
                " 2. Not interested to say\n"
                " 3. Yes"))
            self.con_button.setText(_translate("FourteenthWindow", "Get The Results"))
            self.choice.setText(_translate("FourteenthWindow", "Enter your choice (1-3):\n"
                "")) 
        elif self.objectName() == "LoadingWindow":
            self.setWindowTitle(_translate("LoadingWindow", "DepDetect"))
            self.loading_label.setText(_translate("LoadingWindow", "Analyzing data..."))

        elif self.objectName() == "ResultsWindow":
            self.setWindowTitle(_translate("ResultsWindow", "DepDetect"))
            self.results.setText(_translate("ResultsWindow", "Results"))
            self.try_again_button.setText(_translate("ResultsWindow", " Try Again"))
            self.confusion_matrix_result.setText(_translate("ResultsWindow", ""))
            self.classification_report_result.setText(_translate("ResultsWindow", ""))
            self.output.setText(_translate("ResultsWindow", ""))
            self.message_label.setText(_translate("ResultsWindow", ""))
            
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())
