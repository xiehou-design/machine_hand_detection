import sys
from PyQt5 import QtWidgets
# 引入这个类
from ui import Ui_MainWindow

if __name__ == '__main__':
    # 创建一个应用(Application)对象,sys.argv参数是一个来自命令行的参数列表,
    app = QtWidgets.QApplication(sys.argv)
    # 创建一个widget组件基础类
    windows = QtWidgets.QMainWindow()
    # 实例化ui界面的类
    ui = Ui_MainWindow()
    # 把ui界面放到控件中
    ui.setupUi(windows)
    # 界面显示
    windows.show()
    windows.setFixedSize(537, 592)
    # 循环执行窗口触发事件，结束后不留垃圾的退出，不添加的话新建的widget组件就会一闪而过
    sys.exit(app.exec_())
