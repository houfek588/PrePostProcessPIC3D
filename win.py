import sys
from PyQt6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import main
import time

g_single_config_names = ("Value", "FFT", "Histogram", "Value2D", "FFT2D", "FFT in time")
g_multi_config_names = ("Value", "FFT", "Histogram")

class ControlWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Visualization Control")
        self.setMinimumSize(900, 600)
        self._apply_style()
        self._build_ui()

    def _apply_style(self):
        # Modern dark theme with rounded corners and flat design
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #f0f0f0;
                font-family: 'Segoe UI Variable', Arial, sans-serif;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #3c3f41;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4b5154;
            }
            QLineEdit, QListWidget, QCheckBox, QGroupBox, QSpinBox {
                background-color: #313335;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)


    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)
        grid.setSpacing(12)

        # Available Data List
        grid.addWidget(QtWidgets.QLabel("Available Data:"), 0, 0)
        self.available_list = QtWidgets.QListWidget()
        self.available_list.addItems(main.g_available)
        grid.addWidget(self.available_list, 1, 0, 3, 1)

        # Single Graph Controls
        btn_layout1 = QtWidgets.QVBoxLayout()
        btn_right1 = QtWidgets.QPushButton()
        btn_right1.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowRight))
        btn_right1.clicked.connect(lambda: self._move_item(self.available_list, self.single_list))
        btn_left1 = QtWidgets.QPushButton()
        btn_left1.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowLeft))
        btn_left1.clicked.connect(lambda: self._move_item(self.single_list, self.available_list))
        btn_layout1.addWidget(btn_right1)
        btn_layout1.addWidget(btn_left1)
        grid.addLayout(btn_layout1, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Data for plot in single graph:"), 0, 2)
        single_group = QtWidgets.QGroupBox()
        single_layout = QtWidgets.QVBoxLayout(single_group)
        self.single_list = QtWidgets.QListWidget()
        single_layout.addWidget(self.single_list)
        grid.addWidget(single_group, 1, 2)

        opts1 = QtWidgets.QVBoxLayout()
        # self.cb_valx = QtWidgets.QCheckBox(g_single_config_names[0])
        # self.cb_fft = QtWidgets.QCheckBox(g_single_config_names[1])
        # self.cb_hist = QtWidgets.QCheckBox(g_single_config_names[2])
        # self.cb_val2d = QtWidgets.QCheckBox(g_single_config_names[3])
        # self.cb_fft2d = QtWidgets.QCheckBox(g_single_config_names[4])
        # self.cb_time_fft = QtWidgets.QCheckBox(g_single_config_names[5])
        # self.single_plot_cfg = (self.cb_valx, self.cb_fft, self.cb_hist, self.cb_val2d, self.cb_fft2d, self.cb_time_fft)
        self.single_plot_cfg = tuple(QtWidgets.QCheckBox(name) for name in g_single_config_names)
        self.single_plot_cfg[0].setChecked(True)
        self.single_plot_cfg[3].setChecked(True)
        self.single_plot_cfg[4].setChecked(True)
        self.single_plot_cfg[5].setChecked(True)

        for cb in self.single_plot_cfg:
            opts1.addWidget(cb)
        grid.addLayout(opts1, 1, 3)

        # Multi Graph Controls
        btn_layout2 = QtWidgets.QVBoxLayout()
        btn_right2 = QtWidgets.QPushButton()
        btn_right2.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowRight))
        btn_right2.clicked.connect(lambda: self._move_item(self.available_list, self.multi_list))
        btn_left2 = QtWidgets.QPushButton()
        btn_left2.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowLeft))
        btn_left2.clicked.connect(lambda: self._move_item(self.multi_list, self.available_list))
        btn_layout2.addWidget(btn_right2)
        btn_layout2.addWidget(btn_left2)
        grid.addLayout(btn_layout2, 3, 1)

        grid.addWidget(QtWidgets.QLabel("Multi Graph:"), 2, 2)
        multi_group = QtWidgets.QGroupBox()
        multi_layout = QtWidgets.QVBoxLayout(multi_group)
        self.multi_list = QtWidgets.QListWidget()
        multi_layout.addWidget(self.multi_list)
        grid.addWidget(multi_group, 3, 2)


        opts2 = QtWidgets.QVBoxLayout()
        # self.cb_valm = QtWidgets.QCheckBox("VAL")
        # self.cb_fftm = QtWidgets.QCheckBox("FFT")
        # self.cb_hist = QtWidgets.QCheckBox("Hist")
        self.multi_plot_cfg = tuple(QtWidgets.QCheckBox(name) for name in g_multi_config_names)
        self.multi_plot_cfg[2].setChecked(True)

        for cb in self.multi_plot_cfg:
            opts2.addWidget(cb)
        grid.addLayout(opts2, 3, 3)


        # Parameters Label
        grid.addWidget(QtWidgets.QLabel("Parameters:"), 4, 0)

        # Parameter Settings Group (below label)
        params_group = QtWidgets.QGroupBox()
        params_layout = QtWidgets.QGridLayout(params_group)
        # Paths with browse buttons
        self.le_result_folder = QtWidgets.QLineEdit("../res_data_vth/beam_long_time15/data/")
        btn_browse_res = QtWidgets.QPushButton("Browse...")
        btn_browse_res.clicked.connect(lambda: self._browse_folder(self.le_result_folder))
        params_layout.addWidget(QtWidgets.QLabel("Result Folder:"), 0, 0)
        params_layout.addWidget(self.le_result_folder, 0, 1)
        params_layout.addWidget(btn_browse_res, 0, 2)
        self.le_output_folder = QtWidgets.QLineEdit("../res_data_vth/beam_long_time15/result/")
        btn_browse_out = QtWidgets.QPushButton("Browse...")
        btn_browse_out.clicked.connect(lambda: self._browse_folder(self.le_output_folder))
        params_layout.addWidget(QtWidgets.QLabel("Output Folder:"), 1, 0)
        params_layout.addWidget(self.le_output_folder, 1, 1)
        params_layout.addWidget(btn_browse_out, 1, 2)
        # Boolean
        self.cb_save_graphs = QtWidgets.QCheckBox("Save Graphs")
        self.cb_save_graphs.setChecked(True)
        params_layout.addWidget(self.cb_save_graphs, 2, 0, 1, 1)
        # Integer cycles
        self.spin_field_cycle = QtWidgets.QSpinBox()
        self.spin_field_cycle.setRange(1, 100000)
        self.spin_field_cycle.setValue(10)
        params_layout.addWidget(QtWidgets.QLabel("Field Output Cycle:"), 3, 0)
        params_layout.addWidget(self.spin_field_cycle, 3, 1)

        self.spin_particle_cycle = QtWidgets.QSpinBox()
        self.spin_particle_cycle.setRange(1, 10000000)
        self.spin_particle_cycle.setValue(5000)
        params_layout.addWidget(QtWidgets.QLabel("Particle Output Cycle:"), 4, 0)
        params_layout.addWidget(self.spin_particle_cycle, 4, 1)
        # Number of HDF Files
        self.spin_num_hdf = QtWidgets.QSpinBox()
        self.spin_num_hdf.setRange(1, 1000)
        self.spin_num_hdf.setValue(32)
        params_layout.addWidget(QtWidgets.QLabel("Number HDF Files:"), 5, 0)
        params_layout.addWidget(self.spin_num_hdf, 5, 1)
        grid.addWidget(params_group, 5, 0, 1, 5)

        # Execute Button and Processing Indicator
        self.btn_execute = QtWidgets.QPushButton("Execute")
        self.btn_execute.clicked.connect(self._execute)

        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate busy indicator
        self.progress.setVisible(False)
        grid.addWidget(self.btn_execute, 6, 3)
        grid.addWidget(self.btn_clear, 6, 4)
        grid.addWidget(self.progress, 6, 0, 1, 3)

        # Stretch for responsiveness
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(3, 1)
        grid.setRowStretch(5, 1)

    def _move_item(self, src: QtWidgets.QListWidget, dst: QtWidgets.QListWidget):
        for item in src.selectedItems():
            dst.addItem(item.text())
            src.takeItem(src.row(item))

    def _browse_folder(self, line_edit: QtWidgets.QLineEdit):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if path:
            line_edit.setText(path)

    def _clear(self):
        # Show processing indicator
        self.progress.setVisible(True)
        QtWidgets.QApplication.processEvents()

        # process
        main.clear_script_data()

        # Hide indicator after done
        self.progress.setVisible(False)

    def _execute(self):
        # Show processing indicator
        self.progress.setVisible(True)
        QtWidgets.QApplication.processEvents()

        options_single = {cb.text(): str(cb.isChecked()) for cb in self.single_plot_cfg}
        options_multi = {cb.text(): str(cb.isChecked()) for cb in self.multi_plot_cfg}
        selections = {
            'available': self.get_available_data(),
            'single': self.get_single_data(),
            'multi': self.get_multi_data(),
            'options_single': options_single,
            'options_multi': options_multi,
            'result_folder': self.le_result_folder.text(),
            'output_folder': self.le_output_folder.text(),
            'save_graphs': self.cb_save_graphs.isChecked(),
            'FieldOutputCycle': self.spin_field_cycle.value(),
            'ParticleOutputCycle': self.spin_particle_cycle.value(),
            'NumberHdfFiles': self.spin_num_hdf.value(),
            # 'output_path': self.output_path.text()
        }
        # print("Executing with settings:", selections)

        # save configuration files
        main.save_config(self.le_result_folder.text(), self.le_output_folder.text(), self.cb_save_graphs.isChecked()
                         , self.spin_field_cycle.value(), self.spin_particle_cycle.value(), self.spin_num_hdf.value())
        main.save_plot_config(options_single, options_multi)

        # data processing
        try:
            main.data_process(self.get_single_data(), self.get_multi_data())
        except Exception as ex:
            print(f"SOMETHING WENT WRONG: {ex}")


        # Hide indicator after done
        self.progress.setVisible(False)

    # Utility methods for accessing list contents
    def get_list_items(self, list_widget: QtWidgets.QListWidget) -> list[str]:
        """Return all items currently in the given QListWidget."""
        return [list_widget.item(i).text() for i in range(list_widget.count())]

    def get_available_data(self) -> list[str]:
        return self.get_list_items(self.available_list)

    def get_single_data(self) -> list[str]:
        return self.get_list_items(self.single_list)

    def get_multi_data(self) -> list[str]:
        return self.get_list_items(self.multi_list)

    # Utility methods for populating lists from Python variables
    def set_list_items(self, list_widget: QtWidgets.QListWidget, items: list[str]):
        """Clear and populate the given QListWidget with a list of string items."""
        list_widget.clear()
        list_widget.addItems(items)

    def set_available_data(self, items: list[str]):
        self.set_list_items(self.available_list, items)

    def set_single_data(self, items: list[str]):
        self.set_list_items(self.single_list, items)

    def set_multi_data(self, items: list[str]):
        self.set_list_items(self.multi_list, items)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ControlWindow()
    win.show()
    sys.exit(app.exec())
