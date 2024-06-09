from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QSlider, QLabel, QVBoxLayout, QHBoxLayout, QApplication
from PySide6.QtCore import Qt, SIGNAL

import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from functools import partial

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FLOATING_POINT_PRECISION = 4


class ControlWindowLauncher:
    def __init__(self, num_values, vals_range, self_centering=None, slider_labels=None):
        self.e_exit = mp.Event()
        self.manager = SharedMemoryManager()
        self.manager.start()
        # UI setting
        self.num_values = num_values
        ui_config = {
            "num_values": num_values,
            "vals_range": vals_range,
            "self_centering": self_centering,
            "slider_labels": slider_labels
        }

        assert len(vals_range) == num_values, "range and num_values mismatch"
        for val_range in vals_range:
            assert len(val_range) == 2, "range should be a tuple of two elements"
        if self_centering is not None:
            for val_range in vals_range:
                assert val_range[0] <= self_centering <= val_range[1], "unexpected self_centering value"
        if slider_labels is not None:
            assert len(slider_labels) == num_values, "slider_labels and num_values mismatch"
        self.shared_list = self.manager.ShareableList([0] * num_values)
        self.windows_process = _ControlWindowProcess(self.e_exit, self.shared_list, ui_config)
        self.windows_process.start()

    def close(self):
        self.e_exit.set()
        self.windows_process.join()


    def get_values(self):
        vals = [0] * self.num_values
        for i in range(self.num_values):
            vals[i] = self.shared_list[i]
        return vals

    def is_alive(self):
        return not self.e_exit.is_set()

    def __del__(self):
        self.manager.shutdown()
        self.e_exit.set()
        self.windows_process.terminate()
        logger.info("ControlWindowLauncher::exit")


class _ControlWindowProcess(mp.Process):
    def __init__(self, e_exit, shared_memory_handel, ui_config):
        super().__init__()
        self.shared_memory_handel = shared_memory_handel
        self.ui_config = ui_config
        self.e_exit = e_exit

    def run(self):
        app = QApplication([])
        window = _ControlWindow(self.e_exit, self.shared_memory_handel, self.ui_config)
        window.show()
        app.exec()
        logger.info("_ControlWindowProcess::exit")
        self.e_exit.set()


class _ControlWindow(QtWidgets.QWidget):
    def __init__(self, e_exit, shared_memory_handel, ui_config):
        super().__init__()
        self.e_exit = e_exit
        val_len = ui_config["num_values"]
        val_range = ui_config['vals_range']
        self_centering = ui_config['self_centering']
        slider_labels = ui_config['slider_labels']
        self.self_centering = self_centering

        self.setWindowTitle("value control")
        self.setGeometry(300, 300, 300, 100)
        self.val_list = shared_memory_handel

        self.f_scale = 10 ** FLOATING_POINT_PRECISION

        layout = QVBoxLayout()
        slider_layouts = [QHBoxLayout() for _ in range(val_len)]
        for slider_layout in slider_layouts:
            layout.addLayout(slider_layout)

        for i in range(val_len):
            if slider_labels is not None:
                txt_label = QLabel(slider_labels[i])
                slider_layouts[i].addWidget(txt_label)
            label = QLabel(f'{0:10.{FLOATING_POINT_PRECISION}f}')
            slider_layouts[i].addWidget(label)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(val_range[i][0] * self.f_scale)
            slider.setMaximum(val_range[i][1] * self.f_scale)
            slider.setTickInterval(1)
            slider.valueChanged.connect(partial(self.on_value_changed, i, slider, label))
            slider_layouts[i].addWidget(slider)
            if self_centering is not None:
                slider.sliderReleased.connect(partial(self.center_slider, slider))

        self.setLayout(layout)

    def on_value_changed(self, index, slider, label, value):
        val = float(value) / self.f_scale
        label.setText(f"{val:10.{FLOATING_POINT_PRECISION}f}")
        self.val_list[index] = val

    def center_slider(self, slider):
        slider.setValue(int(self.self_centering*self.f_scale))


    def close(self):
        print("_ControlWindow::exit")
        super().close()

    def closeEvent(self, event):
        print("_ControlWindow::closeEvent")
        self.e_exit.set()
        event.accept()


if __name__ == "__main__":
    import time

    launcher = ControlWindowLauncher(3, [(0, 100), (0, 10), (0, 10)],
                                     slider_labels=["slider1", "slider2", "slider3"])
    try:
        while launcher.is_alive():
            time.sleep(0.1)
            print(launcher.get_values())
    except KeyboardInterrupt:
        print("keyboard interrupt")
    launcher.close()
    time.sleep(1)
    print("exit")
