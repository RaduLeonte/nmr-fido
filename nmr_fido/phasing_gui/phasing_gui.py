import numpy as np
from copy import deepcopy
import nmrglue as ng
from skimage import measure
import contourpy
from typing import Tuple
from time import perf_counter

from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
import pyqtgraph as pg

from nmr_fido.nmrdata import NMRData
from nmr_fido.phasing_gui.ToolbarButton import ToolbarButton

import nmr_fido as nf


COLOR_PALETTE = {
    "--text-color": "#e0e0e0",
    "--bg-color1": "#121212",
    "--bg-color2": "#1e1e1e",
    "--bg-color3": "#252525",
    "--border-color": "#505050",
}


def phase_gui(data: NMRData):
    app: QApplication = QApplication()

    main_window: MainWindow = MainWindow(app, data)
    main_window.show()

    app.exec()


class MainWindow(QMainWindow):
    """Main window class
    """
    def __init__(self, app: QApplication, data: NMRData) -> None:
        """Constructor.

        Args:
            app (QApplication): QApplication context.
        """
        # Parent constructor
        super().__init__()
        
        self.app = app
        self.data = data
        
        self.base_level = None
        self.nr_levels = 16
        self.level_multiplier = 1.2
        self._calculate_levels()
        
        self.positive_color = "m"
        self.negative_color = "c"
        
        self._init_ui()
        
        pass
    

    
    
    #region UI
    def _init_ui(self) -> None:
        self.app.setStyle("fusion")
        
        self.setWindowTitle("NMR Fido - Phasing GUI")
        #self.setWindowIcon(QIcon("icon.png"))
        
        # Minimum size
        min_size = (820, 400)
        self.setMinimumSize(QSize(*min_size))
        
        # Set init size based on screen aspect ratio
        ratio = 0.8
        screen_size = QApplication.primaryScreen().availableSize()
        if screen_size.width() >= screen_size.height():
            app_width = int(screen_size.width()*ratio)
            app_size = QSize(app_width, int(app_width*(min_size[1]/min_size[0])))
        else:
            app_height = int(screen_size.height()*ratio)
            app_size = QSize(int(app_height*(min_size[0]/min_size[1])), app_height)
        self.resize(app_size)
        
        """
        Main container
        """
        central_widget = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        self.setCentralWidget(central_widget)
        

        main_spectrum_group = QWidget()
        main_spectrum_group.setLayout(QVBoxLayout())
        central_widget.addWidget(main_spectrum_group) # Add
        
        main_spectrum_group.layout().addWidget(self._create_main_spectrum_toolbar()) # Add
        
        main_spectrum_group.layout().addWidget(self._create_main_spectrum_plot()) # Add
        
        
        phasing_traces_group = QWidget()
        central_widget.addWidget(phasing_traces_group) # Add
        return
    
    #region Main spectrum toolbar
    def _create_main_spectrum_toolbar(self) -> QWidget:
        axis_labels = [ax["label"] for ax in self.data.axes]
        
        main_spectrum_toolbar = QWidget()
        main_spectrum_toolbar.setLayout(QHBoxLayout())
        main_spectrum_toolbar.layout().setAlignment(Qt.AlignLeft)
        
        # Dimension select groups
        for axis in ["X", "Y"]:
            dim_controls = QWidget()
            dim_controls.setLayout(QHBoxLayout())
            main_spectrum_toolbar.layout().addWidget(dim_controls)
            
            # Label
            dim_controls.layout().addWidget(QLabel(f"{axis}:"))
            
            for label in axis_labels:
                dim_controls.layout().addWidget(ToolbarButton(label)) # Add
        
        
        # Levels controls
        levels_controls = QWidget()
        levels_controls.setLayout(QHBoxLayout())
        main_spectrum_toolbar.layout().addWidget(levels_controls)
        
        levels_controls.layout().addWidget(QLabel("Contours:"))
        
        increase_base_level_button = ToolbarButton("+")
        levels_controls.layout().addWidget(increase_base_level_button)
        increase_base_level_button.setToolTip("Increase the contours base level")
        increase_base_level_button.pressed.connect(self._increase_base_level_button_callback)
        
        decrease_base_level_button = ToolbarButton("-")
        levels_controls.layout().addWidget(decrease_base_level_button)
        decrease_base_level_button.setToolTip("Decrease the contours base level")
        decrease_base_level_button.pressed.connect(self._decrease_base_level_button_callback)
        
        level_multiplier_input = QDoubleSpinBox()
        levels_controls.layout().addWidget(level_multiplier_input)
        level_multiplier_input.setToolTip("Levels multiplier")
        level_multiplier_input.setMinimum(1.0)
        level_multiplier_input.setMaximum(10.0)
        level_multiplier_input.setSingleStep(0.1)
        level_multiplier_input.setDecimals(2)
        level_multiplier_input.setValue(self.level_multiplier)
        level_multiplier_input.valueChanged.connect(self._level_multiplier_input_callback)
        
        nr_levels_input = QSpinBox()
        levels_controls.layout().addWidget(nr_levels_input)
        nr_levels_input.setToolTip("Number of contours to draw")
        nr_levels_input.setMinimum(0)
        nr_levels_input.setMaximum(100)
        nr_levels_input.setValue(self.nr_levels)
        nr_levels_input.valueChanged.connect(self._nr_levels_input_callback)
        
        
        return main_spectrum_toolbar
    
    
    def _increase_base_level_button_callback(self) -> None:
        if self.base_level is None: self._calculate_levels()
        
        self.base_level *= self.level_multiplier
        self._draw_plot()
        return
    
    def _decrease_base_level_button_callback(self) -> None:
        if self.base_level is None: self._calculate_levels()
        
        self.base_level /= self.level_multiplier
        self._draw_plot()
        return
    
    
    def _level_multiplier_input_callback(self, value) -> None:
        self.level_multiplier = value
        self._draw_plot()
        return
    
    
    def _nr_levels_input_callback(self, value) -> None:
        self.nr_levels = value
        self._draw_plot()
        return
    
    #endregion
    
    #region Main spectrum
    def _create_main_spectrum_plot(self) -> QWidget:
        main_spectrum_plot = pg.PlotWidget()
        plot_layout = pg.GraphicsLayout()
        main_spectrum_plot.setCentralItem(plot_layout)
        
        #main_spectrum_plot.setBackground(QColor("#1e1e1e"))
        main_spectrum_plot.setBackground(QColor(0, 0, 0, 0))
        main_spectrum_plot.getAxis("bottom").setTextPen("w")
        main_spectrum_plot.getAxis("left").setTextPen("w")
        
        # Plot
        self.plot_ax = pg.PlotItem()
        plot_layout.addItem(self.plot_ax)
        
        # Remove ticks on the left and top axis
        for axis_code in ["left", "top"]:
            self.plot_ax.showAxis(axis_code)
            axis = self.plot_ax.getAxis(axis_code)
            axis.setTicks([])
            axis.setLabel("")
            axis.setStyle(showValues=False)

        # Set labels
        axis_labels = [ax["label"] for ax in self.data.axes]
        
        self.plot_ax.getAxis("bottom").setLabel(f"{axis_labels[0]} [ppm]")
        self.plot_ax.getAxis("bottom").setTextPen(COLOR_PALETTE["--text-color"])
        
        self.plot_ax.showAxis("right")
        self.plot_ax.getAxis("right").setLabel(f"{axis_labels[1]}\n[ppm]")
        self.plot_ax.getAxis("right").label.setRotation(0)
        self.plot_ax.getAxis("right").label.setTextWidth(60)
        self.plot_ax.getAxis("right").setTextPen(COLOR_PALETTE["--text-color"])
        
        # Styling
        self.plot_ax.getViewBox().setBackgroundColor(COLOR_PALETTE["--bg-color1"])
        
        # Scales
        x_scale = self.data.axes[0]["scale"]
        y_scale = self.data.axes[1]["scale"]
        self.plot_ax.setXRange(x_scale[-1], x_scale[0], padding=0)
        self.plot_ax.setYRange(y_scale[-1], y_scale[0], padding=0)
        # Invert scales if necessary
        if x_scale[0] > x_scale[-1]: self.plot_ax.getViewBox().invertX(True)
        if y_scale[0] > y_scale[-1]: self.plot_ax.getViewBox().invertY(True)
        
        
        # Bounding rectangle
        x_min, x_max = min(x_scale), max(x_scale)
        y_min, y_max = min(y_scale), max(y_scale)
        self.bounding_rect = pg.QtWidgets.QGraphicsRectItem(
            QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
        )
        self.bounding_rect.setPen(pg.mkPen(self.positive_color, width=0.5))
        self.bounding_rect.setBrush(QBrush(Qt.NoBrush))
        
        self.contour_generator = contourpy.contour_generator(z=self.data.real, name='serial')
        self._markers = []
        
        self._draw_plot()
        
        self._add_marker((9.5, 134))

        
        return main_spectrum_plot
    
    
    def _add_marker(self, pos):
        marker_box_width = 0.2
        marker_size = (marker_box_width, marker_box_width*5)
        marker_line_width = 1
        marker_color = "r"
        
        center_x = pos[0] - marker_size[0] / 2
        center_y = pos[1] - marker_size[1] / 2
        
        marker = pg.ROI(pos=(center_x, center_y), size=marker_size, movable=True)
        marker.setPen(pg.mkPen(marker_color, width=marker_line_width))
        marker.setZValue(10)
        self.plot_ax.addItem(marker)

        # Infinite lines (crosshairs)
        vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(marker_color, width=marker_line_width))
        hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(marker_color, width=marker_line_width))
        self.plot_ax.addItem(vline)
        self.plot_ax.addItem(hline)

        # Centered line positions
        vline.setPos(pos[0])
        hline.setPos(pos[1])

        # Callback to update line positions when marker is moved
        def update_lines():
            p = marker.pos()
            s = marker.size()
            cx = p.x() + s[0] / 2
            cy = p.y() + s[1] / 2
            vline.setPos(cx)
            hline.setPos(cy)

        marker.sigRegionChanged.connect(update_lines)
        
        markers = [v[0] for v in self._markers]
        print(marker, markers)
        if marker not in markers:
            self._markers.append(
                [marker, vline, hline]
            )
    

    
    
    def _median_absolute_deviation(self, data, k=1.4826) -> np.ndarray:
        """ Median Absolute Deviation: a "Robust" version of standard deviation.
            Indices variabililty of the sample.
            https://en.wikipedia.org/wiki/Median_absolute_deviation
        """
        data = np.ma.array(data).compressed()
        median = np.median(data)
        return k*np.median(np.abs(data - median))
    
    
    def _calculate_levels(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.base_level is not None:
            levels_positive = np.array([self.base_level * (self.level_multiplier ** j) for j in range(self.nr_levels)])
        else:
            rows, cols = data.shape
            row_start, row_end = rows // 4, rows * 3 // 4
            col_start, col_end = cols // 4, cols * 3 // 4
            central_region = data.real[row_start:row_end, col_start:col_end]
            
            maximum = float(np.max(central_region))
            
            #base_level = self._median_absolute_deviation(central_region, k=3)

            levels_positive = np.array([maximum / (self.level_multiplier ** j) for j in range(self.nr_levels)])
            
            self.base_level = np.min(levels_positive)
        
        levels_negative = levels_positive*-1
        
        return levels_positive, levels_negative
    
    
    def _draw_plot(self) -> None:
        start_time = perf_counter()
        print("Drawing plot...")
        
        # Clear plot
        self.plot_ax.clear()
        
        # Get real data
        data = self.data.real
        
        # Levels
        levels_positive, levels_negative = self._calculate_levels()
        
        # Scales
        x_scale = self.data.axes[0]["scale"]
        y_scale = self.data.axes[1]["scale"]
        
        # Draw bounding rect
        self.plot_ax.addItem(self.bounding_rect)
        
        # Contours
        def _draw_contours(levels, pen_color):
            path = QPainterPath()
            for level in levels:
                lines = self.contour_generator.lines(level)
                
                for line in lines:
                    if line.shape[0] < 2: continue
                    
                    x = np.interp(line[:, 0], [0, data.shape[1] - 1], [x_scale[0], x_scale[-1]])
                    y = np.interp(line[:, 1], [0, data.shape[0] - 1], [y_scale[0], y_scale[-1]])
                    
                    path.moveTo(x[0], y[0])
                    for xi, yi in zip(x[1:], y[1:]): path.lineTo(xi, yi)
            
            item = QGraphicsPathItem(path)
            item.setPen(pg.mkPen(color=pen_color, width=1))
            self.plot_ax.addItem(item)
        
        _draw_contours(levels_positive, self.positive_color)
        _draw_contours(levels_negative, self.negative_color)
        
        for marker, _, _ in self._markers:
            p = marker.pos()
            s = marker.size()
            cx = p.x() + s[0] / 2
            cy = p.y() + s[1] / 2
            self._add_marker((cx, cy))
        
        print(f"Done drawing plot... {perf_counter() - start_time:.3f} s")
        return
    
    
    #endregion UI
    
    
if __name__ == "__main__":
    import nmrglue as ng
    
    ng_dic, ng_data = ng.pipe.read("tests/dnajb1_wt_ctddd.ft2")
    
    
    data = NMRData(
        ng_data,
        axes= [
            {
                "label": "15N",
                "scale": ng.pipe.make_uc(ng_dic, ng_data, dim=1).ppm_scale(),
                "unit": "ppm",
            },
            {
                "label": "1H",
                "scale": ng.pipe.make_uc(ng_dic, ng_data, dim=0).ppm_scale(),
                "unit": "ppm",
            },
        ]
    )
    data = nf.TP(data)
    
    phase_gui(data)