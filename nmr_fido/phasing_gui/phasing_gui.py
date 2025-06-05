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
        
        
class CustomPlotItem(pg.PlotItem):
    def __init__(self, parent_plot_widget, main_window):
        self.parent_plot_widget = parent_plot_widget
        self.main_window = main_window
        super().__init__()
        self.setMenuEnabled(False)
        
        
    def contextMenuEvent(self, event):
        menu = QMenu(self.parent_plot_widget)
        
        pos = event.pos()
        
        offset = [10, 10]
        pos = [pos.x() + offset[0], pos.y() + offset[1]]
        
        position_in_plot = self.vb.mapSceneToView(QPointF(*pos))
        print("---", (position_in_plot.x(), position_in_plot.y()))
        
        # Create custom actions
        action1 = QAction("Add marker", self)
        action1.triggered.connect(lambda: self.main_window._add_marker((position_in_plot.x(), position_in_plot.y())))
        menu.addAction(action1)

        # Show the menu at the mouse position
        global_pos = self.parent_plot_widget.mapToGlobal(QPoint(*pos))
        menu.exec(global_pos)

class Marker():
    def __init__(
        self,
        parent: 'MainWindow',
        pos: tuple,
        trace: pg.PlotWidget,
        width: float = 0.2,
        height:float = 1.0,
        color: str = "r",
    ) -> None:
        self.parent = parent
        
        self.pos = pos
        
        self.trace = trace
        
        self.width = width
        self.height = height
        self.size = (width, height)
        
        self.line_width = 1
        
        self.color = color
        
        self.x = pos[0]
        self.y = pos[1]
        
        rect_x = pos[0] - self.size[0] / 2
        rect_y = pos[1] - self.size[1] / 2
        
        self.rect = pg.ROI(pos=(rect_x, rect_y), size=self.size, movable=True)
        self.rect.setPen(pg.mkPen(self.color, width=self.line_width))
        self.rect.setZValue(10)

        # Infinite lines (crosshairs)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(self.color, width=self.line_width))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(self.color, width=self.line_width))

        self.vline.setPos(pos[0])
        self.hline.setPos(pos[1])
        
        self.rect.sigRegionChanged.connect(self._update_lines)
        
        self._update_trace()
        pass
    
    def _update_lines(self) -> None:
        rect_pos = self.rect.pos()
        rect_size = self.rect.size()
        self.x = rect_pos.x() + rect_size[0] / 2
        self.y = rect_pos.y() + rect_size[1] / 2
        
        self.pos = (self.x, self.y)
        
        self.vline.setPos(self.x)
        self.hline.setPos(self.y)
        
        
        self._update_trace()
        
        return
    
    def _update_trace(self) -> None:
        self.parent._update_trace(self.trace, self.pos)
        return


def phasing_gui(data: NMRData):
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
        
        self.traces = []
        self.trace_mode = "rows"
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
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
        
        self.main_spectrum_plot = self._create_main_spectrum_plot()
        main_spectrum_group.layout().addWidget(self.main_spectrum_plot) # Add        
        
        traces_group = QWidget()
        traces_group.setLayout(QVBoxLayout())
        central_widget.addWidget(traces_group) # Add
        
        traces_group.layout().addWidget(self._create_traces_toolbar())
        
        self.traces_container = self._create_traces_container()
        traces_group.layout().addWidget(self.traces_container) # Add
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
        self.plot_ax = CustomPlotItem(parent_plot_widget=main_spectrum_plot, main_window=self)
        plot_layout.addItem(self.plot_ax)
        
        self.plot_ax.scene().sigMouseMoved.connect(self._on_mouse_move)
        
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

        
        return main_spectrum_plot
    
    
    def _on_mouse_move(self, pos):
        position_in_plot = self.plot_ax.vb.mapSceneToView(pos)
        
        # Update the status bar with the coordinates
        x, y = position_in_plot.x(), position_in_plot.y()
        self.statusBar.showMessage(f"X: {x:.2f} ppm, Y: {y:.2f} ppm")
    
    
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
            rows, cols = self.data.shape
            row_start, row_end = rows // 4, rows * 3 // 4
            col_start, col_end = cols // 4, cols * 3 // 4
            central_region = self.data.real[row_start:row_end, col_start:col_end]
            
            maximum = float(np.max(central_region))
            
            #base_level = self._median_absolute_deviation(central_region, k=3)

            levels_positive = np.array([maximum / (self.level_multiplier ** j) for j in range(self.nr_levels)])
            
            self.base_level = np.min(levels_positive)
        
        levels_negative = levels_positive*-1
        
        return levels_positive, levels_negative
    
    
    def _draw_plot(self) -> None:
        start_time = perf_counter()
        print("Drawing plot...")
        
        
        self._clear_all_markers()
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
        
        self._draw_all_markers()
        
        print(f"Done drawing plot... {perf_counter() - start_time:.3f} s")
        return
    
    
    def _add_marker(self, pos):
        marker = Marker(self, pos, self._create_trace())
        self._markers.append(marker)
        self._draw_marker(marker)
            
            
    def _draw_all_markers(self) -> None:
        for marker in self._markers:
            self._draw_marker(marker) 
        return
    
    
    def _draw_marker(self, marker) -> None:
        self.plot_ax.addItem(marker.rect)
        self.plot_ax.addItem(marker.vline)
        self.plot_ax.addItem(marker.hline)
        return
    
    
    def _clear_all_markers(self) -> None:
        for marker in self._markers:
            self._clear_marker(marker) 
        return
    
    
    def _clear_marker(self, marker) -> None:
        self.plot_ax.removeItem(marker.rect)
        self.plot_ax.removeItem(marker.vline)
        self.plot_ax.removeItem(marker.hline)
        return
    
    
    def _create_traces_toolbar(self):
        traces_toolbar = QWidget()
        traces_toolbar.setLayout(QHBoxLayout())
        traces_toolbar.layout().setAlignment(Qt.AlignLeft)
        
        display_rows_button = ToolbarButton("Rows")
        traces_toolbar.layout().addWidget(display_rows_button)
        display_rows_button.setToolTip("Display rows")
        display_rows_button.pressed.connect(self._display_rows_button_callback)
        
        display_columns_button = ToolbarButton("Columns")
        traces_toolbar.layout().addWidget(display_columns_button)
        display_columns_button.setToolTip("Display columns")
        display_columns_button.pressed.connect(self._display_columns_button_callback)
        
        def create_slider(range):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(*range)
            slider.setValue(0)
            slider.setSingleStep(1)
            
            return slider
        
        def create_spinbox():
            spinbox = QDoubleSpinBox()
            spinbox.setValue(0)
            spinbox.setRange(-360.0, 360.0)
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(2)
            
            return spinbox
        
        def add_callbacks(slider_coarse, slider_fine, spin_box, slider_factor_coarse, slider_factor_fine):
            slider_coarse.valueChanged.connect(
                lambda: slider_callback(slider_coarse, slider_fine, spin_box, slider_factor_coarse, slider_factor_fine)
            )
            slider_fine.valueChanged.connect(
                lambda: slider_callback(slider_coarse, slider_fine, spin_box, slider_factor_coarse, slider_factor_fine)
            )
            spin_box.valueChanged.connect(
                lambda: spin_box_callback(slider_coarse, slider_fine, spin_box, slider_factor_coarse, slider_factor_fine)
            )
        
        def slider_callback(slider_coarse, slider_fine, spin_box, slider_factor_coarse, slider_factor_fine):
            coarse_value = float(slider_coarse.value() / slider_factor_coarse)
            fine_value = float(slider_fine.value() / slider_factor_fine)
            spin_box.blockSignals(True)
            spin_box.setValue(coarse_value + fine_value)
            spin_box.blockSignals(False)
            
            self._update_all_traces()
                
        def spin_box_callback(slider_coarse, slider_fine, spin_box, slider_factor_coarse, slider_factor_fine):
            total_value = spin_box.value()
            coarse_value = int(total_value)
            fine_value = int((total_value - coarse_value) * slider_factor_fine)
            
            slider_coarse.blockSignals(True)
            slider_coarse.setValue(coarse_value * slider_factor_coarse)
            slider_coarse.blockSignals(False)
            
            slider_fine.blockSignals(True)
            slider_fine.setValue(fine_value)
            slider_fine.blockSignals(False)
            
            self._update_all_traces()
        
        slider_factor_coarse = 10
        slider_factor_fine = 10
        
        
        traces_toolbar.layout().addWidget(QLabel("P0"))
        
        p0_slider_coarse = create_slider((-360 * slider_factor_coarse, 360 * slider_factor_coarse))
        traces_toolbar.layout().addWidget(p0_slider_coarse)
        
        p0_slider_fine = create_slider((-10 * slider_factor_coarse, 10 * slider_factor_coarse))
        traces_toolbar.layout().addWidget(p0_slider_fine)
        
        p0_spin_box = create_spinbox()
        self.p0_spin_box = p0_spin_box
        traces_toolbar.layout().addWidget(p0_spin_box)
            
        add_callbacks(p0_slider_coarse, p0_slider_fine, p0_spin_box, slider_factor_coarse, slider_factor_fine)
        
        
        traces_toolbar.layout().addWidget(QLabel("P1"))
        
        p1_slider_coarse = create_slider((-360 * slider_factor_coarse, 360 * slider_factor_coarse))
        traces_toolbar.layout().addWidget(p1_slider_coarse)
        
        p1_slider_fine = create_slider((-10 * slider_factor_coarse, 10 * slider_factor_coarse))
        traces_toolbar.layout().addWidget(p1_slider_fine)
        
        p1_spin_box = create_spinbox()
        self.p1_spin_box = p1_spin_box
        traces_toolbar.layout().addWidget(p1_spin_box)
            
        add_callbacks(p1_slider_coarse, p1_slider_fine, p1_spin_box, slider_factor_coarse, slider_factor_fine)
        
        return traces_toolbar
    
    
    def _display_rows_button_callback(self) -> None:
        self.trace_mode = "rows"
        self._update_all_traces()
        return
    
    def _display_columns_button_callback(self) -> None:
        self.trace_mode = "columns"
        self._update_all_traces()
        return
    
    
    def _create_traces_container(self):
        traces_container = QWidget()
        traces_container.setLayout(QVBoxLayout())
        traces_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        return traces_container
    
    
    def _create_trace(self):
        trace = pg.PlotWidget()
        trace.setBackground(QColor(0, 0, 0, 0))
        trace.getAxis("bottom").setTextPen("w")
        trace.showAxis("right")
        trace.getAxis("right").setTextPen("w")
        for axis_code in ["left", "top"]:
            trace.showAxis(axis_code)
            axis = trace.getAxis(axis_code)
            axis.setTicks([])
            axis.setLabel("")
            axis.setStyle(showValues=False)
        
        self.traces.append(trace)
        self.traces_container.layout().addWidget(trace)
        
        return trace
    
    
    def _update_all_traces(self) -> None:
        for marker in self._markers:
            marker._update_trace()
    
    
    def _update_trace(self, trace, pos):
        trace.clear()
        
        x_scale = self.data.axes[0]["scale"]
        y_scale = self.data.axes[1]["scale"]
        
        axis_labels = [ax["label"] for ax in self.data.axes]
        
        data = np.array(self.data)
        if self.trace_mode == "rows":
            target_ppm = pos[1]
            target_scale = y_scale
            index = min(range(len(target_scale)), key=lambda i: abs(target_scale[i] - target_ppm))
            
            scale = x_scale
            trace_data = data[index]
            
            trace.getAxis("bottom").setLabel(f"{axis_labels[0]} [ppm]")
            
        else:
            target_ppm = pos[0]
            target_scale = x_scale
            index = min(range(len(target_scale)), key=lambda i: abs(target_scale[i] - target_ppm))
            
            scale = y_scale
            trace_data = np.transpose(data)[index]
            
            trace.getAxis("bottom").setLabel(f"{axis_labels[1]} [ppm]")
        
        print(self.trace_mode, target_ppm)
        
        trace_data = nf.PS(trace_data, p0=self.p0_spin_box.value(), p1=self.p1_spin_box.value(), ht=True).real
        
        trace_max = float(np.max(np.abs(trace_data)))
        if scale[0] > scale[-1]: trace.getViewBox().invertX(True)
        trace.setYRange(-trace_max, trace_max)
        trace.plot(scale, trace_data)
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
    
    phasing_gui(data)