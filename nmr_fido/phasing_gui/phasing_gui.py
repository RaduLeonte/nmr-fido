import numpy as np
from copy import deepcopy
import nmrglue as ng
from skimage import measure

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
    
    
    def _create_main_spectrum_toolbar(self) -> QWidget:
        axis_labels = [ax["label"] for ax in self.data.axes]
        
        main_spectrum_toolbar = QWidget()
        main_spectrum_toolbar.setLayout(QHBoxLayout())
        main_spectrum_toolbar.layout().setAlignment(Qt.AlignLeft)
        
        # X group
        for axis in ["X", "Y"]:
            dim_controls = QWidget()
            dim_controls.setLayout(QHBoxLayout())
            main_spectrum_toolbar.layout().addWidget(dim_controls)
            
            # Label
            dim_controls.layout().addWidget(QLabel(f"{axis}:"))
            
            for label in axis_labels:
                main_spectrum_toolbar.layout().addWidget(ToolbarButton(label)) # Add
        
        return main_spectrum_toolbar
    
    
    
    def _create_main_spectrum_plot(self) -> QWidget:
        main_spectrum_plot = pg.PlotWidget()
        plot_layout = pg.GraphicsLayout()
        main_spectrum_plot.setCentralItem(plot_layout)
        
        #main_spectrum_plot.setBackground(QColor("#1e1e1e"))
        main_spectrum_plot.setBackground(QColor(0, 0, 0, 0))
        main_spectrum_plot.getAxis("bottom").setTextPen("w")
        main_spectrum_plot.getAxis("left").setTextPen("w")
        
        plot_ax = pg.PlotItem()
        
        for axis_code in ["left", "top"]:
            plot_ax.showAxis(axis_code)
            axis = plot_ax.getAxis(axis_code)
            axis.setTicks([])
            axis.setLabel("")
            axis.setStyle(showValues=False)

        
        axis_labels = [ax["label"] for ax in self.data.axes]
        
        plot_ax.getAxis("bottom").setLabel(f"{axis_labels[0]} [ppm]")
        plot_ax.getAxis("bottom").setTextPen(COLOR_PALETTE["--text-color"])
        
        plot_ax.showAxis("right")
        plot_ax.getAxis("right").setLabel(f"{axis_labels[1]}\n[ppm]")
        plot_ax.getAxis("right").label.setRotation(0)
        plot_ax.getAxis("right").label.setTextWidth(60)
        plot_ax.getAxis("right").setTextPen(COLOR_PALETTE["--text-color"])
        
        plot_ax.getViewBox().setBackgroundColor(COLOR_PALETTE["--bg-color1"])
        
        plot_layout.addItem(plot_ax)
        
        
        data = self.data.real
        
        base_level = None
        if base_level is None:
            base_level = self._median_absolute_deviation(data, k=6)
        level_multiplier = 1.2
        nr_levels = 12
        
        levels_positive = [base_level * (level_multiplier ** j) for j in range(nr_levels)]
        levels_negative = [-l for l in levels_positive]
        
        pos_color = "m"
        neg_color = "c"
        
        print(self.data.summary())
        print(data)
        x_scale = self.data.axes[0]["scale"]
        y_scale = self.data.axes[1]["scale"]
        
        x_pixel_to_scale = lambda xi: np.interp(xi, [0, x_scale.size-1], [x_scale[0], x_scale[-1]])
        y_pixel_to_scale = lambda yi: np.interp(yi, [0, y_scale.size-1], [y_scale[0], y_scale[-1]])
        invert_x = x_scale[0] > x_scale[-1]
        invert_y = y_scale[0] > y_scale[-1]
        
        def _draw_contours(levels, pen_color):
            path = QPainterPath()
            for level in levels:
                contours = measure.find_contours(data, level=level)
                for contour in contours:
                    if contour.shape[0] < 2:
                        continue
                    x0 = x_pixel_to_scale(contour[0, 1])
                    y0 = y_pixel_to_scale(contour[0, 0])
                    path.moveTo(x0, y0)
                    for pt in contour[1:]:
                        x = x_pixel_to_scale(pt[1])
                        y = y_pixel_to_scale(pt[0])
                        path.lineTo(x, y)
            item = QGraphicsPathItem(path)
            item.setPen(pg.mkPen(color=pen_color, width=1))
            plot_ax.addItem(item)
        
        _draw_contours(levels_positive, pos_color)

        # Draw negative levels with inverted color
        _draw_contours(levels_negative, neg_color)
        
        
        plot_ax.setXRange(x_scale[-1], x_scale[0], padding=0)
        plot_ax.setYRange(y_scale[-1], y_scale[0], padding=0)
        
        if invert_x: plot_ax.getViewBox().invertX(True)
        if invert_y: plot_ax.getViewBox().invertY(True)
        
        return main_spectrum_plot
    
    
    def _median_absolute_deviation(self, data, k=1.4826):
        """ Median Absolute Deviation: a "Robust" version of standard deviation.
            Indices variabililty of the sample.
            https://en.wikipedia.org/wiki/Median_absolute_deviation
        """
        data = np.ma.array(data).compressed()
        median = np.median(data)
        return k*np.median(np.abs(data - median))
    
    
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