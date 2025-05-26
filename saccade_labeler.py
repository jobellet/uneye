import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                               QMenuBar, QFileDialog, QMessageBox, QLabel, QCheckBox, QSpinBox,
                               QSizePolicy, QPushButton) # Added QPushButton
from PySide6.QtGui import QAction, QColor, QPalette
from PySide6.QtCore import Qt 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector # Added SpanSelector
import numpy as np

class SaccadeLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SaccadeLabeler")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height

        self.eye_data = None 
        self.smoothed_eye_data = None 
        self.velocity_data = None 
        self.labels = np.array([]) # For storing 0s (fixation) and 1s (saccade)
        self.marked_saccade_regions = [] # List of (start_idx, end_idx) tuples
        self.saccade_spans = [] # To store axvspan objects for easy removal
        self.span_selector = None # For Matplotlib SpanSelector

        # Create menubar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        load_data_action = QAction("Load Eye Data...", self)
        load_data_action.triggered.connect(self.load_eye_data)
        file_menu.addAction(load_data_action)

        save_labels_action = QAction("Save Labels...", self)
        save_labels_action.triggered.connect(self.save_labels)
        file_menu.addAction(save_labels_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Control Panel (left side)
        control_panel = QWidget()
        control_panel.setMinimumWidth(200)
        control_panel_palette = control_panel.palette()
        control_panel_palette.setColor(QPalette.Window, QColor("lightgray"))
        control_panel.setPalette(control_panel_palette)
        control_panel.setAutoFillBackground(True)
        control_panel_layout = QVBoxLayout(control_panel) 
        control_panel_label = QLabel("Control Panel")
        control_panel_label.setAlignment(Qt.AlignHCenter) # Center align label
        control_panel_layout.addWidget(control_panel_label)

        # Smoothing controls
        self.smoothing_checkbox = QCheckBox("Enable Smoothing")
        control_panel_layout.addWidget(self.smoothing_checkbox)

        smoothing_layout = QHBoxLayout() # Layout for label and spinbox
        smoothing_label = QLabel("Smoothing Window Size:")
        smoothing_layout.addWidget(smoothing_label)
        
        self.smoothing_spinbox = QSpinBox()
        self.smoothing_spinbox.setRange(3, 51) # Range 3-51
        self.smoothing_spinbox.setSingleStep(2) # Step 2 (always odd)
        self.smoothing_spinbox.setValue(5) # Default 5
        self.smoothing_spinbox.setEnabled(False) # Disabled by default
        smoothing_layout.addWidget(self.smoothing_spinbox)
        control_panel_layout.addLayout(smoothing_layout)
        
        # Connect smoothing controls
        self.smoothing_checkbox.stateChanged.connect(self.on_smoothing_settings_changed)
        self.smoothing_spinbox.valueChanged.connect(self.on_smoothing_settings_changed)

        control_panel_layout.addSpacing(20) # Add some space

        # Saccade marking controls
        saccade_label = QLabel("Saccade Marking")
        saccade_label.setAlignment(Qt.AlignHCenter)
        control_panel_layout.addWidget(saccade_label)

        self.mark_saccade_button = QPushButton("Mark Saccade Mode")
        self.mark_saccade_button.setCheckable(True)
        self.mark_saccade_button.toggled.connect(self.toggle_saccade_mode)
        control_panel_layout.addWidget(self.mark_saccade_button)

        self.clear_last_saccade_button = QPushButton("Clear Last Saccade")
        self.clear_last_saccade_button.clicked.connect(self.clear_last_saccade)
        control_panel_layout.addWidget(self.clear_last_saccade_button)

        self.clear_all_saccades_button = QPushButton("Clear All Saccades")
        self.clear_all_saccades_button.clicked.connect(self.clear_all_saccades)
        control_panel_layout.addWidget(self.clear_all_saccades_button)
        
        control_panel_layout.addStretch() # Pushes controls to the top
        main_layout.addWidget(control_panel)

        # Plotting Area (central area)
        plotting_widget = QWidget()
        plotting_layout = QVBoxLayout(plotting_widget) # Layout for the plotting widget

        # Matplotlib Figure and Canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        # Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        plotting_layout.addWidget(self.toolbar)
        plotting_layout.addWidget(self.canvas)

        # Create subplots
        self.ax_position = self.figure.add_subplot(2, 1, 1)
        self.ax_velocity = self.figure.add_subplot(2, 1, 2)

        # Set labels - these will be reapplied in apply_smoothing_and_velocity
        # self.ax_position.set_xlabel("Time (samples)")
        # self.ax_position.set_ylabel("Position")
        # self.ax_velocity.set_xlabel("Time (samples)")
        # self.ax_velocity.set_ylabel("Velocity")
        
        self.setup_span_selector() # Initialize SpanSelector

        # Dummy data for initial display
        x = np.linspace(0, 10, 100)
        y_pos = np.sin(x)
        y_vel = np.cos(x)
        # Dummy data for initial display - will be cleared by apply_smoothing_and_velocity
        # x = np.linspace(0, 10, 100)
        # y_pos = np.sin(x)
        # y_vel = np.cos(x)
        # self.ax_position.plot(x, y_pos, label="X-Position")
        # self.ax_position.plot(x, y_pos * 0.5, label="Y-Position") # Dummy Y
        # self.ax_position.legend()
        # self.ax_velocity.plot(x, y_vel, label="Velocity")
        # self.ax_velocity.legend()

        self.figure.tight_layout() # Adjust layout to prevent overlap
        # self.canvas.draw() # Initial draw will be handled by apply_smoothing_and_velocity

        main_layout.addWidget(plotting_widget, 1)  # Add stretch factor for resizability
        
        self.apply_smoothing_and_velocity() # Initial call to setup plots

        self.show()

    def on_smoothing_settings_changed(self):
        # Enable/disable spinbox based on checkbox
        self.smoothing_spinbox.setEnabled(self.smoothing_checkbox.isChecked())
        self.apply_smoothing_and_velocity()

    def _moving_average(self, data, window_size):
        if window_size % 2 == 0: # Ensure odd window size for symmetry
            window_size +=1
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    def apply_smoothing_and_velocity(self):
        if self.eye_data is None or self.eye_data.size == 0:
            # Clear plots if no data
            self.ax_position.clear()
            self.ax_position.set_xlabel("Time (samples)")
            self.ax_position.set_ylabel("Position")
            self.ax_position.legend([])
            
            self.ax_velocity.clear()
            self.ax_velocity.set_xlabel("Time (samples)")
            self.ax_velocity.set_ylabel("Velocity")
            self.ax_velocity.legend([])

            # Also clear saccade visualizations if no data
            self.update_saccade_visualizations() # Will clear spans and redraw
            
            self.canvas.draw()
            return

        current_eye_data_x = self.eye_data[:, 0]
        current_eye_data_y = self.eye_data[:, 1]
        time_samples = np.arange(current_eye_data_x.shape[0])

        self.smoothed_eye_data = None # Reset

        if self.smoothing_checkbox.isChecked():
            window_size = self.smoothing_spinbox.value()
            if window_size > 1: # Min window size for smoothing
                smoothed_x = self._moving_average(current_eye_data_x, window_size)
                smoothed_y = self._moving_average(current_eye_data_y, window_size)
                self.smoothed_eye_data = np.column_stack((smoothed_x, smoothed_y))
                
                # Data for velocity calculation is the smoothed data
                calc_data_x = smoothed_x
                calc_data_y = smoothed_y
            else: # Window size too small, use original
                calc_data_x = current_eye_data_x
                calc_data_y = current_eye_data_y
        else:
            # Data for velocity calculation is the original data
            calc_data_x = current_eye_data_x
            calc_data_y = current_eye_data_y

        # Velocity Calculation
        if calc_data_x.shape[0] > 1: # Need at least 2 points for diff
            diff_x = np.diff(calc_data_x)
            diff_y = np.diff(calc_data_y)
            self.velocity_data = np.sqrt(diff_x**2 + diff_y**2)
            # Pad with NaN at the beginning to match original time axis length for plotting if desired
            # self.velocity_data = np.insert(self.velocity_data, 0, np.nan)
        else:
            self.velocity_data = np.array([])


        # Plotting
        self.ax_position.clear()
        self.ax_position.plot(time_samples, current_eye_data_x, label="X Position (Raw)", alpha=0.7)
        self.ax_position.plot(time_samples, current_eye_data_y, label="Y Position (Raw)", alpha=0.7)
        if self.smoothed_eye_data is not None:
            self.ax_position.plot(time_samples, self.smoothed_eye_data[:, 0], label=f"X Position (Smoothed {self.smoothing_spinbox.value()})", linestyle='--')
            self.ax_position.plot(time_samples, self.smoothed_eye_data[:, 1], label=f"Y Position (Smoothed {self.smoothing_spinbox.value()})", linestyle='--')
        self.ax_position.set_xlabel("Time (samples)")
        self.ax_position.set_ylabel("Position")
        self.ax_position.legend()

        self.ax_velocity.clear()
        if self.velocity_data is not None and self.velocity_data.size > 0:
             # Plot velocity against time_samples[1:] because np.diff reduces length by 1
            self.ax_velocity.plot(time_samples[1:], self.velocity_data, label="Velocity")
        self.ax_velocity.set_xlabel("Time (samples)")
        self.ax_velocity.set_ylabel("Velocity")
        self.ax_velocity.legend()
        
        self.update_saccade_visualizations() # Draw saccade regions

        self.figure.tight_layout()
        self.canvas.draw()

    def setup_span_selector(self):
        if self.span_selector is None and hasattr(self, 'ax_position'):
            self.span_selector = SpanSelector(
                self.ax_position,
                self.on_saccade_select,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="red"), # Visual properties of the selector itself
                interactive=True,
                drag_from_anywhere=True,
                button=1 # Left mouse button
            )
            self.span_selector.active = False # Initially inactive

    def toggle_saccade_mode(self, checked):
        if self.span_selector:
            self.span_selector.active = checked
            # Update button text or style if needed
            if checked:
                self.mark_saccade_button.setText("Marking Mode (ON)")
                # Could change stylesheet too, e.g.
                # self.mark_saccade_button.setStyleSheet("background-color: lightgreen")
            else:
                self.mark_saccade_button.setText("Mark Saccade Mode")
                # self.mark_saccade_button.setStyleSheet("")
        else:
            if checked: # Tried to activate but selector not ready
                 QMessageBox.warning(self, "Error", "Span selector not initialized. Load data first.")
                 self.mark_saccade_button.setChecked(False)


    def on_saccade_select(self, xmin, xmax):
        if self.eye_data is None:
            return

        idx_min, idx_max = int(np.floor(xmin)), int(np.ceil(xmax))
        
        # Ensure indices are within bounds
        idx_min = max(0, idx_min)
        idx_max = min(self.eye_data.shape[0] - 1, idx_max)

        if idx_min >= idx_max: # Avoid zero-length or negative selections
            return

        self.labels[idx_min:idx_max + 1] = 1
        self.marked_saccade_regions.append((idx_min, idx_max))
        
        # Sort regions by start time to make "clear last" more predictable if overlaps allowed
        self.marked_saccade_regions.sort(key=lambda r: r[0]) 
        
        self.update_saccade_visualizations()
        # Optionally, deactivate after one selection if that's the desired UX
        # self.mark_saccade_button.setChecked(False) 


    def update_saccade_visualizations(self):
        # Clear existing spans
        for span in self.saccade_spans:
            try:
                span.remove()
            except Exception: # In case span was already removed or invalid
                pass
        self.saccade_spans.clear()

        # Draw new spans for all marked regions
        if hasattr(self, 'ax_position'): # Ensure axes exist
            for start_idx, end_idx in self.marked_saccade_regions:
                span = self.ax_position.axvspan(start_idx, end_idx, alpha=0.3, color='yellow', ymax=0.95) 
                # ymax to avoid full height, looks a bit better
                self.saccade_spans.append(span)
        
        if hasattr(self, 'canvas'): # Ensure canvas exists
            self.canvas.draw()

    def clear_last_saccade(self):
        if not self.marked_saccade_regions:
            QMessageBox.information(self, "Info", "No saccades to clear.")
            return
        
        start_idx, end_idx = self.marked_saccade_regions.pop()
        if self.labels.size > 0: # Ensure labels array exists and is not empty
             self.labels[start_idx:end_idx + 1] = 0
        
        self.update_saccade_visualizations()

    def clear_all_saccades(self):
        if not self.marked_saccade_regions and np.all(self.labels == 0):
             QMessageBox.information(self, "Info", "No saccades to clear.")
             return

        self.marked_saccade_regions.clear()
        if self.labels.size > 0:
            self.labels.fill(0) # Reset all labels to 0
        
        self.update_saccade_visualizations()


    def load_eye_data(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Eye Data",
            "", # Start directory
            "Data Files (*.csv *.txt *.dat);;All Files (*)"
        )
        if file_name:
            try:
                # Assuming space or tab delimited, and at least two columns.
                # Modify delimiter if needed, e.g., delimiter=',' for CSV
                data = np.loadtxt(file_name, comments='#') # comments='#' is a common convention

                if data.ndim == 1: # If only one column loaded
                    raise ValueError("Data file must have at least two columns for X and Y position.")
                if data.shape[1] < 2: # If less than 2 columns
                    raise ValueError("Data file must have at least two columns for X and Y position.")

                self.eye_data = data[:, :2]  
                self.smoothed_eye_data = None 
                self.velocity_data = None 
                
                # Initialize labels and clear previous markings
                self.labels = np.zeros(self.eye_data.shape[0])
                self.marked_saccade_regions.clear()
                # self.saccade_spans are cleared by update_saccade_visualizations

                if self.span_selector is None: # If first time data loaded
                    self.setup_span_selector()
                
                self.apply_smoothing_and_velocity() # This will handle plotting & call update_saccade_visualizations

                QMessageBox.information(self, "Success", "Eye data loaded successfully.")

            except ValueError as ve: 
                QMessageBox.critical(self, "Error Loading Data", f"Failed to load or parse data file: {ve}\nEnsure the file has at least two numeric columns.")
                self.eye_data = None
                self.labels = np.array([])
                self.marked_saccade_regions.clear()
                self.apply_smoothing_and_velocity() 
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Data", f"An error occurred: {e}")
                self.eye_data = None
                self.labels = np.array([])
                self.marked_saccade_regions.clear()
                self.apply_smoothing_and_velocity()

    def save_labels(self):
        if self.labels is None or self.labels.size == 0:
            QMessageBox.warning(self, "No Labels", "There is no label data to save.")
            return

        # Define file filters
        filters = "CSV files (*.csv);;Text files (*.txt);;NumPy files (*.npy);;All Files (*.*)"
        # Get default save directory (optional, can be empty)
        # default_dir = "" # Or use a remembered path

        file_name, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Labels",
            "", # Start directory
            filters
        )

        if file_name:
            try:
                file_ext = ""
                if "." in file_name:
                    file_ext = file_name.split('.')[-1].lower()

                if selected_filter == "CSV files (*.csv)" or file_ext == 'csv':
                    if not file_ext == 'csv': file_name += '.csv'
                    np.savetxt(file_name, self.labels, fmt='%d', delimiter=',')
                elif selected_filter == "Text files (*.txt)" or file_ext == 'txt':
                    if not file_ext == 'txt': file_name += '.txt'
                    np.savetxt(file_name, self.labels, fmt='%d')
                elif selected_filter == "NumPy files (*.npy)" or file_ext == 'npy':
                    if not file_ext == 'npy': file_name += '.npy'
                    np.save(file_name, self.labels)
                else: # Default or All Files with unknown extension
                    # Defaulting to .txt if no specific extension or filter match
                    if not file_ext in ['csv', 'txt', 'npy']:
                         QMessageBox.information(self, "Defaulting Format", "No specific format selected or recognized, saving as .txt file.")
                         if not file_name.endswith(".txt"): file_name += ".txt"
                    np.savetxt(file_name, self.labels, fmt='%d')
                
                QMessageBox.information(self, "Success", f"Labels saved successfully to {file_name}")

            except Exception as e:
                QMessageBox.critical(self, "Error Saving Labels", f"An error occurred while saving labels: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = SaccadeLabeler()
    sys.exit(app.exec())
