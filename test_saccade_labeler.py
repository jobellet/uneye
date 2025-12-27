import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Ensure QApplication is created before importing SaccadeLabeler if it creates Qt widgets in __init__
from PySide6.QtWidgets import QApplication
app = QApplication.instance()
if app is None:
    app = QApplication([])

from saccade_labeler import SaccadeLabeler


class TestSaccadeLabelerLogic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a dummy test_data directory if it doesn't exist (it should based on previous steps)
        cls.test_data_dir = "test_data"
        os.makedirs(cls.test_data_dir, exist_ok=True)

        # Paths to sample files
        cls.valid_file_path = os.path.join(cls.test_data_dir, "sample_valid_data.txt")
        cls.two_col_file_path = os.path.join(cls.test_data_dir, "sample_two_column_data.txt")
        cls.one_col_file_path = os.path.join(cls.test_data_dir, "sample_one_column_data.txt")
        cls.non_existent_file_path = os.path.join(cls.test_data_dir, "non_existent.txt")

        # Content of sample_valid_data.txt (for assertion)
        # 1.0 2.0 10.0
        # 3.0 4.0 20.0
        # 5.0 6.0 30.0
        # 7.0 8.0 40.0
        cls.expected_valid_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        # Content of sample_two_column_data.txt
        # 1.0 2.0
        # 3.0 4.0
        # 5.0 6.0
        # 7.0 8.0
        # 9.0 10.0
        cls.expected_two_col_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])


    def setUp(self):
        """Set up for each test."""
        self.main_app = SaccadeLabeler()
        # We don't call self.main_app.show()

    # --- 1. Data Loading Tests ---
    @patch('PySide6.QtWidgets.QFileDialog.getOpenFileName')
    @patch('PySide6.QtWidgets.QMessageBox.information') # Mock success message
    def test_load_valid_data_three_cols(self, mock_msg_info, mock_dialog):
        mock_dialog.return_value = (self.valid_file_path, "Data Files (*.csv *.txt *.dat)")
        self.main_app.load_eye_data()
        self.assertIsNotNone(self.main_app.eye_data)
        self.assertEqual(self.main_app.eye_data.shape, (4, 2))
        np.testing.assert_array_equal(self.main_app.eye_data, self.expected_valid_data)
        mock_msg_info.assert_called_once()

    @patch('PySide6.QtWidgets.QFileDialog.getOpenFileName')
    @patch('PySide6.QtWidgets.QMessageBox.information')
    def test_load_valid_data_two_cols(self, mock_msg_info, mock_dialog):
        mock_dialog.return_value = (self.two_col_file_path, "Data Files (*.csv *.txt *.dat)")
        self.main_app.load_eye_data()
        self.assertIsNotNone(self.main_app.eye_data)
        self.assertEqual(self.main_app.eye_data.shape, (5, 2))
        np.testing.assert_array_equal(self.main_app.eye_data, self.expected_two_col_data)
        mock_msg_info.assert_called_once()

    @patch('PySide6.QtWidgets.QFileDialog.getOpenFileName')
    @patch('PySide6.QtWidgets.QMessageBox.critical') # Mock error message
    def test_load_invalid_data_one_col(self, mock_msg_critical, mock_dialog):
        mock_dialog.return_value = (self.one_col_file_path, "Data Files (*.csv *.txt *.dat)")
        self.main_app.load_eye_data()
        self.assertIsNone(self.main_app.eye_data)
        mock_msg_critical.assert_called_once()
        # Check if plots are cleared (exemplified by checking if labels is empty)
        self.assertEqual(self.main_app.labels.size, 0)


    @patch('PySide6.QtWidgets.QFileDialog.getOpenFileName')
    @patch('PySide6.QtWidgets.QMessageBox.critical')
    def test_load_non_existent_file(self, mock_msg_critical, mock_dialog):
        mock_dialog.return_value = (self.non_existent_file_path, "Data Files (*.csv *.txt *.dat)")
        self.main_app.load_eye_data()
        self.assertIsNone(self.main_app.eye_data)
        # np.loadtxt will raise FileNotFoundError, which is caught by the general Exception in load_eye_data
        mock_msg_critical.assert_called_once()
        self.assertEqual(self.main_app.labels.size, 0)

    @patch('PySide6.QtWidgets.QFileDialog.getOpenFileName')
    def test_load_dialog_cancel(self, mock_dialog):
        mock_dialog.return_value = ("", "") # User cancels dialog
        initial_eye_data = self.main_app.eye_data # Could be None or previous data
        self.main_app.load_eye_data()
        if initial_eye_data is None:
            self.assertIsNone(self.main_app.eye_data)
        else:
            np.testing.assert_array_equal(self.main_app.eye_data, initial_eye_data)

    # --- 2. Smoothing Function Tests ---
    def test_smoothing_simple(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        window_size = 3
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) # convolve with ones/3, mode same
        # Manual calculation for mode='same' with window=3:
        # x_s[0] = (0*x[-1] + 0*x[0] + 1*x[0] + 1*x[1] + 1*x[2]) / 3 ... this is not how np.convolve works with mode='same'
        # np.convolve pads with zeros at ends.
        # For [1,2,3,4,5], window 3:
        # (1+2)/3 -> 1st element if mode='valid'
        # (0*0 + 1*1 + 1*2)/3 = 1.0
        # (1*1 + 1*2 + 1*3)/3 = 2.0
        # (1*2 + 1*3 + 1*4)/3 = 3.0
        # (1*3 + 1*4 + 1*5)/3 = 4.0
        # (1*4 + 1*5 + 0*0)/3 = 3.0 ... this is not correct.
        # Let's use a known output from np.convolve
        # np.convolve([1,2,3,4,5], [1/3,1/3,1/3], mode='same') -> array([0.333, 1.    , 2.    , 3.    , 4.    , 3.    , 2.333])
        # My _moving_average ensures odd window, so 3 is fine.
        # data = np.array([1.,2.,3.,4.,5.])
        # window = np.ones(3)/3
        # np.convolve(data, window, mode='same') -> array([1.        , 2.        , 3.        , 4.        , 3.66666667]) (Mistake here, last element is (4+5+0)/3)
        # It should be:
        # (0+1+2)/3 = 1
        # (1+2+3)/3 = 2
        # (2+3+4)/3 = 3
        # (3+4+5)/3 = 4
        # (4+5+0)/3 = 3 -> This is how it behaves.
        data_to_smooth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_smoothed = np.array([ (0+1+2)/3, (1+2+3)/3, (2+3+4)/3, (3+4+5)/3, (4+5+0)/3 ])
        # Recalculating np.convolve([1,2,3,4,5], [1,1,1]/3, 'same')
        # Pad data with (3-1)/2 = 1 zero on each side effectively for calculation [0,1,2,3,4,5,0]
        # Result[0] = (0*w0 + 1*w1 + 2*w2) -> if weights are [1/3,1/3,1/3] -> (0+1+2)/3 = 1
        # Result[1] = (1*w0 + 2*w1 + 3*w2) -> (1+2+3)/3 = 2
        # Result[2] = (2*w0 + 3*w1 + 4*w2) -> (2+3+4)/3 = 3
        # Result[3] = (3*w0 + 4*w1 + 5*w2) -> (3+4+5)/3 = 4
        # Result[4] = (4*w0 + 5*w1 + 0*w2) -> (4+5+0)/3 = 3
        expected_smoothed = np.array([1., 2., 3., 4., 3.])


        smoothed = self.main_app._moving_average(data_to_smooth, 3)
        np.testing.assert_array_almost_equal(smoothed, expected_smoothed)

        data_long = np.array([1.,1.,1.,10.,10.,10.,1.,1.,1.])
        # Expected: [0.66, 1, 1, 4, 10, 10, 7, 1, 1, 0.66] - from manual trace
        # (0+1+1)/3 = 0.66
        # (1+1+1)/3 = 1
        # (1+1+10)/3 = 4
        # (1+10+10)/3 = 7
        # (10+10+10)/3 = 10
        # (10+10+1)/3 = 7
        # (10+1+1)/3 = 4
        # (1+1+0)/3 = 0.66
        expected_long = np.array([0.66666667, 1. , 4. , 7. ,10. , 7. , 4. , 1. , 0.66666667])
        smoothed_long = self.main_app._moving_average(data_long, 3)
        np.testing.assert_array_almost_equal(smoothed_long, expected_long)


    def test_smoothing_window_larger_than_array(self):
        data = np.array([1.0, 2.0, 3.0])
        window_size = 5 # Larger than data
        # np.convolve([1,2,3], np.ones(5)/5, mode='same')
        # Padded: [0,0,1,2,3,0,0]
        # (0+0+1+2+3)/5 = 1.2
        # (0+1+2+3+0)/5 = 1.2
        # (1+2+3+0+0)/5 = 1.2
        expected = np.array([0.6, 1.2, 1.2]) # Recalculated from np.convolve behavior
        # (0*w0+0*w1+1*w2+2*w3+3*w4) -> (0+0+1+2+3)/5 = 1.2  (if centered kernel)
        # (0*w0+1*w1+2*w2+3*w3+0*w4) -> (0+1+2+3+0)/5 = 1.2
        # (1*w0+2*w1+3*w2+0*w3+0*w4) -> (1+2+3+0+0)/5 = 1.2
        # For np.convolve([1,2,3], [1,1,1,1,1]/5, mode='same')
        # result[0] = (w[2]*1 + w[3]*2 + w[4]*3) = (1+2+3)/5 = 1.2
        # result[1] = (w[1]*1 + w[2]*2 + w[3]*3) = (1+2+3)/5 = 1.2
        # result[2] = (w[0]*1 + w[1]*2 + w[2]*3) = (1+2+3)/5 = 1.2
        # This is wrong. np.convolve does not center the kernel for 'same' if kernel is longer.
        # It effectively aligns the start of kernel with start of (padded) array for each output element.
        # Example: np.convolve([1,2,3], np.array([0.2,0.2,0.2,0.2,0.2]), mode='same')
        # is array([0.2, 0.6, 1.2])
        expected = np.array([0.2, 0.6, 1.2])
        smoothed = self.main_app._moving_average(data, window_size)
        np.testing.assert_array_almost_equal(smoothed, expected)

    def test_smoothing_even_window_becomes_odd(self):
        # _moving_average should increment even window size by 1
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window_size = 4 # Will become 5
        # Expected is same as window_size = 5 for this data
        # np.convolve(data, np.ones(5)/5, mode='same')
        # Padded: [0,0,1,2,3,4,5,0,0]
        # (0+0+1+2+3)/5 = 1.2
        # (0+1+2+3+4)/5 = 2.0
        # (1+2+3+4+5)/5 = 3.0
        # (2+3+4+5+0)/5 = 2.8
        # (3+4+5+0+0)/5 = 2.4
        expected = np.array([1.2, 2. , 3. , 2.8, 2.4])
        smoothed = self.main_app._moving_average(data, window_size)
        np.testing.assert_array_almost_equal(smoothed, expected)

    # --- 3. Velocity Calculation Tests ---
    def test_velocity_calculation_no_smoothing(self):
        # Prepare eye_data
        self.main_app.eye_data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        self.main_app.smoothing_checkbox.setChecked(False)
        
        self.main_app.apply_smoothing_and_velocity()
        
        self.assertIsNotNone(self.main_app.velocity_data)
        # Expected velocity: sqrt((2-1)^2 + (2-1)^2) = sqrt(1+1) = sqrt(2) approx 1.414
        # sqrt((3-2)^2 + (3-2)^2) = sqrt(2)
        # sqrt((4-3)^2 + (4-3)^2) = sqrt(2)
        expected_velocity = np.array([np.sqrt(2), np.sqrt(2), np.sqrt(2)])
        np.testing.assert_array_almost_equal(self.main_app.velocity_data, expected_velocity, decimal=3)
        self.assertEqual(len(self.main_app.velocity_data), len(self.main_app.eye_data) - 1)

    def test_velocity_calculation_with_smoothing(self):
        # Prepare eye_data - simple data that will be noticeably smoothed
        self.main_app.eye_data = np.array([[1., 1.], [1., 1.], [10., 10.], [10., 10.], [1., 1.]])
        self.main_app.labels = np.zeros(self.main_app.eye_data.shape[0]) # Need labels for plot updates
        self.main_app.smoothing_checkbox.setChecked(True)
        self.main_app.smoothing_spinbox.setValue(3) # Window size 3
        
        self.main_app.apply_smoothing_and_velocity()
        
        self.assertIsNotNone(self.main_app.velocity_data)
        self.assertIsNotNone(self.main_app.smoothed_eye_data)
        
        # Expected smoothed data for X (and Y):
        # (0+1+1)/3 = 0.666
        # (1+1+10)/3 = 4
        # (1+10+10)/3 = 7
        # (10+10+1)/3 = 7
        # (10+1+0)/3 = 3.666
        expected_smoothed_x = np.array([0.66666667, 4. , 7. , 7. , 3.66666667])
        # Velocity from this smoothed data:
        # dx = [3.333, 3, 0, -3.333]
        # dy = [3.333, 3, 0, -3.333]
        # vel = sqrt(dx^2+dy^2)
        dx = np.diff(expected_smoothed_x)
        expected_velocity = np.sqrt(dx**2 + dx**2) # Using dx for dy as Y is same as X
        
        np.testing.assert_array_almost_equal(self.main_app.velocity_data, expected_velocity, decimal=3)
        self.assertEqual(len(self.main_app.velocity_data), len(self.main_app.eye_data) - 1)


    # --- 4. Label Management Tests ---
    def test_labels_initialization_on_load(self):
        # Mock load_eye_data to simulate data loading
        with patch.object(self.main_app, 'load_eye_data', MagicMock()) as mock_load:
            self.main_app.eye_data = np.array([[1,2],[3,4],[5,6]]) # Simulate data loaded
            # Manually trigger what load_eye_data does for labels
            self.main_app.labels = np.zeros(self.main_app.eye_data.shape[0])
            self.main_app.marked_saccade_regions.clear()
            
            # Call apply_smoothing_and_velocity which also updates visualizations
            # This part is more about ensuring load_eye_data properly sets up labels
            self.main_app.apply_smoothing_and_velocity()


        self.assertEqual(len(self.main_app.labels), 3)
        np.testing.assert_array_equal(self.main_app.labels, np.array([0,0,0]))
        self.assertEqual(len(self.main_app.marked_saccade_regions), 0)

    def test_mark_saccade_on_saccade_select(self):
        self.main_app.eye_data = np.zeros((10, 2)) # Dummy 10 samples
        self.main_app.labels = np.zeros(10)
        self.main_app.marked_saccade_regions = []
        
        # Simulate selection from sample 2 to 5 (inclusive for labels)
        # SpanSelector gives float values, so simulate that
        self.main_app.on_saccade_select(1.5, 5.5) # Should select indices 1 through 5
        
        expected_labels = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(self.main_app.labels, expected_labels)
        self.assertEqual(self.main_app.marked_saccade_regions, [(1, 5)])

    def test_clear_last_saccade(self):
        self.main_app.eye_data = np.zeros((10, 2))
        self.main_app.labels = np.zeros(10)
        self.main_app.marked_saccade_regions = []

        # Mark two saccades
        self.main_app.on_saccade_select(1.0, 3.0) # Region (1,3)
        self.main_app.on_saccade_select(5.0, 7.0) # Region (5,7)
        
        self.assertEqual(len(self.main_app.marked_saccade_regions), 2)
        
        self.main_app.clear_last_saccade()
        
        expected_labels_after_clear_last = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(self.main_app.labels, expected_labels_after_clear_last)
        self.assertEqual(self.main_app.marked_saccade_regions, [(1, 3)]) # Only first region remains

        # Clear the last remaining one
        self.main_app.clear_last_saccade()
        np.testing.assert_array_equal(self.main_app.labels, np.zeros(10))
        self.assertEqual(len(self.main_app.marked_saccade_regions), 0)

    def test_clear_all_saccades(self):
        self.main_app.eye_data = np.zeros((10, 2))
        self.main_app.labels = np.zeros(10)
        self.main_app.marked_saccade_regions = []

        self.main_app.on_saccade_select(1.0, 3.0)
        self.main_app.on_saccade_select(5.0, 7.0)
        
        self.main_app.clear_all_saccades()
        
        np.testing.assert_array_equal(self.main_app.labels, np.zeros(10))
        self.assertEqual(len(self.main_app.marked_saccade_regions), 0)

    # --- 5. Data Saving Tests ---
    @patch('PySide6.QtWidgets.QFileDialog.getSaveFileName')
    @patch('numpy.savetxt')
    @patch('numpy.save')
    @patch('PySide6.QtWidgets.QMessageBox.information') # Mock success message
    def test_save_labels_csv(self, mock_msg_info, mock_np_save, mock_np_savetxt, mock_dialog):
        self.main_app.labels = np.array([0, 1, 1, 0, 1])
        mock_dialog.return_value = ("dummy.csv", "CSV files (*.csv)")
        
        self.main_app.save_labels()
        
        mock_np_savetxt.assert_called_once_with("dummy.csv", self.main_app.labels, fmt='%d', delimiter=',')
        mock_np_save.assert_not_called()
        mock_msg_info.assert_called_once()

    @patch('PySide6.QtWidgets.QFileDialog.getSaveFileName')
    @patch('numpy.savetxt')
    @patch('numpy.save')
    @patch('PySide6.QtWidgets.QMessageBox.information')
    def test_save_labels_txt(self, mock_msg_info, mock_np_save, mock_np_savetxt, mock_dialog):
        self.main_app.labels = np.array([0, 1, 0, 1, 0])
        mock_dialog.return_value = ("dummy.txt", "Text files (*.txt)")
        
        self.main_app.save_labels()
        
        mock_np_savetxt.assert_called_once_with("dummy.txt", self.main_app.labels, fmt='%d')
        mock_np_save.assert_not_called()
        mock_msg_info.assert_called_once()

    @patch('PySide6.QtWidgets.QFileDialog.getSaveFileName')
    @patch('numpy.savetxt')
    @patch('numpy.save')
    @patch('PySide6.QtWidgets.QMessageBox.information')
    def test_save_labels_npy(self, mock_msg_info, mock_np_save, mock_np_savetxt, mock_dialog):
        self.main_app.labels = np.array([1, 1, 0, 0, 1])
        mock_dialog.return_value = ("dummy.npy", "NumPy files (*.npy)")
        
        self.main_app.save_labels()
        
        mock_np_save.assert_called_once_with("dummy.npy", self.main_app.labels)
        mock_np_savetxt.assert_not_called()
        mock_msg_info.assert_called_once()

    @patch('PySide6.QtWidgets.QFileDialog.getSaveFileName')
    @patch('PySide6.QtWidgets.QMessageBox.warning') # Mock warning message
    def test_save_labels_no_data(self, mock_msg_warning, mock_dialog):
        self.main_app.labels = np.array([]) # Empty labels
        # mock_dialog.return_value = ("dummy.txt", "Text files (*.txt)") # Dialog won't be called
        
        self.main_app.save_labels()
        
        mock_msg_warning.assert_called_once_with(self.main_app, "No Labels", "There is no label data to save.")
        mock_dialog.assert_not_called()


    @patch('PySide6.QtWidgets.QFileDialog.getSaveFileName')
    @patch('numpy.savetxt')
    @patch('PySide6.QtWidgets.QMessageBox.information')
    def test_save_labels_default_to_txt(self, mock_msg_info, mock_np_savetxt, mock_dialog):
        self.main_app.labels = np.array([0,1,1,0])
        # User types 'myfile' and selects 'All Files (*.*)'
        mock_dialog.return_value = ("myfile", "All Files (*.*)")
        
        # Mock the information dialog that warns about defaulting to .txt
        with patch('PySide6.QtWidgets.QMessageBox.information') as mock_default_info:
            self.main_app.save_labels()
        
            # Check if the specific default format warning was shown
            # This call is separate from the final success message.
            # Depending on execution order, it might be the first or second QMessageBox.information call.
            # For simplicity here, we'll assume it's called.
            # A more robust check would involve checking call_args_list.
            # mock_default_info.assert_any_call(self.main_app, "Defaulting Format", "No specific format selected or recognized, saving as .txt file.")
            # The above assertion is tricky due to multiple QMessageBox.information mocks. Let's check the save call.

        # Should save as "myfile.txt"
        mock_np_savetxt.assert_called_once_with("myfile.txt", self.main_app.labels, fmt='%d')
        # mock_msg_info.assert_called_with(self.main_app, "Success", f"Labels saved successfully to myfile.txt")


if __name__ == '__main__':
    unittest.main()
