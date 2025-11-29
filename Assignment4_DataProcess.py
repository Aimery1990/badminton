
import os
from typing import Union
from IPython.display import display
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Keywords:
    classification_column_name = "shot_type"
    datetime_index = 'datetime'
    training = 'training'
    validating = 'validating'


from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq


class TimeDomainAug:
    """
    Performs data augmentation on 3D windowed time series data (N_windows, T, F).
    """

    def jitter(self, windows, std=0.01):
        """Adds random noise to the time series data."""
        noise = np.random.normal(0, std, windows.shape)
        return windows + noise

    def axis_rotate(self, windows, max_angle_deg=5):
        """
        Applies a random 2D rotation (e.g., around Z-axis) to the vector data
        (Accel, Gyro) across all timesteps in all windows.
        """
        windows_aug = windows.copy()

        theta = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
        c, s = np.cos(theta), np.sin(theta)

        # Rotation matrix (rotation around the Z-axis for X-Y plane)
        rot_matrix = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

        # Apply rotation to Accelerometer (indices 0, 1, 2)
        accel_slice = windows_aug[:, :, 0:3]
        windows_aug[:, :, 0:3] = np.einsum('ijk,kl->ijl', accel_slice, rot_matrix)

        # Apply rotation to Gyroscope (indices 3, 4, 5)
        gyro_slice = windows_aug[:, :, 3:6]
        windows_aug[:, :, 3:6] = np.einsum('ijk,kl->ijl', gyro_slice, rot_matrix)

        return windows_aug

    def amplitude_scale(self, windows, scale_range=(0.95, 1.05)):
        """
        Applies a random scaling factor to the amplitude of each feature channel (F)
        across all timesteps (T) and all windows (N).
        """
        # Generate one scale factor per feature channel (F)
        scales = np.random.uniform(scale_range[0], scale_range[1], windows.shape[2])

        # The scale array needs to be broadcastable to the (N, T, F) shape.
        # Reshape scales from (F,) to (1, 1, F)
        scales_reshaped = scales[np.newaxis, np.newaxis, :]

        return windows * scales_reshaped

    def augment_set(self, X_w, y_w, n_aug=1):
        """
        Applies a sequence of augmentation techniques and stacks the results.
        """
        print(f"[TimeDomainAug] Input shape: {X_w.shape}, Generating {n_aug} augmented copies.")

        X_augmented_list = [X_w]  # Start with original data
        y_augmented_list = [y_w]

        for i in range(n_aug):
            print(f"--- Augmentation Pass #{i + 1} ---")

            # Start with a clean copy of the original windows
            X_aug = X_w.copy()

            # 1. Apply Jitter
            X_aug = self.jitter(X_aug, std=0.01)

            # 2. Apply Rotation
            X_aug = self.axis_rotate(X_aug, max_angle_deg=10)

            # 3. Apply Amplitude Scale (New/Fixed)
            X_aug = self.amplitude_scale(X_aug, scale_range=(0.9, 1.1))  # Used slightly wider range for effect

            X_augmented_list.append(X_aug)
            y_augmented_list.append(y_w)

        X_final = np.vstack(X_augmented_list)
        y_final = np.concatenate(y_augmented_list)

        print(f"[TimeDomainAug] Output shape: X={X_final.shape}, Y={y_final.shape}")

        return X_final, y_final



def vector_magnitude(window, idxs):
    """
    Accel: idxs = [0,1,2]
    Gyro: idxs = [3,4,5]
    """
    return np.sqrt(np.sum(window[:, idxs] ** 2, axis=1))


def corr_safe(a, b):
    # 若任一軸恆定，視為沒相關，返回0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0
    return np.corrcoef(a, b)[0, 1]


class FeatureExtractor:
    def __init__(self, sensor_cols):
        self.sensor_cols = sensor_cols

    def extract_features(self, window):
        features = {}
        features.update(self.time_stats(window))
        features.update(self.resultant_features(window))
        features.update(self.peak_features(window))
        features.update(self.jerk_features(window))
        features.update(self.frequency_features(window, sampling_freq=10))
        features['gyro_integration_magnitude'] = self.gyro_integration_magnitude(window, [3, 4, 5])
        features.update(self.cross_axis_corr(window, [0, 1, 2], "accel"))
        features.update(self.cross_axis_corr(window, [3, 4, 5], "gyro"))

        return features

    def time_stats(self, window):
        # window shape: (window_size, n_features)
        stats = {}
        for idx, name in enumerate(self.sensor_cols):
            data = window[:, idx]
            stats[f"{name}_mean"] = np.mean(data)
            stats[f"{name}_std"] = np.std(data)
            stats[f"{name}_mad"] = np.median(np.abs(data - np.median(data)))
            stats[f"{name}_min"] = np.min(data)
            stats[f"{name}_max"] = np.max(data)
            stats[f"{name}_range"] = np.max(data) - np.min(data)
            stats[f"{name}_iqr"] = np.percentile(data, 75) - np.percentile(data, 25)
        return stats

    def resultant_features(self, window):
        # 假設sensor順序分別是（X, Y, Z）,如accel_X, accel_Y, accel_Z
        out = {}
        group_indices = {
            'a': [0, 1, 2], 'g': [3, 4, 5], 'm': [6, 7, 8]
        }
        for key, idxs in group_indices.items():
            mag = np.sqrt(np.sum(window[:, idxs] ** 2, axis=1))
            out[f"{key}_mag_mean"] = np.mean(mag)
            out[f"{key}_mag_std"] = np.std(mag)
        return out

    def peak_features(self, window):
        out = {}
        for key, idxs in {'a': [0, 1, 2], 'g': [3, 4, 5]}.items():
            mag = np.sqrt(np.sum(window[:, idxs] ** 2, axis=1))
            peaks, _ = find_peaks(mag)
            peak_vals = mag[peaks]
            out[f"{key}_peak_count"] = len(peaks)
            out[f"{key}_peak_mean_ampl"] = np.mean(peak_vals) if len(peak_vals) > 0 else 0
            out[f"{key}_peak_max_ampl"] = np.max(peak_vals) if len(peak_vals) > 0 else 0
        return out

    def jerk_features(self, window):
        out = {}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            a = window[:, i]  # Accel_X, Accel_Y, Accel_Z
            jerk = np.diff(a)
            out[f"jerk_{axis}_mean"] = np.mean(jerk)
            out[f"jerk_{axis}_std"] = np.std(jerk)
        return out

    def frequency_features(self, window, sampling_freq=10):
        feats = {}
        N = window.shape[0]

        for key, idxs in {'a': [0, 1, 2], 'g': [3, 4, 5]}.items():
            sig = vector_magnitude(window, idxs)
            # Remove DC分量
            sig_zm = sig - np.mean(sig)
            # 計算實FFT(只返回正頻)
            fft_vals = np.abs(rfft(sig_zm))
            fft_freqs = rfftfreq(N, d=1 / sampling_freq)

            # 主導頻率與功率
            dom_idx = np.argmax(fft_vals)
            dom_freq = fft_freqs[dom_idx]
            dom_power = fft_vals[dom_idx]
            feats[f"{key}_dom_freq"] = dom_freq
            feats[f"{key}_dom_power"] = dom_power

            # 頻帶能量
            def band_energy(fmin, fmax):
                band_mask = (fft_freqs >= fmin) & (fft_freqs < fmax)
                return np.sum(fft_vals[band_mask] ** 2)

            feats[f"{key}_energy_low"] = band_energy(0.2, 1.5)
            feats[f"{key}_energy_mid"] = band_energy(1.5, 4)
        return feats

    def gyro_integration_magnitude(self, window, gyro_idxs=[3, 4, 5], dt=0.1):
        """
        window: shape (n_samples, n_features)
        gyro_idxs: 三個gyroscope欄位的index（如[3,4,5]）
        dt: 單位時間間隔，假設10Hz則dt=0.1s
        """
        gyro_window = window[:, gyro_idxs]  # [n_samples, 3]
        # 對每個軸積分（近似：累加值 * dt），然後計算總位移（旋轉模長）
        integral = np.sum(gyro_window, axis=0) * dt
        mag = np.linalg.norm(integral)
        return mag  # 輸出單個float，代表該窗口總旋轉量

    def cross_axis_corr(self, window, idxs, sensor_prefix):
        vals = window[:, idxs]
        c_xy = corr_safe(vals[:, 0], vals[:, 1])
        c_xz = corr_safe(vals[:, 0], vals[:, 2])
        c_yz = corr_safe(vals[:, 1], vals[:, 2])
        return {
            f"{sensor_prefix}_corr_xy": c_xy,
            f"{sensor_prefix}_corr_xz": c_xz,
            f"{sensor_prefix}_corr_yz": c_yz
        }




class DataPrep:
    def __init__(self, data_path, data_name, encoding='latin1', raw_df=None, debug=True):
        self.data_path = data_path
        self.data_name = data_name
        self.debug = debug

        if raw_df is not None:
            self.data_df = raw_df  # Use the pre-combined DataFrame
        else:
            self.data_df = pd.read_csv(self.data_path, encoding=encoding)

        # self.data_df = pd.read_csv(self.data_path, encoding=encoding)
        if self.debug:
            print(f"{self.data_path} head:")
            display(self.data_df.head())

        self.data_df[Keywords.classification_column_name] = self.data_name
        self.column_dict = {}
        self.columns = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_windows = None
        self.train_windows_labels = None
        self.test_windows = None
        self.test_windows_labels = None

        self.train_windows_df = None
        self.test_windows_df = None

        self.feature_df_train = None
        self.feature_df_test = None
        # self.feature_df = None
        self.feature_df_train_scaled = None
        self.feature_df_test_scaled = None

    def timestamp(self):
        self.data_df['datetime'] = pd.to_datetime(self.data_df['Date'] + ' ' + self.data_df['Time'])

        duplicate_time_entries = self.data_df[self.data_df.duplicated(subset=['datetime'], keep=False)].sort_values(
            by='datetime')

        if not duplicate_time_entries.empty:
            print("Duplicated or Overlapping Time Entries found:")
            display(duplicate_time_entries)
            return duplicate_time_entries
        else:
            print("No duplicated or overlapping time entries found.")
            return None

    def data_columns_reg(self, **kwargs):
        '''
        call like:
        data_columns_reg(
            accel_cols = ['Accel_X (g)', 'Accel_Y (g)', 'Accel_Z (g)'],
            gyro_cols = ['Gyro_X (°/s)', 'Gyro_Y (°/s)', 'Gyro_Z (°/s)'],
            mag_cols = ['Mag_X (G)', 'Mag_Y (G)', 'Mag_Z (G)'],
        )
        '''

        for key, value in kwargs.items():
            self.column_dict[key] = value
            self.columns += value
            if self.debug:
                print(f"{key}: {value}")

    def is_missing_values(self):
        missing_values = self.data_df.isnull().sum()

        if self.debug:
            print(missing_values)
        if (missing_values > 0).any():
            print("Missing values per column:")
            print(missing_values)
            return True
        else:
            return False

    def set_index_sort(self, index=Keywords.datetime_index):
        self.data_df = self.data_df.set_index(index, drop=True)
        self.data_df = self.data_df.sort_index()
        if self.debug:
            print("set_index_sort DataFrame head:")
            self.data_df.info()

    def resample_forward_fill(self, period='100ms', drop_init_na=True):
        self.data_df = self.data_df.resample(period).ffill()
        if drop_init_na:
            self.data_df = self.data_df.iloc[1:]

        if self.debug:
            print("Resampled DataFrame head:")
            display(self.data_df.head())
            display(self.data_df['Mag_X (G)'].mean())
            display(self.data_df['Mag_X (G)'].std())
            display(self.data_df['Mag_Z (G)'].mean())
            display(self.data_df['Mag_Z (G)'].std())
            print("\nResampled DataFrame info:")
            self.data_df.info()

    def data_calibration(self):

        for col in self.columns:
            global_mean = self.data_df[col].mean()
            self.data_df[col] = self.data_df[col] - global_mean
            if self.debug:
                print(f"column: {col} mean: {global_mean} subtracted")

        if self.debug:
            print("Calibrated DataFrame head:")
            display(self.data_df.head())
            display(self.data_df['Mag_X (G)'].mean())
            display(self.data_df['Mag_X (G)'].std())
            display(self.data_df['Mag_Z (G)'].mean())
            display(self.data_df['Mag_Z (G)'].std())
            print("\nCalibrated DataFrame info:")
            self.data_df.info()

    def noise_reduce(self, window_length=5, polyorder=2):
        '''
        window_length (int): The length of the filter window. Must be an odd integer.
        polyorder (int): The order of the fitting polynomial. Must be less than the window_length.
        '''

        if window_length % 2 == 0:
            window_length += 1
            print(f"Warning: window_length adjusted to {window_length} (must be odd).")

        if polyorder >= window_length:
            polyorder = window_length - 1
            print(f"Warning: polyorder adjusted to {polyorder} (must be < window_length).")

        cols_to_filter = self.columns

        if self.debug:
            print(
                f"Applying Savitzky-Golay Filter (Window: {window_length}, PolyOrder: {polyorder}) to columns: {cols_to_filter}")

        for col in cols_to_filter:

            self.data_df[col] = savgol_filter(
                self.data_df[col],
                window_length=window_length,
                polyorder=polyorder
            )
            if self.debug:
                print(f"Column: {col} smoothed.")

        if self.debug:
            print("\nNoise Reduced DataFrame head:")
            display(self.data_df.head())
            print("\nNoise Reduced DataFrame info:")
            self.data_df.info()

    def train_test_split(self, test_size=0.2, random_state=42):

        # Use the correct classification column name defined in Keywords
        target_col = Keywords.classification_column_name

        X = self.data_df.drop(columns=[target_col])
        y = self.data_df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print(f"Shape of X_train: {self.X_train.shape}")
        print(f"Shape of X_test: {self.X_test.shape}")
        print(f"Shape of y_train: {self.y_train.shape}")
        print(f"Shape of y_test: {self.y_test.shape}")


    def window_size_and_overlap(self, sampling_frequency_hz=10, window_duration_seconds=2.0, overlap_percentage=0.5):
        window_size = int(window_duration_seconds * sampling_frequency_hz)
        overlap = int(window_size * overlap_percentage)
        return window_size, overlap


    def create_windows_for_set(self, data_set_type, window_size, overlap_size):
        """
        Creates time windows (samples) from the continuous time series data
        for the specified set (training or validating), ensuring independent processing.

        Args:
            data_set_type (str): Must be Keywords.training or Keywords.validating.
            window_size (int): The length of the window in number of timesteps.
            overlap_size (int): The length of the overlap in number of timesteps.

        Returns:
            tuple: (X_windowed, y_windowed) - A numpy array of 3D windows
                   and a numpy array of corresponding labels.
        """

        # Select the correct dataset based on the input type
        if data_set_type == Keywords.training:
            X_data = self.X_train
            y_data = self.y_train
        elif data_set_type == Keywords.validating:
            X_data = self.X_test
            y_data = self.y_test
        else:
            raise ValueError(f"Invalid keyword for data_set_type: {data_set_type}. Use 'training' or 'validating'.")

        # Check if data sets exist
        if X_data is None or y_data is None:
            print(f"Error: {data_set_type} set is None. Please run train_test_split first.")
            return np.array([]), np.array([])

        windowed_data = []
        windowed_labels = []

        # Calculate step size
        step_size = window_size - overlap_size
        if step_size <= 0:
            raise ValueError("Overlap size cannot be greater than or equal to window size.")

        N = len(X_data)  # Total length of the current dataset

        if self.debug:
            print(f"Creating windows for {data_set_type} set (Total Timesteps: {N}).")
            print(f"Window Size: {window_size}, Overlap: {overlap_size}, Step Size: {step_size}")

        # Iterate through the data using the calculated step size
        for i in range(0, N - window_size + 1, step_size):
            # 1. Extract the time window (sensor data)
            # Use iloc for position-based indexing on the current dataset (X_data)
            window = X_data.iloc[i: i + window_size]

            # 2. Extract the corresponding labels for the window
            labels_in_window = y_data.iloc[i: i + window_size]

            # 3. Assign a single label to the window (most common label within the window)
            # .mode()[0] gets the most frequent label. For single-activity files, this will be the file's label.
            most_common_label = labels_in_window.mode()[0]

            # Convert sensor data to Numpy array and append
            sensor_data = window[self.columns].values
            windowed_data.append(sensor_data)
            windowed_labels.append(most_common_label)

        if self.debug:
            print(f"Successfully created {len(windowed_data)} windows for the {data_set_type} set.")


        if data_set_type == Keywords.training:
            self.train_windows = np.array(windowed_data)
            self.train_windows_labels = np.array(windowed_labels)
            # self.train_windows_df = pd.DataFrame(self.train_windows, columns=self.columns)
            # self.train_windows_df[Keywords.classification_column_name] = self.train_windows_labels

        elif data_set_type == Keywords.validating:
            self.test_windows = np.array(windowed_data)
            self.test_windows_labels = np.array(windowed_labels)
            # self.test_windows_df = pd.DataFrame(self.test_windows, columns=self.columns)
            # self.test_windows_df[Keywords.classification_column_name] = self.test_windows_labels

        return np.array(windowed_data), np.array(windowed_labels)

    def feature_extract(self, data_set_type):
        """
        Extracts features from the windowed time series data for the specified set,
        and saves the resulting 2D feature DataFrame.
        """

        if data_set_type == Keywords.training:
            windows = self.train_windows
            labels = self.train_windows_labels
        elif data_set_type == Keywords.validating:
            windows = self.test_windows
            labels = self.test_windows_labels
        else:
            raise ValueError(f"Invalid keyword for data_set_type: {data_set_type}. Use 'training' or 'validating'.")

        if windows is None:
            print(f"Error: Windowed data for {data_set_type} is None. Run create_windows_for_set first.")
            return

        extractor = FeatureExtractor(self.columns)
        window_features = []

        # 1. Feature Extraction Loop
        for window in windows:
            feature_extracted_window = extractor.extract_features(window)
            window_features.append(feature_extracted_window)

        # 2. Convert List of Dictionaries to DataFrame
        feature_df = pd.DataFrame(window_features)

        # 3. Add the classification label column
        feature_df[Keywords.classification_column_name] = labels

        # 4. Save the resulting feature DataFrame to the correct attribute
        if data_set_type == Keywords.training:
            self.feature_df_train = feature_df
        elif data_set_type == Keywords.validating:
            self.feature_df_test = feature_df

        if self.debug:
            print(
                f"Feature extraction complete for {data_set_type} set. Final Feature DataFrame Shape: {feature_df.shape}")
            display(feature_df.head())

        return feature_df

    def feature_normalize(self):
        """
        Fits a StandardScaler on the training feature set (self.feature_df_train)
        and then applies the transformation to both the training and testing feature sets,
        preventing data leakage.
        """
        if self.feature_df_train is None or self.feature_df_test is None:
            print("Error: Feature DataFrames are not available. Please run feature_extract for both sets first.")
            return

        # 1. Separate features (X) and labels (y) for both sets
        # Get the list of feature columns by dropping the classification column
        feature_cols = self.feature_df_train.columns.drop(Keywords.classification_column_name)

        X_train = self.feature_df_train[feature_cols]
        y_train = self.feature_df_train[Keywords.classification_column_name]

        X_test = self.feature_df_test[feature_cols]
        y_test = self.feature_df_test[Keywords.classification_column_name]

        # 2. Initialize and Fit Scaler on Training Data ONLY
        scaler = StandardScaler()

        # Fit the scaler to the training data
        X_train_scaled_array = scaler.fit_transform(X_train)

        if self.debug:
            print("StandardScaler fitted successfully on training data.")
            print(f"Mean (per feature): {scaler.mean_}")
            print(f"Std Dev (per feature): {scaler.scale_}")

        # 3. Transform Training Data
        # Convert the scaled array back to a DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=feature_cols, index=X_train.index)

        # 4. Transform Testing Data using the FITTED Scaler
        X_test_scaled_array = scaler.transform(X_test)

        # Convert the scaled array back to a DataFrame
        X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=feature_cols, index=X_test.index)

        # 5. Re-integrate labels and update class attributes (optional but good practice)
        self.feature_df_train_scaled = X_train_scaled.copy()
        self.feature_df_train_scaled[Keywords.classification_column_name] = y_train.values

        self.feature_df_test_scaled = X_test_scaled.copy()
        self.feature_df_test_scaled[Keywords.classification_column_name] = y_test.values

        if self.debug:
            print("\nFeature Normalization Complete.")
            print(f"Scaled Training Features Shape: {self.feature_df_train_scaled.shape}")
            display(self.feature_df_train_scaled.head())

        # You can now use self.feature_df_train_scaled and self.feature_df_test_scaled for modeling.

        return self.feature_df_train_scaled, self.feature_df_test_scaled

    def data_augment(self, n_aug=3, max_angle_deg=10, jitter_std=0.01, scale_range=(0.9, 1.1)):
        """
        Performs Time-Domain Data Augmentation ONLY on the training set (self.train_windows).
        The augmented data REPLACES the original training windows for subsequent feature extraction.

        Args:
            n_aug (int): The number of augmented copies to create (Total training size will be N * (n_aug + 1)).
            max_angle_deg (int): Maximum rotation angle in degrees.
            jitter_std (float): Standard deviation for Gaussian noise jitter.
            scale_range (tuple): Amplitude scaling range (min, max).
        """

        if self.train_windows is None or self.train_windows_labels is None:
            print("Error: Training windows are not available. Run create_windows_for_set first.")
            return

        if self.debug:
            print("--- Starting Data Augmentation (Time Domain) ---")
            print(f"Original Training Windows Shape: {self.train_windows.shape}")

        # 1. Instantiate the Augmenter
        # Note: TimeDomainAug methods use fixed parameters in their current implementation,
        # but we can pass specific parameters to augment_set if needed.
        augmenter = TimeDomainAug()

        # 2. Augment the training data
        # We perform jitter, rotation, and amplitude scaling as defined in TimeDomainAug.augment_set
        X_train_aug, y_train_aug = augmenter.augment_set(
            self.train_windows,
            self.train_windows_labels,
            n_aug=n_aug
        )

        # 3. Replace the original training data with the augmented data
        # (This augmented data now includes the original windows + n_aug copies)
        self.train_windows = X_train_aug
        self.train_windows_labels = y_train_aug

        # Crucially, clear existing feature data to force re-extraction on the NEW, augmented data.
        self.feature_df_train = None
        self.feature_df_train_scaled = None

        if self.debug:
            print("--- Data Augmentation Complete ---")
            print(f"NEW Total Training Windows Shape: {self.train_windows.shape}")

    def dump_data(self, file_path="./", train_file_name="A4_features.csv", test_file_name="A4_validation.csv"):
        """
        Exports the final scaled training and testing feature DataFrames to CSV files.

        Args:
            file_path (str): Directory path where the files should be saved.
            train_file_name (str): Filename for the scaled training features.
            test_file_name (str): Filename for the scaled testing features.
        """

        # 1. Check if data is available
        if self.feature_df_train_scaled is None or self.feature_df_test_scaled is None:
            print(
                "❌ Error: Scaled feature dataframes (self.feature_df_train_scaled and self.feature_df_test_scaled) are missing.")
            print("Please run feature_extract() and feature_normalize() first.")
            return

        # 2. Ensure the output directory exists
        os.makedirs(file_path, exist_ok=True)

        # 3. Define full file paths
        train_full_path = os.path.join(file_path, train_file_name)
        test_full_path = os.path.join(file_path, test_file_name)

        # 4. Export Training Set
        try:
            # index=False ensures the DataFrame's index is not saved as a column
            self.feature_df_train_scaled.to_csv(train_full_path, index=False)
            print(f"✅ Successfully exported SCALED training features to: {train_full_path}")
        except Exception as e:
            print(f"❌ Error exporting training data: {e}")

        # 5. Export Testing Set
        try:
            self.feature_df_test_scaled.to_csv(test_full_path, index=False)
            print(f"✅ Successfully exported SCALED testing features to: {test_full_path}")
        except Exception as e:
            print(f"❌ Error exporting testing data: {e}")


def process_single_shot_type(
        input_data_path: Union[list, str],
        shot_type: str,
        output_dir: str,
        train_file_name: str,
        test_file_name: str,
        sampling_freq: int = 10,
        window_duration: float = 2.0,
        overlap_percent: float = 0.5,
        n_augmentations: int = 3,
        e2e_pipeline = False,
        debug: bool = True
):
    """
    Executes the full data preparation, feature engineering, augmentation,
    and dumping pipeline for a single activity/shot type file.

    Args:
        input_data_path (str): Path to the raw CSV file.
        shot_type (str): The classification label for this data.
        output_dir (str): Directory to save the final feature CSVs.
        train_file_name (str): Filename for the scaled training features.
        test_file_name (str): Filename for the scaled testing features.
        ... (optional parameters for pipeline configuration)
    """
    print(f"\n=======================================================")
    print(f"| STARTING PIPELINE for Shot Type: {shot_type} |")
    print(f"=======================================================")

    # 1. Initialize and Register Columns
    print("--- 1. Data Initialization & Setup ---")
    # data_prep = DataPrep(
    #     data_path=input_data_path,
    #     data_name=shot_type,
    #     debug=debug
    # )
    if not isinstance(input_data_path, list):
        input_data_path = [input_data_path]

    print(f"Loading {len(input_data_path)} raw files for shot type: {shot_type}")

    all_raw_dfs = []
    for path in input_data_path:
        try:
            df = pd.read_csv(path, encoding='latin1')
            all_raw_dfs.append(df)
            print(f"Loaded: {path}")
        except FileNotFoundError:
            print(f"WARNING: File not found at {path}. Skipping.")

    # Concatenate all raw dataframes for this shot type
    if not all_raw_dfs:
        raise ValueError(f"No valid data files loaded for shot type: {shot_type}")

    combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)



    data_prep = DataPrep(
        data_path="Combined Data",  # Placeholder for tracking
        data_name=shot_type,
        debug=debug,
        raw_df=combined_raw_df  # Pass the combined DataFrame
    )

    data_prep.data_columns_reg(
        accel_cols=['Accel_X (g)', 'Accel_Y (g)', 'Accel_Z (g)'],
        gyro_cols=['Gyro_X (°/s)', 'Gyro_Y (°/s)', 'Gyro_Z (°/s)'],
        mag_cols=['Mag_X (G)', 'Mag_Y (G)', 'Mag_Z (G)'],
    )

    # 2. Preprocessing Steps
    print("\n--- 2. Preprocessing ---")
    data_prep.timestamp()

    # NOTE: Using the corrected missing values check from previous steps
    if (data_prep.data_df.isnull().sum() > 0).any():
        print("🚨 Warning: Missing values found and handled by imputation/dropping.")
        # If your data is clean, the check below isn't needed, but it's safer to include:
        # data_prep.data_df.dropna(inplace=True)

    data_prep.set_index_sort()
    data_prep.resample_forward_fill(period='100ms', drop_init_na=True)
    data_prep.data_calibration()
    data_prep.noise_reduce(window_length=5, polyorder=2)

    # 3. Splitting and Windowing
    print("\n--- 3. Splitting & Windowing ---")
    data_prep.train_test_split(test_size=0.2, random_state=42)

    window_size, overlap_size = data_prep.window_size_and_overlap(
        sampling_frequency_hz=sampling_freq,
        window_duration_seconds=window_duration,
        overlap_percentage=overlap_percent
    )

    data_prep.create_windows_for_set(Keywords.training, window_size, overlap_size)
    data_prep.create_windows_for_set(Keywords.validating, window_size, overlap_size)

    if e2e_pipeline:
        print(f"| Pipeline for {shot_type} finished. "
              f"Data saved to Matrices"
              f"'{data_prep.train_windows.shape} {data_prep.train_windows_labels.shape}' "
              f"'{data_prep.test_windows.shape} {data_prep.test_windows_labels.shape}' "
              f". |")
        print("=======================================================")

        return data_prep.train_windows, data_prep.train_windows_labels, data_prep.test_windows, data_prep.test_windows_labels

    # 4. Data Augmentation (Training Set Only)
    print("\n--- 4. Data Augmentation ---")
    data_prep.data_augment(n_aug=n_augmentations)

    # 5. Feature Extraction (on Augmented Data)
    print("\n--- 5. Feature Extraction ---")
    data_prep.feature_extract(Keywords.training)
    data_prep.feature_extract(Keywords.validating)

    # 6. Feature Normalization (Fit Train, Transform All)
    print("\n--- 6. Feature Normalization ---")
    data_prep.feature_normalize()

    # 7. Dump Final CSV
    print("\n--- 7. Dumping Final CSV ---")
    data_prep.dump_data(
        file_path=output_dir,
        train_file_name=train_file_name,
        test_file_name=test_file_name
    )
    print(f"| Pipeline for {shot_type} finished. Files saved to '{output_dir}'. |")
    print("=======================================================")

    # Return the processed DataFrames for potential concatenation later
    return data_prep.feature_df_train_scaled, data_prep.feature_df_test_scaled




# --- GLOBAL CONFIGURATION ---
OUTPUT_DIR = "./combined_final_features"


SHOT_CONFIGS = [
    # --- CLEAR Shot Type (Combines two source files) ---
    ('clear',
     [
         './new_10_kyle/clear_10_kyle.csv',
         # './badminton_shots/clear_x30.csv'
     ],
     'A4_features_clear.csv',
     'A4_validation_clear.csv'),

    # --- DRIVE Shot Type (Combines two source files) ---
    ('drive',
     [
         './new_10_kyle/drive_10_kyle.csv',
         # './badminton_shots/drive_x30.csv'
     ],
     'A4_features_drive.csv',
     'A4_validation_drive.csv'),

    # --- LIFT Shot Type (Combines two source files) ---
    ('lift',
     [
         './new_10_kyle/lift_10_kyle.csv',
         # './badminton_shots/lift_x30.csv'
     ],
     'A4_features_lift.csv',
     'A4_validation_lift.csv'),

    # --- SMASH Shot Type (Combines two source files) ---
    ('smash',
     [
         './new_10_kyle/smash_10_kyle.csv',
         # './badminton_shots/smash_x30.csv'
     ],
     'A4_features_smash.csv',
     'A4_validation_smash.csv'),

    ('clear',
     [
         # './new_10_kyle/clear_10_kyle.csv',
         './badminton_shots/clear_x30.csv'
     ],
     'A3_features_clear.csv',
     'A3_validation_clear.csv'),

    # --- DRIVE Shot Type (Combines two source files) ---
    ('drive',
     [
         # './new_10_kyle/drive_10_kyle.csv',
         './badminton_shots/drive_x30.csv'
     ],
     'A3_features_drive.csv',
     'A3_validation_drive.csv'),

    # --- LIFT Shot Type (Combines two source files) ---
    ('lift',
     [
         # './new_10_kyle/lift_10_kyle.csv',
         './badminton_shots/lift_x30.csv'
     ],
     'A3_features_lift.csv',
     'A3_validation_lift.csv'),

    # --- SMASH Shot Type (Combines two source files) ---
    ('smash',
     [
         # './new_10_kyle/smash_10_kyle.csv',
         './badminton_shots/smash_x30.csv'
     ],
     'A3_features_smash.csv',
     'A3_validation_smash.csv')
]


def run_data_processing_mlp():

    all_train_dfs = []
    all_test_dfs = []
    for shot_type, input_paths, train_name, test_name in SHOT_CONFIGS:
        print(f"Processing configurations for: {shot_type}")

        # The input_data_path parameter in the function now correctly receives the list of paths
        train_df, test_df = process_single_shot_type(
            input_data_path=input_paths, # Pass the list of paths
            shot_type=shot_type,
            output_dir=OUTPUT_DIR,
            train_file_name=train_name,
            test_file_name=test_name
        )
        all_train_dfs.append(train_df)
        all_test_dfs.append(test_df)

    # --- COMBINE ALL DATASETS ---
    final_train_df = pd.concat(all_train_dfs, ignore_index=True)
    final_test_df = pd.concat(all_test_dfs, ignore_index=True)

    # --- DUMP COMBINED DATA (OPTIONAL) ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_train_df.to_csv(os.path.join(OUTPUT_DIR, "final_combined_training.csv"), index=False)
    final_test_df.to_csv(os.path.join(OUTPUT_DIR, "final_combined_validation.csv"), index=False)

    print("\n--- FINAL COMBINATION COMPLETE ---")
    print(f"Total Combined Training Features Shape: {final_train_df.shape}")
    print(f"Total Combined Validation Features Shape: {final_test_df.shape}")


def run_data_processing_transformer():
    all_train_windows = []
    all_train_labels = []
    all_test_windows = []
    all_test_labels = []

    for shot_type, input_paths, train_name, test_name in SHOT_CONFIGS:
        print(f"Processing: {shot_type}")

        train_w, train_l, test_w, test_l = process_single_shot_type(
            input_data_path=input_paths,
            shot_type=shot_type,
            output_dir=OUTPUT_DIR,
            train_file_name=train_name,
            test_file_name=test_name,
            e2e_pipeline=True     # ⭐重要：走序列数据模式！
        )

        all_train_windows.append(train_w)
        all_train_labels.append(train_l)
        all_test_windows.append(test_w)
        all_test_labels.append(test_l)

    # ⭐ 合并（按第 0 维拼接）
    final_train_windows = np.concatenate(all_train_windows, axis=0)
    final_train_labels = np.concatenate(all_train_labels, axis=0)
    final_test_windows = np.concatenate(all_test_windows, axis=0)
    final_test_labels = np.concatenate(all_test_labels, axis=0)

    print("Final train windows shape:", final_train_windows.shape)
    print("Final test windows shape:", final_test_windows.shape)

    return final_train_windows, final_train_labels, final_test_windows, final_test_labels







