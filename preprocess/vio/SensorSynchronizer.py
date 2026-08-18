from dataclasses import replace

import numpy as np

from preprocess.data_types.VIOTypes import (
    IMUSample,
    RawSensorData,
    SynchronizedSensorData,
    VIOCalibration,
)


class SensorSynchronizer:
    """将相机和 IMU 数据统一到 Android 单调时钟域。"""

    def synchronize(
        self,
        sensor_data: RawSensorData,
        calibration: VIOCalibration,
    ) -> SynchronizedSensorData:
        camera, clock_sync = self._synchronize_camera_clock(
            sensor_data,
            calibration.timeshift_cam_imu_s,
        )
        imu, imu_update_rate_hz = self._merge_imu_streams(sensor_data)
        if imu[0].timestamp_ns > camera[0].timestamp_ns:
            raise ValueError("IMU data starts after the first aligned camera frame")
        if imu[-1].timestamp_ns < camera[-1].timestamp_ns:
            raise ValueError("IMU data ends before the last aligned camera frame")

        return SynchronizedSensorData(
            camera=camera,
            imu=imu,
            imu_update_rate_hz=imu_update_rate_hz,
            clock_sync=clock_sync,
            raw_gyroscope_samples=len(sensor_data.gyroscope),
            raw_accelerometer_samples=len(sensor_data.accelerometer),
        )

    @staticmethod
    def _synchronize_camera_clock(
        sensor_data: RawSensorData,
        timeshift_cam_imu_s: float,
    ) -> tuple[tuple, dict]:
        rokid_timestamps = np.asarray(
            [sample.rokid_timestamp_ns for sample in sensor_data.camera],
            dtype=np.int64,
        )
        device_timestamps = np.asarray(
            [sample.device_monotonic_ns for sample in sensor_data.camera],
            dtype=np.int64,
        )
        source_delta = (rokid_timestamps - rokid_timestamps[0]).astype(np.float64)
        target_delta = (device_timestamps - device_timestamps[0]).astype(np.float64)
        centered_source = source_delta - source_delta.mean()
        denominator = float(centered_source @ centered_source)
        if denominator <= 0:
            raise ValueError("Unable to estimate camera clock mapping")

        scale = float(centered_source @ (target_delta - target_delta.mean()))
        scale /= denominator
        relative_offset_ns = float(np.mean(target_delta - scale * source_delta))
        affine_timestamps = np.rint(
            device_timestamps[0] + relative_offset_ns + scale * source_delta
        ).astype(np.int64)
        timeshift_ns = int(round(timeshift_cam_imu_s * 1_000_000_000.0))
        aligned_timestamps = affine_timestamps + timeshift_ns
        SensorSynchronizer._validate_increasing(
            aligned_timestamps,
            "aligned camera timestamps",
        )

        residuals = device_timestamps - affine_timestamps
        camera = tuple(
            replace(sample, timestamp_ns=int(timestamp_ns))
            for sample, timestamp_ns in zip(
                sensor_data.camera,
                aligned_timestamps,
            )
        )
        clock_sync = {
            "model": "affine_with_calibrated_timeshift",
            "scale": scale,
            "drift_ppm": (scale - 1.0) * 1_000_000.0,
            "relative_offset_ns": relative_offset_ns,
            "timeshift_cam_imu_s": timeshift_cam_imu_s,
            "timeshift_cam_imu_ns": timeshift_ns,
            "residual_rms_ns": float(
                np.sqrt(np.mean(residuals.astype(np.float64) ** 2))
            ),
            "residual_max_abs_ns": int(np.max(np.abs(residuals))),
        }
        return camera, clock_sync

    @staticmethod
    def _merge_imu_streams(
        sensor_data: RawSensorData,
    ) -> tuple[tuple[IMUSample, ...], float]:
        gyro_timestamps = np.asarray(
            [sample.timestamp_ns for sample in sensor_data.gyroscope],
            dtype=np.int64,
        )
        accel_timestamps = np.asarray(
            [sample.timestamp_ns for sample in sensor_data.accelerometer],
            dtype=np.int64,
        )
        gyro_values = np.stack([sample.values for sample in sensor_data.gyroscope])
        accel_values = np.stack(
            [sample.values for sample in sensor_data.accelerometer]
        )

        valid = (gyro_timestamps >= accel_timestamps[0]) & (
            gyro_timestamps <= accel_timestamps[-1]
        )
        gyro_timestamps = gyro_timestamps[valid]
        gyro_values = gyro_values[valid]
        if len(gyro_timestamps) < 2:
            raise ValueError("Gyroscope and accelerometer time ranges do not overlap")

        interpolated_accel = np.column_stack(
            [
                np.interp(
                    gyro_timestamps,
                    accel_timestamps,
                    accel_values[:, axis],
                )
                for axis in range(3)
            ]
        )
        imu = tuple(
            IMUSample(
                timestamp_ns=int(timestamp_ns),
                gyroscope=gyro.copy(),
                accelerometer=accel.copy(),
            )
            for timestamp_ns, gyro, accel in zip(
                gyro_timestamps,
                gyro_values,
                interpolated_accel,
            )
        )
        median_interval_ns = float(np.median(np.diff(gyro_timestamps)))
        if median_interval_ns <= 0:
            raise ValueError("Unable to determine IMU update rate")
        return imu, 1_000_000_000.0 / median_interval_ns

    @staticmethod
    def _validate_increasing(values: np.ndarray, label: str) -> None:
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"{label} must be strictly increasing")
