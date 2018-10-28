import numpy as np


class AnomalyDetector:

    def __init__(self, meter_readings, meter_history_readings):
        self.meter_readings = meter_readings
        self.meter_history_readings = meter_history_readings

    def moving_average(signal, window=1):
        window = np.ones(int(window)) / float(window)
        return np.convolve(signal, window, "same")

    def find_anomaly(self, readings=None, E1=2, E2=1):
        anomaly_indices = []

        if readings is None:
            return None
        elif type(readings) is list:
            readings = np.asarray(readings)

        averaged = self.moving_average(signal=readings, window=1)
        residuals = np.absolute(readings - averaged)

        std = np.std(residuals)

        right_lim = averaged + (E1 * std)
        left_lim = averaged - (E1 * std)

        filtered_reads = np.where((readings > right_lim) | (readings < left_lim))

        index_value = list(enumerate(filtered_reads))
        anomaly_indices.extend(index_value)

        # for index, x, averaged in zip(count(), readings, averaged):
        #     if x>averaged+(E1*std) or x<averaged-(E1*std):
        #         anomaly_indices.append((index,x))

        return (len(anomaly_indices) >= E2), anomaly_indices

    def detect_from_meters(self, E1=1.5, E2=1):

        _, anomaly_indices = self.find_anomaly(readings=self.meter_readings, E1=E1, E2=E2)

        meter_indices = [index for index, _ in anomaly_indices]
        return meter_indices

    def detect_from_meter_history(self, meter_index, E1, E2):
        readings = self.meter_history_readings

        if meter_index < 0 or meter_index >= len(readings):
            return None

        _, anomaly_indices = self.find_anomaly(readings=readings[meter_index], E1=E1, E2=E2)

        time_indices = [index for index, _ in anomaly_indices]
        return time_indices
