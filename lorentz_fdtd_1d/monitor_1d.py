import numpy as np
from matplotlib import pyplot as plt

from lorentz_fdtd_1d.fdtd_common import C_CONSTANT, MU_0


class Monitor1D:
    def __init__(self, start: float, end: float, name: str, color: str = 'orange') -> None:
        self.start = start
        self.end = end
        self.start_index = 0
        self.end_index = 0
        self.name = name
        self.color = color

    def relocation(self, start_index, end_index, nt, dt, dz, node_e, node_h):
        self.start_index = start_index
        self.end_index = end_index
        self.nt = nt
        self.e_value = np.zeros(shape=(nt, end_index-start_index+1))
        self.h_value = np.zeros(shape=(nt, end_index-start_index+1))
        self.dt = dt
        self.dz = dz
        self.node_e = node_e
        self.node_h = node_h

    def record(self, e, h, t):
        self.e_value[t, :] = e[self.start_index:self.end_index+1]
        self.h_value[t, :] = h[self.start_index:self.end_index+1]

    def plot(self, time_unit='us', frequency_unit='GHz', frequency_range=(0, 10)):
        self.figure_handle = plt.figure()
        self.axes = self.figure_handle.subplots(2, 2)
        # set title
        self.figure_handle.suptitle(f'Monitor: {self.name}')

        e_fft = np.fft.fft(self.e_value, axis=0)
        h_fft = np.fft.fft(self.h_value, axis=0)
        freq = self._format_frequency(
            np.fft.fftfreq(self.nt, d=self.dt), frequency_unit)
        e_fft = e_fft[freq >= frequency_range[0]]
        h_fft = h_fft[freq >= frequency_range[0]]
        freq = freq[freq >= frequency_range[0]]
        e_fft = e_fft[freq <= frequency_range[1]]
        h_fft = h_fft[freq <= frequency_range[1]]
        freq = freq[freq <= frequency_range[1]]

        # plot e
        e_time = self._format_time(self._get_e_time(), time_unit)
        self.axes[0][0].plot(e_time, self.e_value)
        self.axes[0][0].set_xlabel('Time ({})'.format(time_unit))
        self.axes[0][0].set_ylabel('E (V/m)')
        self.axes[0][0].grid()
        self.axes[0][0].set_xlim(e_time[0], e_time[-1])
        self.axes[0][0].set_ylim(np.min(self.e_value), np.max(self.e_value))
        self.axes[0][0].set_title('E')
        self.axes[0][1].plot(freq, np.abs(e_fft))
        self.axes[0][1].set_xlabel('Frequency ({})'.format(frequency_unit))
        self.axes[0][1].grid()
        self.axes[0][1].set_xlim(frequency_range[0], frequency_range[1])
        # mark max amplitude
        max_index = np.argmax(np.abs(e_fft))
        max_freq = freq[max_index]
        max_amp = np.abs(e_fft)[max_index].max()
        self.axes[0][1].annotate(f'{max_amp:.2f} with {max_freq:.4f} {frequency_unit}', xy=(
            max_freq, max_amp), xytext=(max_freq, max_amp))
        print(
            f'monitor {self.name} E field max amplitude: {max_amp:.2f} at {max_freq:.4f} {frequency_unit}')

        # plot h
        h_time = self._format_time(self._get_h_time(), time_unit)
        self.axes[1][0].plot(h_time, self.h_value)
        self.axes[1][0].set_xlabel('Time ({})'.format(time_unit))
        self.axes[1][0].set_ylabel('H (A/m)')
        self.axes[1][0].grid()
        self.axes[1][0].set_xlim(h_time[0], h_time[-1])
        self.axes[1][0].set_ylim(np.min(self.h_value), np.max(self.h_value))
        self.axes[1][0].set_title('H')
        self.axes[1][1].plot(freq, np.abs(h_fft))
        self.axes[1][1].set_xlabel('Frequency ({})'.format(frequency_unit))
        self.axes[1][1].grid()
        self.axes[1][1].set_xlim(frequency_range[0], frequency_range[1])
        # mark max amplitude
        max_index = np.argmax(np.abs(h_fft))
        max_freq = freq[max_index]
        max_amp = np.abs(h_fft)[max_index].max()
        self.axes[1][1].annotate(f'{max_amp:.2f} with {max_freq:.4f} {frequency_unit}', xy=(
            max_freq, max_amp), xytext=(max_freq, max_amp))
        print(
            f'monitor {self.name} H field max amplitude: {max_amp:.2f} at {max_freq:.4f} {frequency_unit}')
        plt.show()

    def _format_time(self, time, time_unit):
        if time_unit == 'us':
            return time * 1e6
        elif time_unit == 'ns':
            return time * 1e9
        elif time_unit == 'ps':
            return time * 1e12
        else:
            return time

    def _format_frequency(self, freq, frequency_unit):
        if frequency_unit == 'GHz':
            return freq * 1e-9
        elif frequency_unit == 'MHz':
            return freq * 1e-6
        elif frequency_unit == 'KHz':
            return freq * 1e-3
        else:
            return freq

    def _get_e_time(self):
        return np.arange(self.nt) * self.dt

    def _get_h_time(self):
        return (np.arange(self.nt) + 0.5) * self.dt


class LorentzFDTDMonitor(Monitor1D):
    def __init__(self, relative_velocity, start: float, end: float, name: str, color: str = 'orange') -> None:
        super().__init__(start, end, name, color)
        self.relative_velocity = relative_velocity
        self.beta = self.relative_velocity / C_CONSTANT
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        if (self.start != self.end):
            raise Exception(
                'LorentzFDTDMonitor start and end should be the same')
        self.static_start = self.start
        self.static_end = self.end
        self.dynamic_start_e = self.start * self.gamma
        self.dynamic_end_e = self.end * self.gamma
        self.dynamic_start_h = self.start * self.gamma
        self.dynamic_end_h = self.end * self.gamma

        self.t_0 = - self.gamma * self.beta * self.start / C_CONSTANT

    def record(self, e, h, t):
        self.dynamic_start_e = self.static_start - self.relative_velocity * t * self.dt
        self.dynamic_start_h = self.static_start - \
            self.relative_velocity * (t + 0.5) * self.dt

        if (self.dynamic_start_e < self.node_e[0]):
            raise Exception(f'Monitor {self.name} out of boundary')
        if (self.dynamic_end_e > self.node_e[-1]):
            raise Exception(f'Monitor {self.name} out of boundary')
        if (self.dynamic_start_h < self.node_h[0]):
            raise Exception(f'Monitor {self.name} out of boundary')
        if (self.dynamic_end_h > self.node_h[-1]):
            raise Exception(f'Monitor {self.name} out of boundary')

        start_index_e = np.argmin(np.abs(self.node_e - self.dynamic_start_e))
        start_index_h = np.argmin(np.abs(self.node_h - self.dynamic_start_h))
        self.start = self.dynamic_start_e

        if (self.node_e[start_index_e] <= self.dynamic_start_e):
            a = (self.dynamic_start_e - self.node_e[start_index_e]) / self.dz
            b = 1 - a
            self.e_value[t, :] = a * e[start_index_e+1] + b * e[start_index_e]
        else:
            a = (self.node_e[start_index_e] - self.dynamic_start_e) / self.dz
            b = 1 - a
            self.e_value[t, :] = a * \
                e[start_index_e - 1] + b * e[start_index_e]

        if (self.node_h[start_index_h] <= self.dynamic_start_h):
            a = (self.dynamic_start_h - self.node_h[start_index_h]) / self.dz
            b = 1 - a
            self.h_value[t, :] = a * h[start_index_h +
                                       1] + b * h[start_index_h]
        else:
            a = (self.node_h[start_index_h] - self.dynamic_start_h) / self.dz
            b = 1 - a
            self.h_value[t, :] = a * h[start_index_h -
                                       1] + b * h[start_index_h]

    def _get_e_time(self):
        t = np.arange(self.nt) * self.dt_0 + self.t_0
        return ((1 - self.beta ** 2) * self.gamma * t + self.beta * self.gamma * self.static_start / C_CONSTANT)

    def _get_h_time(self):
        t = (np.arange(self.nt) + 0.5) * self.dt_0 + self.t_0
        return ((1 - self.beta ** 2) * self.gamma * t + self.beta * self.gamma * self.static_start / C_CONSTANT)

    def plot(self, time_unit='us', frequency_unit='GHz', frequency_range=(0, 10)):
        e_value = self.e_value
        h_value = self.h_value
        self.dt_0 = self.dt
        self.dt =  self.dt_0 / self.gamma
        self.e_value = self.gamma * \
            (e_value + self.relative_velocity * h_value * MU_0)
        self.h_value = self.gamma * \
            (h_value + self.beta * e_value / (C_CONSTANT * MU_0))
        return super().plot(time_unit, frequency_unit, frequency_range)
