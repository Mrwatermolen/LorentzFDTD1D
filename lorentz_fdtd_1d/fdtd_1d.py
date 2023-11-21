from lorentz_fdtd_1d.fdtd_common import *
from lorentz_fdtd_1d.object_1d import Object1D
from lorentz_fdtd_1d.monitor_1d import Monitor1D
from lorentz_fdtd_1d.tfsf_1d import TFSF1D, GaussianWaveform, SinWaveform
import numpy as np
from matplotlib import pyplot as plt


class FDTD1D:
    def __init__(self, dz,  tfsf: TFSF1D, objects: list[Object1D], monitors: list[Monitor1D]) -> None:
        self.clf = 1
        self.dz = dz
        self.dt = self.clf * dz / C_CONSTANT
        self.tfsf = tfsf
        self.object_list = objects
        self.monitor_list = monitors

    def add_object(self, object: Object1D):
        self.object_list.append(object)

    def add_monitor(self, monitor: Monitor1D):
        self.monitor_list.append(monitor)

    def run(self, total_time_step: int, is_plot: bool = False, plot_step: int = 1):
        self.total_time_step = total_time_step
        self.current_time_step = 0
        self.is_plot = is_plot
        self.plot_step = plot_step
        self.define_problem()

        if self.is_plot:
            self.figure_handle = plt.figure(figsize=(10, 5))

        for i in range(total_time_step):
            self.current_time_step = i
            self.updateE()
            self.updateH()
            self.record()
            if self.is_plot and i % self.plot_step == 0:
                self.plot()

    def define_problem(self):
        self._init_space()
        self._init_material()
        self._init_update_coff()
        self._init_tfsf()
        self._init_monitor()

    def _init_space(self):
        self.start_z = np.min([obj.start for obj in self.object_list])
        self.end_z = np.max([obj.end for obj in self.object_list])
        self.num_grid = (int)(round((self.end_z - self.start_z) / self.dz))
        self.size_z = self.num_grid * self.dz
        self.end_z = self.start_z + self.size_z
        self.ex = np.zeros(self.num_grid+1)
        self.hy = np.zeros(self.num_grid)
        self.node_e = np.arange(self.num_grid+1) * self.dz
        self.node_h = (self.node_e[:-1] + self.node_e[1:]) / 2

    def _init_material(self):
        self.eps = np.ones(self.num_grid+1)*EPSILON_0
        self.mu = np.ones(self.num_grid)*MU_0
        self.sigma_e = np.zeros(self.num_grid+1)
        self.sigma_m = np.zeros(self.num_grid)
        for obj in self.object_list:
            start_index = (int)((obj.start - self.start_z) / self.dz)
            end_index = (int)((obj.end - self.start_z) / self.dz)
            self.eps[start_index:end_index+1] = obj.eps_r * EPSILON_0
            self.mu[start_index:end_index] = obj.mu_r * MU_0
            self.sigma_e[start_index:end_index+1] = obj.sigma_e
            self.sigma_m[start_index:end_index] = obj.sigma_m

    def _init_update_coff(self):
        self.cexe = (2 * self.eps - self.sigma_e * self.dt) / \
            (2 * self.eps + self.sigma_e * self.dt)
        self.cexhy = -(2 * self.dt) / \
            ((2 * self.eps + self.dt * self.sigma_e) * self.dz)
        self.chyh = (2 * self.mu - self.sigma_m * self.dt) / \
            (2 * self.mu + self.sigma_m * self.dt)
        self.chyex = -(2 * self.dt) / \
            ((2 * self.mu + self.dt * self.sigma_m) * self.dz)

        # auxiliary variable
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.x = 0
        self.y = 0
        self.p = 0
        self.q = 0
        self.boundary_coff_1 = (C_CONSTANT * self.dt - self.dz) / \
            (C_CONSTANT * self.dt + self.dz)
        self.boundary_coff_2 = 2 * self.dz / \
            (C_CONSTANT * self.dt + self.dz)

    def _init_tfsf(self):
        self.tfsf.relocate(self.num_grid-self.tfsf.k, self.dt, self.dz)

    def _init_monitor(self):
        for monitor in self.monitor_list:
            start_index = (int)((monitor.start - self.start_z) / self.dz)
            end_index = (int)((monitor.end - self.start_z) / self.dz)
            monitor.relocation(start_index, end_index,
                               self.total_time_step, self.dt, self.dz, self.node_e, self.node_h)

    def updateE(self):
        self.x = self.ex[-2]
        self.y = self.ex[-1]
        self.p = self.ex[1]
        self.q = self.ex[0]
        for i in range(1, len(self.ex)-1):
            self.ex[i] = self.cexe[i] * self.ex[i] + self.cexhy[i] * \
                (self.hy[i] - self.hy[i-1])
        self.tfsf.correctE(e=self.ex)
        self.ex[0] = - self.c + self.boundary_coff_1 * \
            (self.ex[1] + self.d) + self.boundary_coff_2 * (self.p + self.q)
        self.ex[-1] = - self.a + self.boundary_coff_1 * \
            (self.ex[-2] + self.b) + self.boundary_coff_2 * (self.x + self.y)
        self.a = self.x
        self.b = self.y
        self.c = self.p
        self.d = self.q

    def updateH(self):
        for i in range(len(self.hy)):
            self.hy[i] = self.chyh[i] * self.hy[i] + self.chyex[i] * \
                (self.ex[i+1] - self.ex[i])
        self.tfsf.update(self.current_time_step)
        self.tfsf.correctH(h=self.hy)

    def record(self):
        for monitor in self.monitor_list:
            monitor.record(self.ex, self.hy, self.current_time_step)

    def plot(self):
        self.figure_handle.clf()
        # plt add rectangle to represent the boundary
        plt.gca().add_patch(plt.Rectangle(
            (self.node_e[0], -1.5), self.node_e[-1]-self.node_e[0], 3, fill=None, alpha=1, edgecolor='black', linewidth=2))
        # plt add rectangle to represent the slab
        for obj in self.object_list:
            plt.gca().add_patch(plt.Rectangle(
                (obj.start, -1.5), obj.end-obj.start, 3, fill=None, alpha=1, edgecolor=obj.color, linewidth=2))
        # plt add line to represent the tfsf boundary
        plt.plot([self.node_e[self.tfsf.k], self.node_e[self.tfsf.k]], [-1.5, 1.5],
                 color=self.tfsf.color, linestyle='--', linewidth=2, label='tfsf')
        # plt add point to represent the monitor
        for i, monitor in enumerate(self.monitor_list):
            plt.plot(monitor.start, 0,
                     'o', color=monitor.color, label=f'monitor{i}')
        plt.plot(self.node_e, self.ex, label='ex')
        plt.xlim([self.node_e[0], self.node_e[-1]])
        plt.ylim([-1.5, 1.5])
        plt.grid()
        plt.title(
            f'time step: {self.current_time_step}/{self.total_time_step}')
        plt.pause(0.05)


class LorentzFDTD1D(FDTD1D):
    def __init__(self, dz, relative_velocity, tfsf: TFSF1D, objects: list[Object1D], monitors: list[Monitor1D]) -> None:
        super().__init__(dz, tfsf, objects, monitors)
        self.relative_velocity = relative_velocity
        self._lorentz_transform_init()

    def _lorentz_transform_init(self):
        self.beta = self.relative_velocity / C_CONSTANT
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        temp_coff = self.gamma * (1 - self.beta)
        waveform = self.tfsf.waveform
        if type(waveform) is SinWaveform:
            self.tfsf.waveform.frequency = self.tfsf.waveform.frequency * temp_coff
            self.tfsf.waveform.amplitude = self.tfsf.waveform.amplitude * temp_coff
        elif type(waveform) is GaussianWaveform:
            self.tfsf.waveform.tau = self.tfsf.waveform.tau * temp_coff
            self.tfsf.waveform.t_0 = self.tfsf.waveform.t_0 * temp_coff
