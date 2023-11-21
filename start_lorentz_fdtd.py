import numpy as np
from lorentz_fdtd_1d.fdtd_common import *
from lorentz_fdtd_1d.fdtd_1d import LorentzFDTD1D
from lorentz_fdtd_1d.monitor_1d import LorentzFDTDMonitor
from lorentz_fdtd_1d.object_1d import Object1D
from lorentz_fdtd_1d.tfsf_1d import TFSF1D, SinWaveform


def single_slab():
    frequency = 2e9
    slab_velocity = 0.01 * C_CONSTANT
    beta = slab_velocity / C_CONSTANT
    f = frequency * (np.sqrt(1 - beta) / np.sqrt(1+beta))
    object_list = []
    monitor_list = []
    dz = C_CONSTANT / (f * 20)

    object_list.append(Object1D(start=0*dz, end=160*dz, eps_r=1,
                       mu_r=1, sigma_e=0, sigma_m=0, name='vacuum', color='blue'))
    object_list.append(Object1D(start=100*dz, end=150*dz, eps_r=4,
                                mu_r=1, sigma_e=0, sigma_m=0, name='slab', color='red'))
    monitor_list.append(LorentzFDTDMonitor(relative_velocity=slab_velocity, start=97*dz, end=97*dz,
                        name='reflect', color='orange'))
    monitor_list.append(LorentzFDTDMonitor(relative_velocity=slab_velocity,
                        start=145*dz, end=145*dz, name='transmit', color='purple'))
    tfsf = TFSF1D(waveform=SinWaveform(frequency=frequency),
                  inject_index=98, color='green')

    fdtd = LorentzFDTD1D(
        dz, slab_velocity, tfsf=tfsf, objects=object_list, monitors=monitor_list)
    fdtd.run(total_time_step=2000, is_plot=True, plot_step=5)
    fdtd.monitor_list[0].plot(frequency_range=(1.9, 2.0))
    fdtd.monitor_list[1].plot(frequency_range=(2.0, 2.1))


def multilayer_slab():
    frequency = 2e9
    slab_velocity = 0.01 * C_CONSTANT
    beta = slab_velocity / C_CONSTANT
    f = frequency * (np.sqrt(1 - beta) / np.sqrt(1+beta))
    object_list = []
    monitor_list = []
    dz = C_CONSTANT / (f * 20)
    
    object_list.append(Object1D(0, 250*dz, 1, 1, 0, 0, 'vacuum', 'blue'))
    slab_0 = Object1D(130*dz, 160*dz, 4, 1, 0, 0, 'slab_0', 'red')
    slab_1 = Object1D(160*dz, 190*dz, 10, 1, 0, 0, 'slab_1', 'brown')
    slab_2 = Object1D(190*dz, 220*dz, 1, 1, 1e8, 0, 'slab_2', 'black')
    object_list.append(slab_0)
    object_list.append(slab_1)
    object_list.append(slab_2)
    
    monitor_list.append(LorentzFDTDMonitor(relative_velocity=slab_velocity, start=125*dz, end=125*dz,
                        name='reflect', color='orange'))
    
    tfsf= TFSF1D(waveform=SinWaveform(frequency=frequency), inject_index=128)
    
    fdtd = LorentzFDTD1D(
        dz, slab_velocity, tfsf=tfsf, objects=object_list, monitors=monitor_list)
    fdtd.run(total_time_step=2000)
    fdtd.monitor_list[0].plot(frequency_range=(1.9, 2.0))

if __name__ == '__main__':
    single_slab()
    multilayer_slab()
