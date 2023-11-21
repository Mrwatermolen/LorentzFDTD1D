import numpy as np
from lorentz_fdtd_1d.fdtd_common import *
from lorentz_fdtd_1d.fdtd_1d import FDTD1D
from lorentz_fdtd_1d.monitor_1d import Monitor1D
from lorentz_fdtd_1d.object_1d import Object1D
from lorentz_fdtd_1d.tfsf_1d import TFSF1D, SinWaveform


def basic_fdtd():
    frequency = 2e9
    dz = C_CONSTANT / (frequency * 30)

    tfsf = TFSF1D(waveform=SinWaveform(frequency=frequency), inject_index=20, color='green')
    object_list = []
    monitor_list = []

    object_list.append(Object1D(start=0*dz, end=100*dz, eps_r=1,
                       mu_r=1, sigma_e=0, sigma_m=0, name='vacuum', color='blue'))
    object_list.append(Object1D(start=30*dz, end=50*dz, eps_r=4,
                        mu_r=1, sigma_e=0, sigma_m=0, name='slab', color='red'))
    monitor_list.append(Monitor1D(start=15*dz, end=15*dz,
                        name='monitor', color='orange'))
    
    fdtd = FDTD1D(dz=dz, tfsf=tfsf, objects=object_list, monitors=monitor_list)
    fdtd.run(total_time_step=1200, is_plot=True, plot_step=5)
    fdtd.monitor_list[0].plot()
    
if __name__ == "__main__":
    basic_fdtd()
