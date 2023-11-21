from lorentz_fdtd_1d.fdtd_common import *
import numpy as np

class TFSFWaveform:
    def value(self, t):
        pass

class SinWaveform(TFSFWaveform):
    def __init__(self, frequency: float, amplitude: float = 1) -> None:
        self.frequency = frequency
        self.amplitude = amplitude

    def value(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t)
    
class GaussianWaveform(TFSFWaveform):
    def __init__(self, tau, t_0):
        self.tau = tau
        self.t_0 = t_0
    
    def value(self, t):
        return np.exp(-(t-self.t_0)**2/self.tau**2)

class TFSF1D:
    def __init__(self, waveform:TFSFWaveform,  inject_index, color:str='green') -> None:
        self.waveform = waveform
        self.k = inject_index
        self.color = color

    def relocate(self, aux_size, dt, dl):
        self.e_i = np.zeros(aux_size+3)
        self.h_i = np.zeros(aux_size+2)
        self.dt = dt
        self.dl = dl
        self.ceihi = -(dt / (EPSILON_0 * dl))
        self.chiei = -(dt / (MU_0 * dl))
        # boundary
        self.c1 = (C_CONSTANT * self.dt - self.dl) / \
            (C_CONSTANT * self.dt + self.dl)
        self.c2 = 2 * self.dl / (C_CONSTANT * self.dt + self.dl)
        self.x = 0
        self.y = 0
        self.a = 0
        self.b = 0

    def update(self, nt):
        self.e_i[0] = self.waveform.value((nt) * self.dt)
        self.x = self.e_i[-2]
        self.y = self.e_i[-1]

        for i in range(1, len(self.e_i)-1):
            self.e_i[i] = self.e_i[i] + self.ceihi * \
                (self.h_i[i] - self.h_i[i-1])

        self.e_i[-1] = - self.a + self.c1 * \
            (self.e_i[-2] + self.b) + self.c2 * (self.x + self.y)
        self.a = self.x
        self.b = self.y

        for i in range(len(self.h_i)):
            self.h_i[i] = self.h_i[i] + self.chiei * \
                (self.e_i[i+1] - self.e_i[i])

    def correctE(self, e):
        coff = self.dt / (self.dl * EPSILON_0)
        e[self.k] += coff * self.h_i[1]

    def correctH(self, h):
        coff = self.dt / (self.dl * MU_0)
        h[self.k-1] += coff * self.e_i[2]
