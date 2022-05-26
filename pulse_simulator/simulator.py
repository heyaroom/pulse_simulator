import numpy as np
import scipy.linalg as lin

class Simulator:
    def __init__(self):
        pass
        
    def set_system(self, system, frame_frequency=None):
        system.compile(frame_frequency)
        self.dim = system.dim
        self.static_hamiltonian = system.static_hamiltonian
        self.operators = system.dynamic_operators
        self.detunings = system.dynamic_detunings
        self.frame = system.frame_difference
        
    def set_sequence(self, sequence, step=0.1, visualize=False):
        for port in sequence.port_list:
            port.if_freq = self.detunings[port.name]
            port.DAC_STEP = step
        sequence.compile()
        
        if visualize:
            sequence.draw(baseband=True)
        
        self.waveforms = {}
        for port in sequence.port_list:
            self.waveforms[port.name] = (port.waveform.real, port.waveform.imag)
        self.time = port.time
        self.trigger_position_list = sequence.trigger_position_list

    def run(self, frame_inverse=True, return_all=True):

        def ith_hamiltonian(i):
            tmp = 0j + self.static_hamiltonian
            for key in self.operators.keys():
                waveforms = self.waveforms[key]
                operators = self.operators[key]
                for waveform, operator in zip(waveforms, operators):
                    tmp += waveform[i]*operator
            return tmp

        def precompile(time, ith_hamiltonian):
            h0 = ith_hamiltonian(0)
            t_list = [time[0]]
            s_list = []
            h_list = []
            for i in range(1,time.size):
                h1 = ith_hamiltonian(i)
                if (return_all) or (not np.allclose(h0,h1)) or (i==time.size-1):
                    t_list.append(time[i])
                    s_list.append(t_list[-1] - t_list[-2])
                    h_list.append(0.5*(h0+h1))
                    h0 = h1
            return t_list, s_list, h_list

        def time_evolution(s_list, h_list, frame):
            u = np.identity(self.dim)
            f = np.identity(self.dim)

            unitary = [u]
            for h,s in zip(h_list, s_list):
                u = lin.expm(-1j*h*s)@u
                if return_all and frame_inverse:
                    f = lin.expm(+1j*frame*s)@f
                    unitary.append(f@u)

            if return_all:
                return unitary
            else:
                t = sum(s_list)
                f = lin.expm(+1j*frame*t)
                return f@u

        t_list, s_list, h_list = precompile(2*np.pi*self.time, ith_hamiltonian)

        self.unitary = time_evolution(s_list, h_list, self.frame)