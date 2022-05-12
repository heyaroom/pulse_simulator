import numpy as np
import scipy.linalg as lin

class Simulator:
    def __init__(self, frame_frequency):
        self.frame_frequency = frame_frequency
        
    def set_system(self, system):
        system.compile(self.frame_frequency)
        self.static_hamiltonian = system.static_hamiltonian
        self.operators = system.dynamic_operators
        self.detunings = system.dynamic_detunings
        
    def set_sequence(self, sequence, step=0.1, visualize=False):
        for port in sequence.port_list:
            port.if_freq = self.detunings[port.name]
            port.DAC_STEP = step
        sequence.compile()
        
        if visualize:
            sequence.draw(baseband=False)
        
        self.waveforms = {}
        for port in sequence.port_list:
            self.waveforms[port.name] = (port.waveform.real, port.waveform.imag)
        self.time = port.time
            
    def run(self):
        def ith_hamiltonian(i):
            tmp = 0j + self.static_hamiltonian
            for key in self.operators.keys():
                waveforms = self.waveforms[key]
                operators = self.operators[key]
                for waveform, operator in zip(waveforms, operators):
                    tmp += waveform[i]*operator
            return tmp
        
        def time_evolution(time, ith_hamiltonian):
            dim = ith_hamiltonian(0).shape[0]
            step = np.gradient(time)

            unitary = []
            u = np.identity(dim)
            for i in range(time.size):
                u = lin.expm(-1j*ith_hamiltonian(i)*step[i])@u
                unitary.append(u)
            return unitary
        
        self.unitary = time_evolution(2*np.pi*self.time, ith_hamiltonian)