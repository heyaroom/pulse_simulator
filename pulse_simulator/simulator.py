import numpy as np
import scipy.linalg as lin

class Simulator:
    """Class for computing the time-evolution with the arbitral pulse sequence"""
    
    def __init__(self):
        pass
        
    def set_system(self, system, frame_frequency=None):
        """register the quantum system to be simulated
        Args:
            system (System) : class for the target quantum system
            frame_frequency (float) : rotation frequency of the system simulating the time evolution
        """
        system.compile(frame_frequency)
        self.dim = system.dim
        self.static_hamiltonian = system.static_hamiltonian
        self.operators = system.dynamic_operators
        self.detunings = system.dynamic_detunings
        self.frame = system.frame_difference
        
    def set_sequence(self, sequence, step=0.1, visualize=False):
        """register the pulse sequence to be simulated
        Args:
            sequence (Sequence) : class for the target pulse schedule (imported from sequence_parser)
            step (float) : time step width for simulation (ns)
        """
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
        """run the simulation
        Args:
            frame_inverse (bool) : whether to correct the simulation results to the qubit frame
            return_all (float) : whether to return the simulation results during pulse execution
        """

        def ith_hamiltonian(i):
            tmp = 0j + self.static_hamiltonian
            for key in self.operators.keys():
                waveforms = self.waveforms[key]
                operators = self.operators[key]
                for waveform, operator in zip(waveforms, operators):
                    tmp += waveform[i]*operator
            return tmp
        
        def compare_waveform(i,j):
            for key in self.operators.keys():
                waveforms = self.waveforms[key]
                for waveform in waveforms:
                    if waveform[i] != waveform[j]:
                        return False
            return True
        
        def precompile(time, ith_hamiltonian):
            h0 = ith_hamiltonian(0)
            i_list = [0]
            s_list = []
            h_list = []
            for i in range(1,time.size):
                flag_wave = compare_waveform(i, i_list[-1])
                if return_all or (not flag_wave) or (i==time.size-1):
                    h1 = ith_hamiltonian(i)
                    if i - i_list[-1] >= 2:
                        i_list.append(i-1)
                        s_list.append(time[i-1] - time[i_list[-2]])
                        h_list.append(h0)
                    i_list.append(i)
                    s_list.append(time[i] - time[i-1])
                    h_list.append(0.5*(h0+h1))
                    h0 = h1
            return i_list, s_list, h_list

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