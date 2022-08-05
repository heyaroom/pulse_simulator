# pulse_simulator
Pulse Simulator is a library supporting time-domain simulations.


Users can use [sequence_parser](https://github.com/qipe-nlab/sequence_parser.git) to easily describe time-domain pulse sequences and perform numerical calculations for the gate operations on Transmon qubits.


Pulse Simulator will streamline simulations by increasing the reusability of pulse sequences.

## Usage

1. Import Modules
```python
import numpy as np
from sequence_parser.sequence import Sequence
from sequence_parser.port import Port
from sequence_parser.instruction import *
from pulse_simulator.system import System
from pulse_simulator.simulator import Simulator
from pulse_simulator.util.visualize import *
```

2. Example (Simulating TPCX echo sequence)
```python
# system parameter
w0, a0 = + 8.0, -0.4 # GHz
w1, a1 = + 8.8, -0.4 # GHz
g01 = + 0.01 # GHz

# dressed target frequency
w0d = w0 + g01**2/(w0-w1) + g01**2*(a0+a1)/(((w0+a0)-w1)*(w0-(w1+a1)))
w1d = w1 + g01**2/(w1-w0) + g01**2*(a0+a1)/(((w0+a0)-w1)*(w0-(w1+a1)))

# control parameter
x0_amp = + 0.06 # GHz
x0_dur = 1/x0_amp # ns
x0_beta = 0.5/a0/(2*np.pi)
cr01_amp = + 0.3 # GHz
cr01_dur = + 100 # ns
cr01_edge = + 30 # ns
ct01_amp = + 0.002 # GHz

sys = System()
sys.add_qubit(idx=0, dim=3, frequency=w0, anharmonicity=a0)
sys.add_qubit(idx=1, dim=3, frequency=w1, anharmonicity=a1)
sys.add_coupling((0,1), coupling=g01)
sys.add_drive(0, qubit=0, amplitude=cr01_amp, frequency=w1d) # corresponds to Port(0)
sys.add_drive(1, qubit=1, amplitude=ct01_amp, frequency=w1d) # corresponds to Port(1)
sys.add_drive(2, qubit=0, amplitude=x0_amp, frequency=w0d) # corresponds to Port(2)

# Two-Pulse echoed Control-X
seq = Sequence()
seq.add(FlatTop(RaisedCos(+1,cr01_edge), cr01_dur), Port(0))
seq.add(FlatTop(RaisedCos(+1,cr01_edge), cr01_dur), Port(1))
seq.trigger([Port(0), Port(1), Port(2)])
seq.add(HalfDRAG(RaisedCos(+1, x0_dur), x0_beta), Port(2))
seq.trigger([Port(0), Port(1), Port(2)])
seq.add(FlatTop(RaisedCos(-1,cr01_edge), cr01_dur), Port(0))
seq.add(FlatTop(RaisedCos(-1,cr01_edge), cr01_dur), Port(1))
seq.trigger([Port(0), Port(1), Port(2)])
seq.add(HalfDRAG(RaisedCos(+1, x0_dur), x0_beta), Port(2))

sim = Simulator()
sim.set_system(sys)
sim.set_sequence(seq, visualize=True)
sim.run(return_all=True)

time = sim.time
unitary = sim.unitary

data = hamiltonian_tomography_zx_data(sys, sim.unitary, control=0)
visualize_hamiltonian_tomography(sim.time, data, sim.trigger_position_list)
```

Visualized Pulse Sequence

![TPCX sequence](/figures/tpcx_sequence.png)

Visualized Hamiltonian Tomography (Control Qubit)

![HT control](/figures/hamiltonian_tomography_control.png)

Visualized Hamiltonian Tomography (Target Qubit)

![HT target](/figures/hamiltonian_tomography_target.png)

## Citation
No obligation. Use the following as needed.
```
@Misc{PulseSimulator,
  author = {Heya, Kentaro},
  title = {{Pulse Simulator}: A pulse simulator for quantum engineering},
  year = {2021--},
  url = "https://github.com/heyaroom/pulse_simulator",
  note = {[Online; accessed <today>]}
}
```
