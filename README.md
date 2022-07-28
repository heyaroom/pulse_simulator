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

2. Most simple example
```python
w0 = + 8.0 # GHz
a0 = - 0.4 # GHz
o0 = + 0.06 # GHz
dur = 1/o0 # ns

sys = System()
sys.add_qubit(idx=0, dim=3, frequency=w0, anharmonicity=a0)
sys.add_drive(idx=0, qubit=0, amplitude=o0, frequency=w0)

seq = Sequence()
seq.add(RaisedCos(+1,dur), Port(0))

sim = Simulator()
sim.set_system(sys)
sim.set_sequence(seq, visualize=True)
sim.run()

unitary = sim.unitary
```

## Citation
'''
@Misc{PulseSimulator,
  author = {Heya, Kentaro},
  title = {{Pulse Simulator}: A pulse simulator for quantum engineering},
  year = {2021--},
  url = "https://github.com/heyaroom/pulse_simulator",
  note = {[Online; accessed <today>]}
}
'''
