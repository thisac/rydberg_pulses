# Rydberg simulations

Optimizations of pulse sequences for quantum gates applied on neutral atom arrays using the Rydberg blockade mechanism.

## Installation:

Clone or download the repository and install via pip:

    $ pip install -e ./ --user

## Usage:
Begin by importing the module.

```python
from rydberg import System
```

Create a rydberg-system-object, choosing what gate to optimize over (implemented ones are CNOT, CZ, Toffoli/CCNOT and CCZ). Default parameters are chosen if no parameters are given.

```python
params = {
    "operation": "CNOT",
    "pulse_sequence": "CNOT",
    "opt_algorithm": "BOBYQA",
    "density": false,
    "num_of_tests": 0,
    "num_of_evaluations": 20000,

    "print_every": 1,
    "verbose": false,
    "save_data": false
    }

rbs = System("CNOT", parameters=params)
```

Next, load and set up the inputs and targets of the optimization, and then optimize.

```python
rbs.build_codes()
rbs.optimize()
```

Both rows can also be run as one as `rbs.build_codes().optimize()`. When the optimization is done, the results can be printed and plotted with the display_results function.

```python
rbs.display_results()
```

The plot will show the pulses applied from left to right, where the heights of the bars correspond to the strength of each respective pulse (in $\pi$), and the numbers above the bars corresponds to the qubit number (in a row).

Default parameters are stored in [parameters.json](rydberg/system/parameters.json), the different gates are stored in [codes.json](rydberg/system/codes.json) and the pulse sequence templates are stored in [pulse_sequences.json](rydberg/system/pulse_sequences.json).