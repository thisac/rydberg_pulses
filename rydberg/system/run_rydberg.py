from becc import System
from becc.rydberg import rydberg
from becc.utils import print_array as pa
import sys

parameters = {
    "opt_algorithm": "BOBYQA",
    "num_of_tests": 0,
    "print_every": 100,
    "save_data": False,
}

try:
    gate = sys.argv[1]
except IndexError:
    print("No argument. Using CNOT.")
    gate = "CNOT"

if gate == "CZ":
    new_params = {
        "num_of_layers": 3,
        "code_name": "CZ_rydberg",
        "pulse_sequence": "CZ",
    }
elif gate == "CNOT":
    new_params = {
        "num_of_layers": 5,
        "code_name": "CNOT_rydberg",
        "pulse_sequence": "CNOT",
    }
elif gate == "CNOT2_long":
    new_params = {
        "num_of_layers": 21,
        "code_name": "CNOT2_rydberg",
        "pulse_sequence": "long_cnot2",
    }
elif gate == "CNOT2_short":
    new_params = {
        "num_of_layers": 6,
        "code_name": "CNOT_rydberg",
        "pulse_sequence": "shortest_cnot2",
    }
elif gate == "Toffoli_long":
    new_params = {
        "num_of_layers": 34,
        "code_name": "Toffoli_rydberg",
        "pulse_sequence": "long",
    }
elif gate == "Toffoli_short" or gate == "Toffoli":
    new_params = {
        "num_of_layers": 10,
        "code_name": "Toffoli_rydberg",
        "pulse_sequence": "shortest_toffoli",
    }
else:
    print("No gate!")

parameters.update(new_params)

rbs = System("rydberg", parameters)
rbs.build_codes().optimize()

rydberg.display_results(rbs.results["bestX"], parameters["pulse_sequence"])
