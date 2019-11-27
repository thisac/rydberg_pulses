import json
import time
import warnings

from pathlib import Path
from operator import mul
from functools import reduce
from itertools import product
from datetime import timedelta, datetime

import seaborn
import qutip as qt
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad as agrad

from .state_set_optimization import optimize_state_set_lossless as opt
from ..utils import print_dict

seaborn.set()


class System:
    def __init__(self, operation):
        self.inputs = []
        self.targets = []
        self.rhos_inputs = []
        self.rhos_targets = []
        self.operation = operation

        self.parameters = self.load_parameters(operation)
        self.verbose = self.parameters["verbose"]
        self.codes_dict = self.get_codes(
            self.parameters["operation"],
            self.verbose
            )
        if self.codes_dict is None:
            raise NameError(f"{self.parameters['operation']} not found.")
        self.pulse_sequence = self.get_sequence(
            self.parameters["pulse_sequence"]
            )

    def print_parameters(self):
        """ Print parameters of system """
        print("\nParameters:")
        print_dict(self.parameters)
        print("\n")

    def load_parameters(self, name):
        """ Load parameters from a local JSON file """
        codes = None
        custom_code_path = Path.cwd() / "parameters.json"
        script_path = Path(__file__).parent / "parameters.json"
        try:
            with open(custom_code_path) as read_file:
                codes = json.load(read_file)[name]
        except FileNotFoundError:
            pass
        except KeyError:
            warnings.warn(
                f"System parameters for {name} not found in custom directory",
                Warning
                )

        try:
            with open(script_path) as read_file:
                codes = json.load(read_file)[name]
        except KeyError:
            warnings.warn(
                f"System parameters for {name} not found in default directory",
                Warning
                )

        return codes

    def optimize(self, **kwargs):
        t_start = time.perf_counter()

        if self.parameters["num_of_tests"] == 0:
            num_of_tests = 999999
            stop_at_minimum = True
        else:
            num_of_tests = self.parameters["num_of_tests"]
            stop_at_minimum = False

        best_error = 1
        best_num_of_evals = None
        for j in range(num_of_tests):

            results = opt(
                self,
                **kwargs,
                )

            if results["bestError"] < best_error:
                best_error = results["bestError"]
                best_num_of_evals = results["numEvaluations"]

            if self.parameters["verbose"]:
                print(
                    f"{j:3d}  {results['returnCodeMessage']:>15}    "
                    f"Min error: {results['bestError']:.2e}\n"
                )

            if stop_at_minimum and best_error < 1e-8:
                num_of_tests = j + 1
                break

        t_stop = time.perf_counter()
        if self.parameters["verbose"]:
            test_time = timedelta(seconds=((t_stop - t_start) / num_of_tests))
            run_time = timedelta(seconds=(t_stop - t_start))
            print(f"\nMid-time:        {datetime.now()}")
            print(f"\nTime per test:   {test_time}")
            print(f"Runtime [h:m:s]: {run_time}")
            print(f"\nMinimum error: {best_error}")
            print(f"\n# of evaluations: {best_num_of_evals}\n")

        self.results = results

        return self

    def build_system_function(self):
        raw_pulse_sequence = self.pulse_sequence["pulse_sequence"]
        try:
            pt_out = self.pulse_sequence["pt_out"]
        except KeyError:
            pt_out = None

        def build_system(x):
            return self.pulse(raw_pulse_sequence, x)

        return build_system, pt_out

    def get_sequence(self, sequence):
        sequence_data = None
        custom_code_path = Path.cwd() / "pulse_sequences.json"
        script_path = Path(__file__).parent / "pulse_sequences.json"
        try:
            with open(custom_code_path) as read_file:
                sequence_data = json.load(read_file)[sequence]
                if self.verbose:
                    print(f"Found {sequence} in user pulse_sequences.json")
        except FileNotFoundError:
            try:
                with open(script_path) as read_file:
                    sequence_data = json.load(read_file)[sequence]
                    if self.verbose:
                        print(f"Found {sequence} in pulse_sequences.json")
            except KeyError:
                if self.verbose:
                    print(f"Code {sequence} not found in pulse_sequences.json")
        except KeyError:
            if self.verbose:
                print(f"Code {sequence} not found in user pulse_sequences.json")

        return sequence_data

    def pulse(self, pulse_sequence, x):
        assert len(np.array(pulse_sequence).shape) == 2

        list_of_pulses = []
        for i, qb in enumerate(pulse_sequence):
            list_of_pulses.append(self.single_pulse(qb, x[i]))

        return reduce(mul, list_of_pulses)

    def single_pulse(self, qb, theta, phi=0):
        """ Single pulse on any number of qubits (corresponding to len(qb))

        parameters:
            qb (list[int]): List of pulses. Index corresponds to which qubit
                the pulse should be applied, while the number determines which
                pulse:
                1 = single qubit rotation
                2 = |1> <--> |r> rotation depending on control qubits
            theta (float): theta value in SU(2) rotational matrix
            phi (float): phi value in SU(2) rotational matrix
        """
        qb = np.array(qb)
        SU2 = self.get_SU2(theta, phi)

        mat_S01 = np.eye(3, dtype=complex)
        mat_S1r = np.eye(3, dtype=complex)

        # NOTE: Switch if (not) using autograd
        # mat_S01 = array_assignment(mat_S01, SU2, (0, 0))
        # mat_S1r = array_assignment(mat_S1r, SU2, (1, 1))
        mat_S01[:SU2.shape[0], :SU2.shape[1]] = SU2
        mat_S1r[1:SU2.shape[0] + 1, 1:SU2.shape[1] + 1] = SU2

        mat_S01 = qt.Qobj(mat_S01)
        mat_S1r = qt.Qobj(mat_S1r)

        mat_rr = qt.Qobj([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        mat_01 = qt.Qobj([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        mat_eye = qt.qeye(3)

        qb_list = []
        last_el = 0
        for i, el in enumerate(qb):
            try:
                next_el = qb[i + 1]
            except IndexError:
                next_el = 0

            if el == 1 and next_el != 2 and last_el != 2:
                qb_list.append(mat_S01)
            elif el == 2:
                if i == 0:
                    pulse_2 = (
                        qt.tensor(mat_eye, mat_rr)
                        + qt.tensor(mat_S1r, mat_01)
                        )
                elif i == len(qb) - 1:
                    pulse_2 = (
                        qt.tensor(mat_rr, mat_eye)
                        + qt.tensor(mat_01, mat_S1r)
                        )
                else:
                    pulse_2 = (
                        qt.tensor(mat_rr, qt.tensor(mat_eye, mat_eye))
                        + qt.tensor(mat_eye, qt.tensor(mat_eye, mat_rr))
                        + qt.tensor(mat_01, qt.tensor(mat_S1r, mat_01))
                        - qt.tensor(mat_rr, qt.tensor(mat_eye, mat_rr))
                        )
                qb_list.append(pulse_2)
            elif next_el != 2 and last_el != 2:
                qb_list.append(mat_eye)
            last_el = el.copy()
        pulse = reduce(qt.tensor, qb_list)

        return pulse

    def get_SU2(self, theta, phi):
        SU2 = np.array([
            [np.cos(theta / 2), -np.exp(-1j * phi) * np.sin(theta / 2)],
            [np.exp(-1j * phi) * np.sin(theta / 2), np.cos(theta / 2)]
            ])
        return SU2

    def apply_error():
        pass

    def build_cost(self):
        build_system, pt_out = self.build_system_function()

        if self.parameters["density"]:
            inputs = self.rhos_inputs
            outputs = self.rhos_targets

            def cost_fn(x):
                S = build_system(x)
                SH = S.dag()

                cost = 0
                for i, single_rho in enumerate(inputs):

                    rho = S * single_rho * SH

                    if pt_out:
                        rho = rho.ptrace(pt_out)

                    cost += 1 - np.real(np.trace(np.dot(outputs[i], rho)))

                return cost / len(inputs)

        else:
            def cost_fn(x):
                S = build_system(x)
                cost = 0
                for i in range(len(self.inputs)):
                    c_part = self.targets[i] * S * self.inputs[i].dag()
                    cost += 1.0 - np.real(np.sum(c_part))
                return cost / len(self.inputs)

        cost_grad = agrad(cost_fn)

        def cost_wrapper(x, grad):
            if grad.size > 0:
                grad[:] = cost_grad(x)
            return cost_fn(x)

        return cost_wrapper

    def build_codes(self):
        qb_dict = {
            "0": qt.Qobj([[1, 0, 0]]),
            "1": qt.Qobj([[0, 1, 0]]),
            "r": qt.Qobj([[0, 0, 1]]),
            "-0": qt.Qobj([[-1, 0, 0]]),
            "-1": qt.Qobj([[0, -1, 0]]),
            "-r": qt.Qobj([[0, 0, -1]])
            }

        inputs = np.array(self.codes_dict["inputs"], dtype=str)
        targets = np.array(self.codes_dict["targets"], dtype=str)

        self.inputs = [qt.tensor([qb_dict[i] for i in inp]) for inp in inputs]
        self.targets = [qt.tensor([qb_dict[i] for i in inp]) for inp in targets]
        print(self.inputs)
        self.rhos_inputs = []
        self.rhos_targets = []
        for el in self.inputs:
            self.rhos_inputs.append(qt.ket2dm(el))
        for el in self.targets:
            self.rhos_targets.append(qt.ket2dm(el))

        return self

    def get_output(self, x):
        raw_pulse_sequence = self.pulse_sequence["pulse_sequence"]
        best_x = self.results["bestX"]

        return np.dot(self.pulse(raw_pulse_sequence, best_x), x)

    def display_results(self):
        results = self.results["bestX"]
        raw_pulse_sequence = self.pulse_sequence["pulse_sequence"]

        pulse_one_results = []
        pulse_two_results = []
        non_pulses = []

        pulse_one_range = []
        pulse_two_range = []
        non_pulses_range = []

        qb_one = []
        qb_two = []
        non_qb = []

        for i, el in enumerate(raw_pulse_sequence):
            print(el, results[i])
            el = np.array(el)
            if np.any(el == 1):
                pulse_one_results.append(results[i] / np.pi)
                pulse_one_range.append(i + 1)
                qb_one.append(int(np.where(el == 1)[0] + 1))
            elif np.any(el == 2):
                pulse_two_results.append(results[i] / np.pi)
                pulse_two_range.append(i + 1)
                qb_two.append(int(np.where(el == 2)[0] + 1))
            else:
                non_pulses.append(results[i] / np.pi)
                non_pulses_range.append(i)
                non_qb.append(0)

        plt.figure(
            figsize=(4 + len(results), 4 + np.sqrt(len(results))),
            dpi=80,
            )
        ax = plt.subplot(111)

        if pulse_one_results:
            bar1 = ax.bar(
                pulse_one_range, pulse_one_results[::-1],
                width=1, color='b', label="Pulse one"
                )
            self.autolabel(ax, bar1, qb_one)
        if pulse_two_results:
            bar2 = ax.bar(
                pulse_two_range, pulse_two_results[::-1],
                width=1, color='r', label="Pulse two"
                )
            self.autolabel(ax, bar2, qb_two)

        ax.legend()
        ax.set_xlabel("Pulse")
        ax.set_ylabel(r"$\theta$ / $\pi$")
        ax.set_xticks(1 + np.arange(len(results)))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def autolabel(ax, rects, label):
        if len(rects) < 10:
            size = 'large'
        elif len(rects) >= 10 and len(rects) < 30:
            size = 'medium'
        elif len(rects) >= 30 and len(rects) < 50:
            size = 'small'
        elif len(rects) >= 50 and len(rects) < 70:
            size = 'x-small'
        else:
            size = 'xx-small'

        for i, rect in enumerate(rects):
            height = np.max([0.0, rect.get_height()])
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{label[i]}",
                ha='center',
                va='bottom',
                size=size,
                weight='bold'
                )

    @staticmethod
    def get_codes(operation, verbose=True):
        codes = None
        custom_code_path = Path.cwd() / "codes.json"
        script_path = Path(__file__).parent / "codes.json"
        try:
            with open(custom_code_path) as read_file:
                codes = json.load(read_file)[operation]
                if verbose:
                    print(f"Found {operation} in user codes.json")
        except FileNotFoundError:
            try:
                with open(script_path) as read_file:
                    codes = json.load(read_file)[str(operation)]
                    if verbose:
                        print(f"Found {operation} in codes.json")
            except KeyError:
                if verbose:
                    print(f"Code {operation} not found in codes.json")
        except KeyError:
            if verbose:
                print(f"Code {operation} not found in user codes.json")

        return codes


def get_basis(n, print_basis=False):
    state_list = [["0", "1", "r"]]
    many_lists = state_list * n
    state_set = list(product(*many_lists))
    if print_basis:
        for i, el in enumerate(state_set):
            print(f"{i:2d}: ", *el)
    else:
        return state_set
