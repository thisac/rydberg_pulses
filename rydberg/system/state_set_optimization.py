import numpy as np
import nlopt

from .nlopt_wrapper import nlopt_optimize


def optimize_state_set_lossless(system, **kwargs):
    numPhases = system.pulse_sequence["layers"]
    cost = system.build_cost()

    # run the optimization
    lowerBounds = -2 * np.pi * np.ones((numPhases, ))
    upperBounds = 2 * np.pi * np.ones((numPhases, ))
    optKwargs = {
        'stopval': 1e-12,
        'ftolRel': 1e-8,
        'xtolRel': 1e-8,
        # 'maxTime': 60,    #  1m
        # 'maxTime': 300,   #  5m
        # 'maxTime': 1800,  # 30m
        # 'maxTime': 3600,  #  1h
        # 'maxTime': 86400,  # 24h
        # 'guess': np.zeros((numPhases, ), dtype=float),
        'guess': 2 * np.pi * np.random.random((numPhases, )) - np.pi,  # random uniform guess over -pi -> pi
        # 'guess': 4 * np.pi * np.random.random((numPhases, )) - 2 * np.pi,  # random uniform guess over -2pi -> 2pi
        # 'guess': 8 * np.pi * np.random.random((numPhases, )) - 4 * np.pi,  # random uniform guess over -4pi -> 4pi
        # 'guess': [np.pi / 2, -np.pi, -np.pi, 2 * np.pi, np.pi, np.pi, np.pi / 2],
        'printEvery': system.parameters["print_every"],
        'maxEval': system.parameters["num_of_evaluations"],
        }

    optKwargs.update((key, value) for key, value in kwargs.items() if value is not None)
    opt_algorithm = get_algorithm(str(system.parameters["opt_algorithm"]))

    results = nlopt_optimize(cost, opt_algorithm, lowerBounds, upperBounds, **optKwargs)

    if results['caughtError'] is not None:
        print("Warning: optimization returned an error.")
        print(results['caughtError'])
    if system.parameters["verbose"]:
        print("\nRunning time: {:0.02f} seconds".format(results['runTime']))
        print("Number of cost function evaluations: {:d}".format(results['numEvaluations']))
        print("Cost function evaluations/second: {:0.02f}".format(results['numEvaluations'] / results['runTime']))
        print("Optimization result: {}".format(results['returnCodeMessage']))
        print("Best error: {}".format(results['bestError']))

    return results


def get_algorithm(opt_algorithm):
    default_algorithm = "LBFGS"
    algorithms = {
        "LBFGS": nlopt.LD_LBFGS,
        "BOBYQA": nlopt.LN_BOBYQA,
        "CRS": nlopt.GN_CRS2_LM,
        "MLSL": nlopt.G_MLSL_LDS,
        "STOGO": nlopt.GD_STOGO,
        "STOGO_RAND": nlopt.GD_STOGO_RAND,
    }

    try:
        opt_function = algorithms[opt_algorithm]
    except KeyError:
        print(f"Warning: {opt_algorithm} not found. Using default ({default_algorithm}).")
        input("Press Enter to continue...")
        opt_function = algorithms[default_algorithm]

    return opt_function
