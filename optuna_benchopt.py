#!/usr/bin/env python3
import optuna
import click
from pathlib import Path
import os
from numbers import Real

from benchopt.runner import run_one_solver
import benchopt
from benchopt.benchmark import Benchmark
from benchopt.runner import _run_benchmark
from benchopt.config import get_setting
from benchopt.cli.completion import (
    complete_solvers,
    complete_datasets,
)
from benchopt.utils.terminal_output import TerminalOutput
import ast
import numpy as np
import pandas as pd


class OptunaObjective:
    def __init__(
        self, benchmark, dataset, objective, solver_klass, solver_param, trial_params
    ):
        self.benchmark = benchmark
        self.dataset = dataset
        self.objective = objective
        self.solver_klass = solver_klass
        self.solver_params = solver_param
        self.trial_params = trial_params

        self.output = TerminalOutput(1, True)
        self.output.set(verbose=True)

    @staticmethod
    def _add_suggestion(trial, ttype, name, *args, **kwargs):

        if ttype == "categorical":
            return trial.suggest_categorical(name, *args, **kwargs)
        elif ttype == "float":
            return trial.suggest_float(name, *args, **kwargs)
        elif ttype == "int":
            return trial.suggest_int(name, *args, **kwargs)
        else:
            raise ValueError(f"Invalid trial type {ttype}")

    def __call__(self, trial):

        # reparametrize the solver with the trial parameters
        new_params = self.solver_params.copy()
        for ttype, name, args, kwargs in self.trial_params:
            new_params[name] = self._add_suggestion(trial, ttype, name, *args, **kwargs)

        # List all datasets, objective and solvers to run based on the filters
        # provided. Merge the solver_names and forced to run all necessary solvers.
        all_runs = self.benchmark.get_all_runs(
            [(self.solver_klass, new_params)],
            None,
            self.dataset,
            self.objective,
            output=self.output,
        )  # should only contains a single solver config (but potentially multiple dataset points)

        common_kwargs = dict(
            benchmark=self.benchmark,
            n_repetitions=1,
            max_runs=10,
            timeout=100,
            pdb=False,
            collect=False,
        )
        run_stats = []
        peak = []
        for kwargs in all_runs:
            results = run_one_solver(**common_kwargs, **kwargs)
            peak_value = results[-1]["objective_psnr"]
            # Do the pruning here (if the solver gives bad results)
            run_stats.extend(results)
            peak.append(peak_value)
        return np.median(peak_value)
        # extract the best scores from the run_stats
        return run_stats


def parse_sweep(sweep):
    parsed = ast.parse(sweep).body[0]
    match parsed:
        case ast.Assign(
            [ast.Name(id=name)], ast.Tuple(elts=[*args]) | ast.List(elts=[*args])
        ) if all(
            isinstance(a, ast.Constant) for a in args
        ):  # a= (...)
            return ("categorical", name, [a.value for a in args])
        case ast.Assign(
            [ast.Name(id=name)],
            ast.Call(
                func=ast.Name(id="range"),
                args=[ast.Constant(low), ast.Constant(high), *step],
            ),
        ):  # a=range(low, high, <step>)
            step = step[0].value if step else 0
            print("here we are", low, high, step)
            if all(isinstance(a, int) for a in (low, high, step)):
                return ("int", name, [low, high], {"step": step or 1})
            elif any(isinstance(a, Real) for a in (low, high, step)):
                return ("float", name, [low, high], {"step": step or None})
            else:
                raise ValueError("Invalid linpsace parametrization {s}")
        case ast.Assign(
            [ast.Name(id=name)],
            ast.Call(
                func=ast.Name(id="logrange"),
                args=[ast.Constant(low), ast.Constant(high)],
            ),
        ) if all(isinstance(a, float) for a in (low, high)):
            # a = logrange(a,b)
            return ("float", name, [low, high], {"log": True})
        case _:  # default, raise error
            raise ValueError(f"Invalid sweep parameter {s}")
    return trial_params


@click.command()
@click.argument("benchmark", default=Path.cwd(), type=click.Path(exists=True))
@click.option(
    "--solver",
    "-s",
    "solver_name",
    metavar="<solver_name>",
    multiple=False,  # Only one solver can be swept over.
    type=str,
    help="Include <solver_name> in the installation. "
    "By default, all solvers are included except "
    "when -d flag is used. If -d flag is used, then "
    "no solver is included by default. "
    "When `-s` is used, only listed estimators are included. "
    "To include multiple solvers, use multiple `-s` options."
    "To include all solvers, use -s 'all' option.",
    shell_complete=complete_solvers,
)
@click.option(
    "--dataset",
    "-d",
    "dataset_names",
    metavar="<dataset_name>",
    multiple=True,
    type=str,
    help="Install the dataset <dataset_name>. By default, all "
    "datasets are included, except when -s flag is used. "
    "If -s flag is used, then no dataset is included. "
    "When `-d` is used, only listed datasets "
    "are included. Note that <dataset_name> can include parameters "
    "with the syntax `dataset[parameter=value]`. "
    "To include multiple datasets, use multiple `-d` options."
    "To include all datasets, use -d 'all' option.",
    shell_complete=complete_datasets,
)
@click.option(
    "--objective",
    "-o",
    "objective_filters",
    metavar="<objective_filter>",
    multiple=True,
    type=str,
    help="Select the objective based on its parameters, with the "
    "syntax `objective[parameter=value]`. This can be used to only "
    "include one set of parameters. ",
)
@click.option(
    "--sweep",
    "-w",
    multiple=True,
    type=str,
    help="Describe a sweep of parameters for a solver."
    "The syntax is `solver_name[parameter=value]`."
    "where value can be a list, or a tuple, or `range(start, stop, [step])` or ` logrange(start, stop)`",
)
def main(benchmark, solver_name, dataset_names, objective_filters, sweep):

    benchmark = Benchmark(benchmark)

    objective = benchmark.check_objective_filters(objective_filters)

    # Check that the dataset/solver patterns match actual dataset
    datasets = benchmark.check_dataset_patterns(dataset_names)
    solver_klass, solver_params = benchmark.check_solver_patterns([solver_name])[0]

    sweep_params = [parse_sweep(s) for s in sweep]
    # use the trials values to get the solver parametrization
    # run_one_benchmark
    # get the objective and return it.
    optuna_objective = OptunaObjective(
        benchmark, datasets, objective, solver_klass, solver_params, sweep_params
    )
    # Initialize the Optuna study
    study = optuna.create_study()
    study.optimize(optuna_objective, n_trials=10)


if __name__ == "__main__":
    main()
