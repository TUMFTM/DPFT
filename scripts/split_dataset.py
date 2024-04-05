import argparse
import os.path as osp
import json

from operator import itemgetter
from typing import Callable, Dict, List, Tuple

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.termination.default import DefaultMultiObjectiveTermination

from dprt.datasets import prepare
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed


class DatasetSplitting(ElementwiseProblem):
    def __init__(self,
                 elements: List[Tuple[np.ndarray, ...]],
                 splits: List[float],
                 objectives: List[Callable],
                 **kwargs):
        """Dataset split problem.

        Attributes:
            elements: Dataset element (sample or scenes) properties, given as a list
                of N dataset elements with O properties of shape (M, C).
            splits: Desired dataset splits given by their share in the range of [0, 1]
                and a sum of 1 (100%).
            objectives: Optimization objective functions as list of length O.
        """
        # Check input arguments
        assert sum(splits) == 1.0

        # Initialize instance attributes
        self.elements = elements
        self.splits = np.asarray(splits)
        self.objectives = objectives

        # Initialize problem
        super().__init__(
            n_var=len(elements),
            n_obj=len(objectives) * (len(splits)),
            n_ieq_constr=0,
            n_eq_constr=1,
            xl=0,
            xu=len(splits) - 1,
            vtype=int,
            **kwargs
        )

        # Convert elements form [(M, C)] with len N -> ((N, C),) with len O
        self.elements = tuple((np.vstack(p) for p in zip(*self.elements)))

        # Determine target distribution
        self.target = self.get_target(self.elements)

    @staticmethod
    def get_target(elements) -> Tuple[np.ndarray, ...]:
        """Returns the target value distribution.

        Arguments:
            elements: Dataset element properties with
                shape (N, C) for each objective O.

        Returns:
            Target value distributions with shape (C,)
            for each objective O.
        """
        return tuple([np.sum(t, axis=0) for t in elements])
    
    def _evaluate(self, x: List[int], out: Dict[str, float], *args, **kwargs) -> None:
        """Evaluates the current population.

        See: https://pymoo.org/interface/result.html

        Arguments:
            x: Design space values with shape (N,).
            out: Optimization output instance.
        """
        # Get fitness score
        out["F"] = self.deviation(x)

        # Apply equality constraints
        _, counts = np.unique(x, return_counts=True)
        out["H"] = np.sum(np.abs((counts / np.sum(counts)) - self.splits))

    def deviation(self, x) -> List[float]:
        """Returns the deviation for each objective and split.

        Arguments:
            x: Design space values given as dataset indices
                with shape (N,).

        Returns:
            Objective spaces values. A fitness score (loss)
            for each objective and dataset split.
        """
        return [
            objective(self.elements[i][x == n], self.target[i])
            for i, objective in enumerate(self.objectives)
            for n in range(len(self.splits))
        ]


def discrete_dist_diff(inputs: np.ndarray, targets: np.ndarray) -> float:
    """Returns the total deviation between two discrete distributions.

    Arguments:
        inputs: Distribution for N elements over C categories
            with shape (N, C).
        targets: Target distribution for the C categories with
            shape (C,).

    Returns:
        The absolute difference between the input and target
        distribution as sum over all categories.
    """
    # Get number of elements per category
    count = np.sum(inputs, axis=0)

    difference = (targets / np.sum(targets)) - (count / np.sum(count))

    return np.sum(np.abs(difference))


def get_kradar_elements(src, preperator) -> List[Tuple[np.ndarray, ...]]:
    """Returns the K-Radar dataset element properties

    Arguments:
        src: Source directory of the raw dataset.
        preperator: Dataset specific preperator
            to access (interact with) the dataset.
    """
    # Get dataset element paths
    dataset_paths = preperator.get_dataset_paths(src)

    # Initialize dataset elements
    elements = {}

    # Collect dataset element properties
    for sequence_paths in dataset_paths.values():
        for seq_id, sequence in sequence_paths.items():
            for sample in sequence:
                sample_id = osp.splitext(osp.basename(sample))[0]
                box = preperator.get_boxes(sample)

                categories = np.zeros((box.shape[0], len(preperator.categories)), dtype=int)
                categories[np.arange(box.shape[0]), box[:, 7].astype(int)] = 1

                # Get sequence path
                sequence_path = osp.split(osp.dirname(sample))[0]
                description = osp.join(sequence_path, 'description.txt')

                # Get sequence description (tags)
                description = preperator.get_description(description)

                # Encode road structures description
                structures = np.zeros((1, len(preperator.road_structures)), dtype=int)
                structures[0, preperator.road_structures[description[0]]] = 1

                # Encode time zone description
                time = np.zeros((1, len(preperator.time_zone)), dtype=int)
                time[0, preperator.time_zone[description[1]]] = 1

                # Encode weather conditions description
                weather = np.zeros((1, len(preperator.weather_conditions)), dtype=int)
                weather[0, preperator.weather_conditions[description[2]]] = 1

                # Add dataset element
                elements[f"{seq_id}_{sample_id}"] = (categories, structures, time, weather)

    # Summarize properties
    elements = {
        sample_id: tuple(np.sum(prop, axis=0) for prop in element)
        for sample_id, element in elements.items()
    }

    return elements


def optimize_splits(elements: List[Tuple[np.ndarray, ...]],
                    splits: List[float],
                    objectives: List[Callable],
                    seed: int = 42) -> List[int]:
    """Returns indices of optimized dataset splits.

    Findes the best assignment of dataset elements to the individual
    dataset splits such that the objectives are minimized.

    Arguments:
        elements: Dataset elements represented by their individual
            properties to optimize for with shape (N, O). The lenght of
            the list corresponds to the number of dataset elements N and
            the length of the tuple must be equal to the number of objectives O.
        splits: Shares of the individual dataset splits with shape (S,).
        objectives: Objective functions to optimize (minimize) for with shape (O,).

    Returns:
        List of indices mapping a dataset element to a split with shape (N,)
        in the range of [0, S]. Returns the last (best) population if no
        feasible solution was found. For example if the constraints were not meet.
    """

    # Initialize optimization problem
    problem = DatasetSplitting(elements, splits=splits, objectives=objectives)

    # Define optimization algorithm
    algorithm = NSGA2(
        pop_size=100,
        eliminate_duplicates=True,
        sampling=IntegerRandomSampling(),
        mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair())
    )

    # Define termination cretireon
    termination = DefaultMultiObjectiveTermination(n_max_gen=1000)

    # Minimize objectives to find optimal solution
    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=True,
        seed=seed
    )

    # Return optimized dataset split indices
    # if res.X is not None:
    #     return res.X

    # Get index of best objective spaces value
    idx = np.argmax(np.sum(res.pop.get('F'), axis=1), axis=0)

    # Return best design space values
    return res.pop.get('X')[idx]


def save(dst: str, split_names: List[str], keys: List[str], indices: List[int]) -> None:
    """Saves the dataset splits.

    Arguments:
        dst: Destination folder name to store the dataset
            split file.
        split_names: Names of the individual dataset splits.
        keys: Unique identifiers of the dataset elements
            with shape (N,).
        indices: Indices mapping a dataset element to a split.
    """
    # Convert indices to array
    indices = np.asarray(indices, dtype=int)

    # Get dataset splits
    splits = {name: itemgetter(*np.where(indices == i)[0].tolist())(keys) for i, name in enumerate(split_names)}

    # Desine output file name
    filename = osp.join(dst, 'splits.json')

    # Save dataset splits to file
    with open(filename, 'w') as f:
        json.dump(splits, f, indent=4)


def main(src: str, cfg: str, dst: str):
    """ Data preparation for subsequent model training or evaluation.

    Arguments:
        src: Source directory path to the raw dataset folder.
        cfg: Path to the configuration file.
        dst: Destination directory to save the processed dataset files.
    """
    # Load dataset configuration
    config = load_config(cfg)

    # Set global random seed
    set_seed(config['computing']['seed'])

    # Initialize dataset preparator
    preperator = prepare(config['dataset'], config)

    # Get dataset elements
    elements = get_kradar_elements(src, preperator)

    # Define target split
    splits = {'train': 0.8, 'val': 0.2}

    # Define objectives to optimize for (one for each property)
    objectives = [discrete_dist_diff, discrete_dist_diff,
                  discrete_dist_diff, discrete_dist_diff]

    # Optimize dataset split
    indices = optimize_splits(list(elements.values()), list(splits.values()),
                              objectives, seed=config['computing']['seed'])

    # Save dataset splits
    save(dst, list(splits.keys()), list(elements.keys()), indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='/data/kradar/KRadar_refined_label_by_UWIPL',
                        help="Path to the raw dataset folder.")
    parser.add_argument('--cfg', type=str, default='/app/config/kradar.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='/data/kradar/processed',
                        help="Path to save the processed dataset.")
    args = parser.parse_args()

    main(src=args.src, cfg=args.cfg, dst=args.dst)
