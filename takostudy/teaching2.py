from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar
import typing
import pandas as pd
from torch.utils import data as data_utils


class Progress(object):
    
    def __init__(self):
        
        self.df = pd.DataFrame()
    
    def accessor(self, name: str, n_iterations: int=None):
        
        return ProgressAccessor(self, name, n_iterations=n_iterations)
    
    def add_result(self, result: dict):
        
        cur = pd.DataFrame(result)
        self.df = pd.concat([self.df, cur], ignore_index=True)


class ProgressAccessor(object):

    def __init__(self, name: str, progress: Progress, n_iterations: int= None, iter_name: str=None, state: dict=None):
        
        self._progress = progress
        self._state = state or {}
        self._name = name
        self._iter_name = iter_name or name
        self._n_iterations = n_iterations
        self._cur_iteration = 0
    
    def update(self):
        self._cur_iteration += 1

    @property
    def iteration_state(self):
        return {
            self._iter_name: self._cur_iteration,
            f'{self._iter_name}_Last': self._n_iterations, 
        }

    def accessor(self, name: str, iter_name: str=None, n_iterations: int=None):

        state = {
            **self.iteration_state,
            **self._state
        }
        return ProgressAccessor(
            name, self._progress, n_iterations, iter_name, state
        )

    def add_result(self, result: dict):
        
        full_result = {
            **self.iteration_state,
            **result,
            **self._state
        }
        self._progress.add_result(full_result)


T = TypeVar('T')


class Trainer(Generic[T]):

    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def teach(self, learner: T, progress: ProgressAccessor) -> Iterator:
        raise NotImplementedError


class Teacher(Trainer):

    def __init__(self, dataset: data_utils.Dataset, batch_size: int, shuffle: bool=True):
        
        self._data_loader = data_utils.DataLoader(
            dataset, batch_size, shuffle=shuffle
        )

    def teach(self, learner: T, progress: ProgressAccessor) -> Iterator:
        
        for x, t in self._data_loader:
            result = learner.learn(x, t)
            progress.add_result(result)
            progress.update()
            yield result


class Validator(Trainer):

    def __init__(self, dataset: data_utils.Dataset, batch_size: int):
        
        self._data_loader = data_utils.DataLoader(
            dataset, batch_size
        )

    def teach(self, learner: T, progress: ProgressAccessor) -> Iterator:
        
        for x, t in self._data_loader:
            result = learner.test(x, t)
            progress.add_result(result)
            progress.update()
            yield result


class Assistant(ABC):

    def assist(self, learner, progress: Progress):
        pass


class CompositeTrainer(Trainer):

    def __init__(self, name: str, trainers: typing.List[Trainer]):

        self._name = name
        self._trainers = trainers

    def teach(self, learner: T, progress: ProgressAccessor) -> Iterator:
        
        # progress = progress.accessor(self._name, n_iterations=len(self._trainer))
        for trainer in self._trainers:

            for result in trainer.teach(learner, progress):
                yield result


class Lesson(Trainer):

    def __init__(self, name: str, trainer: Trainer, assistants: typing.List[Assistant]):
        
        self._trainer = trainer
        self._assistants = assistants
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name

    def teach(self, learner: T, progress: ProgressAccessor) -> Iterator:
        
        progress = progress.accessor(self._name, n_iterations=len(self._trainer))
        for result in self._trainer.teach(learner, progress):
            
            for assistant in self._assistants:
                assistant.assist(learner, progress)
            yield result

# need to add the ability to specify iterations...


# class Builder <- build up specific types of teachers
# studying will depend on studying

# class Ability to run a full lesson

# 
