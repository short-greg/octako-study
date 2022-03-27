from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar
import typing
import pandas as pd
from sklearn import preprocessing
from torch.utils import data as data_utils
from tqdm import tqdm

# i like this better
# could be conflicts in naming
# Epoch Teacher Iteration
#       X       0
# Epoch   Epoch/Iter  Epoch/Teacher Epoch/Teacher/Iter Epoch/Teacher/Results (other results)
# <>      1           Trainer       0                    ...
# 
# make sure the name does not have / in it
# this should make it easy to query

# y, weight = MemberOut(ModFactory, ['weight'])
# MemberSet() <- sets the member based on the input
# probably just need these two

# would need to make it so if the necessary data is available
# it does not execute the module
# Shared() <- maybe i don't need this

# Lesson(
#   'Epoch', [Team(trainer, assistants), Team(validator, assistants)]
# )


from enum import Enum

class Status(Enum):
    
    READY = 0
    IN_PROGRESS = 1
    FINISHED = 2


class Chart(object):
    
    def __init__(self):
        
        self.df = pd.DataFrame()
    
    def accessor(self, name: str, n_iterations: int=None):
        
        return ChartAccessor(self, name, n_iterations=n_iterations)
    
    def add_result(self, result: dict):
        
        cur = pd.DataFrame(result)
        self.df = pd.concat([self.df, cur], ignore_index=True)


class ChartAccessor(object):

    def __init__(self,  category: str, name: str, iter_name: str, progress: Chart, n_iterations: int= None, state: dict=None):
        """initializer

        Args:
            category (str): Name of the teacher category
            name (str): Name of the teacher
            iter_name (str): Name of the iterator
            progress (Chart): 
            n_iterations (int, optional): _description_. Defaults to None.
            state (dict, optional): _description_. Defaults to None.
        """
        self._category = category
        self._progress = progress
        self._state = state or {}
        self._name = name
        self._iter_name = iter_name
        self._n_iterations = n_iterations
        self._cur_iteration = 0
    
    def update(self):
        self._cur_iteration += 1

    @property
    def local_state(self):
        return {
            self._category: self._name,
            self._iter_name: self._cur_iteration,
            f'N_{self._iter_name}': self._n_iterations, 
        }
    
    @property
    def progress(self) -> Chart:
        return self._progress

    @property
    def iteration(self) -> int:
        return self._cur_iteration

    @property
    def n_iterations(self) -> int:
        return self._n_iterations
    
    @property
    def results(self) -> pd.DataFrame:
        return self._progress.df[self._progress.df[self._category] == self._name]

    def child(self, category: str, name: str, iter_name: str=None, n_iterations: int=-1):
        """Create a child progress accessor

        Args:
            category (str): Name of the category
            name (str): Name of the teacher
            iter_name (str): Name of the iteration column
            n_iterations (int, optional): Total number of iterations

        Returns:
            ChartAccesssor: Chart accessor with state 
        """
        state = {
            **self._state,
            **self.local_state
        }
        return ChartAccessor(
            category, name, iter_name, self._progress, n_iterations, state
        )

    def add_result(self, result: dict):
        """_summary_

        Args:
            result (dict): _description_
        """
        
        full_result = {
            **self.local_state,
            **result,
            **self._state
        }
        self._progress.add_result(
            full_result
        )



class DatasetIterator(ABC):
    """For conveniently iterating over a dataset in a behavior tree
    """

    @abstractmethod
    def adv(self):
        raise NotImplementedError
    
    @abstractproperty
    def cur(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def pos(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def is_end(self) -> bool:
        raise NotImplementedError


class DataLoaderIter(DatasetIterator):

    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self._cur_iter = iter(self._dataloader)
        self._finished = False
        self._cur = None
        self._is_start = True
        self._pos = 0
        self._iterate()

    def reset(self, dataloader: DataLoader=None):

        self._dataloader = dataloader or self._dataloader
        self._cur_iter = iter(self._dataloader)
        self._finished = False
        self._cur = None
        self._is_start = True
        self._pos = 0
        self._iterate()

    def _iterate(self):
        if self._finished:
            raise StopIteration
        try:
            self._cur = next(self._cur_iter)
        except StopIteration:
            self._cur = None
            self._finished = True

    def adv(self):
        self._iterate()
        self._pos += 1
        if self._is_start:
            self._is_start = False
    
    @property
    def cur(self):
        return self._cur
    
    def __len__(self) -> int:
        return len(self._dataloader)
    
    @property
    def pos(self) -> int:
        return self._pos

    def is_end(self) -> bool:
        return self._finished

    def is_start(self):
        return self._is_start




T = TypeVar('T')

class Assistant(ABC):

    def __init__(self, name: str):
        self.name = name

    def start(self, progress: ChartAccessor):
        pass

    def assist(self, progress: ChartAccessor):
        pass

    def finish(self, progress: ChartAccessor):
        pass


class AssistantGroup(object):

    def __init__(self, assistants: typing.List[Assistant]=None):

        self._assistants = assistants

    def start(self, progress: ChartAccessor):

        if self._assistants is None:
            return
        
        for assistant in self._assistants:
            assistant.start(progress)

    def assist(self, progress: ChartAccessor):
        if self._assistants is None:
            return

        for assistant in self._assistants:
            assistant.assist(progress)

    def finish(self, progress: ChartAccessor):
        if self._assistants is None:
            return

        for assistant in self._assistants:
            assistant.finish(progress)


class Task(ABC):

    def __init__(self, name: str):
        self._name = name
        self._status = Status.READY

    def name(self) -> str:
        return self._name

    def status(self) -> str:
        return self._status

    @abstractmethod
    def advance(self, progress: ChartAccessor) -> Status:
        raise NotImplementedError
    
    def reset(self):
        self._status = Status.READY


class Teacher(Task):

    def __init__(self, learner, dataset: data_utils.Dataset, batch_size: int, shuffle: bool=True):
        
        self._learner = learner
        self._dataset = dataset
        self._batch_size = batch_size
        self._dataloader = None
        self._shuffle = shuffle

    def advance(self, progress: ChartAccessor) -> Status:

        if self._status == Status.READY:
            self._dataloader = DataLoaderIter(data_utils.DataLoader(
                self._dataset, self._batch_size, shuffle=self._shuffle
            ))

        if self._dataloader.is_end():
            self._status = Status.FINISHED
            return self._status
        
        x, t = self._dataloader.cur
        result = self._learner.learn(x, t)
        progress.add_result(result)
        progress.update()
        self._status = Status.IN_PROGRESS

    def reset(self):
        super().reset()
        self._dataloader = None
        

class Validator(Trainer):

    def __init__(self, learner, dataset: data_utils.Dataset, batch_size: int):
        
        self._learner = learner
        self._dataset = dataset
        self._batch_size = batch_size
        self._dataloader = None

    def advance(self, progress: ChartAccessor) -> Status:

        if self._status == Status.READY:
            self._dataloader = DataLoaderIter(data_utils.DataLoader(
                self._dataset, self._batch_size
            ))

        if self._dataloader.is_end():
            self._status = Status.FINISHED
            return self._status
        
        x, t = self._dataloader.cur
        result = self._learner.test(x, t)
        progress.add_result(result)
        progress.update()
        self._status = Status.IN_PROGRESS

    def reset(self):
        super().reset()
        self._dataloader = None


class ProgressBar(Assistant):

    def __init__(self):

        self._pbar: tqdm = None
    
    def start(self, progress: ChartAccessor):
        self._pbar = tqdm()
        self._pbar.refresh()
        self._pbar.reset()
        self._pbar.total = progress.n_iterations

    def finish(self, progress: ChartAccessor):
        self._pbar.close()

    def assist(self, progress: ChartAccessor):
        
        self._pbar.update(1)
        self._pbar.refresh()
        # ctx.pbar.set_description_str(course.cur.name)

        # ctx.pbar.set_postfix({
        #     "Epoch": f"{progress.cur_epoch} / {course.n_epochs}",
        #     **course.epoch_results.mean(axis=0).to_dict()
        # })

    def reset():
        super().reset()
        if self._pbar:
            self._pbar.close()
        self._pbar = None

class Lecture(Task):

    def __init__(self, name: str, trainer: Trainer, assistants: typing.List[Assistant]=None):
        super().__init__(name)
        self._trainer = trainer
        self._assistants = AssistantGroup(assistants)

    def advance(self, progress: ChartAccessor) -> Status:
        
        self._assistants.start(progress)
        # progress = progress.accessor(self.name, n_iterations=len(self._trainer))
        
        self._status = self._trainer.advance(progress)
        self._assistants.assist(progress)

        if self._status == Status.FINISHED:
            self._assistants.finish()
        
        return self._status

    def reset(self):
        super().reset()
        self._assistants.reset()
        self._trainer.reset()


class Workshop(Task):

    def __init__(self, name: str, lessons: typing.List[Lesson], assistants: typing.List[Assistant]=None, iterations: int=1):

        self._name = name
        self._lessons = lessons
        self._assistants = AssistantGroup(assistants)
        self._iterations = iterations
        self._cur_iteration = 0
        self._cur_lesson = 0

    def advance(self, progress: ChartAccessor) -> Status:
        
        if self._status == Status.FINISHED:
            return self._status

        if self._status == Status.READY:
            self._assistants.start(progress)
        
        status = self._lessons[self._cur_lesson].advance(progress)
        if status == Status.FINISHED:
            self._cur_lesson += 1
            if self._cur_lesson == len(self._lessons):
                self._cur_iteration += 1
                if self._cur_iteration == len(self._iterations):
                    self._status = Status.FINISHED
                    self._assistants.finish(progress)
                    return self._status
                for lesson in self._lessons:
                    lesson.reset()
        self._status = Status.IN_PROGRESS
        return self._status
    
    def iteration(self) -> int:
        return self._cur_iteration

    def reset(self):
        super().__init__()
        self._assistants.reset()
        for lesson in self._lessons:
            lesson.reset()
        self._cur_iteration = 0


class Notifier(Assistant):
    """Assistant that 'triggers' another assistant to begine
    """

    def __init__(self, name: str, assistants: typing.List[Assistant]):
        """initializer

        Args:
            assistants (typing.List[Assistant]): Assitants to notify
        """
        super().__init__(name)
        self._assistants = AssistantGroup(assistants)

    def start(self, progress: ChartAccessor):
        return self._assistants.start(progress)

    def finish(self, progress: ChartAccessor):
        self._assistants.finish(progress)

    @abstractmethod
    def to_notify(self, progress: ChartAccessor) -> bool:
        raise NotImplementedError

    def assist(self, progress: ChartAccessor):
        if self.to_notify(progress):
            self._assistants.assist(progress)
    
    def reset(self):
        super().reset()
        self._assistants.reset()


class NotifierF(Notifier):
    
    def __init__(self, name: str, assistants: typing.List[Assistant], notify_f: typing.Callable):
        super().__init__(name, assistants)
        self._notify_f = notify_f
    
    def to_notify(self, progress: ChartAccessor) -> bool:
        return self._notify_f(learner)


class IterationNotifier(Notifier):

    def __init__(self, name: str, assistants: typing.List[Assistant], frequency: int):

        super().__init__(name, assistants)
        self._frequency = frequency
    
    def to_notify(self, progress: ChartAccessor) -> bool:
        return ((progress.iteration + 1) % self._frequency) == 0


class TrainerBuilder(object):

    def __init__(self):
        self.n_epochs = 1
        self.teacher = None
        self.validator = None
        self.tester = None
    
    def n_epochs(self, n_epochs: int):
        self.n_epochs = n_epochs
        return self

    def teacher(self, dataset: data_utils.Dataset, batch_size: int=2**5):
        self.teacher = Teacher(dataset, batch_size)
        return self

    def validator(self, dataset: data_utils.Dataset, batch_size: int=2**5):
        self.validator = Validator(dataset, batch_size)

    def tester(self, dataset: data_utils.Dataset, batch_size: int=2**5):
        self.tester = Validator(dataset, batch_size)

    def build(self):

        sub_teachers = []
        if self.teacher is not None:
            sub_teachers.append(self.teacher)
        if self.validator is not None:
            sub_teachers.append(self.validator)
        
        teachers = []
        if sub_teachers:
            teachers.append(Lesson(
                'Epoch',
                [self.teacher, self.validator], iterations=self.n_epochs
            ))
        

        if self.tester is not None:
            teachers.append(self.tester)
        
        return Lesson('Training', teachers)



# TODO: Add Standard teaching modules
# Trainer
# Validator
# 


# Main <- 
# Assistant
# Team <- 

# Lesson(
#     'Training',
#     Lesson( 
#         'Epoch',
#         [Trainer(), Validator()],
#         assistants=['ChartBar'],
#         iterations=10
#     ),
#     Validator(),
#     iterations=1
# )


# need to add the ability to specify iterations...


# class Builder <- build up specific types of teachers
# studying will depend on studying

# class Ability to run a full lesson

# 
