from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar
import typing
import pandas as pd
from sklearn import preprocessing
from torch.utils import data as data_utils
from tqdm import tqdm


class Progress(object):
    
    def __init__(self):
        
        self.df = pd.DataFrame()
    
    def accessor(self, name: str, n_iterations: int=None):
        
        return ProgressAccessor(self, name, n_iterations=n_iterations)
    
    def add_result(self, result: dict):
        
        cur = pd.DataFrame(result)
        self.df = pd.concat([self.df, cur], ignore_index=True)


# i like this better
# could be conflicts in naming
# Epoch Teacher Iteration
#       X       0
# Epoch   Epoch/Iter  Epoch/Teacher Epoch/Teacher/Iter Epoch/Teacher/Results (other results)
# <>      1           Trainer       0                    ...
# 
# make sure the name does not have / in it
# this should make it easy to query


class ProgressAccessor(object):

    def __init__(self,  category: str, name: str, iter_name: str, progress: Progress, n_iterations: int= None, state: dict=None):
        """initializer

        Args:
            category (str): Name of the teacher category
            name (str): Name of the teacher
            iter_name (str): Name of the iterator
            progress (Progress): 
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
    def progress(self) -> Progress:
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
            ProgressAccesssor: Progress accessor with state 
        """
        state = {
            **self._state,
            **self.local_state
        }
        return ProgressAccessor(
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

# y, weight = MemberOut(ModFactory, ['weight'])
# MemberSet() <- sets the member based on the input
# probably just need these two

# would need to make it so if the necessary data is available
# it does not execute the module
# Shared() <- maybe i don't need this

T = TypeVar('T')

class Assistant(ABC):

    def __init__(self, name: str):
        self.name = name

    def start(self, learner, progress: ProgressAccessor, ctx: dict=None):
        pass

    def assist(self, learner, progress: ProgressAccessor, ctx: dict):
        pass

    def finish(self, learner, progress: ProgressAccessor, ctx: dict):
        pass


class AssistantGroup(object):

    def __init__(self, assistants: typing.List[Assistant]):

        self._assistants = assistants

    def start(self, learner, progress: ProgressAccessor, ctx: dict=None):
        
        ctx = ctx or dict()
        for assistant in self._assistants:
            ctx[assistant.name] = assistant.start(learner, progress, ctx.get(assistant.name))
        return ctx

    def assist(self, learner, progress: ProgressAccessor, ctx: dict):
        for assistant in self._assistants:
            ctx[assistant.name] = assistant.assist(learner, progress, ctx.get(assistant.name))

    def finish(self, learner, progress: ProgressAccessor, ctx: dict):
        for assistant in self._assistants:
            ctx[assistant.name] = assistant.finish(learner, progress, ctx.get(assistant.name))


class Trainer(Generic[T]):

    def __init__(self, name: str, assistants: typing.List[Assistant]=None):

        self._name = name
        # TODO: I don't really want to have to call the assistants for each
        # teacher.. 
        # self._assistants = AssistantGroup(assistants)

    def name(self) -> str:
        return self._name

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



class ProgressBar(Assistant):
    
    def start(self, learner, progress: ProgressAccessor, ctx: dict):
        ctx = ctx or dict(pbar=tqdm(total=progress.n_iterations))
        pbar: tqdm = ctx['pbar']
        pbar.refresh()
        pbar.reset()
        pbar.total = progress.n_iterations
        return ctx

    def finish(self, learner, progress: ProgressAccessor, ctx: dict):
        ctx['pbar'].close()

    def assist(self, learner, progress: ProgressAccessor, ctx: dict):
        pbar: tqdm = ctx['pbar']
        pbar.update(1)
        pbar.refresh()
        # ctx.pbar.set_description_str(course.cur.name)

        # ctx.pbar.set_postfix({
        #     "Epoch": f"{progress.cur_epoch} / {course.n_epochs}",
        #     **course.epoch_results.mean(axis=0).to_dict()
        # })


class Lesson(Trainer):

    def __init__(self, name: str, trainers: typing.List[Trainer], assistants: typing.List[Assistant]=None, iterations: int=1):

        self._name = name
        self._trainers = trainers
        self._assistants = AssistantGroup(assistants)
        self._iterations = iterations

    def teach(self, learner: T, progress: ProgressAccessor) -> Iterator:
        
        ctx = self._assistants.start(learner, progress)
        for i in range(self._iterations):
            for trainer in self._trainers:
                for result in trainer.teach(learner, progress):
                    self._assistants.assist(learner, progress, ctx)
                    yield result

        self._assistants.finish(learner, progress)


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

    def start(self, learner, progress: ProgressAccessor, ctx: dict=None):
        return self._assistants.start(learner, progress, ctx)

    def finish(self, learner, progress: ProgressAccessor, ctx: dict):
        self._assistants.finish(learner, progress, ctx)

    @abstractmethod
    def to_notify(self, learner, progress: Progress) -> bool:
        raise NotImplementedError

    def assist(self, learner, progress: ProgressAccessor, ctx: dict):
        if self.to_notify(learner, progress):
            self._assistants.assist(learner, progress, ctx)


class NotifierF(Notifier):
    
    def __init__(self, name: str, assistants: typing.List[Assistant], notify_f: typing.Callable):
        super().__init__(name, assistants)
        self._notify_f = notify_f
    
    def to_notify(self, learner, progress: ProgressAccessor) -> bool:
        return self._notify_f(learner, progress)


class IterationNotifier(Notifier):

    def __init__(self, name: str, assistants: typing.List[Assistant], frequency: int):

        super().__init__(name, assistants)
        self._frequency = frequency
    
    def to_notify(self, learner, progress: ProgressAccessor) -> bool:
        return ((progress.iteration + 1) % self._frequency) == 0

# Lesson(
#   'Epoch', [Team(trainer, assistants), Team(validator, assistants)]
# )

# TODO: Need to fix this.. Is Team? really necessary??
class Team(Trainer):

    def __init__(self, trainer: Trainer, assistants: typing.List[Assistant]):
        
        self._trainer = trainer
        self._assistants = AssistantGroup(assistants)
    
    @property
    def name(self) -> str:
        return self._trainer.name

    def teach(self, learner: T, progress: ProgressAccessor) -> Iterator:
        
        ctx = self._assistants.start(learner, progress, ctx)
        # progress = progress.accessor(self.name, n_iterations=len(self._trainer))
        for result in self._trainer.teach(learner, progress):
            
            self._assistants.assist(learner, progress, ctx)
            yield result
        self._assistants.finish(learner, progress, ctx)


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



# Lesson(
#     'Training',
#     Lesson( 
#         'Epoch',
#         [Trainer(), Validator()],
#         assistants=['ProgressBar'],
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
