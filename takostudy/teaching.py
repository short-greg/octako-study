from abc import ABC, abstractmethod, abstractproperty
import dataclasses
import typing
from sango.nodes import (
    STORE_REF, Action, Status, Fallback, Var, Shared, Tree, Sequence, Parallel, 
    action, condf, actionf, cond, loads_, loads, neg, task, task_, until, var_
)
from sango.vars import ref_
from torch.types import Storage

from tqdm import tqdm
from dataclasses import dataclass, is_dataclass
import pandas as pd
from torch.utils.data import DataLoader
import math


@dataclass
class Progress:

    name: str
    total_epochs: int
    cur_epoch: int = 0
    total_iterations: int = 0 
    cur_iteration: int = 0

    def to_dict(self):
        return dataclasses.asdict(self)


class ProgressRecorder(object):

    def __init__(self, default_epochs: int=1):
        self._progresses = {}
        self._cur_progress: str = None
        self._default_epochs = default_epochs
        self._completed = False
    
    def add(self, name: str, n_epochs: int=None, total_iterations: int=0, switch=True):
        if name in self._progresses:
            raise ValueError(f'Progress named {name} already exists.')
        
        n_epochs = n_epochs if n_epochs is not None else self._default_epochs

        self._progresses[name] = Progress(
            name, n_epochs, total_iterations=total_iterations
        )
        if switch: 
            self.switch(name)

    def switch(self, name: str):
        if name not in self._progresses:
            raise ValueError(f'Progress named {name} does not exist.')
        
        self._cur_progress = name
    
    def get(self, name: str):
        return self._progresses[name]
    
    @property
    def names(self):
        return set(self._progresses.keys())

    @property
    def cur(self) -> Progress:
        return self._progresses.get(self._cur_progress)
    
    def complete(self):
        self._completed = True
        
    @property
    def completed(self):
        return self._completed
    
    def adv_epoch(self, total_iterations=0):
        self.cur.cur_epoch += 1
        self.cur.total_iterations = total_iterations

    def adv_iter(self):
        self.cur.cur_iteration += 1


class Results(object):
    
    def __init__(self, score_name: str='Validation'):
        
        self.df = pd.DataFrame()
        self._progress_cols = set()
        self._result_cols = set()
        self._score_name = score_name
    
    def add_result(
        self, teacher: str, progress: Progress, 
        results: typing.Dict[str, float], score: float
    ):

        self._progress_cols.update(
            set(progress.to_dict().keys())
        )

        self._result_cols.update(
            set(results.keys())
        )

        self.df = self.df.append({
            self.teacher_col: teacher,
            self.score_col: score,
            **progress.to_dict(),
            **results
        }, ignore_index=True)
    
    @property
    def teacher_col(self):
        return "Teacher"
    
    @property
    def score_col(self):
        return self._score_name

    @property
    def result_cols(self):
        return set(self._result_cols)
    
    @property
    def progress_cols(self):
        return set(self._progress_cols)


class ShowProgress(Action):
    
    progress = var_()

    def act(self):
        pass


class Score(Action):

    results = var_[Results]()
    score = var_[float]()
    scored_by = var_[str]()

    def __init__(self, score_col: str, score_last: bool=True):
        super().__init__()
        self._score_col = score_col
        self._score_last = score_last

    def act(self):

        results: Results = self.results.val
        sub = results.df[results.df[results.teacher_col] == self._score_col]
        if self._score_last:
            # TODO: Change this so it does not use a specific field literal
            sub = sub[sub['cur_epoch'] == sub['cur_epoch'].max()]
        self.score.val = sub[results.score_col].mean()
        if math.isnan(self.score.val):
            return Status.FAILURE
        return Status.SUCCESS


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
        self._cur_iter = None
        self._finished = False
        self._cur = None
        self._pos = None
        self._is_start = True
        self.reset()
    
    def adv(self):
        if self._finished:
            raise StopIteration
        if self._is_start:
            self._is_start = False

        self._pos += 1
        try:
            self._cur = next(self._cur_iter)
        except StopIteration:
            self._cur = None
            self._finished = True

    @property
    def cur(self):
        return self._cur
        
    def reset(self, dataloader: DataLoader=None):

        self._dataloader = dataloader or self._dataloader
        self._cur_iter = iter(self._dataloader)
        self._finished = False
        self._cur = None
        self._pos = 0
        self.adv()
        self._pos -= 1
        self._is_start = True
    
    def __len__(self) -> int:
        return len(self._dataloader)
    
    @property
    def pos(self) -> int:
        return self._pos

    def is_end(self) -> bool:
        return self._finished

    def is_start(self):
        return self._is_start



class Teach(Action):
    
    results = var_[Results]()
    data_iter = var_[DataLoaderIter]()
    progress = var_[ProgressRecorder]()
    learner = var_()

    def _update_progress(self):
        n_iterations = len(self.data_iter)

        if not self.progress.val.contains(self._name):
            self.progress.val.add(
                self._name, total_iterations=len(self.data_iter), switch=True
            )
        elif self.data_iter.val.is_start():
            self.progress.val.adv_epoch(self._name, n_iterations)
        
        self.progress.val.switch(self._name)

    @abstractmethod
    def perform_action(self, x, t):
        pass

    def reset(self):
        super().reset()
        self.data_iter.val.reset()
    
    def is_prepared(self):
        return self.data_iter.val is not None

    def act(self):
        if self.data_iter.val is None:
            return Status.FAILURE
        
        if self.data_iter.val.is_end():
            return Status.SUCCESS
        self._update_progress()
        
        x, t = self.data_iter.val.cur
        result = self.perform_action(x, t)
        # TODO: update the score
        self.results.val.add_result(
            self._name, self.progress.val.cur, 
            result, score=0.0
        )
        self.data_iter.val.adv()
        self.progress.val.adv_iter()
        return Status.RUNNING


class Train(Teach):

    def perform_action(self, x, t):
        return self.learner.val.learn(x, t)


class Validate(Teach):
    
    def perform_action(self, x, t):
        return self.learner.val.test(x, t)


class Trainer(Tree):

    n_epochs = var_[int](1)
    batch_size = var_[int](32)
    validation_data = var_[DataLoaderIter]()
    training_data = var_[DataLoaderIter]()
    testing_data = var_[DataLoaderIter]()
    learner = var_()
    score = var_[float]()
    scored_by = var_[str]()

    @task
    class entry(Parallel):
        update_progress_bar = actionf('_update_progress_bar', STORE_REF)

        @task
        class execute(Sequence):

            @task
            @until
            @neg
            class epoch(Sequence):

                train = action('training')
                @task
                class validation(Fallback):
                    can_skip = condf('needs_validation') << loads(neg)
                    validate = action('validation')
                to_continue = condf('_to_continue', STORE_REF)
            
            @task
            class testing(Fallback):
                can_skip = condf('needs_testing') << loads(neg)
                test = action('testing')
    
            complete = actionf('_complete')

            @task
            class scoring(Fallback):
                score_testing = action('score_testing')
                score_validation = action('score_validation')
                score_training = action('score_training')


    def __init__(self, name: str):
        super().__init__(name)

        progress = ProgressRecorder(1)
        self._progress = Var(progress)
        self._results = Var(Results())
        self.validation = self._create_teacher("Training", Train, self.training_data)
        self.validation = self._create_teacher("Validation", Validate, self.validation_data)
        self.testing = self._create_teacher("Testing", Validate, self.testing_data)
        self.score_training = self._create_scorer("Training")
        self.score_validation = self._create_scorer("Validation")
        self.score_testing = self._create_scorer("Testing")
    
    def _create_teacher(self, name, teacher_cls, data):
        return teacher_cls(
            name,
            data_iter=Shared(data), progress=Shared(self._progress),
            learner=Shared(self.learner), batch_size=Shared(self.batch_size)
        )
    
    def _create_scorer(self, name):
        return Score(
            name, True, results=Shared(self.results), 
            score=Shared(self.score), scored_by=Shared(self.scored_by)
        )

    def reset(self):
        super().reset()
        self._completed = False
        self.learner.val = None

    def needs_validation(self):
        return self.validation.is_prepared()

    def needs_testing(self):
        return self.testing.is_prepared()
    
    def run(self, learner, to_evaluate: str='Validation'):
        self.reset()
        self.learner.val = learner
        status = None
        while True:
            status = self.tick()
            if status.done:
                break
        
        return status
    
    def results(self):
        pass

    def _to_continue(self, store: Storage):
        cur_epoch = store.get_or_add('n_epochs', 0)

        if cur_epoch.val < self.n_epochs.val:
            cur_epoch.val += 1
            return True
        return False
    
    def _complete(self):
        self._progress.val.complete()
        return Status.SUCCESS

    def _update_progress_bar(self, store: Storage):
        
        pbar = store.get_or_add('pbar', recursive=False)

        if self._progress.val.completed:
            if not pbar.is_empty(): 
                pbar.val.close()
                pbar.empty()
            return Status.SUCCESS
        
        if self._progress.val.cur is None:
            return Status.RUNNING
        if pbar.is_empty():
            pbar.val = tqdm(total=self._progress.val.cur.total_iterations)
        pbar.val.set_description_str(self._progress.val.cur.name)
        pbar.val.update(1)

        # self.pbar.total = lecture.n_lesson_iterations
        # self.pbar.set_postfix(lecture.results.mean(axis=0).to_dict())
        # pbar.refresh()
        return Status.RUNNING
