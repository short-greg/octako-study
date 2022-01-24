from abc import abstractmethod
import dataclasses
import typing

from torch import neg_
from torchaudio import datasets
from sango.nodes import (
    STORE_REF, Action, Status, Fallback, Var, Shared, Tree, Sequence, Parallel, 
    action, condf, actionf, cond, loads_, loads, neg, task, task_, until, var_
)
from sango.vars import ref_
from torch.types import Storage

from tqdm import tqdm
from dataclasses import dataclass, is_dataclass
import pandas as pd
from torch.utils.data import  DataLoader, Dataset


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


class Results:
    
    def __init__(self):
        
        self.df = pd.DataFrame()
        self._progress_cols = set()
        self._result_cols = set()
    
    def add_result(self, teacher: str, progress: Progress, results: typing.Dict[str, float]):

        self._progress_cols.update(
            set(progress.to_dict().keys())
        )

        self._result_cols.update(
            set(results.keys())
        )

        self.df = self.df.append({
            self.teacher_col: teacher,
            **progress.to_dict(),
            **results
        }, ignore_index=True)
    
    @property
    def teacher_col(self):
        return "Teacher"
    
    @property
    def result_cols(self):
        return set(self._result_cols)
    
    @property
    def progress_cols(self):
        return set(self._progress_cols)


class Teach(Action):
    
    results = var_()
    dataset = var_()
    progress = var_()
    batch_size = var_()
    learner = var_()

    def __init__(self, name: str):
        super().__init__(name)
        self._iter = None

    def _setup_progress(self):
        self._iter = iter(DataLoader(
            self.dataset.val, self.batch_size.val, shuffle=True
        ))
        n_iterations = len(self._iter)
        if self._name in self.progress.val.names:
            self.progress.val.switch(self._name)
            self.progress.val.adv_epoch(n_iterations)
        else:
            self.progress.val.add(
                self._name, total_iterations=n_iterations, switch=True
            )
    
    def _setup_iter(self):

        is_setup = self._iter is not None
        if not is_setup:
            self._setup_progress()
        else:
            self.progress.val.adv_iter()

    @abstractmethod
    def perform_action(self, x, t):
        pass

    def reset(self):
        super().reset()
        self._iter = None
    
    def is_prepared(self):
        return self.dataset.val is not None

    def act(self):
        if self.dataset.val is None:
            return Status.FAILURE
        self._setup_iter()

        try:
            x, t = next(self._iter)
        except StopIteration:
            self._iter = None
            return Status.SUCCESS
        
        result = self.perform_action(x, t)

        self.results.val.add_result(self._name, self.progress.val.cur, result)
        return Status.RUNNING


class Train(Teach):

    def perform_action(self, x, t):
        return self.learner.val.learn(x, t)


class Validate(Teach):
    
    def perform_action(self, x, t):
        return self.learner.val.test(x, t)


class Trainer(Tree):

    n_batches = var_(1)
    batch_size = var_(32)
    validation_dataset = var_()
    training_data = var_()
    testing_data = var_()
    validation_data = var_()
    learner = var_()

    @task
    class entry(Parallel):
        update_progress_bar = actionf('_update_progress_bar', STORE_REF)

        @task
        class execute(Sequence):
            @task
            @until
            class epoch(Sequence):

                train = action('train')
                @task
                class validation(Fallback):
                    can_skip = condf('needs_validation') << neg
                    validate = action('validate')
                stop = condf('stop')
                
            @task
            class testing(Sequence):
                can_skip = neg() << condf('needs_validation')
                test = action('test')
    
            complete = actionf('_complete')

    def __init__(self, name: str):
        super().__init__(name)
        self._progress = Var(ProgressRecorder())
        self._results = Var(Results())

        self.training = Train(
            "Trainer", results=Shared(self._results), 
            dataset=Shared(self.training_data), progress=Shared(self._progress),
            learner=Shared(self.learner), batch_size=Shared(self.batch_size)
        )
        self.validation = Validate(
            "Validator", results=Shared(self._results), 
            dataset=Shared(self.validation_data), progress=Shared(self._progress),
            learner=Shared(self.learner), batch_size=Shared(self.batch_size)
        )
        self.testing = Validate(
            "Tester", results=Shared(self._results), 
            dataset=Shared(self.testing_data), progress=Shared(self._progress),
            learner=Shared(self.learner), batch_size=Shared(self.batch_size)
        )

    def reset(self):
        self._completed = False

    def needs_validation(self):
        return self.validation.is_prepared()

    def needs_testing(self):
        return self.testing.is_prepared()

    def _to_continue(self, store: Storage):
        cur_batch = store.get_or_add('cur_batch', 0)

        if cur_batch.val < self.n_batches.val:
            cur_batch.val += 1
            return True

        return False
    
    def _complete(self):
        self._completed = True

    def _update_progress_bar(self, store: Storage):
        
        pbar = store.get_or_add('pbar', recursive=False)

        if self._progress.completed:
            if not pbar.is_empty(): 
                pbar.close()
                pbar.empty()
            return Status.SUCCESS
        
        if self._progress.cur is None:
            return Status.RUNNING
        if pbar.is_empty():
            pbar.val = tqdm(total=self._progress.cur.total_iterations)
        pbar.val.set_description_str(self._progress.cur.name)
        pbar.val.update(1)

        # self.pbar.total = lecture.n_lesson_iterations
        # self.pbar.set_postfix(lecture.results.mean(axis=0).to_dict())
        # pbar.refresh()
        return Status.RUNNING
