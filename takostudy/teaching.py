from abc import ABC, abstractmethod, abstractproperty
import dataclasses
from functools import partial
import typing
from omegaconf import ValidationError
from sango.nodes import (
    STORE_REF, Action, Status, Fallback, Var, Shared, Tree, Sequence, Parallel, 
    action, condf, actionf, cond, loads_, loads, neg, task, task_, until, var_
)
from sango.vars import ref_
from torch.types import Storage

from tqdm import tqdm
from dataclasses import dataclass, is_dataclass, field
import pandas as pd
from torch.utils.data import DataLoader
import math
from tako.learners import Learner


class ShowProgress(Action):
    
    progress = var_()

    def act(self):
        pass

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


@dataclass
class TeacherProgress:

    n_epochs: int
    n_iterations: int=None
    cur_iter: int=None
    cur_epoch: int=None

    def to_dict(self):
        return dataclasses.asdict(self)
    
    @property
    def CUR_EPOCH_COL(self):
        return 'cur_epoch'

    @property
    def N_EPOCHS_COL(self):
        return 'n_epochs'

@dataclass
class TeacherData:
    name: str
    data_iter: DataLoaderIter
    learner: Learner
    score_col: str
    progress: TeacherProgress = field(default_factory=partial(TeacherProgress, 0))


class Course:
    """[summary]
    """
    TEACHER_COL = 'teacher'

    def __init__(self, teacher_data: typing.List[TeacherData]):
        
        self._teacher_data = {
            datum.name: datum
            for datum in teacher_data
        }
        self._cur = None
        self._completed = False
        self._df = pd.DataFrame()
        self._result_cols = set()
    
    def validate(self):
        if self._completed is True:
            raise ValueError("Course has already been completed")
        if self._cur is None:
            raise ValueError("Current teacher has not been set.")

    def switch_teacher(self, teacher: str):
        
        if not teacher in self._teacher_data:
            raise ValueError(f"Teacher {teacher} is not a part of this course")
        if self._completed:
            raise ValueError("Course has already been completed.")
        self._cur = teacher

    @property
    def cur(self) -> TeacherData:
        if self._cur is None:
            return None
        return self._teacher_data[self._cur]

    def adv_epoch(self):
        self.validate()
        self.cur.data_iter.reset()
        self.cur.progress.n_iterations = len(self.cur.data_iter)
        self.cur.progress.cur_iter = 0
        if self.cur.progress.cur_epoch is None:
            self.cur.progress.cur_epoch = 0 
        else: self.cur.progress.cur_epoch += 1

    def adv_iter(self, results: dict):
        self.validate()
        if self.cur.progress.n_iterations is None:
            self.cur.progress.n_iterations = 0
        else:
            self.cur.progress.n_iterations += 1

        self._result_cols.update(
            set(results.keys())
        )
        self.cur.data_iter.adv()
        self._df = self.df.append({
            self.TEACHER_COL: self._cur,
            **self.cur.progress.to_dict(),
            **results
        }, ignore_index=True)

    def eval(self, teacher: str=None):
        if self._cur is None and teacher is None:
            raise ValueError(f'Must set the current teacher or pass in teacher name')
        teacher = teacher or self._cur
        teacher_data = self._teacher_data[teacher]
        progress = teacher_data.progress
        epoch = progress.cur_epoch
        df = self._df

        return df[(df[self.TEACHER_COL] == teacher) & (df[progress.CUR_EPOCH_COL] == epoch)][
            teacher_data.score_col
        ].mean()

    @property
    def learner(self):
        self.validate()
        return self._teacher_data[self._cur].learner

    @property
    def data(self):
        self.validate()
        return self._teacher_data[self._cur].data_iter.cur

    def pos(self):
        self.validate()
        if self.cur.progress is None:
            raise ValueError(f"Progress for teacher {self._cur} has not been started")
        return self._teacher_data[self._cur].teacher_progress.cur_iter

    def complete(self):
        self._completed = True
        self._cur = None
    
    @property
    def completed(self):
        return self._completed

    @property
    def df(self):
        return self._df

    def progress(self, teacher: str) -> TeacherProgress:
        return self._teacher_data[teacher].progress
    
    def __contains__(self, teacher: str):
        return teacher in self._teacher_data


class Score(Action):

    course = var_[Course]()
    score = var_[float]()
    scored_by = var_[str]()
    teacher = var_[str]()

    def __init__(self, score_last: bool=True):
        super().__init__()
        self._score_last = score_last

    def act(self):
        score = self.course.val.eval(self.teacher.val)

        if math.isnan(score):
            return Status.FAILURE
        
        self.score.val = score
        self.scored_by.val = self.teacher.val
        return Status.SUCCESS


class Teach(Action):
    
    course = var_[Course]()

    @abstractmethod
    def perform_action(self, x, t):
        pass

    def reset(self):
        super().reset()

    def is_prepared(self):
        return self._name in self.course.val

    def act(self):
        course: Course = self.course.val
        course.switch_teacher(self._name)
        if course.data is None or course.cur.progress.cur_epoch is None:
            course.adv_epoch()

        x, t = course.data
        result = self.perform_action(x, t)
        course.adv_iter(result)

        if course.data is None:
            return Status.SUCCESS
        return Status.RUNNING


class Train(Teach):

    def perform_action(self, x, t):
        return self.course.val.learner.learn(x, t)


class Validate(Teach):
    
    def perform_action(self, x, t):
        return self.course.val.learner.test(x, t)


class Trainer(Tree):

    course = var_[Course]()
    n_epochs = var_[int](1)
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
                to_continue = condf('_to_continue')
            
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

        self.training = Train("Training", course=self.course)
        self.validation = Validate("Validation", course=self.course)
        self.testing = Validate("Test", course=self.course)
        self.score_training = Score(
            'Training Scorer',
            teacher="Training", score=Shared(self.score), 
            scored_by=Shared(self.scored_by), course=Shared(self.course)
        )
        self.score_validation = Score('Validation Scorer', teacher="Validation", score=Shared(self.score), 
            scored_by=Shared(self.scored_by), course=Shared(self.course))
        self.score_testing = Score('Testing Scorer', teacher="Validation", score=Shared(self.score), 
            scored_by=Shared(self.scored_by), course=Shared(self.course))

    def needs_validation(self):
        return self.validation.is_prepared()

    def needs_testing(self):
        return self.testing.is_prepared()
    
    def run(self, course, to_evaluate: str='Validation'):
        self.reset()
        status = None
        self.course.val = course
        while True:
            status = self.tick()
            if status.done:
                break
        
        return status
    
    def results(self):
        pass

    def _to_continue(self):
        progress: TeacherProgress = self.course.val.progress("Training")
        if progress.n_epochs > progress.cur_epoch + 1:
            return True
        return False
    
    def _complete(self):
        self.course.val.complete()
        return Status.SUCCESS

    def _update_progress_bar(self, store: Storage):
        
        pbar = store.get_or_add('pbar', recursive=False)

        course: Course = self.course.val

        if course.completed:
            if not pbar.is_empty(): 
                pbar.val.close()
                pbar.empty()
            return Status.SUCCESS
        
        if course.cur is None:
            return Status.RUNNING
        
        progress = course.cur.progress

        if pbar.is_empty():
            pbar.val = tqdm(total=progress.n_iterations)
        pbar.val.set_description_str(course.cur.name)
        pbar.val.update(1)

        # self.pbar.total = lecture.n_lesson_iterations
        # self.pbar.set_postfix(lecture.results.mean(axis=0).to_dict())
        # pbar.refresh()
        return Status.RUNNING
