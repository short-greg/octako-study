from abc import ABC, abstractmethod, abstractproperty
import dataclasses
from functools import partial
import typing
from sango.nodes import (
    STORE_REF, Action, Status, Fallback, Var, Shared, Tree, Sequence, Parallel, 
    action, condf, actionf, cond, loads_, success, loads, neg, task, task_, until, var_
)
from sango.vars import ref_
from torch.types import Storage

from tqdm import tqdm
from dataclasses import dataclass, is_dataclass, field
import pandas as pd
from torch.utils.data import DataLoader
import math
from tako.learners import Learner
import torch



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
class MaterialProgress:

    n_iterations: int=None
    cur_iter: int=None
    cur_epoch: int=None

    def to_dict(self):
        return dataclasses.asdict(self)
    
    @property
    def CUR_EPOCH_COL(self):
        return 'cur_epoch'

    def reset(self):
        self.n_iteration = None
        self.cur_iter = None
        self.cur_epoch = None


@dataclass
class Material:
    name: str
    data_iter: DataLoaderIter
    score_col: str
    progress: MaterialProgress = field(default_factory=partial(MaterialProgress, 0))


class Course:
    """[summary]
    """
    MATERIAL_COL = 'material'
    N_EPOCHS_COL = 'n_epochs'

    def __init__(self, epochs: int, materials: typing.List[Material]):
        
        self._materials: typing.Dict[str, Material] = {
            material.name: material
            for material in materials
        }
        self._cur = None
        self._completed = False
        self._df = pd.DataFrame()
        self._result_cols = set()
        self._n_epochs = epochs
    
    def validate(self, material: str=None):
        if self._completed is True:
            raise ValueError("Course has already been completed")
        if material is None and self._cur is None:
            raise ValueError("Current material has not been set.")
        if material is not None and material not in self._materials:
            raise ValueError("Material specified is not a teacher for the course")

    def switch_material(self, material: str):
        
        if not material in self._materials:
            raise ValueError(f"Teacher {material} is not a part of this course")
        if self._completed:
            raise ValueError("Course has already been completed.")
        self._cur = material

    @property
    def cur(self) -> Material:
        if self._cur is None:
            return None
        return self._materials[self._cur]
    
    @property
    def cur_iteration(self):
        return self.cur.progress.cur_iter

    def progress(self, material: str=None):
        material = material or self._cur
        return self._materials[material].progress
    
    def adv_epoch(self):
        self.validate()
        cur = self.cur
        cur.data_iter.reset()
        cur.progress.n_iterations = len(self.cur.data_iter)
        cur.progress.cur_iter = 0
        if cur.progress.cur_epoch is None:
            cur.progress.cur_epoch = 0 
        else: cur.progress.cur_epoch += 1

    def adv_iter(self, results: typing.Dict[str, torch.Tensor]):
        self.validate()
        cur = self.cur
        if cur.progress.n_iterations is None:
            cur.progress.n_iterations = len(cur.data_iter)
        if cur.progress.cur_iter is None:
            cur.progress.cur_iter = 0
        else:
            cur.progress.cur_iter += 1
        self._result_cols.update(
            set(results.keys())
        )
        cur.data_iter.adv()
        results = {k: v.detach().cpu().numpy() for k, v in results.items()}
        self._df = self._df.append({
            self.MATERIAL_COL: self._cur,
            'n_epochs': self._n_epochs,
            **self.cur.progress.to_dict(),
            **results
        }, ignore_index=True)

    def eval(self, material: str=None):
        if material not in self:
            raise ValueError(f'Material {material} is not a part of this course')
        material: Material = material or self._cur
        material_obj = self._materials[material]
        progress = material_obj.progress
        epoch = progress.cur_epoch
        df = self._df

        return df[(df[self.MATERIAL_COL] == material) & (df[progress.CUR_EPOCH_COL] == epoch)][
            material_obj.score_col
        ].mean()

    @property
    def data(self):
        self.validate()
        return self._materials[self._cur].data_iter.cur

    def pos(self, material: str=None):
        self.validate()
        material = material or self._cur
        if self.cur.progress is None:
            raise ValueError(f"Progress for material {self._cur} has not been started")
        return self._materials[self._cur].progress.cur_iter

    def complete(self):
        self._completed = True
        self._cur = None
    
    def reset(self):
        self._cur = None
        self._completed = False
        self._df = pd.DataFrame()
        self._result_cols = set()
        for k, material in self._materials.items():
            material.progress.reset()
            material.data_iter.reset()
    
    @property
    def epoch_results(self):
        progress = self.progress()
        return self._df[
            (self._df[self.MATERIAL_COL] == self._cur) & 
            (self._df[progress.CUR_EPOCH_COL] == progress.cur_epoch)
        ][list(self._result_cols)]

    @property
    def completed(self):
        return self._completed

    @property
    def results(self):
        return self._df
    
    def __contains__(self, material: str):
        return material in self._materials


class ShowProgress(Action):
    
    course = var_[Course]()

    def __init__(self, name: str):
        super().__init__(name)
        self._pbar = None
        self._cur = None

    def act(self):

        course: Course = self.course.val

        if course.completed:
            if not self._pbar is None: 
                self._pbar.close()
                self._pbar = None
            return Status.SUCCESS
        
        if course.cur is None:
            return Status.FAILURE
        
        progress = course.progress()

        if self._pbar is None:
            self._pbar = tqdm(total=progress.n_iterations)
        if progress.cur_iter == 0:
            self._pbar.reset()
        else:
            self._pbar.update(1)
        self._pbar.set_description_str(course.cur.name)

        # self.pbar.total = self.course.val.cur.n_lesson_iterations
        # self.pbar.set_postfix(lecture.results.mean(axis=0).to_dict())
        # pbar.refresh()
        return Status.SUCCESS


class Teach(Action):
    
    course = var_[Course]()
    learner = var_[Learner]()

    def __init__(self, material: str, name: str=''):
        super().__init__(name)
        self._material = material

    @abstractmethod
    def perform_action(self, x, t):
        pass

    def is_prepared(self):
        
        if self.learner.val is None or self.course.val is None:
            return False
        
        return self._material in self.course.val

    def act(self):
        course: Course = self.course.val
        course.switch_material(self._material)
        if course.data is None or course.progress().cur_epoch is None:
            course.adv_epoch()

        x, t = course.data
        result = self.perform_action(x, t)
        course.adv_iter(result)
        if course.data is None:
            return Status.SUCCESS
        return Status.RUNNING


class Train(Teach):

    def perform_action(self, x, t):
        return self.learner.val.learn(x, t)


class Validate(Teach):
    
    def perform_action(self, x, t):
        return self.learner.val.test(x, t)


class Trainer(Tree):

    score = var_[float]()
    scored_by = var_[str]()
    learner = var_[Learner]()

    VALIDATION = "Validation"
    TRAINING = "Training"
    TESTING = "Testing"

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
                score_testing = actionf('_score', "Testing")
                score_validation = actionf('_score', "Validation")
                score_training = actionf('_score', "Training")

    def __init__(
        self, name: str, n_epochs: int, train: DataLoaderIter, 
        validate: DataLoaderIter=None, test: DataLoaderIter=None,

    ):
        super().__init__(name)
        self._n_epochs = n_epochs
        self._train_iter = train
        self._validation_iter = validate
        self._test_iter = test

        self.course = Var(self._setup_course())

        self.training = Train(
            self.TRAINING, learner=Shared(self.learner), course=Shared(self.course)
        )
        self.validation = Validate(self.VALIDATION, learner=Shared(self.learner), course=Shared(self.course))
        self.testing = Validate(self.TESTING, learner=Shared(self.learner), course=Shared(self.course))

    def _setup_course(self):
        score_col = "Validation"
        materials = []
        materials.append(Material(self.TRAINING, self._train_iter, score_col))
        if self._validation_iter is not None:
            materials.append(Material(self.VALIDATION, self._validation_iter, score_col))
        if self._test_iter is not None:
            materials.append(Material(self.TESTING, self._test_iter, score_col))
        
        return Course(epochs=self._n_epochs, materials=materials)

    def output(self, val):
        return Status.SUCCESS

    def needs_validation(self):
        return self.validation.is_prepared()

    def needs_testing(self):
        return self.testing.is_prepared()
    
    def reset(self):
        super().reset()
        self.course.val.reset()

    def run(self):
        while True:
            status = self.tick()
            if status.done:
                break
        
        return self._cur_status
    
    def results(self):
        pass

    def _to_continue(self):
        progress: MaterialProgress = self.course.val.progress(self.TRAINING)
        if self._n_epochs > progress.cur_epoch + 1:
            return True
        return False
    
    def _complete(self):
        self.course.val.complete()
        return Status.SUCCESS
    
    def _score(self, material_name: str):
        if material_name not in self.course.val:
            return Status.FAILURE
        score = self.course.val.eval(material_name)
        if math.isnan(score):
            return Status.FAILURE
        
        self.score.val = score
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
        
        progress = course.progress()
        if pbar.is_empty():
            pbar.val = tqdm(total=progress.n_iterations)
        if progress.cur_iter == 1 or progress.cur_iter == 0:
            pbar.val.refresh()
            pbar.val.reset()
            pbar.val.total = progress.n_iterations
        else:
            pbar.val.update(1)

        pbar.val.set_description_str(course.cur.name)

        pbar.val.set_postfix({
            "Epoch": f"{progress.cur_epoch} / {self._n_epochs}",
            **course.epoch_results.mean(axis=0).to_dict()
        })

        pbar.val.refresh()
        return Status.RUNNING


# class Score(Action):

#     course = var_[Course]()
#     score = var_[float]()
#     scored_by = var_[str]()
#     teacher = var_[str]()

#     def __init__(self, score_last: bool=True):
#         super().__init__()
#         self._score_last = score_last

#     def act(self):
#         score = self.course.val.eval(self.teacher.val)

#         if math.isnan(score):
#             return Status.FAILURE
        
#         self.score.val = score
#         self.scored_by.val = self.teacher.val
#         return Status.SUCCESS
