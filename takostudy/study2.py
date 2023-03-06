# 1st party
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import InitVar, dataclass, fields, field
from dataclasses import asdict, dataclass
import typing
import typing
from pytest import param
from itertools import chain
from datetime import datetime
import inspect
from uuid import uuid4
import os
import json

# 3rd party
from numpy import isin
import optuna
import optuna
import hydra
import hydra
from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig
from omegaconf import OmegaConf
import functools
import pandas as pd
from pandas import DataFrame

# my libraries
from octako.components import Learner
from octako import teach
import octako


PDELIM = "/"



class TrialSelector(ABC):

    def __init__(self, name, default):

        self._default = default
        self._name = name
    
    @property
    def name(self):
        return self._name

    @property
    def default(self):
        return self._default

    def select(self, trial: optuna.Trial=None, path: str='', best: dict=None):

        if trial:
            return self.suggest(trial, path)
        return self.update_best(best, path)

    def cat_path(self, path: str=None, sub: str=None):
        full_path = self._name
        if path is not None:
            full_path = f'{path}/{full_path}'
        if sub is not None:
            full_path = f'{full_path}_{sub}' 
        return full_path

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @abstractmethod
    def suggest(self, trial: optuna.Trial, path: str):
        raise NotImplementedError

    @abstractmethod
    def update_best(self, best_val: dict, path: str=None):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, **params: dict):
        raise NotImplementedError
    

class Default(TrialSelector):

    def __init__(self, name: str, val):
        super().__init__(name, val)
        self._val = val
    
    def suggest(self, trial: optuna.Trial, path: str):
        return self._val
    
    def update_best(self, best_val: dict, path: str=None):
        return self.default

    def to_dict(self):
        return dict(
            name=self.name, val=self._val
        )

    @classmethod
    def from_dict(cls, **params: dict):
        return cls(params['name'], params["val"])


class Int(TrialSelector):

    def __init__(self, name: str, low: int, high: int, default: int=0):
        super().__init__(name, default)

        self._low = low
        self._high = high
    
    @property
    def low(self):
        return self._low
    
    @property
    def high(self):
        return self._high
    
    def suggest(self, trial: optuna.Trial, path: str):
        return trial.suggest_int(
            self.cat_path(path) , self._low, self._high
        )

    def update_best(self, best_val: dict, path: str=None):
        return best_val.get(self.cat_path(path), self.default)

    def to_dict(self):
        return dict(
            name=self.name, low=self._low, high=self._high, default=self._default
        )

    @classmethod
    def from_dict(cls, **params: dict):
        return cls(
            params['name'], 
            low=params['low'], high=params['high'], 
            default=params.get('default')
        )


class ExpInt(TrialSelector):

    def __init__(self, name: str, low: int, high: int, base: int=10, default: int=0, to_int: bool=False):
        """Suggests an int which gets exponentiated to the value specified by base. 
        Outputs a float by default, but can postprocess to be an int

        Args:
            name (str): Name of the trial selector
            low (int): The lower bound on the int that is suggested
            high (int): The upper bound on the int that is suggested
            base (int, optional): The value to exponentiate by. Defaults to 10.
            default (int, optional): The default integer. Defaults to 0.
            to_int (bool, optional): Whether to postprocess to an integer value. Defaults to False.
        """
        super().__init__(name, default)

        self._low = low
        self._high = high
        self._base = base
        self.to_int = to_int

    def postprocess(self, value: float):
        if self.to_int:
            return int(value)
        return value
    
    @property
    def default(self):
        return self.postprocess(self._base ** self._default)

    def suggest(self, trial: optuna.Trial, path: str):
        return self.postprocess(self._base ** trial.suggest_int(
            self.cat_path(path) , self._low, self._high
        ))

    def update_best(self, best_val: dict, path: str=None):
        return self.postprocess(
            self._base ** best_val.get(self.cat_path(path), self.default)
        )

    def to_dict(self):
        return dict(
            name=self.name, low=self._low, high=self._high, base=self._base, default=self._default
        )
    
    @classmethod
    def from_dict(cls, **params: dict):
        return cls(
            params['name'], 
            low=params['low'], high=params['high'], 
            base=params.get('base', None), default=params.get('default')
        )


class Bool(TrialSelector):

    def __init__(self, name: str, default: bool=True):
        super().__init__(name, default)

    def suggest(self, trial: optuna.Trial, path: str):

        return bool(trial.suggest_discrete_uniform(
            self.cat_path(path) , 0, 1, 1
        ) > 0.5)

    def update_best(self, best_val: dict, path: str=None):
        val = best_val.get(self.cat_path(path), self.default)
        return bool(val)

    def to_dict(self):
        return dict(
            name=self.name, default=self._default
        )

    @classmethod
    def from_dict(cls, **params: dict):
        return cls(params['name'], default=bool(params.get('default')))


class Float(TrialSelector):

    def __init__(self, name: str, low: float=0., high: float=1., default: float=1.0):
        super().__init__(name, default)
        self._low = low
        self._high = high

    def suggest(self, trial: optuna.Trial, path: str):

        return trial.suggest_uniform(
            self.cat_path(path) , self._low, self._high
        )

    def update_best(self, best_val: dict, path: str=None):
        val = best_val.get(self.cat_path(path), self.default)
        return val

    def to_dict(self):
        return dict(
            name=self.name, low=self._low, high=self._high, default=self._default
        )

    @classmethod
    def from_dict(cls, **params: dict):
        return cls(params['name'], low=params['low'], high=params['high'], default=params.get('default'))


class Categorical(TrialSelector):

    def __init__(self, name: str, categories: typing.List[str], default: str):
        super().__init__(name, default)
        self._categories = categories

    def suggest(self, trial: optuna.Trial, path: str):
        return trial.suggest_categorical(
            self.cat_path(path), self._categories
        )

    def update_best(self, best_val: dict, path: str=None):
        return best_val.get(self.cat_path(path), self.default)

    
    def to_dict(self):
        return dict(
            name=self.name, categories=self._categories, default=self._default
        )
    
    @classmethod
    def from_dict(cls, **params: dict):
        return cls(params['name'], params["categories"], default=params.get('default'))


class ConditionalCategorical(TrialSelector):

    def __init__(self, name: str, categories: typing.Dict[str, str], default: str):
        super().__init__(name, default)
        self._categories = categories
    
    def _get_paths(self, path):
        base_path = self.cat_path(path)
        sub_path = self.cat_path(path, "sub")
        return base_path, sub_path
    
    def suggest(self, trial: optuna.Trial, path: str):
        base_path, sub_path = self._get_paths(path)
        base = trial.suggest_categorical(base_path, list(self._categeries.keys()))
        sub_categories = self._categories[base]
        sub = trial.suggest_categorical(sub_path, sub_categories)
        return (base, sub)
    
    def update_best(self, best_val: dict, path: str=None):
        base_path, sub_path = self._get_paths(path)
        if base_path not in best_val or sub_path not in best_val:
            return self.default
        return best_val[base_path], best_val[sub_path] 

    def to_dict(self):
        return dict(
            name=self.name, categories=self._categories, default=self._default
        )
    
    @classmethod
    def from_dict(cls, **params: dict):
        return cls(params['name'], params["categories"], default=params.get('default'))


class Non(object):

    @classmethod
    def from_dict(cls, **params: dict):
        return params['value']


class LogUniform(TrialSelector):

    def __init__(self, name: str, low: int, high: int, default: int):
        super().__init__(name, default)
        self._low = low
        self._high = high

    def suggest(self, trial: optuna.Trial, path: str):
        return trial.suggest_loguniform(
            self.cat_path(path), self._low, self._high
        )

    def update_best(self, best_val: dict, path: str=None):
        return best_val.get(self.cat_path(path), self.default)

    def to_dict(self):
        return dict(
            name=self.name, low=self._low, high=self._high, default=self._default
        )

    @classmethod
    def from_dict(cls, **params: dict):
        return cls(params['name'], params["low"], params["high"], default=params.get('default'))


def prepend_dict_names(prepend_with: str, d: dict):

    return {
        f'{prepend_with}/{key}': val 
        for key, val in d.items()
    }


class Array(TrialSelector):

    def __init__(
        self, name: str, low: int, high: int, 
        params: typing.Dict[(str, TrialSelector)], 
        default=typing.List
    ):
        super().__init__(name, default)

        self._low = low
        self._high = high
        self._params = params

    def suggest(self, trial: optuna.Trial, path: str):
        params = []
        path = self.cat_path(path)
        
        size = trial.suggest_int(
            path + '/size',
            self._low,
            self._high
        )

        # TODO: HOW TO DEAL WITH THIS??
        for i in range(size):
            params.append({})
            for k, v in self._params.items():
                params[i][k] = v.suggest(
                    trial, f'{path}/{i}'
                )
        return params

    def _get_param_by_name(self, name: str):

        for k, param in self._params.items():
            if param._name == name:
                return k, param

    def update_best(self, best_val: dict, path: str=None):
        
        path = self.cat_path(path)
        size_path = f'{path}/size'
        if size_path not in best_val:
            return self.default
    
        size = best_val[f'{path}/size']
        result = []
        for i in range(size):
            i_str = str(i)
            result.append({})

            cur_path = f'{path}/{i}'
            # cur_params = best_val[cur_path]
            # cur_params = best[i_str]
            # for k in cur_params.keys():
            for k, v in self._params.items():
                v = v.update_best(best_val, cur_path)
                result[i][k] = v
                # key, param = self._get_param_by_name(k)
                # v = param.update_best(cur_params)
        
        return result
    
    def to_dict(self):
        
        result = dict()

        for name, selector in self._params.items():
            cur_result = selector.to_dict()
            result[name] = cur_result
        return result
    
    @classmethod
    def from_dict(cls, **params: dict):
        selectors: typing.Dict[str, TrialSelector] = {}
        
        for k, p in params['sub'].items():
            selectors[k] = ParamMap[p["type"]].from_dict(**p)

        return cls(
            params['name'], low=params["low"], high=params["high"], params=selectors,
            default=params.get('default')
        )


ParamMap: typing.Dict[str, TrialSelector] = {
    "Array": Array,
    "Int": Int,
    'ExpInt': ExpInt, 
    "LogUniform": LogUniform,
    "Float": Float,
    "Categorical": Categorical,
    "ConditionalCategorical": ConditionalCategorical,
    "Bool": Bool,
    "Default": Default,
    "Non": Non
}


class Params(object):

    def __init__(self, params: typing.Dict):
        """initializer
        Args:
            params (typing.Dict): the parameters for the model
        """
        self._params = params

    def suggest(self, trial, path: str='') -> 'Params':
        params = {}
        for k, v in self._params.items():
            if isinstance(v, TrialSelector):
                params[k] = v.suggest(trial, path)
            else:
                params[k] = v

        return Params(params)
    
    def default(self) -> 'Params':
        result = {}
        for k, v in self._params.items():
            if isinstance(v, TrialSelector):
                result[k] = v.default
            else:
                result[k] = v

        return Params(result)

    def to_dict(self) -> typing.Dict:
        return self._params
    
    def update(self, **kwargs):
        self._params.update(**kwargs)

    def update_sub(self, sub: typing.List[str], **kwargs):
        """update sub dictionary

        Args:
            sub (typing.List[str]): The sub parameter dicts to update
        """
        cur = self._params
        for k in sub:
            cur = cur[k]
        cur.update(**kwargs)

    def get_sub(self, key: str):
        return Params(self._params[key])


def convert_params(trial_param_config: typing.Dict) -> typing.Dict:
    result = {}
    
    for k, v in trial_param_config.items():
        if k.lower() == 'type':
            continue
        if isinstance(v, dict) and 'type' in v and v['type'].lower() == 'sub':
            result[k] = convert_params(v)
        elif isinstance(v, dict) and 'type' in v:
            result[k] = ParamMap[v['type']].from_dict(**v)
        else:
            result[k] = v
    return result 


def to_params(trial_param_config: typing.Dict) -> Params:
    """Convert 

    Args:
        trial_params (typing.Dict): _description_

    Returns:
        Params: _description_
    """

    return Params(
        convert_params(trial_param_config)
    )


def asdict_shallow(obj):
    return dict((field.name, getattr(obj, field.name)) for field in fields(obj))


def is_trial_selector(value) -> bool:    
    return (
        isinstance(value, dict) or isinstance(value, DictConfig)
        and 'type' in value
    )


class LearnerFactory(object):

    @abstractmethod
    def __call__(self, params: Params) -> octako.Learner:
        pass


class LMFactory(LearnerFactory):

    def __init__(self, lm_cls: typing.Type[octako.zen.LearningMachine], device):
        self.lm_cls = lm_cls
        self.device = device

    def __call__(self, params: Params) -> octako.zen.LearningMachine:
        lm = self.lm_cls(**params.to_dict())
        lm.to(self.device)
        return lm


class TeacherFactory(object):

    @abstractmethod
    def __call__(self, params: Params) -> teach.Teacher:
        pass


EXPERIMENT_COL = 'Experiment'
DATASET_COL = 'Dataset' 
TRIAL_COL = 'Trial'
SCORE_COL = 'Score'
DATE_COL = "Date"
TIME_COL = "Time"
TEST_COL = "Test Type"
MACHINE_COL = "Machine Type"
STUDY_ID = "Study ID"
EXPERIMENT_ID = "Experiment ID"
DESCRIPTION_COL = "Description"


@dataclass
class Experiment(object):

    name: str
    score: float
    chart: teach.Chart
    description: str
    trial_name: str
    dataset: str
    is_validation: bool
    research_id: teach.ResearchID
    study_builder: 'StudyBuilder'

    def __post_init__(self):
        self.experiment_id = uuid4()
        date_, time = self.datetime_to_str(self.research_id.experiment_date)
        self.date = date_
        self.time = time

    def datetime_to_str(self, experiment_date: datetime):
        return experiment_date.strftime("%Y/%m/%d"),  experiment_date.strftime("%H:%M:%S")
    
    def test_type_to_str(self, is_validation: bool):
        if is_validation:
            test_type = "Validation"
        else: test_type = "Test"
        return test_type

    def summarize(self):
        # TODO: Implement a better way to do the averaging
        results = self.chart.df
        cur_result = results[['Teacher', 'Epoch', 'loss', 'validation']].groupby(
            by=['Teacher', 'Epoch']
        ).mean().reset_index()
        cur_result[[EXPERIMENT_ID, EXPERIMENT_COL, STUDY_ID, DATASET_COL, TRIAL_COL, DATE_COL, TIME_COL, TEST_COL]] = [
            self.research_id.experiment_id, 
            self.research_id.experiment_name, 
            self.research_id.study_name, 
            self.dataset, 
            self.trial_name, 
            self.date, 
            self.time, 
            self.test_type_to_str(self.is_validation)
        ]    
        return cur_result


class ExperimentLog(object):

    def __init__(self, maximize: bool):
        self.maximize = maximize
        self._experiments: typing.List[Experiment] = []
    
    def add(self, experiment: Experiment):
        self._experiments.append(experiment)

    def best(self) -> typing.Tuple[int, Experiment]:
        if self.maximize:
            return functools.reduce(
                lambda x, y: x if x[1].score > y[1].score else y,
                enumerate(self._experiments)
            )
        return functools.reduce(
            lambda x, y: x if x[1].score < y[1].score else y,
            enumerate(self._experiments)
        )
    
    def summarize(self):
        summaries = [experiment.summarize() for experiment in self._experiments]
        return pd.concat(summaries)


class StudyBuilder(ABC):
    """Store experiment params in a single dataclass to organize them
    effectively
    Subclasses must be dataclasses
    """

    @abstractmethod
    def learner(self) -> Learner:
        pass
    
    @abstractmethod
    def teacher(self) -> teach.Teacher:
        pass

    @abstractmethod
    def dataset_loader(self) -> teach.DatasetLoader:
        pass
    
    def suggest(self, trial: optuna.Trial, path: str='') -> 'StudyBuilder':
        kwargs = {}
        for k, v in asdict(self).items():
            if isinstance(v, TrialSelector):
                kwargs[k] = v.suggest(trial, path)
            elif isinstance(v, Params):
                kwargs[k] = v.suggest(trial, path)
            else: kwargs[k] = v
        return self.__class__(**kwargs)

    def default(self) -> 'StudyBuilder':
        kwargs = {}
        for k, v in asdict(self).items():
            if isinstance(v, TrialSelector):
                kwargs[k] = v.default
            elif isinstance(v, Params):
                kwargs[k] = v.default()
            else: kwargs[k] = v
        return self.__class__(**kwargs)

    def dict_update(self, update_params: typing.Dict):
        kwargs = asdict(self)
        kwargs.update(update_params)
        return self.__class__(**kwargs)


@dataclass
class StandardStudyBuilder(StudyBuilder):

    lesson_name: str
    batch_size: int
    n_epochs: int
    maximize: bool
    learner_factory: LearnerFactory
    learner_params: Params
    dataset_loader_: teach.DatasetLoader

    def learner(self) -> Learner:
        return self.learner_factory(self.learner_params)
    
    def teacher(self) -> teach.MainTeacher:
        return teach.MainTeacher(
            self.lesson_name,
            self.batch_size, self.n_epochs, self.dataset_loader(),
            loss_window=30
        )

    def dataset_loader(self) -> teach.DatasetLoader:
        return self.dataset_loader_


class OptunaStudy(object):

    best_idx = 'BEST'
    final_idx = 'FINAL'

    def __init__(
        self, study_builder: StudyBuilder, maximize: bool,
        research: str='', study: str='', description: str=''
        # learner_factory: LearnerFactory, 
        # dataset_loader: teach.DatasetLoader,
        # learner_params: Params=None,
        # batch_size: int=64, n_epochs: int=10, base_name: str='', 
        # maximize: bool=False, device: str='cpu'
    ):
        """initializer

        Args:
            learner_factory (LearnerFactory): 
            learner_params (Params, optional): . Defaults to None.
            batch_size (int, optional): . Defaults to 64.
            n_epochs (int, optional): . Defaults to 10.
            base_name (str, optional): . Defaults to ''.
            maximize (bool, optional): . Defaults to False.
            device (str, optional): . Defaults to 'cpu'.
        """
        self._maximize = maximize
        self._study_builder = study_builder
        self._direction = self.get_direction(self._maximize)
        self.research = research
        self.description = description
        self.study = study
        self.study_id = uuid4()
        # TODO: generalize this and add a class method to simplify creating
        # standard studies
        # self._learner_factory = learner_factory
        # self._learner_params = learner_params
        # self._teacher_factory: TeacherFactory = teach.MainTeacher
        # self._teacher_params = Params(
        #     {'batch_size': batch_size, 'n_epochs': n_epochs, 'dataset_loader': dataset_loader}
        # )
        # self._base_name = base_name
        # self._device = device

    @classmethod
    def build_standard(
        cls, lesson_name: str, batch_size: int, n_epochs: int, learner_factory: 'LMFactory', 
        learner_params: Params, dataset_loader: teach.DatasetLoader, maximize: bool
    ):
        """Build an OptunaStudy

        Args:
            batch_size (int): _description_
            n_epochs (int): _description_
            learner_factory (LearnerFactory): _description_
            learner_params (Params): _description_
            dataset_loader (teach.DatasetLoader): _description_

        Returns:
            _type_: _description_
        """
        return OptunaStudy(
            StandardStudyBuilder(
                lesson_name, batch_size, n_epochs, maximize, learner_factory,  
                learner_params, dataset_loader
            ), maximize
        )

    @staticmethod
    def get_direction(to_maximize):
        return optuna.study.StudyDirection.MAXIMIZE if to_maximize else optuna.study.StudyDirection.MINIMIZE

    # def update_teacher(self, teacher_factory: 'TeacherFactory', params: Params):
    #     """Update the teacher factory to use

    #     Args:
    #         teacher_factory (TeacherFactory): 
    #         params (Params): 
    #     """
    #     self._teacher_factory = teacher_factory
    #     self._teacher_params = params
    
    def get_objective(self, experiment_name: str, experiments: ExperimentLog) -> typing.Callable:
        cur: int = 0

        def objective(trial: optuna.Trial):
            nonlocal cur
            nonlocal experiments

            study_builder = self._study_builder.suggest(trial)
            teacher = study_builder.teacher()
            learner = study_builder.learner()

            score, chart = teacher.validate(learner)
            experiments.add(Experiment(
                score, chart, self.description, 
                str(cur), True, 
                teach.ResearchID(
                    self.research, self.study, experiment_name,
                    self.study_id
                ), 
                study_builder)
            )

            cur += 1
            return score
        return objective
    
    def run_trials(self, n_trials: int, experiment_name: str='') -> ExperimentLog:

        experiment_log = ExperimentLog(self._maximize)
        optuna_study = optuna.create_study(direction=self._direction)
        objective = self.get_objective(experiment_name, experiment_log)
        optuna_study.optimize(objective, n_trials)

        _, best = experiment_log.best()
        teacher = best.study_builder.teacher()
        learner = best.study_builder.learner()

        score, chart = teacher.train(learner)
        final = Experiment("best", score, chart, best)
        experiment_log.add(final)
        return final, experiment_log


class Config:

    name: str  # name fo config to choose
    path: str = 'blueprints/'


class HydraStudyConfig(object):

    def __init__(self, name, dir='./', overrides: dict=None):
    
        overrides = overrides or {}
        overrides_list = {f'{k}={v}' for k, v in overrides.items()}
        initialize_config_dir(dir)

        # overrides=["db=mysql", "db.user=me"])
        cfg = compose(config_name=name, overrides=overrides_list) 
        hydra.utils.call(
            cfg.paths
        )
        self._cfg = cfg
        # self._params = to_params(self.experiment_cfg)

    # TODO: Ensure i can use this
    # @property
    # def experiment(self):
    #     return self._cfg.experiment
    
    @property
    def experiment_type(self):
        return self._cfg.type
    
    @property
    def full(self):
        return self._cfg.full_study
    
    @property
    def cfg(self):
        return self._cfg
    
    @property
    def study_cfg(self):
        return self._cfg.study

    # @property
    # def experiment_cfg(self):
    #     return self.study_cfg['experiment']

    @property
    def device(self):
        return self._cfg.device
    
    @property
    def name(self) -> str:
        return self.study_cfg.name
    
    @property
    def n_trials(self) -> int:
        return self.study_cfg.n_trials
    
    @property
    def maximize(self) -> bool:
        return self.study_cfg.maximize
    
    # @property
    # def params(self) -> Params:
    #     return self._params


def combine_results(label_col: str, results: typing.Dict[str, DataFrame]) -> DataFrame:

    results_with_label = []
    for k, result in results.items():
        cur_result = result.results.copy()
        cur_result[label_col] = k
        results_with_label.append(
            cur_result
        )
    return pd.concat(results_with_label)


def mkdir(dir):
    if not  os.path.exists(dir):
        os.makedirs(dir)


# class ResultManager(object):

#     EXPERIMENT_COL = 'Experiment'
#     DATASET_COL = 'Dataset' 
#     TRIAL_COL = 'Trial'
#     SCORE_COL = 'Score'
#     DATE_COL = "Date"
#     TIME_COL = "Time"
#     TEST_COL = "Test Type"
#     MACHINE_COL = "Machine Type"
#     STUDY_ID = "Study ID"
#     EXPERIMENT_ID = "Experiment ID"
#     DESCRIPTION_COL = "Description"
    
#     # TODO: STORE THE COLUMN USED BY AN EXPERIMENT SOMEWHERE

#     def __init__(self, directory_path: str, study_name: str, load_if_available: bool=True):
#         """initializer
#         """
#         # load the results
#         self._directory_path = directory_path
#         # self._dataset = dataset
#         self._study_name = study_name
#         loaded = False
#         if load_if_available:
#             try:
#                 self.reload_file()
#                 loaded = True
#             except (FileNotFoundError, AttributeError):
#                 loaded = False

#         if not loaded:
#             self._info = {} # {self.DATASET_COL: self._dataset}
#             self._key_df = pd.DataFrame(
#                 columns=[self.DATASET_COL, self.EXPERIMENT_COL, self.STUDY_ID, self.EXPERIMENT_ID, self.DESCRIPTION_COL]
#             )
#             self._result_df = pd.DataFrame(
#                 columns=[self.DATASET_COL, self.EXPERIMENT_COL, self.STUDY_ID, self.EXPERIMENT_ID]
#             )

#     @property
#     def n_results(self):
#         return len(self._result_df)
    
#     @property
#     def n_experiments(self):
#         return len(self._key_df)

#     def add_experiment(
#         self, summary: Experiment
#     ) -> bool:
#         # case 1: 

#         if self._key_df[self.EXPERIMENT_ID].str.contains(
#             summary.research_id.experiment_id
#         ).any():
#             return False
#         test_type = summary.test_type_to_str(summary.is_validation)
#         idx = len(self._key_df.index)
#         date_, time = summary.datetime_to_str(summary.research_id.experiment_date)
#         self._key_df.loc[idx] = {
#             self.EXPERIMENT_ID: summary.research_id.experiment_id, 
#             self.STUDY_ID: summary.research_id.study_id, 
#             self.EXPERIMENT_COL: summary.research_id.experiment_name, 
#             self.DATASET_COL: summary.dataset, 
#             self.DATE_COL: date_, 
#             self.TIME_COL: time, 
#             self.TEST_COL: test_type, 
#             self.DESCRIPTION_COL: summary.description
#         }
#         return True
    
#     def delete_study(self, study_id: str):
#         self._key_df = self._key_df.loc[self._key_df[self.STUDY_ID] != study_id]
#         self._result_df = self._result_df.loc[self._key_df[self.STUDY_ID] != study_id]

#     def drop_experiment(self, experiment_id: str):
#         self._key_df = self._key_df.loc[self._key_df[self.EXPERIMENT_ID] != experiment_id]
#         self._result_df = self._result_df.loc[self._key_df[self.EXPERIMENT_ID] != experiment_id]

#     def add_summary(
#         self, experiment: Experiment
#     ):

#         self.add_experiment(experiment)
#         cur_results = experiment.summarize()
#         self._result_df = pd.concat([self._result_df, cur_results], ignore_index=True)

#     @property
#     def dir(self):
#         return f'{self._directory_path}/{self._study_name}'
#         # return f'{self._directory_path}/{self._dataset}/'

#     @property
#     def info_file(self):
#         return f'{self.dir}/info.json'
    
#     @property
#     def key_file(self):
#         return f'{self.dir}/key.csv'

#     @property
#     def result_file(self):
#         return f'{self.dir}/results.csv' 

#     def reload_file(self):
#         with open(self.info_file, 'r') as fp:
#             self._info = json.load(fp)
#         # self._dataset = self._info[self.DATASET_COL]
#         self._key_df = pd.read_csv(self.key_file)
#         self._result_df = pd.read_csv(self.result_file)

#     def to_file(self):
#         mkdir(self.dir)
#         with open(self.info_file, 'w') as fp:
#             json.dump(self._info, fp)
#         self._key_df.to_csv(self.key_file, index=False)
#         self._result_df.to_csv(self.result_file, index=False)


# def output_results_to_file(
#     summaries: typing.List[Experiment], directory: str, study_name: str
# ):
#     result_manager = ResultManager(directory, study_name, True)
    
#     for summary in summaries:
#         result_manager.add_summary(summary)
    
#     result_manager.to_file()

