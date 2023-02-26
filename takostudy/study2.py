# 1st party
from abc import ABC, abstractmethod, abstractproperty
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, fields, field
from dataclasses import asdict, dataclass
import typing
import typing
from pytest import param
from itertools import chain
import inspect

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


# my libraries
from octako.components import Learner
from octako import teach


PDELIM = "/"

LearnerFactory = typing.Callable[[typing.Any], Learner]
TeacherFactory = typing.Callable[[typing.Any], teach.Teacher]


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

        return Params(**params)

    def to_dict(self) -> typing.Dict:
        result = {}
        for k, v in self._params:
            if isinstance(v, TrialSelector):
                result[k] = v.default
            else:
                result[k] = v

        return result


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


class Study(ABC):
    
    @abstractmethod
    def run(self, name: str) -> dict:
        raise NotImplementedError


@dataclass
class Experiment(object):

    name: str
    score: float
    chart: teach.Chart
    learner_params: Params
    teacher_params: Params


class ExperimentLog(object):

    def __init__(self, maximize: bool):
        self.maximize = maximize
        self._experiments: typing.List[Experiment] = []
    
    def add(self, experiment: Experiment):
        self._experiments.append(experiment)

    def best(self) -> typing.Tuple[int, Experiment]:
        if self.maximize:
            return functools.reduce(
                lambda x, y: x if x.score[1] > y.score[1] else y,
                enumerate(self._experiments)
            )
        return functools.reduce(
            lambda x, y: x if x.score[1] < y.score[1] else y,
            enumerate(self._experiments)
        )


class OptunaStudy(Study):

    best_idx = 'BEST'
    final_idx = 'FINAL'

    def __init__(
        self, learner_factory: LearnerFactory, learner_params: Params=None, 
        batch_size: int=64, n_epochs: int=10, base_name: str='', 
        maximize: bool=False, device: str='cpu'
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
        self._learner_factory = learner_factory
        self._learner_params = learner_params
        self._teacher_factory: TeacherFactory = teach.MainTeacher
        self._teacher_params = {'batch_size': batch_size, 'n_epochs': n_epochs}
        self._base_name = base_name
        self._device = device
        self._maximize = maximize
        self._direction = self.get_direction(self._maximize)

    def update_teacher(self, teacher_factory: TeacherFactory, params: Params):
        """Update the teacher factory to use

        Args:
            teacher_factory (TeacherFactory): 
            params (Params): 
        """
        self._teacher_factory = teacher_factory
        self._teacher_params = params
    
    def get_objective(self, experiments: ExperimentLog) -> typing.Callable:
        cur: int = 0

        def objective(trial: optuna.Trial):
            nonlocal cur
            nonlocal experiments

            teacher_params = self._teacher_params.suggest(trial, self._base_name)
            learner_params = self._learner_params.suggest(trial, self._base_name)
            learner = self._learner_factory(**learner_params)
            teacher = self._teacher_factory(**teacher_params)

            score, chart = teacher.validate(learner)
            experiments.add(Experiment(str(cur), score, chart, learner_params, teacher_params))

            cur += 1
            return score
        return objective
    
    def run_trials(self, n_trials: int) -> typing.List[Experiment]:

        experiment_log = ExperimentLog(self._maximize)
        optuna_study = optuna.create_study(direction=self._direction)
        objective = self.get_objective(experiment_log)
        optuna_study.optimize(objective, n_trials)

        _, best = experiment_log.best()
        teacher = self._teacher_factory(**best.teacher_params)
        learner = self._learner_factory(**best.learner_params)
        score, chart = teacher.train(learner)
        final = Experiment("best", score, chart, best.learner_params, best.teacher_params)
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
        self._params = to_params(self.experiment_cfg)

    # TODO: Ensure i can use this
    @property
    def experiment(self):
        return self._cfg.experiment
    
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

    @property
    def experiment_cfg(self):
        return self.study_cfg['experiment']

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
    
    @property
    def params(self) -> Params:
        return self._params
    