from abc import ABC, abstractmethod, abstractproperty
from dataclasses import InitVar, dataclass, fields, field
from numpy import isin
import optuna
import typing
import hydra
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import optuna
import typing
import hydra
from omegaconf import DictConfig
from pytest import param

from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
from itertools import chain
import inspect


PDELIM = "/"


class Study(ABC):

    @abstractmethod
    def perform(self) -> float:
        pass

    @abstractproperty
    def params(self):
        pass


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


# TODO: Do I need this?
# class Non(object):


#     def to_dict(self, flat=False):
#         return dict(self.)

#     @staticmethod
#     def from_dict(**params: dict):
#         return params['value']

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


def convert_params(trial_params: dict):
    result = {}
    
    for k, v in trial_params.items():
        if k.lower() == 'type':
            continue
        if isinstance(v, dict) and 'type' in v and v['type'].lower() == 'sub':
            result[k] = convert_params(v)
        elif isinstance(v, dict) and 'type' in v:
            result[k] = ParamMap[v['type']].from_dict(**v)
        else:
            result[k] = v
    return result 


# sub can be optuna params
# sub can be tunablefactory

def asdict_shallow(obj):
    return dict((field.name, getattr(obj, field.name)) for field in fields(obj))

def is_trial_selector(value) -> bool:    
    return (
        isinstance(value, dict) or isinstance(value, DictConfig)
        and 'type' in value
    )


@dataclass
class OptunaParams(object):

    # TODO: This does not support
    # "Selectors"
    def load_state_dict(self, state_dict: dict):
        for k, _ in asdict(self).items():
            setattr(self, k, state_dict[k])

    # TODO: doesn't work with selectors
    def state_dict(self):
        return asdict(self)
    
    def __post_init__(self):
        self._extra = []
    
    def to_dict(self):

        result = dict()
        for k, v in asdict(self).items():
            if (
                isinstance(v, OptunaSelector) or 
                isinstance(v, TrialSelector) or
                isinstance(v, OptunaParams)
            ):
                result[k] = v.to_dict()
            else:
                result[k] = v
        
        return result

    @classmethod
    def from_dict(cls, **overrides: dict):
        data_fields = {item.name: item for item in fields(cls)}

        args = []
        updated = dict()
        for k, v in overrides.items():
            if k == 'type':
                continue
            elif k not in data_fields:
                updated[k] = v
            elif data_fields[k].type == OptunaSelector:
                updated[k] = data_fields[k].type(**v)
            elif isinstance(data_fields[k].type, type) and issubclass(data_fields[k].type, OptunaParams):
                updated[k] = data_fields[k].type.from_dict(**v)
            elif is_trial_selector(v):
                updated[k] = ParamMap[v['type']].from_dict(**v)
            else:
                updated[k] = v
        result = cls(*args, **updated)
        return result
    
    def define(self, **kwargs):
        return self.__class__(**kwargs)
    
    def extra_params(self) -> typing.Iterator:
        for k in self._extra:
            yield k, getattr(self, k)

    def suggest(self, trial=None, path: str=''):
        """Suggest params

        Args:
            trial (_type_, optional): optuna.Trial. Defaults to None.
            path (str, optional): path for the variable. Defaults to ''.

        Returns:
            OptunaParams
        """
        args = {}
        params = asdict_shallow(self)
        for k, v in chain(params.items(), self.extra_params()):
            if isinstance(v, TrialSelector):
                if trial is None:
                    args[k] = v.default
                else:
                    args[k] = v.suggest(trial, path)
            elif isinstance(v, OptunaParams):
                args[k] = v.suggest(trial, path)
            else:
                args[k] = v
        
        result = self.__class__(**args)
        return result
    
    @property
    def defined(self):

        params = asdict(self)
        for k, v in chain(params.items(), self.extra_params()):
            if isinstance(v, TrialSelector):
                return False
            if isinstance(v, OptunaParams):
                if not v.defined:
                    return False
        return True


@dataclass
class Validation(object):

    params: OptunaParams
    score: float
    maximize: bool
    for_validation: bool
    trial_params: dict

    def bests(self, other: 'Validation') -> 'Validation':
        if self.maximize and other.maximize:
            return self if self.score > other.score else other

        elif not self.maximize and not other.maximize:
            return self if self.score < other.score else other

        raise ValueError(
            'Cannot compare two summaries unless the value for maximize is the same: ' 
            f'Self: {self.maximize} Other: {other.maximize}'
        )

    def load_state_dict(self, state_dict):

        self.params.load_state_dict(state_dict['params'])
        self.score = state_dict['score']
        self.maximize = state_dict['maximize']
        self.trial_params = state_dict['trial_params']
        self.for_validation = state_dict['for_validation']

    def state_dict(self):

        state_dict_ = {}
        state_dict_['score'] = self.score
        state_dict_['maximize'] = self.maximize
        state_dict_['trial_params'] = self.trial_params
        state_dict_['params'] = self.params.state_dict()
        state_dict_['for_validation'] = self.for_validation
        return state_dict_


class Experiment(ABC):

    @abstractmethod
    def run(self) -> Validation:
        raise NotImplementedError


Selection = typing.Dict[str, typing.Type[OptunaParams]]
ParamArg = typing.Union[OptunaParams, dict]


class ParamClass(object):

    class Params(OptunaParams):
        pass

    def __init__(self, param_overrides: dict=None):
        param_overrides = param_overrides or {}
        self._params_base = self.Params.from_dict(**param_overrides)

    def to_dict(self):

        return dict(
            params=self._params_base.to_dict()
        )
    
    @classmethod
    def from_dict(cls, **overrides):

        return cls(selected=overrides['params'])

@dataclass
class OptunaSelector(object):

    selected: str
    params: InitVar[ParamArg] = None
    selections: typing.ClassVar[Selection] = {}

    def to_dict(self):

        return dict(
            selected=self.selected,
            selection=self.selection.to_dict()
        )
    
    @classmethod
    def from_dict(cls, **overrides):

        return cls(selected=overrides['selected'], params=overrides['selection'])

    def __post_init__(self, params):
        params = params or {}

        if isinstance(params, OptunaParams):
            self.selection = self.selections[self.selected](params)
        else:
            self.selection = self.selections[self.selected].from_dict(
                **params
            )
    

class OptunaExperiment(ParamClass):

    def __init__(self, param_overrides: dict=None):
        super().__init__(param_overrides)
        self._params = self._params_base.suggest()

    def reset(self):
        self._params = self._params_base.suggest()
    
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    @abstractmethod
    def trial_experiment(self, trial: optuna.Trial, path: str='') -> Validation:
        raise NotImplementedError

    @abstractmethod
    def experiment(self, params: OptunaParams) -> Validation:
        raise NotImplementedError


class Study(ABC):
    
    @abstractmethod
    def run(self, name: str) -> dict:
        raise NotImplementedError


class OptunaStudy(Study):

    best_idx = 'BEST'
    final_idx = 'FINAL'

    @staticmethod
    def get_direction(to_maximize):
        return optuna.study.StudyDirection.MAXIMIZE if to_maximize else optuna.study.StudyDirection.MINIMIZE

    def __init__(
        self, experiment: OptunaExperiment, base_name: str, n_trials: int, to_maximize: bool
    ):
        self._experiment = experiment
        self._base_name = base_name
        self._n_trials = n_trials
        self._direction = self.get_direction(to_maximize)
    
    def get_objective(self, summaries: typing.Dict) -> typing.Callable:
        cur: int = 0

        def objective(trial: optuna.Trial):
            nonlocal cur
            nonlocal summaries
            summary = self._experiment.trial_experiment(trial)
            if self.best_idx not in summaries:
                summaries[self.best_idx] = summary
            else:
                summaries[self.best_idx] = summary.bests(summaries[self.best_idx])

            summaries[cur] = summary
            cur += 1
            return summary.score
        return objective
    
    @property
    def experiment(self) -> OptunaExperiment:
        return self._experiment

    def run(self, name) -> typing.Tuple[Validation, typing.Dict[str, Validation]]:

        summaries: typing.Dict[str, Validation] = {}
        optuna_study = optuna.create_study(direction=self._direction)
        objective = self.get_objective(summaries)
        optuna_study.optimize(objective, self._n_trials)
        summary = self._experiment.experiment(summaries[self.best_idx].params)
        summaries[self.final_idx] = summary
        return summary, summaries


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
        self._params = convert_params(self.experiment_cfg)

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
    def params(self) -> OptunaParams:
        return self._params
    
    def create_study(self, experiment_cls: typing.Type[OptunaExperiment]):
        # cur = self.study_cfg
        experiment = experiment_cls(self.params, device=self.device)
        return OptunaStudy(experiment, self.name, self.n_trials, self.maximize)

    def create_experiment(self, experiment_cls: typing.Type[OptunaExperiment]):
        params = convert_params(self.experiment_cfg)
        return experiment_cls(params, device=self.device)


def flatten_params(d: dict, prepend=None):

    flattened = {}

    for k, v in d.items():
        
        name = f'{prepend}/{k}' if prepend is not None else k

        if isinstance(v, dict):
            flattened.update(
                flatten_params(v, name)
            )
        elif isinstance(v, tuple) or isinstance(v, list):
            flattened.update(
                flatten_params(dict(enumerate(v)), name)
            )
        else:
            flattened[name] = v
    return flattened
