from abc import ABC, abstractmethod, abstractproperty
import copy
from dataclasses import InitVar, dataclass, field
import functools
import optuna
import typing
import hydra
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import optuna
import typing
import hydra
from omegaconf import DictConfig

PDELIM = "/"


class Study(ABC):

    @abstractmethod
    def perform(self) -> float:
        pass

    @abstractproperty
    def params(self):
        pass


class OptunaStudy(Study):

    @abstractmethod
    def perform(self) -> float:
        pass

    @abstractmethod
    def resample(self):
        """Use to resample the study given the parameters of the study

        Returns: OptunaStudy
        """
        pass

    @abstractmethod
    def from_best(self, best: dict):
        """Use to resample the study given the parameters of the study

        Returns: OptunaStudy
        """
        pass


class TrialSelector(ABC):

    def __init__(self, name, default):

        self.default = default
        self._name = name

    def select(self, path: str='', trial: optuna.Trial=None, best: dict=None):

        if trial:
            return self.suggest(path, trial)
        return self.update_best(best, path)

    def cat_path(self, path: str=None, sub: str=None):
        full_path = self._name
        if path is not None:
            full_path = f'{path}/{full_path}'
        if sub is not None:
            full_path = f'{full_path}_{sub}' 
        return full_path

    @abstractmethod
    def suggest(self, path: str, trial: optuna.Trial):
        raise NotImplementedError

    @abstractmethod
    def update_best(self, best_val: dict, path: str=None):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, params: dict):
        raise NotImplementedError
    

class Default(TrialSelector):

    def __init__(self, name: str, val):
        super().__init__(name, val)
        self._val = val
    
    def suggest(self, path: str, trial: optuna.Trial):
        return self._val
    
    def update_best(self, best_val: dict, path: str=None):
        return self.default

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], params["val"])


class Int(TrialSelector):

    def __init__(self, name: str, low: int, high: int, default: int=0, base: int=None):
        super().__init__(name, default)

        self._low = low
        self._high = high
        self._base = base
    
    def suggest(self, path: str, trial: optuna.Trial):
        result = trial.suggest_int(
            self.cat_path(path) , self._low, self._high
        )
        if self._base is None:
            return result
        return self._base ** result

    def update_best(self, best_val: dict, path: str=None):
        val = best_val.get(self.cat_path(path), self.default)
        if self._base is None:
            return val
        return self._base ** val

    @classmethod
    def from_dict(cls, params: dict):
        return cls(
            params['name'], 
            low=params['low'], high=params['high'], 
            base=params.get('base', None), default=params.get('default')
        )


class Bool(TrialSelector):

    def __init__(self, name: str, default: bool=True):
        super().__init__(name, default)

    def suggest(self, path: str, trial: optuna.Trial):

        return bool(trial.suggest_uniform(
            self.cat_path(path) , 0, 1
        ))

    def update_best(self, best_val: dict, path: str=None):
        val = best_val.get(self.cat_path(path), self.default)
        return bool(val)

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], default=params.get('default'))


class Float(TrialSelector):

    def __init__(self, name: str, low: float=0., high: float=1., default: float=1.0):
        super().__init__(name, default)
        self._low = low
        self._high = high

    def suggest(self, path: str, trial: optuna.Trial):

        return trial.suggest_uniform(
            self.cat_path(path) , self._low, self._high
        )

    def update_best(self, best_val: dict, path: str=None):
        val = best_val.get(self.cat_path(path), self.default)
        return val

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], default=params.get('default'))


class Categorical(TrialSelector):

    def __init__(self, name: str, categories: typing.List[str], default: str):
        super().__init__(name, default)
        self._categories = categories

    def suggest(self, path: str, trial: optuna.Trial):
        return trial.suggest_categorical(
            self.cat_path(path), self._categories
        )

    def update_best(self, best_val: dict, path: str=None):
        return best_val.get(self.cat_path(path), self.default)

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], params["categories"], default=params.get('default'))


class ConditionalCategorical(TrialSelector):

    def __init__(self, name: str, categories: typing.Dict[str, str], default: str):
        super().__init__(name, default)
        self._categeries = categories
    
    def _get_paths(self, path):
        base_path = self.cat_path(path)
        sub_path = self.cat_path(path, "sub")
        return base_path, sub_path
    
    def suggest(self, path: str, trial: optuna.Trial):
        base_path, sub_path = self._get_paths(path)
        base = trial.suggest_categorical(base_path, list(self._categeries.keys()))
        sub_categories = self._categeries[base]
        sub = trial.suggest_categorical(sub_path, sub_categories)
        return (base, sub)
    
    # def _update_best_helper(self, best_val: dict, path:str):
    def update_best(self, best_val: dict, path: str=None):
        base_path, sub_path = self._get_paths(path)
        if base_path not in best_val or sub_path not in best_val:
            return self.default
        return best_val[base_path], best_val[sub_path] 

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], params["categories"], default=params.get('default'))


class LogUniform(TrialSelector):

    def __init__(self, name: str, low: int, high: int, default: int):
        super().__init__(name, default)
        self._low = low
        self._high = high

    def suggest(self, path: str, trial: optuna.Trial):
        return trial.suggest_loguniform(
            self.cat_path(path), self._low, self._high
        )

    def update_best(self, best_val: dict, path: str=None):
        return best_val.get(self.cat_path(path), self.default)
    
    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], params["low"], params["high"], default=params.get('default'))


class Non(object):

    @staticmethod
    def from_dict(params: dict):
        return params['value']


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

    def suggest(self, path: str, trial: optuna.Trial):
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
                    f'{path}/{i}', trial
                )
        return params

    def _get_param_by_name(self, name: str):

        for k, param in self._params.items():
            if param._name == name:
                return k, param

    # def _update_best_helper(self, best_val: dict, path:str):
    def update_best(self, best_val: dict, path: str=None):
        # best = best_val[self.cat_path(path)]
        
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
    
    @classmethod
    def from_dict(cls, params: dict):
        selectors: typing.Dict[str, TrialSelector] = {}
        
        for k, p in params:
            selectors[k] = ParamMap[p["type"]].from_dict(p)

        return cls(
            params['name'], low=params["low"], high=params["high"], params=selectors,
            default=params.get('default')
        )


ParamMap: typing.Dict[str, TrialSelector] = {
    "Array": Array,
    "Int": Int, 
    "LogUniform": LogUniform,
    "Categorical": Categorical,
    "ConditionalCategorical": ConditionalCategorical,
    "Bool": Bool,
    "Default": Default,
    "Non": Non
}


def convert_params(trial_params: dict):

    return {
        k: ParamMap[params['type']].from_dict(params)
        for k, params in trial_params.items()
    }


@dataclass
class OptunaParams(object):

    trial: InitVar[optuna.Trial]=None
    path: InitVar[str] = ''

    def __post_init__(self, trial, path:str):
        params = asdict(self)
        for k, v in params.items():
            if isinstance(v, TrialSelector):
                setattr(self, k, v.suggest(path, trial))
    
    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            if isinstance(v, TrialSelector):
                setattr(self, k, v)

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, trial_params, trial=None, path: str=''):
        return cls(**{
            k: ParamMap[params['type']].from_dict(params)
            for k, params in trial_params.items()
        }, trial=trial, path=path)


class Components(object):

    def __init__(self, components: dict):

        self._components = components

    def load_state_dict(self, state_dict):

        for k, v in state_dict.items():
            self._components[k].load_state_dict(v)

    def state_dict(self):

        state_dict_ = {}
        for k, v in self._components:
            state_dict_[k] = v.state_dict()
        return state_dict_


@dataclass
class Summary(object):

    params: OptunaParams
    score: float
    maximize: bool
    components: Components = field(default_factory=Components)
    # add in results in here

    def bests(self, other):
        if self.maximize and other.maximize:
            return self.score > other.score

        elif not self.maximize and not other.maximize:
            return self.score < other.score

        raise ValueError(
            'Cannot compare two summaries unless the value for maximize is the same: ' 
            f'Self: {self.maximize} Other: {other.maximize}'
        )

    def load_state_dict(self, state_dict):

        self.params.load_state_dict(state_dict['params'])
        self.score = state_dict['score']
        self.maximize = state_dict['maximize']
        self.components.load_state_dict(state_dict['components'])

    def state_dict(self):

        state_dict_ = {}
        state_dict_['score'] = self.score
        state_dict_['maximize'] = self.maximize
        state_dict_['params'] = self.params.state_dict()
        state_dict_['components'] = self.components.state_dict()
        return state_dict_


class Experiment(ABC):

    @abstractmethod
    def validate(self) -> Summary:
        pass

    @abstractmethod
    def test(self) -> Summary:
        pass


class OptunaExperiment(Experiment):

    class Params(OptunaParams):
        pass

    def _get_params(self, param_overrides: dict, trial: None, path: str='', ):
        return self.Params(
            **param_overrides, trial, path
        )

    @abstractmethod
    def validate(self, param_overrides: dict, trial: None, path: str='', ) -> Summary:
        raise NotImplementedError

    @abstractmethod
    def test(self, param_overrides: dict, trial: None, path: str='') -> Summary:
        raise NotImplementedError


class Study(ABC):
    
    @abstractmethod
    def run(self, name: str) -> dict:
        raise NotImplementedError


class OptunaStudy(Study):

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
    
    def get_objective(self, name: str, studies: typing.List) -> typing.Callable:
        cur: int = 0

        def objective(trial: optuna.Trial):
            nonlocal cur
            nonlocal studies
            evaluation = self._experiment.validate(trial=trial, validation=True)
            cur += 1
            return evaluation.score
        return objective

    def run(self, name) -> typing.Tuple[Summary, typing.List[Summary]]:

        summaries = []
        optuna_study = optuna.create_study(direction=self._direction)
        objective = self.get_objective(name, summaries)
        optuna_study.optimize(objective, self._n_trials)
        best = functools.reduce(lambda x, y: x if x.bests(y) else y)
        summary = self._experiment.test(overrides=best.params)
        return summary, summaries


class Config:

    name: str  # name fo config to choose
    path: str = 'blueprints/'


def create_runner(config: Config, study_cls: typing.Type[OptunaStudy]):

    @hydra.main(config_path=config.path, config_name=config.name)
    def _(cfg : DictConfig):

        hydra.utils.call(
            cfg.paths
        )
        params = convert_params(cfg.params)
        study = study_cls(params)
        if bool(cfg.full_study):
            return OptunaStudy(study, cfg.name, cfg.n_trials, cfg.to_maximize)
        # else:
        #     return StudyRunner(study)
    
    return _()
            

# class StudyRunner(object):
    
#     def __init__(
#         self, study: Study, base_name: str
#     ):
#         self._study = study
#         self._base_name = base_name
    

#     def run(self, name) -> dict:
        
#         evaluation = self._study.perform()
#         return evaluation, [self._study]


# class OptunaStudy(studies.Study):
    
#     @abstractmethod
#     def perform(self, trial=None, best=None, validation=False) -> typing.List[dojos.Course]:
#         pass

# class Runner(object):

#     def __init__(self, cfg):

#         self._cfg = cfg

#     def run(self):

#         experiment_cfg = list(self._cfg.blueprints.experiments.items())[0][1]

#         override_base_params = hydra.utils.instantiate(experiment_cfg.override_base_params)
#         override_trial_params = hydra.utils.instantiate(experiment_cfg.override_trial_params)

#         study_builder: base.StudyBuilder = hydra.utils.instantiate(
#             experiment_cfg.study_builder, override_base_params=override_base_params, 
#             override_trial_params=override_trial_params
#         )
#         study: base.Study = hydra.utils.instantiate(experiment_cfg.study_type, study_builder=study_builder)

#         study.run(experiment_cfg.name)


# OLD STUDY RUn
        # parameters.append(optuna_study.best_params)

        # best_params = ParamConverter(study.best_params).to_dict()
        # parameters.append(study.best_params)
        
        # courses.append(self._study.perform(best=study.best_params))

        # trial_param_dict = experiment_cfg.trial_params or {}
        # base_param_dict = experiment_cfg.base_params or {}

        # if experiment_cfg.experiment_type == "single":
        #     base_params = study.base_param_cls(**base_param_dict)
        #     study.run_single(experiment_cfg.name, experiment_cfg.base_params)
        # else:
        #     trial_params = study.trial_param_cls(**trial_param_dict)
        #     base_params = study.base_param_cls(**base_param_dict)
        #     study.run_study(
        #         experiment_cfg.name, experiment_cfg.n_studies, 
        #         base_params, trial_params
        #     )

# import src.fuzzy.studies as fuzzy_studies


# class ObjectiveRunner(object):

#     def __init__(
#         self, name, study_builder: studies.StudyBuilder, study_accessor: study_controller.StudyAccessor
#     ):
#         self.name = name
#         self.study_builder = study_builder
#         self.study_accessor = study_accessor
#         self._cur_study = 0

#     def __call__(self, trial):

#         # experiment_accessor = self.study_accessor.create_experiment(f'{self.name} - Trial {self._cur_study}')

#         # pass the name of the experiment
#         learner, my_dojo, experiment_accessor = self.study_builder.build_for_trial(
#             f'{self.name} - Trial {self._cur_study}', trial, self.study_accessor
#         )
#         my_dojo.run()

#         # TODO: Evaluate results of experiment accessor
#         self._cur_study += 1
        
#         module_accessor = experiment_accessor.get_module_accessor_by_name(
#             self.study_builder.optimize_module
#         )

#         results = module_accessor.get_results(
#             retrieve_fields=[self.study_builder.optimize_field], 
#             round_ids_filter=[module_accessor.progress.round]
#         )
#         return results.mean(axis=0).to_dict()[self.study_builder.optimize_field]


# class FullStudy(studies.Study):

#     def __init__(self, name: str, n_trials: int, study_builder: studies.StudyBuilder):
#         self.study_builder = study_builder
#         self.n_trials = n_trials
#         self.name = name

#     def run(self, name):
#         # create the experiment accessor
#         data_controller = study_controller.DataController()
#         study_accessor = data_controller.create_study(self.name)

#         objective = ObjectiveRunner(
#             name, self.study_builder, study_accessor
#         )

#         my_study = optuna.create_study(direction=self.study_builder.direction)
#         my_study.optimize(objective, n_trials=self.n_trials)

#         # experiment_accessor = study_accessor.create_experiment("Best")
        
#         learner, dojo, experiment_accessor = self.study_builder.build_best(
#             "Best", params.ParamConverter(my_study.best_params).to_dict(), study_accessor
#         )
#         dojo.run()


# class Optunable(ABC):

#     @property
#     def path(self):
#         raise NotImplementedError

#     @singledispatchmethod
#     def _sample_value(self, v ,trial=None, best: dict=None):
#         return v        

#     @_sample_value.register
#     def _(self, v: TrialSelector, trial=None, best: dict=None):
#         return v.select(self.path, trial, best)

#     def _sample(self, trial=None, best: dict=None):

#         for k, v in asdict(self).items():
#             v = self._sample_value(v, trial, best)
#             self.__setattr__(k, v)


# @dataclass
# class TunableLearner(Learner, Optunable):

#     trial: InitVar[optuna.Trial] = None
#     best: InitVar[optuna.Trial] = None

#     @abstractmethod
#     def _build(self):
#         raise NotImplementedError

#     def __post_init__(self, trial, best):
#         self._sample(trial, best)
#         self._build()

# @dataclass
# class TunableDojo(dojos.Dojo, Optunable):

#     trial: InitVar[optuna.Trial] = None
#     best: InitVar[optuna.Trial] = None

#     def _build(self):
#         raise NotImplementedError

#     def __post_init__(self, trial, best):
#         self._sample(trial, best)
#         self._build()

# class MonoStudy(OptunaStudy):

#     def __init__(
#         self, learner_cls: typing.Type[TunableLearner], 
#         dojo_cls: typing.Type[TunableDojo], params, device='cpu'
#     ):
#         self._learner_cls = learner_cls
#         self._dojo_cls = dojo_cls
#         self._params = params
#         self._device = device
    
#     def perform(self, trial=None, validation=False, best=None):
        
#         dojo = self._dojo_cls(**self._params)
#         learner = self._learner_cls(**self._device)

#         if validation:
#             return dojo.validate(learner, trial, best)
#         return dojo.test(learner, trial, best)


# TODO: FIX!!

# class ParamConverter(object):
#     """Convert 'best params' to params to update a class"""

#     def __init__(self, best_params: dict):

#         self._best_params = best_params
    
#     def to_dict(self):

#         return self._nest_params(self._best_params)

#     def _replace_with_lists(self):
#         """[convert dictionaries in in the nested parameters 
#         to lists where the param dictionary consists of the keys 'size' plus integers
#         this is somewhat of a hack]

#         Args:
#             nested_params ([type]): [description]

#         Returns:
#             [type]: [description]
#         """

#         nested_params = self._best_params
#         if 'size' in nested_params and len(nested_params) > 1:
#             all_digits = True
#             for k in nested_params:
#                 if k != 'size' and not k.isdigit():
#                     all_digits = False
#                     break
#             result = []
#             if all_digits is True:
#                 for k, v in nested_params.items():
#                     if k == 'size': continue;
#                     cur = int(k)
#                     if len(result) >= cur:
#                         result.extend([None] * (cur + 1 - len(result)))
#                     result[cur] = self._replace_with_lists(v)
#                 return result

#         for k, v in nested_params.items():
#             if type(v) == dict:
#                 nested_params[k] = self._replace_with_lists(v)
#             else:
#                 nested_params[k] = v

#         return nested_params

#     def _nest_params_helper(self, key_tuple, value, nested_params):
        
#         cur = key_tuple[0]
#         # if cur not in nested_params:
#         #    nested_params[cur] = None

#         if len(key_tuple) == 1:
#             nested_params[cur] = value
#         else:
#             if cur not in nested_params:
#                 nested_params[cur] = {}
#             self._nest_params_helper(
#                 key_tuple[1:], value, nested_params[cur]
#             )

#     def _nest_params(self, best_params):
#         nested_params = {}
#         for key, value in best_params.items():

#             s = key.split(PDELIM)
#             cur = s[0]
#             # nested_params[cur] = 
#             self._nest_params_helper(s, value, nested_params)

#         return nested_params


# import typing
# import hydra
# from omegaconf import DictConfig

# from .studies import OptunaStudy, OptunaStudyRunner, StudyRunner
# # import src.fuzzy.studies as fuzzy_studies


# def run(config_path, config_name, study_cls: typing.Type[OptunaStudy]):

#     @hydra.main(config_path=config_path, config_name=config_name)
#     def run_experiment(cfg : DictConfig):

#         hydra.utils.call(
#             cfg.paths
#         )

#         if bool(cfg.full_study):
#             return OptunaStudyRunner(study_cls(cfg.params), cfg.name, cfg.n_trials, cfg.to_maximize)
#         else:
#             return StudyRunner(study_cls(cfg.params))
            

    # if cfg.type == "Fuzzy":

    #     if cfg.fuzzy.full_study is True:
    #         fuzzy_studies.run_full_study(cfg)
    #     else:
    #         fuzzy_studies.run_single_study(cfg)

# if __name__=='__main__':
#     run_experiment()
