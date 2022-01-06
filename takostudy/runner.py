import typing
import hydra
from omegaconf import DictConfig

from .studies import OptunaStudy, OptunaStudyRunner, StudyRunner
# import src.fuzzy.studies as fuzzy_studies


def run(config_path, config_name, study_cls: typing.Type[OptunaStudy]):

    @hydra.main(config_path=config_path, config_name=config_name)
    def run_experiment(cfg : DictConfig):

        hydra.utils.call(
            cfg.paths
        )

        if bool(cfg.full_study):
            return OptunaStudyRunner(study_cls(cfg.params), cfg.name, cfg.n_trials, cfg.to_maximize)
        else:
            return StudyRunner(study_cls(cfg.params))
            

    # if cfg.type == "Fuzzy":

    #     if cfg.fuzzy.full_study is True:
    #         fuzzy_studies.run_full_study(cfg)
    #     else:
    #         fuzzy_studies.run_single_study(cfg)

# if __name__=='__main__':
#     run_experiment()
