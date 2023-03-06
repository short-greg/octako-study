# # 1st part
# from dataclasses import dataclass
# from datetime import datetime
# import os
# import typing
# import json

# # 3rd party
# from pandas import DataFrame
# import pandas as pd

# # Local
# from octako import Learner
# from octako.teach import Chart, ResearchID

# EXPERIMENT_COL = 'Experiment'
# DATASET_COL = 'Dataset' 
# TRIAL_COL = 'Trial'
# SCORE_COL = 'Score'
# DATE_COL = "Date"
# TIME_COL = "Time"
# TEST_COL = "Test Type"
# MACHINE_COL = "Machine Type"
# STUDY_ID = "Study ID"
# EXPERIMENT_ID = "Experiment ID"
# DESCRIPTION_COL = "Description"


# @dataclass
# class ExperimentSummary:

#     chart: Chart
#     learner: Learner
#     research_id: ResearchID
#     dataset: str
#     is_validation: bool
#     description: str
#     trial_name: str
#     name: str
#     score: float=0.0

#     def datetime_to_str(self, experiment_date: datetime):
#         return  experiment_date.strftime("%Y/%m/%d"),  experiment_date.strftime("%H:%M:%S")
    
#     def test_type_to_str(self, is_validation: bool):
#         if is_validation:
#             test_type = "Validation"
#         else: test_type = "Test"
#         return test_type

#     def summarize(self):
#         # TODO: Implement a better way to do the averaging
#         results = self.chart.df
#         cur_result = results[['Teacher', 'Epoch', 'loss', 'validation']].groupby(
#             by=['Teacher', 'Epoch']
#         ).mean().reset_index()
#         date_, time = self.datetime_to_str(self.research_id.experiment_date)
#         cur_result[[EXPERIMENT_ID, EXPERIMENT_COL, STUDY_ID, DATASET_COL, TRIAL_COL, DATE_COL, TIME_COL, TEST_COL]] = [
#             self.research_id.experiment_id, 
#             self.research_id.experiment_name, 
#             self.research_id.study_name, 
#             self.dataset, 
#             self.trial_name, 
#             date_, 
#             time, 
#             self.test_type_to_str(self.is_validation)
#         ]    
#         return cur_result


# def combine_results(label_col: str, results: typing.Dict[str, DataFrame]) -> DataFrame:

#     results_with_label = []
#     for k, result in results.items():
#         cur_result = result.results.copy()
#         cur_result[label_col] = k
#         results_with_label.append(
#             cur_result
#         )
#     return pd.concat(results_with_label)


# def mkdir(dir):
#     if not  os.path.exists(dir):
#         os.makedirs(dir)


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
#         self, summary: ExperimentSummary
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
#         self, summary: ExperimentSummary
#     ):

#         self.add_experiment(summary)
#         cur_results = summary.summarize()
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
#     summaries: typing.List[ExperimentSummary], directory: str, study_name: str
# ):
#     result_manager = ResultManager(directory, study_name, True)
    
#     for summary in summaries:
#         result_manager.add_summary(summary)
    
#     result_manager.to_file()
