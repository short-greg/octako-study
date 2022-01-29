from sango.nodes import Status
import torch

from tako.learners import Learner, Validator
from . import teaching
from torch.utils.data import TensorDataset, DataLoader
from torch import nn



class TestDataLoaderIterator:

    def test_cur_when_one_iteration(self):
        
        dataset = teaching.DataLoaderIter(DataLoader(TensorDataset(
            torch.zeros(3, 2), torch.zeros(3)
        ), batch_size=32))
        x, t = dataset.cur
        assert len(x) == 3

    def test_cur_when_zero_iteration(self):
        
        dataset = teaching.DataLoaderIter(DataLoader(TensorDataset(
            torch.zeros(0, 2), torch.zeros(0)
        ), batch_size=32))

        assert dataset.cur is None

    def test_cur_after_advance(self):
        
        dataset = teaching.DataLoaderIter(DataLoader(TensorDataset(
            torch.zeros(4, 2), torch.zeros(4)
        ), batch_size=2))
        dataset.adv()
        x, t = dataset.cur
        assert len(x) == 2

    def test_finished_after_two_advances(self):
        
        dataset = teaching.DataLoaderIter(DataLoader(TensorDataset(
            torch.zeros(4, 2), torch.zeros(4)
        ), batch_size=2))
        dataset.adv()
        dataset.adv()
        assert dataset.is_end() == True

    def test_pos_after_one_advance_is_one(self):
        
        dataset = teaching.DataLoaderIter(DataLoader(TensorDataset(
            torch.zeros(4, 2), torch.zeros(4)
        ), batch_size=2))
        dataset.adv()
        assert dataset.pos == 1


class LearnerTest(Learner, Validator):

    def __init__(self):
        nn.Module.__init__(self)

    def learn(self, x, t):
        return {'loss': torch.tensor(1.0)}
    
    def test(self, x, t):
        return {'loss': torch.tensor(1.0)}


class TestCourse:

    def create_course(self):
        dataset = teaching.DataLoaderIter(DataLoader(TensorDataset(
            torch.zeros(4, 2), torch.zeros(4)
        ), batch_size=2))
        return teaching.Course([teaching.TeacherData("Validation", dataset, LearnerTest(), 'Validation')])

    def test_results_with_no_data(self):
        course = self.create_course()
        assert len(course.df) == 0

    def test_results_with_one_result(self):
        course = self.create_course()
        course.switch_teacher('Validation')
        course.adv_iter(
            {'x': 2}
        )
        assert len(course.df) == 1

    def test_results_with_two_results(self):
        course = self.create_course()
        course.switch_teacher('Validation')
        course.adv_iter(
            {'x': 2}
        )
        course.adv_iter(
            {'x': 1}
        )
        assert len(course.df) == 2

    def test_results_with_two_result_columns(self):
        course = self.create_course()
        course.switch_teacher('Validation')
        course.adv_iter(
            {'x': 2, 'y': 3}
        )
        course.adv_iter(
            {'x': 1, 'y': 4}
        )
        cols = course.df.columns
        assert 'x' in cols
        assert 'y' in cols

    def test_data_is_correct(self):
        course = self.create_course()
        course.switch_teacher('Validation')
        x, _ = course.data
        assert isinstance(x, torch.Tensor)

    def test_progress_cols_are_correct(self):
        course = self.create_course()
        course.switch_teacher('Validation')
        course.adv_iter(
            {'x': 2, 'y': 3}
        )
        course.adv_iter(
            {'x': 1, 'y': 4}
        )
        cols = course.df.columns
        assert 'cur_iter' in cols
        assert 'n_epochs' in cols


class TestTrain:

    def _create_teach(self):

        dataset = teaching.DataLoaderIter(DataLoader(TensorDataset(
            torch.zeros(4, 2), torch.zeros(4)
        ), batch_size=2))
        validation = teaching.TeacherData("Validation", dataset, LearnerTest(), 'Validation')
        training = teaching.TeacherData("Training", dataset, LearnerTest(), 'Training')
        course = teaching.Course(
            [validation, training]
        )

        return teaching.Train(
            name='Training',
            course=course
        ), teaching.Validate(
            name='Validation',
            course=course
        )

    def test_teach_init_works(self):
        teach, _ = self._create_teach()
        assert isinstance(teach, teaching.Teach)

    def test_teach_tick_returns_running(self):
        teach, _ = self._create_teach()
        assert teach.tick() == Status.RUNNING

    def test_teach_tick_returns_succcess(self):
        teach, _ = self._create_teach()
        teach.tick()
        assert teach.tick() == Status.SUCCESS

    def test_teach_tick_returns_running_after_reset(self):
        teach, val = self._create_teach()
        teach.tick()
        teach.tick()
        teach.reset()
        assert teach.tick() == Status.RUNNING


class TestTest:

    def _create_teach(self):

        dataset = teaching.DataLoaderIter(DataLoader(TensorDataset(
            torch.zeros(4, 2), torch.zeros(4)
        ), batch_size=2))
        validation = teaching.TeacherData("Validation", dataset, LearnerTest(), 'Validation')
        training = teaching.TeacherData("Training", dataset, LearnerTest(), 'Training')
        course = teaching.Course(
            [validation, training]
        )

        return teaching.Validate(
            name='Validation',
            course=course
        )

    def test_teach_init_works(self):
        teach = self._create_teach()
        assert isinstance(teach, teaching.Validate)

    def test_teach_tick_returns_running(self):
        teach = self._create_teach()
        assert teach.tick() == Status.RUNNING

    def test_teach_tick_returns_succcess(self):
        teach = self._create_teach()
        teach.tick()
        assert teach.tick() == Status.SUCCESS

    def test_teach_tick_returns_running_after_reset(self):
        teach = self._create_teach()
        teach.tick()
        teach.tick()
        teach.reset()
        assert teach.tick() == Status.RUNNING


def create_trainer():

    dataset = teaching.DataLoaderIter(DataLoader(TensorDataset(
        torch.zeros(4, 2), torch.zeros(4)
    ), batch_size=2))
    validation = teaching.TeacherData("Validation", dataset, LearnerTest(), 'Validation')
    training = teaching.TeacherData("Training", dataset, LearnerTest(), 'Training')
    course = teaching.Course(
        [validation, training]
    )

    training = teaching.Trainer(
        name='Trainer',
        course=course
    )

    return teaching.Trainer(
        name='Trainer',
        course=course, 
        n_epochs=1
    )


def create_learner():

    class Learn(Learner, Validator):

        def __init__(self):
            nn.Module.__init__(self)

        def learn(self, x, t):
            return torch.tensor(0.0)
        
        def test(self, x, t):
            return torch.tensor(0.0)
    return Learn()


class TestValidationTrainer:

    def test_teach_init_works(self):
        teach = create_trainer()
        assert isinstance(teach, teaching.Trainer)

    def test_teach_tick_returns_running(self):
        teach = create_trainer()
        assert teach.tick() == Status.RUNNING

#     def test_teach_tick_returns_succcess(self):
#         teach = create_trainer()
#         assert teach.run(learner=LearnerTest()) == Status.FAILURE

#     def test_teach_tick_returns_running_after_reset(self):
#         teach = create_trainer()
#         teach.run(learner=LearnerTest())
#         teach.reset()
#         assert teach.tick() == Status.RUNNING
