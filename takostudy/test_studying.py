from dataclasses import dataclass
import random
from unittest.mock import MagicMock, Mock
from . import studying
import optuna
from optuna.storages import InMemoryStorage
import pytest


class TestDefault:

    def test_suggest_equals_correct_number(self, mocker):

        default = studying.Default(
            "x", 1
        )
        
        assert default.suggest('', object()) == 1

    def test_from_dict_equals_correct_number(self, mocker):

        default = studying.Default.from_dict({
            'name': 'x',
            'val': 1
        })
        
        assert default.default == 1
        assert default.name == 'x'


class TestInt:

    def test_suggest_equals_number_between_zero_and_four(self, mocker):

        int_ = studying.Int(
            "x", 0, 4, 1
        )
        mock = Mock()
        mock.suggest_int.side_effect = lambda self, low, high: 3
        assert int_.suggest('', mock) == 3

    def test_suggest_equals_value_from_deick(self):

        int_ = studying.Int.from_dict({
            'name': 'x',
            'low': 0,
            'high': 4,
            'default': 0
        })
        assert int_.name == 'x'
        assert int_.default == 0


class TestExpInt:

    def test_suggest_equals_eight(self, mocker):

        int_ = studying.ExpInt(
            "x", 0, 4, 2
        )
        mock = Mock()
        mock.suggest_int.side_effect = lambda self, low, high: 3
        assert int_.suggest('', mock) == 8

    def test_suggest_equals_value_from_dict(self):

        int_ = studying.ExpInt.from_dict({
            'name': 'x',
            'low': 0,
            'high': 4,
            'default': 0,
            'base': 1
        })
        assert int_.name == 'x'
        assert int_.default == 0


class TestBool:

    def test_suggest_equals_true(self, mocker):

        int_ = studying.Bool(
            "x", True
        )
        mock = Mock()
        mock.suggest_uniform.side_effect = lambda self, low, high: 0
        assert int_.suggest('', mock) is False

    def test_suggest_equals_value_from_dict(self):

        int_ = studying.Bool.from_dict({
            'name': 'x',
            'default': False
        })
        assert int_.name == 'x'
        assert int_.default is False


class TestFloat:

    def test_suggest_equals_correct_value(self):

        int_ = studying.Float(
            "x", 1, 4.5, 1.5
        )
        mock = Mock()
        mock.suggest_uniform.side_effect = lambda self, low, high: 2.5
        assert int_.suggest('', mock) == 2.5

    def test_suggest_equals_value_from_dict(self):

        int_ = studying.Float.from_dict({
            'name': 'x',
            'low': 1.,
            'high': 4.5,
            'default': 1.5
        })
        assert int_.name == 'x'
        assert int_.default == 1.5


class TestCategorical:

    def test_suggest_equals_correct_value(self):

        int_ = studying.Categorical(
            "x", ['big', 'small'], 'small'
        )
        mock = Mock()
        mock.suggest_categorical.side_effect = lambda self, cats: 'big'
        assert int_.suggest('', mock) == 'big'

    def test_suggest_equals_value_from_dict(self):

        int_ = studying.Categorical.from_dict({
            'name': 'x',
            'categories': ['big', 'small'],
            'default': 'big'
        })
        assert int_.name == 'x'
        assert int_.default == 'big'


class TestArray:

    def test_suggest_equals_correct_value(self):

        int_ = studying.Array(
            "x", low=1, high=3, params={
                'sub': studying.Bool('sub')
            }
        )
        mock = Mock()
        mock.suggest_int.side_effect = lambda self, low, high: 2
        mock.suggest_uniform.side_effect = lambda self, low, high: 0

        assert int_.suggest('', mock) == [{'sub': False}, {'sub': False}]

    def test_suggest_equals_value_from_dict(self):

        arr = studying.Array.from_dict({
            'name': 'x',
            'low': 1,
            'high': 3,
            'sub': {'x': {'type': 'Bool', 'name': 'x', 'default': True }},
            'default': [[False]]
        })
        assert arr.name == 'x'
        assert arr.default == [[False]]


class TestConvertParams:

    def test_convert_equals_correct_params_int(self):

        params = {
            'b': {
                'type': 'Int',
                'low': 0,
                'high': 4,
                'name': 'b'
            }
        }
        params =studying.convert_params(params)
        assert isinstance(params['b'], studying.Int)


class TestOptunaParams:

    @dataclass
    class MyParams(studying.OptunaParams):

        x: int = studying.Int('x', 0, 4, 1)

    def test_sample_produces_my_params_with_defined_value(self):

        mock = Mock()
        mock.suggest_int.side_effect = lambda self, low, high: 2

        params = TestOptunaParams.MyParams().sample(mock, '')
        assert params.x == 2

    def test_defined_produces_my_params_with_defined_value(self):

        mock = Mock()
        mock.suggest_int.side_effect = lambda self, low, high: 2

        params = TestOptunaParams.MyParams().define(x=3)
        assert params.x == 3

    def test_from_dict_produces_my_params_with_defined_value(self):

        params = {
            'x': {
                'type': 'Int',
                'low': 2,
                'high': 4,
                'name': 'b'
            }
        }
        params = TestOptunaParams.MyParams.from_dict(params)
        assert params.x.low == 2

    def test_load_state_dict_produces_correct_values(self):

        p = {
            'x': 2
        }
        params = TestOptunaParams.MyParams()
        params.load_state_dict(p)
        assert params.x == 2


    def test_state_dict_produces_correct_values(self):

        params = TestOptunaParams.MyParams(x=2).state_dict()
        
        assert params['x'] == 2

