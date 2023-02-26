from .. import study2 as _study
from dataclasses import InitVar, dataclass
from unittest.mock import Mock
from dataclasses import field


class TestDefault:

    def test_suggest_equals_correct_number(self):

        default = _study.Default(
            "x", 1
        )
        
        assert default.suggest(object(), '') == 1

    def test_from_dict_equals_correct_number(self):

        default = _study.Default.from_dict(
            name= 'x',
            val= 1
        )
        
        assert default.default == 1
        assert default.name == 'x'


class TestInt:

    def test_suggest_equals_number_between_zero_and_four(self):

        int_ = _study.Int(
            "x", 0, 4, 1
        )
        mock = Mock()
        mock.suggest_int.side_effect = lambda self, low, high: 3
        assert int_.suggest(mock, '') == 3

    def test_suggest_equals_value_from_deick(self):

        int_ = _study.Int.from_dict(
            name= 'x',
            low= 0,
            high= 4,
            default= 0
        )
        assert int_.name == 'x'
        assert int_.default == 0


class TestExpInt:

    def test_suggest_equals_eight(self):

        int_ = _study.ExpInt(
            "x", 0, 4, 2
        )
        mock = Mock()
        mock.suggest_int.side_effect = lambda self, low, high: 3
        assert int_.suggest(mock, '') == 8

    def test_suggest_equals_value_from_dict(self):

        int_ = _study.ExpInt.from_dict(
            name= 'x',
            low= 0,
            high= 4,
            default= 0,
            base= 1
        )
        assert int_.name == 'x'
        assert int_.default == 1

    def test_suggest_equals_value_from_dict_when_2(self):

        int_ = _study.ExpInt.from_dict(
            name= 'x',
            low= 0,
            high= 4,
            default= 2,
            base= 2
        )
        assert int_.name == 'x'
        assert int_.default == 4


class TestBool:

    def test_suggest_equals_false(self):

        int_ = _study.Bool(
            "x", True
        )
        mock = Mock()
        mock.suggest_discrete_uniform.side_effect = lambda self, low, high, other: 0.2
        assert int_.suggest(mock, '') is False

    def test_suggest_equals_true(self):

        int_ = _study.Bool(
            "x", True
        )
        mock = Mock()
        mock.suggest_discrete_uniform.side_effect = lambda self, low, high, other: 0.6
        assert int_.suggest(mock, '') is True

    def test_suggest_equals_value_from_dict(self):

        int_ = _study.Bool.from_dict(
            name= 'x',
            default= False
        )
        assert int_.name == 'x'
        assert int_.default is False


class TestFloat:

    def test_suggest_equals_correct_value(self):

        int_ = _study.Float(
            "x", 1, 4.5, 1.5
        )
        mock = Mock()
        mock.suggest_uniform.side_effect = lambda self, low, high: 2.5
        assert int_.suggest(mock, '') == 2.5

    def test_suggest_equals_value_from_dict(self):

        int_ = _study.Float.from_dict(
            name= 'x',
            low= 1.,
            high= 4.5,
            default= 1.5
        )
        assert int_.name == 'x'
        assert int_.default == 1.5


class TestCategorical:

    def test_suggest_equals_correct_value(self):

        int_ = _study.Categorical(
            "x", ['big', 'small'], 'small'
        )
        mock = Mock()
        mock.suggest_categorical.side_effect = lambda self, cats: 'big'
        assert int_.suggest(mock, '') == 'big'

    def test_suggest_equals_value_from_dict(self):

        int_ = _study.Categorical.from_dict(
            name='x',
            categories= ['big', 'small'],
            default= 'big'
        )
        assert int_.name == 'x'
        assert int_.default == 'big'


class TestArray:

    def test_suggest_equals_correct_value(self):

        int_ = _study.Array(
            "x", low=1, high=3, params={
                'sub': _study.Bool('sub')
            }
        )
        mock = Mock()
        mock.suggest_int.side_effect = lambda self, low, high: 2
        mock.suggest_discrete_uniform.side_effect = lambda self, low, high, other: 0.5

        assert int_.suggest(mock, '') == [{'sub': False}, {'sub': False}]

    def test_suggest_equals_value_from_dict(self):

        arr = _study.Array.from_dict(
            name= 'x',
            low= 1,
            high= 3,
            sub= {'x': {'type': 'Bool', 'name': 'x', 'default': True }},
            default= [[False]]
        )
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
        params =_study.convert_params(params)
        assert isinstance(params['b'], _study.Int)


class TestOptunaParams:

    @dataclass
    class MyParams(_study.OptunaParams):

        x: int = _study.Int('x', 0, 4, 1)

    def test_sample_produces_my_params_with_defined_value(self):

        mock = Mock()
        mock.suggest_int.side_effect = lambda self, low, high: 2

        params = TestOptunaParams.MyParams().suggest(mock, '')
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
        params = TestOptunaParams.MyParams.from_dict(**params)
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

# my_params2 = {
#     'z':_study.Float('y', 0, 2)
# }

# my_params

# @dataclass
# class MyParams2(_study.OptunaParams):
#     z: float = _study.Float('y', 0, 2)


# @dataclass
# class MyParams(_study.OptunaParams):

#     x: int = _study.Int('x', 0, 4, 1)
#     y: MyParams2 = field(default_factory=MyParams2)


# @dataclass
# class MyParamsB(_study.OptunaParams):

#     x: int = _study.Int('x', 0, 4, 1)
#     z: InitVar[MyParams2] = None

#     def __post_init__(self, z: MyParams2):
#         super().__post_init__()
#         self.z = MyParams2() if z is None else z
#         self._extra.append('z')
