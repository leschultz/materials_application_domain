from mad.datasets import load_data
import unittest


class ml_test(unittest.TestCase):

    def test_loads(self):
        '''
        Test data import for all supported sets.
        '''

        load_data.test()
        load_data.friedman()
        load_data.diffusion()
        load_data.super_cond()
        load_data.perovskite_stability()
        load_data.electromigration()
        load_data.thermal_conductivity()
        load_data.dielectric_constant()
        load_data.double_perovskites_gap()
        load_data.elastic_tensor()
        load_data.heusler_magnetic()
        load_data.piezoelectric_tensor()
        load_data.steel_yield_strength()


if __name__ == '__main__':
    unittest.main()
