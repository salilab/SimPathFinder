import warnings
from run_server import RunServerClassifier
import sys
import unittest


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)
    return do_test


class Testing(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Testing, self).__init__(*args, **kwargs)

    def test_check_format(self):
        R = RunServerClassifier()
        out = R.check_format('ec:1.1.1,ec:1.1,ec:2.4.5.1,ec:1.1.1.1')
        self.assertEqual(0, out[0])
        out = R.check_format('ec:1.1.1.1')
        self.assertEqual(0, out[0])
        out = R.check_format('ec:1.1.1.1.1')
        self.assertEqual(1, out[0])
        out = R.check_format('ec:1.1.1.8')
        self.assertEqual(0, out[0])
        out = R.check_format('ec:1.1.1.1.8')
        self.assertEqual(1, out[0])
        out = R.check_format('ec:1.1.1.3m')
        self.assertEqual(1, out[0])
        out = R.check_format('ec:1.1.1')
        self.assertEqual(0, out[0])
        out = R.check_format('ec:1.1')
        self.assertEqual(0, out[0])
        out = R.check_format('ec:8.1')
        self.assertEqual(1, out[0])
        out = R.check_format('ec:1.42.1.7')
        self.assertEqual(0, out[0])
        out = R.check_format('ec:1.2.3.4,ec:2.3.4.44,ec:3.1.23.1')
        self.assertEqual(0, out[0])
        out = R.check_format('JR:1')
        self.assertEqual(1, out[0])
        out = R.check_format('JR..')
        self.assertEqual(1, out[0])


if __name__ == '__main__':
    unittest.main(warnings='ignore')
