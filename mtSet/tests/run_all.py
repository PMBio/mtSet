"""run all tests"""

import unittest

if __name__ == '__main__':

    # Gather all tests in suite
    tests = unittest.TestLoader().discover('.','test*')
    suite = unittest.TestSuite(tests)

    # run all tests
    unittest.TextTestRunner(verbosity=2).run(suite)

