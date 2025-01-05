import unittest
import pysiglib
import iisignature
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

EPSILON = 1e-10

class GeneralTests(unittest.TestCase):

    def test_polyLength(self):
        self.assertEqual(1, pysiglib.polyLength(0, 0))
        self.assertEqual(1, pysiglib.polyLength(0, 1))
        self.assertEqual(1, pysiglib.polyLength(1, 0))

        self.assertEqual(435848050, pysiglib.polyLength(9, 9))
        self.assertEqual(11111111111, pysiglib.polyLength(10, 10))
        self.assertEqual(313842837672, pysiglib.polyLength(11, 11))

        self.assertEqual(10265664160401, pysiglib.polyLength(400, 5))

class SignatureTests(unittest.TestCase):

    def test_trivial(self):
        sig = pysiglib.signature(np.array([[0,0], [1,1]]), 0)
        self.assertTrue(not np.any(sig - np.array([1.]) > EPSILON))

        sig = pysiglib.signature(np.array([[0, 0], [1, 1]]), 1)
        self.assertTrue(not np.any(sig - np.array([1., 1., 1.]) > EPSILON))

        sig = pysiglib.signature(np.array([[0, 0]]), 1)
        self.assertTrue(not np.any(sig - np.array([1., 0., 0.]) > EPSILON))

    def test_random(self):
        for deg in range(1, 6):
            X = np.random.uniform(size=(100, 5))
            iisig = iisignature.sig(X, deg)
            sig = pysiglib.signature(X, deg)
            self.assertTrue(not np.any(iisig - sig[1:] > EPSILON))

    def test_randomBatch(self):
        for deg in range(1, 6):
            X = np.random.uniform(size=(32, 100, 5))
            iisig = iisignature.sig(X, deg)
            sig = pysiglib.signature(X, deg, parallel = False)
            self.assertTrue(not np.any(iisig - sig[:, 1:] > EPSILON))
            sig = pysiglib.signature(X, deg, parallel = True)
            self.assertTrue(not np.any(iisig - sig[:, 1:] > EPSILON))

    def test_randomInt(self):
        for deg in range(1, 6):
            X = np.random.randint(low=-2, high=2, size=(100, 5))
            iisig = iisignature.sig(X, deg)
            sig = pysiglib.signature(X, deg)
            self.assertTrue(not np.any(iisig - sig[1:] > EPSILON))

    def test_randomIntBatch(self):
        for deg in range(1, 6):
            X = np.random.randint(low=-2, high=2, size=(32, 100, 5))
            iisig = iisignature.sig(X, deg)
            sig = pysiglib.signature(X, deg, parallel = False)
            self.assertTrue(not np.any(iisig - sig[:, 1:] > EPSILON))
            sig = pysiglib.signature(X, deg, parallel = True)
            self.assertTrue(not np.any(iisig - sig[:, 1:] > EPSILON))


if __name__ == '__main__':
    unittest.main()