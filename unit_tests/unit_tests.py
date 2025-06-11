import warnings
import unittest

import iisignature
import sigkernel
import numpy as np
import torch

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

EPSILON = 1e-10

class GeneralTests(unittest.TestCase):

    def test_poly_length(self):
        self.assertEqual(1, pysiglib.poly_length(0, 0))
        self.assertEqual(1, pysiglib.poly_length(0, 1))
        self.assertEqual(1, pysiglib.poly_length(1, 0))

        self.assertEqual(435848050, pysiglib.poly_length(9, 9))
        self.assertEqual(11111111111, pysiglib.poly_length(10, 10))
        self.assertEqual(313842837672, pysiglib.poly_length(11, 11))

        self.assertEqual(10265664160401, pysiglib.poly_length(400, 5))

class SignatureTests(unittest.TestCase):

    def check_close(self, a, b):
        a_ = np.array(a)
        b_ = np.array(b)
        self.assertTrue(not np.any(np.abs(a_ - b_) > EPSILON))

    def test_trivial(self):
        sig = pysiglib.signature(np.array([[0,0], [1,1]]), 0)
        self.check_close(sig, [1.])

        sig = pysiglib.signature(np.array([[0, 0], [1, 1]]), 1)
        self.check_close(sig, [1., 1., 1.])

        sig = pysiglib.signature(np.array([[0, 0]]), 1)
        self.check_close(sig, [1., 0., 0.])

    def test_random(self):
        for deg in range(1, 6):
            X = np.random.uniform(size=(100, 5))
            iisig = iisignature.sig(X, deg)
            sig = pysiglib.signature(X, deg)
            self.check_close(iisig, sig[1:])

    def test_random_batch(self):
        for deg in range(1, 6):
            X = np.random.uniform(size=(32, 100, 5))
            iisig = iisignature.sig(X, deg)
            sig = pysiglib.signature(X, deg, parallel = False)
            self.check_close(iisig, sig[:, 1:])
            sig = pysiglib.signature(X, deg, parallel = True)
            self.check_close(iisig, sig[:, 1:])

    def test_random_int(self):
        for deg in range(1, 6):
            X = np.random.randint(low=-2, high=2, size=(100, 5))
            iisig = iisignature.sig(X, deg)
            sig = pysiglib.signature(X, deg)
            self.check_close(iisig, sig[1:])

    def test_random_int_batch(self):
        for deg in range(1, 6):
            X = np.random.randint(low=-2, high=2, size=(32, 100, 5))
            iisig = iisignature.sig(X, deg)
            sig = pysiglib.signature(X, deg, parallel = False)
            self.check_close(iisig, sig[:, 1:])
            sig = pysiglib.signature(X, deg, parallel = True)
            self.check_close(iisig, sig[:, 1:])

class SigKernelTests(unittest.TestCase):

    def check_close(self, a, b):
        a_ = np.array(a)
        b_ = np.array(b)
        self.assertTrue(not np.any(np.abs(a_ - b_) > EPSILON))

    def run_random(self, device):
        for _ in range(5):
            for dyadic_order in range(3):
                X = np.random.uniform(size=(32, 100, 5))
                Y = np.random.uniform(size=(32, 100, 5))

                X = torch.tensor(X, device=device)
                Y = torch.tensor(Y, device=device)

                static_kernel = sigkernel.LinearKernel()
                signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
                kernel1 = signature_kernel.compute_kernel(X, Y, 100)
                kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order)

                self.check_close(kernel1.cpu(), kernel2.cpu())

    def test_random_cpu(self):
        self.run_random("cpu")

    def test_random_cuda(self):
        if pysiglib.BUILT_WITH_CUDA:
            self.run_random("cuda")
        else:
            warnings.warn("Package built without CUDA, skipping test...")

    def test_sig_kernel_numpy1(self):
        x = np.array([[0,1],[3,2]])
        pysiglib.sig_kernel(x,x,0)

    def test_sig_kernel_numpy2(self):
        x = np.array([[[0,1],[3,2]]])
        pysiglib.sig_kernel(x,x,0)



if __name__ == '__main__':
    unittest.main()
