# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

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
class PolyTests(unittest.TestCase):

    def check_close(self, a, b):
        a_ = np.array(a)
        b_ = np.array(b)
        self.assertTrue(not np.any(np.abs(a_ - b_) > EPSILON))

    def test_poly_length(self):
        self.assertEqual(1, pysiglib.poly_length(0, 0))
        self.assertEqual(1, pysiglib.poly_length(0, 1))
        self.assertEqual(1, pysiglib.poly_length(1, 0))

        self.assertEqual(435848050, pysiglib.poly_length(9, 9))
        self.assertEqual(11111111111, pysiglib.poly_length(10, 10))
        self.assertEqual(313842837672, pysiglib.poly_length(11, 11))

        self.assertEqual(10265664160401, pysiglib.poly_length(400, 5))

    def test_poly_mult_random(self):
        for deg in range(1, 6):
            X1 = np.random.uniform(size=(100, 5))
            X2 = np.random.uniform(size=(100, 5))
            X = np.concatenate((X1, X2), axis=0)
            X2 = np.concatenate((X1[[-1], :], X2), axis = 0)
            sig1 = pysiglib.signature(X1, deg)
            sig2 = pysiglib.signature(X2, deg)
            sig = pysiglib.signature(X, deg)
            sig_mult = pysiglib.poly_mult(sig1, sig2, 5, deg)
            self.check_close(sig, sig_mult)

    def test_poly_mult_random_batch(self):
        for deg in range(1, 6):
            X1 = np.random.uniform(size=(32, 100, 5))
            X2 = np.random.uniform(size=(32, 100, 5))
            X = np.concatenate((X1, X2), axis=1)
            X2 = np.concatenate((X1[:, [-1], :], X2), axis=1)
            sig1 = pysiglib.signature(X1, deg)
            sig2 = pysiglib.signature(X2, deg)
            sig = pysiglib.signature(X, deg)
            sig_mult = pysiglib.poly_mult(sig1, sig2, 5, deg)
            self.check_close(sig, sig_mult)

    def test_mult_poly_cuda_err(self):
        x = torch.tensor([[0.]], dtype = torch.float64, device = "cuda")
        with self.assertRaises(ValueError):
            pysiglib.poly_mult(x, x, 1, 0)


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

    def test_cuda_err(self):
        x = torch.tensor([[0.],[1.]], device = "cuda")
        with self.assertRaises(ValueError):
            pysiglib.signature(x, 2)

class SigKernelTests(unittest.TestCase):

    def check_close(self, a, b):
        a_ = np.array(a)
        b_ = np.array(b)
        self.assertTrue(not np.any(np.abs(a_ - b_) > EPSILON))

    def run_random(self, device):
        for _ in range(5):
            for dyadic_order in range(3):
                X = np.random.uniform(size=(32, 50, 5))
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

class InvalidParameterTests(unittest.TestCase):
    def test_poly_length_value_error(self):
        with self.assertRaises(ValueError):
            pysiglib.poly_length(-1, 2)

        with self.assertRaises(ValueError):
            pysiglib.poly_length(1, -2)


    def test_signature_type_error(self):
        with self.assertRaises(TypeError):
            pysiglib.signature('a', 2, False, False, False, False)

        with self.assertRaises(TypeError):
            pysiglib.signature(np.array(['a', 'b']), 2, False, False, False, False)

        with self.assertRaises(TypeError):
            pysiglib.signature(np.array([[0], [1]]), 'a', False, False, False, False)

        with self.assertRaises(TypeError):
            pysiglib.signature(np.array([[0], [1]]), 2, 'a', False, False, False)

        with self.assertRaises(TypeError):
            pysiglib.signature(np.array([[0], [1]]), 2, False, 'a', False, False)

        with self.assertRaises(TypeError):
            pysiglib.signature(np.array([[0], [1]]), 2, False, False, 'a', False)

        with self.assertRaises(TypeError):
            pysiglib.signature(np.array([[[0], [1]]]), 2, False, False, False, 'a')

    def test_signature_value_error(self):
        with self.assertRaises(ValueError):
            pysiglib.signature(np.array([0, 1]), 2)

        with self.assertRaises(ValueError):
            pysiglib.signature(np.array([[[[0]]], [[[1]]]]), 2)

        with self.assertRaises(ValueError):
            pysiglib.signature(np.array([[0], [1]]), -1)

    def test_sig_kernel_type_error(self):
        with self.assertRaises(TypeError):
            pysiglib.sig_kernel('a', np.array([[0], [1]]), 2)

        with self.assertRaises(TypeError):
            pysiglib.sig_kernel(np.array([[0], [1]]), 'a', 2)

        with self.assertRaises(TypeError):
            pysiglib.sig_kernel(np.array([[0], [1]]), np.array([[0], [1]]), 'a')

    def test_sig_kernel_value_error(self):
        with self.assertRaises(ValueError):
            pysiglib.sig_kernel(np.array([0, 1]), np.array([[0], [1]]), 2)

        with self.assertRaises(ValueError):
            pysiglib.sig_kernel(np.array([[0], [1]]), np.array([0, 1]), 2)

        with self.assertRaises(ValueError):
            pysiglib.sig_kernel(np.array([[[[0]]], [[[1]]]]), np.array([[0], [1]]), 2)

        with self.assertRaises(ValueError):
            pysiglib.sig_kernel(np.array([[0], [1]]), np.array([[[[0]]], [[[1]]]]), 2)

        with self.assertRaises(ValueError):
            pysiglib.sig_kernel(np.array([[0], [1]]), np.array([[0], [1]]), -2)



if __name__ == '__main__':
    unittest.main()
