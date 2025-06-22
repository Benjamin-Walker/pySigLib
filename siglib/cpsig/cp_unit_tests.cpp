/* Copyright 2025 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#include "CppUnitTest.h"
#include "cpsig.h"
#include "cp_tensor_poly.h"
#include "cp_path.h"
#include "cp_signature.h"
#include "cp_sig_kernel.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <span>
#include <cmath>

#define EPSILON 1e-13

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

double dot_product(double* a, double* b, int n) {
    double res = 0;
    for (int i = 0; i < n; ++i) {
        res += *(a + i) * *(b + i);
    }
    return res;
}

void gram_(
    double* path1,
    double* path2,
    double* out,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2
) {
    double* out_ptr = out;

    uint64_t flat_path1_length = length1 * dimension;
    uint64_t flat_path2_length = length2 * dimension;

    double* path1_start = path1;
    double* path1_end = path1 + flat_path1_length;

    double* path2_start = path2;
    double* path2_end = path2 + flat_path2_length;

    for (uint64_t b = 0; b < batch_size; ++b) {

        for (double* path1_ptr = path1_start; path1_ptr < path1_end - dimension; path1_ptr += dimension) {
            for (double* path2_ptr = path2_start; path2_ptr < path2_end - dimension; path2_ptr += dimension) {
                *(out_ptr++) = dot_product(path1_ptr + dimension, path2_ptr + dimension, dimension)
                    - dot_product(path1_ptr + dimension, path2_ptr, dimension)
                    - dot_product(path1_ptr, path2_ptr + dimension, dimension)
                    + dot_product(path1_ptr, path2_ptr, dimension);
            }
        }

        path1_start += flat_path1_length;
        path1_end += flat_path1_length;
        path2_start += flat_path2_length;
        path2_end += flat_path2_length;
    }
}


std::vector<int> int_test_data(uint64_t dimension, uint64_t length) {
    std::vector<int> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back(i);
    }
    return data;
}

template<typename FN, typename T, typename... Args>
void check_result(FN f, std::vector<T>& path, std::vector<double>& true_, Args... args) {
    std::vector<double> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    f(path.data(), out.data(), args...);

    for (uint64_t i = 0; i < true_.size(); ++i)
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs( - 1. - out[true_.size()]) < EPSILON);
}

template<typename FN, typename T, typename... Args>
void check_result_2(FN f, std::vector<T>& path1, std::vector<T>& path2, std::vector<double>& true_, Args... args) {
    std::vector<double> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    f(path1.data(), path2.data(), out.data(), args...);

    for (uint64_t i = 0; i < true_.size(); ++i)
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs(-1. - out[true_.size()]) < EPSILON);
}

namespace cpSigTests
{
    TEST_CLASS(PolyTest)
    {
    public:
        TEST_METHOD(PolyLengthTest)
        {
            Assert::AreEqual((uint64_t)1, sig_length(0, 0));
            Assert::AreEqual((uint64_t)1, sig_length(0, 0));
            Assert::AreEqual((uint64_t)1, sig_length(0, 1));
            Assert::AreEqual((uint64_t)1, sig_length(1, 0));

            Assert::AreEqual((uint64_t)435848050, sig_length(9, 9));
            Assert::AreEqual((uint64_t)11111111111, sig_length(10, 10));
            Assert::AreEqual((uint64_t)313842837672, sig_length(11, 11));

            Assert::AreEqual((uint64_t)10265664160401, sig_length(400, 5));
        }

        TEST_METHOD(PolyMultTestLinear)
        {
            // Test signatures of linear 2d paths
            auto f = sig_combine;
            std::vector<double> poly = { 1., 1., 1., 1./2, 1./2, 1./2, 1./2 };
            std::vector<double> true_res = { 1., 2., 2., 2., 2., 2., 2. };

            check_result_2(f, poly, poly, true_res, 2, 2);
        }

        TEST_METHOD(PolyMultSigTest)
        {
            uint64_t dimension = 2, length = 4, degree = 5;
            auto f = sig_combine;
            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
            std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

            uint64_t poly_len_ = sig_length(dimension, degree);

            std::vector<double> poly1;
            poly1.resize(poly_len_);
            signature_double(path1.data(), poly1.data(), dimension, 3, degree);

            std::vector<double> poly2;
            poly2.resize(poly_len_);
            signature_double(path2.data(), poly2.data(), dimension, 3, degree);

            std::vector<double> true_sig;
            true_sig.resize(poly_len_);
            signature_double(path.data(), true_sig.data(), dimension, 5, degree);
            check_result_2(f, poly1, poly2, true_sig, dimension, degree);
        }

        TEST_METHOD(BatchPolyMultSigTest)
        {
            uint64_t batch_size = 3, dimension = 2, length = 4, degree = 2;
            auto f = batch_sig_combine;
            std::vector<double> path1 = { 0., 0., 0.25, 0.25, 0.5, 0.5,
                0., 0., 0.4, 0.4, 0.6, 0.6,
                0., 0., 1., 0.5, 4., 0. };
            std::vector<double> path2 = { 0.5, 0.5, 1., 1.,
                0.6, 0.6, 1., 1.,
                4., 0., 0., 1. };
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            uint64_t res_len_ = sig_length(dimension, degree) * batch_size;

            std::vector<double> poly1;
            poly1.resize(res_len_);
            batch_signature_double(path1.data(), poly1.data(), batch_size, dimension, 3, degree);

            std::vector<double> poly2;
            poly2.resize(res_len_);
            batch_signature_double(path2.data(), poly2.data(), batch_size, dimension, 2, degree);

            std::vector<double> true_sig;
            true_sig.resize(res_len_);
            batch_signature_double(path.data(), true_sig.data(), batch_size, dimension, 4, degree);
            check_result_2(f, poly1, poly2, true_sig, batch_size, dimension, degree, false);
            check_result_2(f, poly1, poly2, true_sig, batch_size, dimension, degree, true);
        }

        TEST_METHOD(BatchPolyMultStressTest)
        {
            uint64_t batch_size = 1000, dimension = 5, degree = 5;

            std::vector<double> poly;
            poly.resize(batch_size * sig_length(dimension, degree));
            std::fill(poly.data(), poly.data() + poly.size(), 1.);

            std::vector<double> out;
            out.resize(batch_size * sig_length(dimension, degree));

            int err = batch_sig_combine(poly.data(), poly.data(), out.data(), batch_size, dimension, degree, true);
            Assert::IsFalse(err);
        }
    };

    TEST_CLASS(PathTest)
    {
    public:
        TEST_METHOD(ConstructorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Path<int> path2(std::span<int>(data), dimension, length);
            Path<int> path3(path2);

            Assert::IsTrue(path == path2);
            Assert::IsTrue(path == path3);
        }
        TEST_METHOD(SqBracketOperatorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Point<int> pt = path[3];
            Assert::AreEqual(data.data() + 3 * dimension, pt.data());
        }
        TEST_METHOD(FirstLastTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);
            
            Point<int> first = path.begin();
            Point<int> last = path.end();
            --last;

            for (uint64_t j = 0; j < dimension; ++j){
                Assert::AreEqual(data[j], first[j]);
                Assert::AreEqual(data[(length - 1) * dimension + j], last[j]);
            }
        }

#ifdef _DEBUG
        TEST_METHOD(OutOfBoundsTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);

            try {
                path[length];
            }
            catch(const std::out_of_range& e){
                Assert::AreEqual("Argument out of bounds in Path::operator[]", e.what());
            }
            catch (...) {
                Assert::Fail();
            }

        }
#endif
    };

    TEST_CLASS(PointTest) {
    public:
        TEST_METHOD(ConstructorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);

            Point<int> pt1(&path, 0);
            Point<int> pt2(&path, length - 1);
            Point<int> pt3(pt2);

            Assert::IsTrue(pt1 != pt2);
            Assert::IsTrue(pt2 == pt3);
        }

        TEST_METHOD(SqBracketOperatorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Point<int> pt(&path, 0);

            for (uint64_t i = 0; i < dimension; ++i)
                Assert::AreEqual(data[i], pt[i]);
        }

        TEST_METHOD(IncrementTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Point<int> pt1(&path, 0);
            Point<int> pt2(&path, 0);

            for (uint64_t i = 0; i < length; ++i) {
                for (uint64_t j = 0; j < dimension; ++j) {
                    Assert::AreEqual(data[i * dimension + j], pt1[j]);
                    Assert::AreEqual(data[i * dimension + j], pt2[j]);
                }
                ++pt1;
                pt2++;
            }
        }

        TEST_METHOD(DecrementTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Point<int> pt1 = --path.end();
            Point<int> pt2 = --path.end();

            for (int64_t i = length - 1; i >= 0; --i) {
                for (uint64_t j = 0; j < dimension; ++j) {
                    Assert::AreEqual(data[i * dimension + j], pt1[j]);
                    Assert::AreEqual(data[i * dimension + j], pt2[j]);
                }
                --pt1;
                pt2--;
            }
        }

        TEST_METHOD(AssignmentTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Point<int> pt1 = path.begin();
            Point<int> pt2 = pt1;

            for (uint64_t i = 0; i < dimension; ++i) {
                Assert::AreEqual(data[i], pt1[i]);
                Assert::AreEqual(data[i], pt2[i]);
            }
        }

        TEST_METHOD(AdvanceTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Point<int> pt(&path, 0);

            for (uint64_t i = 0; i < length; ++i) {
                for (uint64_t j = 0; j < dimension; ++j) {
                    Assert::AreEqual(data[i * dimension + j], pt[j]);
                }
                pt.advance(1);
            }
        }
        TEST_METHOD(TimeAugTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length, true);

            int index = 0;

            for (Point<int> pt = path.begin(); pt != path.end(); ++pt) {
                for (int i = 0; i < dimension; i++) {
                    int val = data[index * dimension + i];
                    Assert::AreEqual(val, pt[i]);
                }
                Assert::AreEqual(index, pt[dimension]);
                index++;
            }
        }
        TEST_METHOD(LeadLagTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length, false, true);

            int index = 0;
            bool parity = false;

            for (Point<int> pt = path.begin(); pt != path.end(); ++pt) {
                for (int i = 0; i < dimension; i++) {
                    int val = data[index * dimension + i];
                    Assert::AreEqual(val, pt[i]);
                }

                for (int i = 0; i < dimension; i++) {
                    int val = 0;
                    if (!parity)
                        val = data[(index + 1) * dimension + i];
                    else
                        val = data[(index + 2) * dimension + i];
                    Assert::AreEqual(val, pt[dimension + i]);
                }
                if(parity)
                    index++;
                parity = !parity;
            }
        }
        TEST_METHOD(TimeAugLeadLagTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length, true, true);

            int time = 0;
            int index = 0;
            bool parity = false;

            for (Point<int> pt = path.begin(); pt != path.end(); ++pt) {
                for (int i = 0; i < dimension; i++) {
                    int val = data[index * dimension + i];
                    Assert::AreEqual(val, pt[i]);
                }

                for (int i = 0; i < dimension; i++) {
                    int val = 0;
                    if (!parity)
                        val = data[(index + 1) * dimension + i];
                    else
                        val = data[(index + 2) * dimension + i];
                    Assert::AreEqual(val, pt[dimension + i]);
                }

                Assert::AreEqual(time, pt[2 * dimension]);

                if (parity) {
                    index++;
                    time--;
                }
                parity = !parity;
                time += 2;
            }
        }
#ifdef _DEBUG
        TEST_METHOD(OutOfBoundsTest) 
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Point<int> pt = path.end();

            try { pt[0]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual("Point is out of bounds for given path in Point::operator[]", e.what()); }
            catch (...) { Assert::Fail(); }

            pt = path.begin();
            try { pt[5]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual("Argument out of bounds in Point::operator[]", e.what()); }
            catch (...) { Assert::Fail(); }

            Path<int> path2(path, true, false);
            pt = path2.begin();
            try { pt[5]; }
            catch (...) { Assert::Fail(); }

            try { pt[6]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual("Argument out of bounds in Point::operator[]", e.what()); }
            catch (...) { Assert::Fail(); }

            Path<int> path3(path, false, true);
            pt = path3.begin();
            try { pt[9]; }
            catch (...) { Assert::Fail(); }

            try { pt[10]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual(e.what(), "Argument out of bounds in Point::operator[]"); }
            catch (...) { Assert::Fail(); }

            Path<int> path4(path, true, true);
            pt = path4.begin();
            try { pt[10]; }
            catch (...) { Assert::Fail(); }

            try { pt[11]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual("Argument out of bounds in Point::operator[]", e.what()); }
            catch (...) { Assert::Fail(); }
        }
#endif
    };

    TEST_CLASS(signatureDoubleTest)
    {
    public:
        TEST_METHOD(TrivialCases) {
            auto f = signature_double;
            std::vector<double> path;
            std::vector<double> true_sig;
            Assert::AreEqual(2, f(path.data(), true_sig.data(), 0, 0, 0, false, false, true));

            true_sig.push_back(1.);
            check_result(f, path, true_sig, 1, 0, 0, false, false, true);

            path.push_back(0.);
            check_result(f, path, true_sig, 1, 1, 0, false, false, true);

            true_sig.push_back(0.);
            check_result(f, path, true_sig, 1, 0, 1, false, false, true);
            check_result(f, path, true_sig, 1, 1, 1, false, false, true);

            path.push_back(1.);
            true_sig[1] = 1.;
            check_result(f, path, true_sig, 1, 2, 1, false, false, true);
        }
        TEST_METHOD(LinearPathTest) {
            auto f = signature_double;
            uint64_t dimension = 2, length = 3, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
            std::vector<double> true_sig;
            true_sig.resize(level_4_start);
            true_sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { true_sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { true_sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { true_sig[i] = 1 / 6.; }
            check_result(f, path, true_sig, dimension, length, degree, false, false, true);
        }

        TEST_METHOD(LinearPathTest2) {
            auto f = signature_double;
            uint64_t dimension = 2, length = 4, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> path = { 0.,0., 0.25, 0.25, 0.75, 0.75, 1.,1. };
            std::vector<double> true_sig;
            true_sig.resize(level_4_start);
            true_sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { true_sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { true_sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { true_sig[i] = 1 / 6.; }
            check_result(f, path, true_sig, dimension, length, degree, false, false, true);
        }

        TEST_METHOD(ManualSigTest) {
            auto f = signature_double;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            check_result(f, path, true_sig, dimension, length, degree, false, false, true);
        }
        TEST_METHOD(ManualSigTest2) {
            auto f = signature_int32;
            uint64_t dimension = 3, length = 4, degree = 3;
            std::vector<int> path = { 9, 5, 8, 5, 3, 0, 0, 2, 6, 4, 0, 2 };
            std::vector<double> true_sig = { 1., -5., - 5., - 6., 12.5, 24.5,
                                                5., 0.5, 12.5, 9., 25.,
                                               21., 18., - 20.5 - 1./3, - 77.5 - 1./3, 11.,
                                               33. + 1./6, - 45.5 - 1./3, - 42. - 1./3, - 47., 5. + 2./3,
                                              - 18., - 17.5 - 1./3, - 30.5 - 1./3, 11. + 2./3, 14. + 1./6,
                                              - 20.5 - 1./3, - 19., - 14. - 1./3, - 7., - 16. - 2./3,
                                              - 39., - 110. - 1./3, 6., - 1./3, - 49.,
                                              - 20. - 2./3, - 78., - 52. - 2./3, - 36. };
            check_result(f, path, true_sig, dimension, length, degree, false, false, true);
        }

        TEST_METHOD(BatchSigTest) {
            auto f = batch_signature_double;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> true_sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 0., 1., 0., 1., -1., 0.5 };

            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, true, false);
            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, true, true);
        }

        TEST_METHOD(BatchSigTestDegree1) {
            auto f = batch_signature_double;
            uint64_t dimension = 2, length = 4, degree = 1;
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> true_sig = { 1., 1., 1.,
                1., 1., 1.,
                1., 0., 1. };

            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, true, false);
            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, true, true);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = signature_int32;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<int> path = { 0, 5, 2, 4, 9 };
            std::vector<double> true_sig = { 1., 9., 4., 40.5, 15.5, 20.5, 8., 121.5, 37.5,
                                64.5, 24.5, 60., 13., 34.5, 10. + 2./3 };
            check_result(f, path, true_sig, dimension, length, degree, true, false, true);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = signature_int32;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<int> path = { 0, 5, 2, 4, 9 };
            std::vector<double> true_sig = { 1., 4., 4., 8., 20., -4., 8., 10. + 2./3, 35., 10., 85., -13., -90., 37., 10. + 2./3};
            check_result(f, path, true_sig, dimension, length, degree, false, true, true);
        }

        TEST_METHOD(BigLeadLagTest) {
            auto f = batch_signature_double;
            uint64_t dimension = 2, length = 10, degree = 2, batch = 1;
            std::vector<double> path;
            path.resize(batch * length * dimension);
            std::vector<double> out;
            out.resize(batch * sig_length(dimension * 2, degree));
            f(path.data(), out.data(), batch, dimension, length, degree, false, true, true, false);
        }
    };

    TEST_CLASS(sigKernelTest) {
    public:
        TEST_METHOD(LinearPathTest) {
            auto f = sig_kernel;
            uint64_t dimension = 2, length = 3;
            std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
            std::vector<double> true_sig = { 4.256702149748847 };
            std::vector<double> gram(length * length);
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            check_result(f, gram, true_sig, dimension, length, length, 2, 2);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_kernel;
            uint64_t dimension = 3, length = 4;
            std::vector<double> path = { .9, .5, .8, .5, .3, .0, .0, .2, .6, .4, .0, .2 };
            std::vector<double> true_sig = { 2.1529809076880486 };
            std::vector<double> gram(length * length);
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            check_result(f, gram, true_sig, dimension, length, length, 2, 2);
        }

        TEST_METHOD(NonSquare) {
            auto f = sig_kernel;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> path2 = { 0., 1., 2. };
            std::vector<double> true_sig = { 11. };
            std::vector<double> gram(length1 * length2);
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, dimension, length1, length2, 0, 0);
        }
    };
}