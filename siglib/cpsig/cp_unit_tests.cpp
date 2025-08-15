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

#define EPSILON 1e-10

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

double dot_product(double* a, double* b, uint64_t n) {
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
            check_result_2(f, poly1, poly2, true_sig, batch_size, dimension, degree, 1);
            check_result_2(f, poly1, poly2, true_sig, batch_size, dimension, degree, -1);
        }

        TEST_METHOD(BatchPolyMultStressTest)
        {
            uint64_t batch_size = 1000, dimension = 5, degree = 5;

            std::vector<double> poly;
            poly.resize(batch_size * sig_length(dimension, degree));
            std::fill(poly.data(), poly.data() + poly.size(), 1.);

            std::vector<double> out;
            out.resize(batch_size * sig_length(dimension, degree));

            int err = batch_sig_combine(poly.data(), poly.data(), out.data(), batch_size, dimension, degree, -1);
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
                Assert::AreEqual(static_cast<double>(data[j]), first[j]);
                Assert::AreEqual(static_cast<double>(data[(length - 1) * dimension + j]), last[j]);
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
                Assert::AreEqual(static_cast<double>(data[i]), pt[i]);
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
                    Assert::AreEqual(static_cast<double>(data[i * dimension + j]), pt1[j]);
                    Assert::AreEqual(static_cast<double>(data[i * dimension + j]), pt2[j]);
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
                    Assert::AreEqual(static_cast<double>(data[i * dimension + j]), pt1[j]);
                    Assert::AreEqual(static_cast<double>(data[i * dimension + j]), pt2[j]);
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
                Assert::AreEqual(static_cast<double>(data[i]), pt1[i]);
                Assert::AreEqual(static_cast<double>(data[i]), pt2[i]);
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
                    Assert::AreEqual(static_cast<double>(data[i * dimension + j]), pt[j]);
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
                    double val = data[index * dimension + i];
                    Assert::AreEqual(val, pt[i]);
                }
                Assert::IsTrue(abs(static_cast<double>(index) / (length - 1) - pt[dimension]) < EPSILON);
                index++;
            }
        }
        TEST_METHOD(LeadLagTest)
        {
            uint64_t dimension = 2, length = 5;
            std::vector<int> data = {2, 6, 7, 1, 7, 0, 1, 7, 6, 3};
            std::vector<int> true_ = { 2, 6, 2, 6, 2, 6, 7, 1, 7, 1, 7, 1, 7, 1, 7, 0, 7, 0, 7, 0, 7, 0, 1, 7, 1, 7, 1, 7, 1, 7, 6, 3, 6, 3, 6, 3};

            Path<int> path(data.data(), dimension, length, false, true);

            int index = 0;
            bool parity = false;

            for (Point<int> pt = path.begin(); pt != path.end(); ++pt) {
                for (int i = 0; i < path.dimension(); ++i) {
                    double val = pt[i];
                    Assert::AreEqual(static_cast<double>(true_[index]), val);
                    ++index;
                }
            }
        }
        TEST_METHOD(TimeAugLeadLagTest)
        {
            uint64_t dimension = 2, length = 5;
            std::vector<int> data = { 2, 6, 7, 1, 7, 0, 1, 7, 6, 3 };
            std::vector<double> true_ = { 2., 6., 2., 6., 0., 
                2., 6., 7., 1., 1. / 8,
                7., 1., 7., 1., 2. / 8,
                7., 1., 7., 0., 3. / 8,
                7., 0., 7., 0., 4. / 8,
                7., 0., 1., 7., 5. / 8,
                1., 7., 1., 7., 6. / 8,
                1., 7., 6., 3., 7. / 8,
                6., 3., 6., 3., 1. };

            Path<int> path(data.data(), dimension, length, true, true);

            int index = 0;
            bool parity = false;

            for (Point<int> pt = path.begin(); pt != path.end(); ++pt) {
                for (int i = 0; i < path.dimension(); ++i) {
                    double val = pt[i];
                    Assert::IsTrue(abs(static_cast<double>(true_[index]) - pt[i]) < EPSILON);
                    ++index;
                }
            }
        }

        TEST_METHOD(ReverseTimeAugTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = int_test_data(dimension, length);

            Path<int> path(data.data(), dimension, length, true);

            int index = static_cast<int>(length) - 1;

            for (Point<int> pt = --path.end(); pt != --path.begin(); --pt) {
                for (int i = 0; i < dimension; i++) {
                    double val = static_cast<double>(data[index * dimension + i]);
                    Assert::AreEqual(val, pt[i]);
                }
                Assert::IsTrue(abs(static_cast<double>(index) / (length - 1) - pt[dimension]) < EPSILON);
                --index;
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
            Assert::AreEqual(2, f(path.data(), true_sig.data(), 0, 0, 0, false, false, 1., true));

            true_sig.push_back(1.);
            check_result(f, path, true_sig, 1, 0, 0, false, false, 1., true);

            path.push_back(0.);
            check_result(f, path, true_sig, 1, 1, 0, false, false, 1., true);

            true_sig.push_back(0.);
            check_result(f, path, true_sig, 1, 0, 1, false, false, 1., true);
            check_result(f, path, true_sig, 1, 1, 1, false, false, 1., true);

            path.push_back(1.);
            true_sig[1] = 1.;
            check_result(f, path, true_sig, 1, 2, 1, false, false, 1., true);
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
            check_result(f, path, true_sig, dimension, length, degree, false, false, 1., true);
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
            check_result(f, path, true_sig, dimension, length, degree, false, false, 1., true);
        }

        TEST_METHOD(ManualSigTest) {
            auto f = signature_double;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            check_result(f, path, true_sig, dimension, length, degree, false, false, 1., true);
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
            check_result(f, path, true_sig, dimension, length, degree, false, false, 1., true);
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

            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, 1);
            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, -1);
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

            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, 1);
            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, -1);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = signature_int32;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<int> path = { 0, 5, 2, 4, 9 };
            std::vector<double> true_sig = { 1., 9., 4., 40.5, 15.5, 20.5, 8., 121.5, 37.5,
                                64.5, 24.5, 60., 13., 34.5, 10. + 2./3 };
            double end_time = length - 1.;
            check_result(f, path, true_sig, dimension, length, degree, true, false, end_time, true);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = signature_int32;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<int> path = { 0, 5, 2, 4, 9 };
            std::vector<double> true_sig = { 1., 9., 9., 40.5, 9., 72., 40.5, 121.5, 6.5, 68., -8.5, 290., 98., 275., 121.5 };
            check_result(f, path, true_sig, dimension, length, degree, false, true, 1., true);
        }

        TEST_METHOD(BigLeadLagTest) {
            auto f = batch_signature_double;
            uint64_t dimension = 2, length = 10, degree = 2, batch = 1;
            std::vector<double> path;
            path.resize(batch * length * dimension);
            std::vector<double> out;
            out.resize(batch * sig_length(dimension * 2, degree));
            f(path.data(), out.data(), batch, dimension, length, degree, false, true, 1., true, 1);
        }
    };

    TEST_CLASS(sigBackpropTest) {
    public:
        TEST_METHOD(LinearPathTest) {
            auto f = sig_backprop_double;
            uint64_t dimension = 2, length = 2, degree = 2;
            std::vector<double> path = { 0., 0., 1.,1. };
            std::vector<double> deriv = { 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -3., -3., 3., 3. };
            std::vector<double> sig = {1., 1., 1., 1./2, 1./2, 1./2, 1./2};
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, false, false, 1.);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_backprop_double;
            uint64_t dimension = 2, length = 3, degree = 2;
            std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { -7.5, -10., -0.5, 0.25, 8., 9.75 };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, false, false, 1.);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_backprop_double;
            uint64_t dimension = 2, length = 3, degree = 3;
            std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { -19.625, -23.625, -1.25, 0.625, 20.875, 23. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, false, false, 1.);
        }

        TEST_METHOD(ManualTestAsBatch) {
            auto f = batch_sig_backprop_double;
            uint64_t dimension = 2, length = 3, degree = 2;
            std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { -7.5, -10., -0.5, 0.25, 8., 9.75 };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result(f, path, true_, deriv.data(), sig.data(), 1, dimension, length, degree, false, false, 1., 1);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            auto f = batch_sig_backprop_double;
            uint64_t dimension = 2, length = 3, degree = 3;
            std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { -19.625, -23.625, -1.25, 0.625, 20.875, 23. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result(f, path, true_, deriv.data(), sig.data(), 1, dimension, length, degree, false, false, 1., 1);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = batch_sig_backprop_double;
            uint64_t dimension = 2, length = 3, degree = 3, batch_size = 3;
            std::vector<double> path = { 0., 0., 1., 2., 0.5, 1., 0., 0., 3., 2., 5., 2., 0., 0., -1., 2., 0.5, -1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12., 13., -14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -19.625, -23.625, -1.25, 0.625, 20.875, 23., -162.5, -103.5, -81.0, 245.5, 243.5, -142.0, -0.625, -0.625, 0., 0., 0.625, 0.625 };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1./48, 1./24, 1./24, 1./12, 1./24, 1./12, 1./12, 1./6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5./6, 3., 9., 2., 13., 2., 6., 1. + 1./3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1./48, -1./24, -1./24,  1./12, -1./24, 1./48, 1./48, -1./6 };
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, false, 1., 1);
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, false, 1., -1);
        }

        TEST_METHOD(TimeAugTest) {
            auto f = sig_backprop_double;
            uint64_t dimension = 1, length = 3, degree = 3;
            std::vector<double> path = { 0., 2., 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { -54., -4.5, 58.5 };
            std::vector<double> sig = { 1., 1., 2., 0.5, 2.5, -0.5, 2., 1./6, 1.5 + 1./3, -1-1./6, 2 + 1./6, 1./3, 2./3, -0.5-1./3, 1 + 1./3 };
            double end_time = length - 1.;
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, true, false, end_time);
        }

        TEST_METHOD(BatchTimeAugTest) {
            auto f = batch_sig_backprop_double;
            uint64_t dimension = 1, length = 3, degree = 3, batch_size = 2;
            std::vector<double> path = { 0., 2., 1., 0., 3., 6. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -54., -4.5, 58.5, -41., 0., 41. };
            std::vector<double> sig = { 1., 1., 2., 0.5, 2.5, -0.5, 2., 1. / 6, 1.5 + 1. / 3, -1 - 1. / 6, 2 + 1. / 6, 1. / 3, 2. / 3, -0.5 - 1. / 3, 1 + 1. / 3,
            1., 6., 2., 18., 6., 6., 2., 36., 12., 12., 4., 12., 4., 4., 4./3};
            double end_time = length - 1.;
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, false, end_time, 1);
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, false, end_time, -1);
        }

        TEST_METHOD(LeadLagTest) {
            auto f = sig_backprop_double;
            uint64_t dimension = 1, length = 3, degree = 3;
            std::vector<double> path = { 0., 2., 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { -76., 5.5, 70.5 };
            std::vector<double> sig = { 1., 1., 1., .5, -2., 3., .5, 1./6, -2., 2., 1., .5, -4., 3.5, 1./6 };
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, false, true, 1.);
        }

        TEST_METHOD(BatchLeadLagTest) {
            auto f = batch_sig_backprop_double;
            uint64_t dimension = 1, length = 3, degree = 3, batch_size = 2;
            std::vector<double> path = { 0., 2., 1., 0., 3., 6. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -76., 5.5, 70.5, -170., 0., 170. };
            std::vector<double> sig = { 1., 1., 1., .5, -2., 3., .5, 1. / 6, -2., 2., 1., .5, -4., 3.5, 1. / 6,
            1., 6., 6., 18., 9., 27., 18., 36., 13.5, 27., 13.5, 67.5, 27., 67.5, 36. };
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, true, 1., 1);
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, true, 1., -1);
        }

        TEST_METHOD(TimeAugLeadLagTest) {
            auto f = sig_backprop_double;
            uint64_t dimension = 1, length = 3, degree = 2;
            std::vector<double> path = { 0., 2., 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. };
            std::vector<double> true_ = { -98., -6., 104. };
            std::vector<double> sig = { 1., 1., 1., 4., .5, -2., 4.5, 3., .5, 5.5, -.5, -1.5, 8. };
            double end_time = length * 2. - 2.;
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, true, true, end_time);
        }

        TEST_METHOD(BatchTimeAugLeadLagTest) {
            auto f = batch_sig_backprop_double;
            uint64_t dimension = 1, length = 3, degree = 2, batch_size = 2;
            std::vector<double> path = { 0., 2., 1., 0., 3., 6. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -98., -6., 104., -34., 0., 34. };
            std::vector<double> sig = { 1., 1., 1., 4., .5, -2., 4.5, 3., .5, 5.5, -.5, -1.5, 8.,
            1., 6., 6., 4., 18., 9., 9., 27., 18., 15., 15., 9., 8. };
            double end_time = length * 2. - 2.;
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, true, end_time, 1);
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, true, end_time, -1);
        }
    };

    TEST_CLASS(sigCombineBackpropTest) {
    public:
        TEST_METHOD(ManualTest) {
            auto f = sig_combine_backprop;
            uint64_t dimension = 2, degree = 2;
            uint64_t result_length = 7;
            std::vector<double> sig1 = { 1., 1., 1., .5, .5, .5, .5 };
            std::vector<double> sig2 = { 1., 0., 1., 0., 1., -1., .5 };
            std::vector<double> derivs = {1., 1., 2., 3., 4., 5., 6.};
            std::vector<double> true_ = { 1., 5., 8., 3., 4., 5., 6., 1., 9., 12., 3., 4., 5., 6. };

            auto func = [&](double* sig_combined_derivs, double* out, double* sig1, double* sig2, uint64_t dimension, uint64_t degree) {
                f(sig_combined_derivs, out, out + result_length, sig1, sig2, dimension, degree);
                };


            check_result(func, derivs, true_, sig1.data(), sig2.data(), dimension, degree);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = batch_sig_combine_backprop;
            uint64_t dimension = 2, degree = 2, batch_size = 2;
            uint64_t result_length = 7 * batch_size;
            std::vector<double> sig1 = { 1., 1., 1., .5, .5, .5, .5, 
                1., 0., 1., 0., 1., -1., .5 };
            std::vector<double> sig2 = { 1., 0., 1., 0., 1., -1., .5, 
                1., 1., 1., .5, .5, .5, .5 };
            std::vector<double> derivs = { 1., 1., 2., 3., 4., 5., 6., 
                1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 1., 5., 8., 3., 4., 5., 6., 
                1., 8., 13., 3., 4., 5., 6., 
                1., 9., 12., 3., 4., 5., 6., 
                1., 6., 8., 3., 4., 5., 6. };

            auto func = [&](double* sig_combined_derivs, double* out, double* sig1, double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) {
                f(sig_combined_derivs, out, out + result_length, sig1, sig2, batch_size, dimension, degree, n_jobs);
                };


            check_result(func, derivs, true_, sig1.data(), sig2.data(), batch_size, dimension, degree, 1);
            check_result(func, derivs, true_, sig1.data(), sig2.data(), batch_size, dimension, degree, -1);
        }
    };

    TEST_CLASS(sigKernelTest) {
    public:

        TEST_METHOD(Trivial) {
            auto f = sig_kernel;
            uint64_t dimension = 1, length = 1;
            std::vector<double> path = { 0. };
            std::vector<double> true_sig = { 1. };
            std::vector<double> gram = {};
            check_result(f, gram, true_sig, dimension, length, length, 0, 0, false);
        }

        TEST_METHOD(TrivialBatch) {
            auto f = batch_sig_kernel;
            uint64_t dimension = 1, length = 1, batch_size = 5;
            std::vector<double> path = { 0. };
            std::vector<double> true_sig = { 1., 1., 1., 1., 1. };
            std::vector<double> gram = {};
            check_result(f, gram, true_sig, batch_size, dimension, length, length, 0, 0, 1, false);
        }
        TEST_METHOD(LinearPathTest) {
            auto f = sig_kernel;
            uint64_t dimension = 2, length = 3;
            std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
            std::vector<double> true_sig = { 4.256702149748847 };
            std::vector<double> gram((length - 1) * (length - 1));
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            check_result(f, gram, true_sig, dimension, length, length, 2, 2, false);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_kernel;
            uint64_t dimension = 3, length = 4;
            std::vector<double> path = { .9, .5, .8, .5, .3, .0, .0, .2, .6, .4, .0, .2 };
            std::vector<double> true_sig = { 2.1529809076880486 };
            std::vector<double> gram((length - 1) * (length - 1));
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            check_result(f, gram, true_sig, dimension, length, length, 2, 2, false);
        }

        TEST_METHOD(NonSquare1) {
            auto f = sig_kernel;
            uint64_t dimension = 1, length1 = 3, length2 = 2;
            std::vector<double> path1 = { 0., 1., 2. };
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> true_sig = { 11. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, dimension, length1, length2, 0, 0, false);
        }

        TEST_METHOD(NonSquare2) {
            auto f = sig_kernel;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path2 = { 0., 1., 2. };
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> true_sig = { 11. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, dimension, length1, length2, 0, 0, false);
        }

        TEST_METHOD(FullGrid) {
            auto f = sig_kernel;
            uint64_t dimension = 1, length1 = 3, length2 = 2;
            std::vector<double> path1 = { 0., 1., 2. };
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> true_sig = { 1., 1.,
                1., 4.,
                1., 11. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, dimension, length1, length2, 0, 0, true);
        }
    };

    TEST_CLASS(sigKernelBackpropTest) {
    public:
        TEST_METHOD(ManualTest1) {
            auto f = sig_kernel_backprop;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> path2 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 4.5 + 1./6, 4.5 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, dimension, length1, length2, 0, 0);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_kernel_backprop;
            uint64_t dimension = 1, length1 = 3, length2 = 3;
            std::vector<double> path1 = { 0., 2., 3. };
            std::vector<double> path2 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 761./72, 7.125, 133./24, 12.5 + 1. / 6 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, dimension, length1, length2, 0, 0);
        }

        TEST_METHOD(ManualTest3) {
            auto f = sig_kernel_backprop;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> path2 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 5.1602194279800226, 5.1185673607720270 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, dimension, length1, length2, 1, 1);
        }

        TEST_METHOD(BatchManualTest1) {
            auto f = batch_sig_kernel_backprop;
            uint64_t batch_size = 2, dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2., 0., 2. };
            std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
            std::vector<double> derivs = { 1., 1. };
            std::vector<double> true_ = { 4.5 + 1. / 6, 4.5, 4.5 + 1. / 6, 4.5 };
            std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            gram_(path1.data(), path2.data(), gram.data() + 2, 1, dimension, length1, length2);
            check_result(f, gram, true_, derivs.data(), batch_size, dimension, length1, length2, 0, 0, 1);
        }

        TEST_METHOD(BatchManualTest2) {
            auto f = batch_sig_kernel_backprop;
            uint64_t batch_size = 2, dimension = 1, length1 = 3, length2 = 3;
            std::vector<double> path1 = { 0., 2., 3., 0., 2., 3. };
            std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
            std::vector<double> derivs = { 1., 1. };
            std::vector<double> true_ = { 761. / 72, 7.125, 133. / 24, 12.5 + 1. / 6, 761. / 72, 7.125, 133. / 24, 12.5 + 1. / 6 };
            std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            gram_(path1.data(), path2.data(), gram.data() + 4, 1, dimension, length1, length2);
            check_result(f, gram, true_, derivs.data(), batch_size, dimension, length1, length2, 0, 0, 1);
        }
    };

    TEST_CLASS(transformPathBackprop) {
    public:

        TEST_METHOD(TimeAugTest) {
            auto f = transform_path_backprop;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs((dimension + 1) * length, 1.);
            std::vector<double> true_ = { 1., 1., 1., 1., 1., 1. };
            check_result(f, derivs, true_, dimension, length, true, false, 1.);
        }
        TEST_METHOD(LeadLagTest) {
            auto f = transform_path_backprop;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs(2 * dimension * (2 * length - 1));
            for (int i = 0; i < derivs.size(); ++i)
                derivs[i] = i;
            std::vector<double> true_ = { 6., 9., 36., 40., 48., 51. };
            check_result(f, derivs, true_, dimension, length, false, true, 1.);
        }

        TEST_METHOD(TimeAugLeadLagTest) {
            auto f = transform_path_backprop;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs((2 * dimension + 1) * (2 * length - 1), 1.);
            std::vector<double> true_ = { 3., 3., 4., 4., 3., 3. };
            check_result(f, derivs, true_, dimension, length, true, true, 1.);
        }
    };
}
//TODO: add tests for transform_path