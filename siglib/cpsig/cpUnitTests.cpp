#include "CppUnitTest.h"
#include "cpsig.h"
#include "cpTensorPoly.h"
#include "cpPath.h"
#include "cpSignature.h"
#include "cpSigKernel.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <span>
#include <cmath>

#define EPSILON 1e-13

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

void gram_(
    double* path1,
    double* path2,
    double* out,
    uint64_t batchSize,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2
) {
    double* outPtr = out;

    uint64_t flatPath1Length = length1 * dimension;
    uint64_t flatPath2Length = length2 * dimension;

    double* path1Start = path1;
    double* path1End = path1 + flatPath1Length;

    double* path2Start = path2;
    double* path2End = path2 + flatPath2Length;

    for (uint64_t b = 0; b < batchSize; ++b) {

        for (double* path1Ptr = path1Start; path1Ptr < path1End - dimension; path1Ptr += dimension) {
            for (double* path2Ptr = path2Start; path2Ptr < path2End - dimension; path2Ptr += dimension) {
                *(outPtr++) = dot_product(path1Ptr + dimension, path2Ptr + dimension, dimension)
                    - dot_product(path1Ptr + dimension, path2Ptr, dimension)
                    - dot_product(path1Ptr, path2Ptr + dimension, dimension)
                    + dot_product(path1Ptr, path2Ptr, dimension);
            }
        }

        path1Start += flatPath1Length;
        path1End += flatPath1Length;
        path2Start += flatPath2Length;
        path2End += flatPath2Length;
    }
}


std::vector<int> intTestData(uint64_t dimension, uint64_t length) {
    std::vector<int> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back(i);
    }
    return data;
}

template<typename FN, typename T, typename... Args>
void checkResult(FN f, std::vector<T>& path, std::vector<double>& true_, Args... args) {
    std::vector<double> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    f(path.data(), out.data(), args...);

    for (uint64_t i = 0; i < true_.size(); ++i)
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs( - 1. - out[true_.size()]) < EPSILON);
}

template<typename FN, typename T, typename... Args>
void checkResult2(FN f, std::vector<T>& path1, std::vector<T>& path2, std::vector<double>& true_, Args... args) {
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
    TEST_CLASS(PolyLengthTest)
    {
    public:
        TEST_METHOD(ValueTest)
        {
            Assert::AreEqual((uint64_t)1, polyLength(0, 0));
            Assert::AreEqual((uint64_t)1, polyLength(0, 0));
            Assert::AreEqual((uint64_t)1, polyLength(0, 1));
            Assert::AreEqual((uint64_t)1, polyLength(1, 0));

            Assert::AreEqual((uint64_t)435848050, polyLength(9, 9));
            Assert::AreEqual((uint64_t)11111111111, polyLength(10, 10));
            Assert::AreEqual((uint64_t)313842837672, polyLength(11, 11));

            Assert::AreEqual((uint64_t)10265664160401, polyLength(400, 5));
        }
    };

    TEST_CLASS(PathTest)
    {
    public:
        TEST_METHOD(ConstructorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = intTestData(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Path<int> path2(std::span<int>(data), dimension, length);
            Path<int> path3(path2);

            Assert::IsTrue(path == path2);
            Assert::IsTrue(path == path3);
        }
        TEST_METHOD(SqBracketOperatorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = intTestData(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Point<int> pt = path[3];
            Assert::AreEqual(data.data() + 3 * dimension, pt.data());
        }
        TEST_METHOD(FirstLastTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

            Path<int> path(data.data(), dimension, length);
            Point<int> pt(&path, 0);

            for (uint64_t i = 0; i < dimension; ++i)
                Assert::AreEqual(data[i], pt[i]);
        }

        TEST_METHOD(IncrementTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

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
            std::vector<int> data = intTestData(dimension, length);

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
            auto f = signatureDouble;
            std::vector<double> path;
            std::vector<double> trueSig;
            Assert::AreEqual(2, f(path.data(), trueSig.data(), 0, 0, 0, false, false, true));

            trueSig.push_back(1.);
            checkResult(f, path, trueSig, 1, 0, 0, false, false, true);

            path.push_back(0.);
            checkResult(f, path, trueSig, 1, 1, 0, false, false, true);

            trueSig.push_back(0.);
            checkResult(f, path, trueSig, 1, 0, 1, false, false, true);
            checkResult(f, path, trueSig, 1, 1, 1, false, false, true);

            path.push_back(1.);
            trueSig[1] = 1.;
            checkResult(f, path, trueSig, 1, 2, 1, false, false, true);
        }
        TEST_METHOD(LinearPathTest) {
            auto f = signatureDouble;
            uint64_t dimension = 2, length = 3, degree = 3;
            uint64_t level3Start = polyLength(dimension, 2);
            uint64_t level4Start = polyLength(dimension, 3);
            std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
            std::vector<double> trueSig;
            trueSig.resize(level4Start);
            trueSig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { trueSig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level3Start; ++i) { trueSig[i] = 1 / 2.; }
            for (uint64_t i = level3Start; i < level4Start; ++i) { trueSig[i] = 1 / 6.; }
            checkResult(f, path, trueSig, dimension, length, degree, false, false, true);
        }

        TEST_METHOD(LinearPathTest2) {
            auto f = signatureDouble;
            uint64_t dimension = 2, length = 4, degree = 3;
            uint64_t level3Start = polyLength(dimension, 2);
            uint64_t level4Start = polyLength(dimension, 3);
            std::vector<double> path = { 0.,0., 0.25, 0.25, 0.75, 0.75, 1.,1. };
            std::vector<double> trueSig;
            trueSig.resize(level4Start);
            trueSig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { trueSig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level3Start; ++i) { trueSig[i] = 1 / 2.; }
            for (uint64_t i = level3Start; i < level4Start; ++i) { trueSig[i] = 1 / 6.; }
            checkResult(f, path, trueSig, dimension, length, degree, false, false, true);
        }

        TEST_METHOD(ManualSigTest) {
            auto f = signatureDouble;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> trueSig = { 1., 0., 1., 0., 1., -1., 0.5 };
            checkResult(f, path, trueSig, dimension, length, degree, false, false, true);
        }
        TEST_METHOD(ManualSigTest2) {
            auto f = signatureInt32;
            uint64_t dimension = 3, length = 4, degree = 3;
            std::vector<int> path = { 9, 5, 8, 5, 3, 0, 0, 2, 6, 4, 0, 2 };
            std::vector<double> trueSig = { 1., -5., - 5., - 6., 12.5, 24.5,
                                                5., 0.5, 12.5, 9., 25.,
                                               21., 18., - 20.5 - 1./3, - 77.5 - 1./3, 11.,
                                               33. + 1./6, - 45.5 - 1./3, - 42. - 1./3, - 47., 5. + 2./3,
                                              - 18., - 17.5 - 1./3, - 30.5 - 1./3, 11. + 2./3, 14. + 1./6,
                                              - 20.5 - 1./3, - 19., - 14. - 1./3, - 7., - 16. - 2./3,
                                              - 39., - 110. - 1./3, 6., - 1./3, - 49.,
                                              - 20. - 2./3, - 78., - 52. - 2./3, - 36. };
            checkResult(f, path, trueSig, dimension, length, degree, false, false, true);
        }

        TEST_METHOD(BatchSigTest) {
            auto f = batchSignatureDouble;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> trueSig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 0., 1., 0., 1., -1., 0.5 };

            checkResult(f, path, trueSig, 3, dimension, length, degree, false, false, true, false);
            checkResult(f, path, trueSig, 3, dimension, length, degree, false, false, true, true);
        }

        TEST_METHOD(BatchSigTestDegree1) {
            auto f = batchSignatureDouble;
            uint64_t dimension = 2, length = 4, degree = 1;
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> trueSig = { 1., 1., 1.,
                1., 1., 1.,
                1., 0., 1. };

            checkResult(f, path, trueSig, 3, dimension, length, degree, false, false, true, false);
            checkResult(f, path, trueSig, 3, dimension, length, degree, false, false, true, true);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = signatureInt32;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<int> path = { 0, 5, 2, 4, 9 };
            std::vector<double> trueSig = { 1., 9., 4., 40.5, 15.5, 20.5, 8., 121.5, 37.5,
                                64.5, 24.5, 60., 13., 34.5, 10. + 2./3 };
            checkResult(f, path, trueSig, dimension, length, degree, true, false, true);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = signatureInt32;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<int> path = { 0, 5, 2, 4, 9 };
            std::vector<double> trueSig = { 1., 4., 4., 8., 20., -4., 8., 10. + 2./3, 35., 10., 85., -13., -90., 37., 10. + 2./3};
            checkResult(f, path, trueSig, dimension, length, degree, false, true, true);
        }

        TEST_METHOD(BigLeadLagTest) {
            auto f = batchSignatureDouble;
            uint64_t dimension = 2, length = 10, degree = 2, batch = 1;
            std::vector<double> path;
            path.resize(batch * length * dimension);
            std::vector<double> out;
            out.resize(batch * polyLength(dimension * 2, degree));
            f(path.data(), out.data(), batch, dimension, length, degree, false, true, true, false);
        }
    };

    TEST_CLASS(sigKernelTest) {
    public:
        TEST_METHOD(LinearPathTest) {
            auto f = sigKernel;
            uint64_t dimension = 2, length = 3;
            std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
            std::vector<double> trueSig = { 4.256702149748847 };
            std::vector<double> gram(length * length);
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            checkResult(f, gram, trueSig, dimension, length, length, 2, 2);
        }

        TEST_METHOD(ManualTest) {
            auto f = sigKernel;
            uint64_t dimension = 3, length = 4;
            std::vector<double> path = { .9, .5, .8, .5, .3, .0, .0, .2, .6, .4, .0, .2 };
            std::vector<double> trueSig = { 2.1529809076880486 };
            std::vector<double> gram(length * length);
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            checkResult(f, gram, trueSig, dimension, length, length, 2, 2);
        }
    };
}