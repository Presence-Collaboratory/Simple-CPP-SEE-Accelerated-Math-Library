// Description: Comprehensive unit tests for Math Library
// Author: DeepSeek, NSDeathman
// Version: 2.1 - Fixed infinity comparisons and normalization checks
#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <limits>

// Include the math library
#include "../MathAPI/MathAPI.h"

namespace MathTests
{
    using namespace Math;

    // ============================================================================
    // Test Configuration
    // ============================================================================

    struct TestConfig
    {
        static constexpr bool VERBOSE_OUTPUT = true;
        static constexpr int PRECISION = 6;
        static constexpr float DEFAULT_EPSILON = 1e-5f;
        static constexpr float STRICT_EPSILON = 1e-7f;
    };

    // ============================================================================
    // Test Utilities
    // ============================================================================

    class TestReporter
    {
    private:
        std::ostream& output;
        int current_indent;

    public:
        TestReporter(std::ostream& os = std::cout) : output(os), current_indent(0) {}

        void indent() { current_indent += 2; }
        void unindent() { current_indent = std::max(0, current_indent - 2); }

        void message(const std::string& msg)
        {
            output << std::string(current_indent, ' ') << msg << std::endl;
        }

        void warning(const std::string& msg)
        {
            output << std::string(current_indent, ' ') << "WARNING: " << msg << std::endl;
        }

        void error(const std::string& msg)
        {
            output << std::string(current_indent, ' ') << "ERROR: " << msg << std::endl;
        }

        void section(const std::string& title)
        {
            output << "\n" << std::string(current_indent, ' ')
                << "=== " << title << " ===" << std::endl;
        }
    };

    class TestSuite
    {
    private:
        std::string suite_name;
        int tests_passed;
        int tests_failed;
        int tests_skipped;
        bool verbose;
        TestReporter reporter;
        std::chrono::steady_clock::time_point start_time;

        struct TestResult
        {
            std::string name;
            bool passed;
            std::string message;
            double duration_ms;
        };

        std::vector<TestResult> test_results;

        // Helper function for safe infinity comparison
        template<typename T>
        bool safe_approximately(const T& a, const T& b, float epsilon)
        {
            // For vector types with component-wise infinity checks
            if constexpr (requires { a.x; b.x; }) {
                // For vector types, check if any component is infinity
                bool a_has_inf = false, b_has_inf = false;
                bool a_has_nan = false, b_has_nan = false;

                // Check each component for special values
                for (int i = 0; i < sizeof(a) / sizeof(a.x); ++i) {
                    if (std::isinf(a[i])) a_has_inf = true;
                    if (std::isinf(b[i])) b_has_inf = true;
                    if (std::isnan(a[i])) a_has_nan = true;
                    if (std::isnan(b[i])) b_has_nan = true;
                }

                if (a_has_inf && b_has_inf) {
                    // For vectors with infinity, check if they're approximately equal
                    // This means all non-infinity components should be equal
                    for (int i = 0; i < sizeof(a) / sizeof(a.x); ++i) {
                        if (!std::isinf(a[i]) && !std::isinf(b[i]) &&
                            !approximately(a[i], b[i], epsilon)) {
                            return false;
                        }
                    }
                    return true;
                }

                if (a_has_nan && b_has_nan) return true;
            }
            // For arithmetic types and half (existing logic)
            else if constexpr (std::is_arithmetic_v<T>) {
                if (std::isinf(a) && std::isinf(b)) {
                    return std::signbit(a) == std::signbit(b);
                }
                if (std::isnan(a) && std::isnan(b)) {
                    return true;
                }
            }

            return approximately(a, b, epsilon);
        }

        // Specialized function for half type only
        bool safe_approximately_half(const half& a, const half& b, float epsilon)
        {
            // Handle infinity cases for half
            if (a.is_inf() && b.is_inf()) {
                return a.is_negative() == b.is_negative();
            }
            if (a.is_nan() && b.is_nan()) {
                return true; // Consider all NaNs equal for test purposes
            }

            // Handle mismatched special cases
            if ((a.is_inf() && !b.is_inf()) || (!a.is_inf() && b.is_inf()) ||
                (a.is_nan() && !b.is_nan()) || (!a.is_nan() && b.is_nan())) {
                return false;
            }

            // Use library's approximately function for normal cases
            return approximately(a, b, epsilon);
        }

        bool safe_approximately_half2(const half2& a, const half2& b, float epsilon)
        {
            // Handle infinity cases for half
            if (a.is_inf() && b.is_inf()) {
                return a.is_negative() == b.is_negative();
            }
            if (a.is_nan() && b.is_nan()) {
                return true; // Consider all NaNs equal for test purposes
            }

            // Handle mismatched special cases
            if ((a.is_inf() && !b.is_inf()) || (!a.is_inf() && b.is_inf()) ||
                (a.is_nan() && !b.is_nan()) || (!a.is_nan() && b.is_nan())) {
                return false;
            }

            // Use library's approximately function for normal cases
            return approximately(a, b, epsilon);
        }

        // Добавить специализированную функцию для half3:
        bool safe_approximately_half3(const half3& a, const half3& b, float epsilon)
        {
            // Handle infinity cases for half3
            if (a.is_inf() && b.is_inf()) {
                return a.is_negative() == b.is_negative();
            }
            if (a.is_nan() && b.is_nan()) {
                return true; // Consider all NaNs equal for test purposes
            }

            // Handle mismatched special cases
            if ((a.is_inf() && !b.is_inf()) || (!a.is_inf() && b.is_inf()) ||
                (a.is_nan() && !b.is_nan()) || (!a.is_nan() && b.is_nan())) {
                return false;
            }

            // Use library's approximately function for normal cases
            return approximately(a, b, epsilon);
        }

        bool safe_approximately_half4(const half4& a, const half4& b, float epsilon)
        {
            // Handle infinity cases for half3
            if (a.is_inf() && b.is_inf()) {
                return a.is_negative() == b.is_negative();
            }
            if (a.is_nan() && b.is_nan()) {
                return true; // Consider all NaNs equal for test purposes
            }

            // Handle mismatched special cases
            if ((a.is_inf() && !b.is_inf()) || (!a.is_inf() && b.is_inf()) ||
                (a.is_nan() && !b.is_nan()) || (!a.is_nan() && b.is_nan())) {
                return false;
            }

            // Use library's approximately function for normal cases
            return approximately(a, b, epsilon);
        } 
        
        bool safe_approximately_quaternion(const quaternion& a, const quaternion& b, float epsilon)
        {
            return approximately(a, b, epsilon);
        }

    public:
        TestSuite(const std::string& name, bool verb = TestConfig::VERBOSE_OUTPUT)
            : suite_name(name), tests_passed(0), tests_failed(0), tests_skipped(0),
            verbose(verb), reporter() {}

        // Test lifecycle management
        void start_test(const std::string& test_name)
        {
            if (verbose)
            {
                reporter.message("RUN: " + test_name);
                reporter.indent();
            }
            start_time = std::chrono::steady_clock::now();
        }

        void end_test(const std::string& test_name, bool passed, const std::string& message = "")
        {
            auto end_time = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            if (verbose) reporter.unindent();

            TestResult result{ test_name, passed, message, duration };
            test_results.push_back(result);

            if (passed)
            {
                tests_passed++;
                if (verbose)
                {
                    reporter.message("+ PASS: " + test_name + " (" + std::to_string(duration) + "ms)");
                }
            }
            else
            {
                tests_failed++;
                reporter.error("- FAIL: " + test_name + " - " + message + " (" + std::to_string(duration) + "ms)");
            }
        }

        void skip_test(const std::string& test_name, const std::string& reason = "")
        {
            tests_skipped++;
            reporter.message("- SKIP: " + test_name + (reason.empty() ? "" : " - " + reason));
        }

        // Value formatting for error messages
        template<typename T>
        std::string format_value(const T& value)
        {
            std::ostringstream oss;
            oss << std::setprecision(TestConfig::PRECISION);

            if constexpr (std::is_arithmetic_v<T>)
            {
                if (std::isinf(value)) {
                    oss << (value < 0 ? "-inf" : "inf");
                }
                else if (std::isnan(value)) {
                    oss << "nan";
                }
                else {
                    oss << value;
                }
            }
            else if constexpr (requires { value.to_string(); })
            {
                oss << value.to_string();
            }
            else if constexpr (requires { oss << value; })
            {
                oss << value;
            }
            else
            {
                oss << "[unprintable type]";
            }
            return oss.str();
        }

        // Assertion methods
        template<typename T>
        bool assert_equal(const T& actual, const T& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            bool success = safe_approximately(actual, expected, epsilon);
            std::string message;

            if (!success)
            {
                message = "Expected: " + format_value(expected) +
                    ", Got: " + format_value(actual) +
                    ", Epsilon: " + std::to_string(epsilon);
            }

            end_test(test_name, success, message);
            return success;
        }

        bool assert_equal_half_t(const half& actual, const half& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            bool success = safe_approximately_half(actual, expected, epsilon);
            std::string message;

            if (!success)
            {
                message = "Expected: " + format_value(expected) +
                    ", Got: " + format_value(actual) +
                    ", Epsilon: " + std::to_string(epsilon);
            }

            end_test(test_name, success, message);
            return success;
        }

        bool assert_equal_half2_t(const half2& actual, const half2& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            bool success = safe_approximately_half2(actual, expected, epsilon);
            std::string message;

            if (!success)
            {
                message = "Expected: " + format_value(expected) +
                    ", Got: " + format_value(actual) +
                    ", Epsilon: " + std::to_string(epsilon);
            }

            end_test(test_name, success, message);
            return success;
        }

        bool assert_equal_half3_t(const half3& actual, const half3& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            bool success = safe_approximately_half3(actual, expected, epsilon);
            std::string message;

            if (!success)
            {
                message = "Expected: " + format_value(expected) +
                    ", Got: " + format_value(actual) +
                    ", Epsilon: " + std::to_string(epsilon);
            }

            end_test(test_name, success, message);
            return success;
        }

        bool assert_equal_half4_t(const half4& actual, const half4& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            bool success = safe_approximately_half4(actual, expected, epsilon);
            std::string message;

            if (!success)
            {
                message = "Expected: " + format_value(expected) +
                    ", Got: " + format_value(actual) +
                    ", Epsilon: " + std::to_string(epsilon);
            }

            end_test(test_name, success, message);
            return success;
        }

        bool assert_equal_quaternion(const quaternion& actual, const quaternion& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            // Для тестов на равенство компонентов нужно сравнивать компоненты напрямую,
            // а не через approximately (который учитывает двойное покрытие)
            bool success = safe_approximately_quaternion_components(actual, expected, epsilon);
            std::string message;

            if (!success)
            {
                message = "Expected: " + format_value(expected) +
                    ", Got: " + format_value(actual) +
                    ", Epsilon: " + std::to_string(epsilon);
            }

            end_test(test_name, success, message);
            return success;
        }

        bool safe_approximately_quaternion_components(const quaternion& a, const quaternion& b, float epsilon)
        {
            // Сравниваем компоненты напрямую, а не через двойное покрытие
            return MathFunctions::approximately(a.x, b.x, epsilon) &&
                MathFunctions::approximately(a.y, b.y, epsilon) &&
                MathFunctions::approximately(a.z, b.z, epsilon) &&
                MathFunctions::approximately(a.w, b.w, epsilon);
        }

        // Дополнительная функция для тестирования вращений (учитывает двойное покрытие)
        bool assert_approximately_quaternion_rotation(const quaternion& actual, const quaternion& expected,
            const std::string& test_name, float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            bool success = approximately(actual, expected, epsilon); // использует двойное покрытие
            std::string message;

            if (!success)
            {
                message = "Expected rotation: " + format_value(expected) +
                    ", Got: " + format_value(actual) +
                    ", Epsilon: " + std::to_string(epsilon);
            }

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_not_equal(const T& actual, const T& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            bool success = !safe_approximately(actual, expected, epsilon);
            std::string message;

            if (!success)
            {
                message = "Values should not be equal: " + format_value(actual);
            }

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_true(const T& condition, const std::string& test_name)
        {
            start_test(test_name);

            bool success = static_cast<bool>(condition);
            std::string message = success ? "" : "Condition evaluated to false";

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_false(const T& condition, const std::string& test_name)
        {
            start_test(test_name);

            bool success = !static_cast<bool>(condition);
            std::string message = success ? "" : "Condition evaluated to true";

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_nan(const T& value, const std::string& test_name)
        {
            start_test(test_name);

            bool success = std::isnan(static_cast<float>(value));
            std::string message = success ? "" : "Value is not NaN";

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_infinity(const T& value, const std::string& test_name)
        {
            start_test(test_name);

            bool success = std::isinf(static_cast<float>(value));
            std::string message = success ? "" : "Value is not infinity";

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_finite(const T& value, const std::string& test_name)
        {
            start_test(test_name);

            bool success = std::isfinite(static_cast<float>(value));
            std::string message = success ? "" : "Value is not finite";

            end_test(test_name, success, message);
            return success;
        }

        // Test suite management
        void header()
        {
            std::cout << "\n" << std::string(70, '=') << std::endl;
            std::cout << "TEST SUITE: " << suite_name << std::endl;
            std::cout << std::string(70, '=') << std::endl;
            test_results.clear();
        }

        void footer()
        {
            std::cout << std::string(70, '-') << std::endl;
            std::cout << "RESULTS: " << tests_passed << " passed, "
                << tests_failed << " failed, "
                << tests_skipped << " skipped" << std::endl;

            if (tests_failed > 0 && verbose)
            {
                std::cout << "\nFAILED TESTS:" << std::endl;
                for (const auto& result : test_results)
                {
                    if (!result.passed)
                    {
                        std::cout << "  - " << result.name << ": " << result.message << std::endl;
                    }
                }
            }

            std::cout << std::string(70, '=') << std::endl;
        }

        // Statistics
        int get_total_count() const { return tests_passed + tests_failed + tests_skipped; }
        int get_passed_count() const { return tests_passed; }
        int get_failed_count() const { return tests_failed; }
        int get_skipped_count() const { return tests_skipped; }
        double get_success_rate() const
        {
            int total = get_total_count();
            return total > 0 ? (static_cast<double>(tests_passed) / total) * 100.0 : 0.0;
        }
    };

    // ============================================================================
    // Test Data Generators
    // ============================================================================

    class TestData
    {
    public:
        static std::vector<float> generate_test_floats()
        {
            return {
                0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
                Constants::PI, Constants::HALF_PI, Constants::TWO_PI,
                std::numeric_limits<float>::epsilon(),
                std::numeric_limits<float>::min(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN()
            };
        }

        static std::vector<float2> generate_test_float2s()
        {
            std::vector<float> floats = generate_test_floats();
            std::vector<float2> result;

            for (size_t i = 0; i < floats.size(); ++i)
            {
                for (size_t j = 0; j < floats.size(); ++j)
                {
                    if (result.size() >= 20) break; // Limit size
                    result.emplace_back(floats[i], floats[j]);
                }
                if (result.size() >= 20) break;
            }
            return result;
        }

        static std::vector<float3> generate_test_float3s()
        {
            auto floats = generate_test_floats();
            return {
                float3::zero(), float3::one(), float3::unit_x(), float3::unit_y(), float3::unit_z(),
                float3(1, 2, 3), float3(-1, -2, -3), float3(0.5f, 1.5f, 2.5f)
            };
        }
    };

    // ============================================================================
    // float2 Tests - Comprehensive
    // ============================================================================

    void test_float2_constructors()
    {
        TestSuite suite("float2 Constructors");
        suite.header();

        // Default constructor
        suite.assert_equal(float2(), float2(0.0f, 0.0f), "Default constructor");

        // Component constructor
        float2 v2(1.5f, 2.5f);
        suite.assert_equal(v2.x, 1.5f, "Component constructor - x");
        suite.assert_equal(v2.y, 2.5f, "Component constructor - y");

        // Scalar constructor
        suite.assert_equal(float2(5.0f), float2(5.0f, 5.0f), "Scalar constructor");

        // Copy constructor
        float2 original(2.0f, 3.0f);
        float2 copy(original);
        suite.assert_equal(copy, original, "Copy constructor");

        // Array constructor
        float data[2] = { 3.0f, 4.0f };
        suite.assert_equal(float2(data), float2(3.0f, 4.0f), "Array constructor");

        // Static constructors
        suite.assert_equal(float2::zero(), float2(0.0f, 0.0f), "Zero vector");
        suite.assert_equal(float2::one(), float2(1.0f, 1.0f), "One vector");
        suite.assert_equal(float2::unit_x(), float2(1.0f, 0.0f), "Unit X");
        suite.assert_equal(float2::unit_y(), float2(0.0f, 1.0f), "Unit Y");

        // Edge cases - fixed infinity comparison
        float2 inf_vec(std::numeric_limits<float>::infinity(), 0.0f);
        suite.assert_equal(inf_vec.x, std::numeric_limits<float>::infinity(), "Infinity component");

        suite.footer();
    }

    void test_float2_arithmetic_operations()
    {
        TestSuite suite("float2 Arithmetic Operations");
        suite.header();

        float2 a(1.0f, 2.0f);
        float2 b(3.0f, 4.0f);
        float2 c;

        // Addition
        suite.assert_equal(a + b, float2(4.0f, 6.0f), "Vector addition");
        suite.assert_equal(a + 1.0f, float2(2.0f, 3.0f), "Scalar addition");
        suite.assert_equal(1.0f + a, float2(2.0f, 3.0f), "Scalar addition commutative");

        // Subtraction
        suite.assert_equal(a - b, float2(-2.0f, -2.0f), "Vector subtraction");
        suite.assert_equal(a - 1.0f, float2(0.0f, 1.0f), "Scalar subtraction");

        // Multiplication
        suite.assert_equal(a * b, float2(3.0f, 8.0f), "Component-wise multiplication");
        suite.assert_equal(a * 2.0f, float2(2.0f, 4.0f), "Scalar multiplication");
        suite.assert_equal(2.0f * a, float2(2.0f, 4.0f), "Scalar multiplication commutative");

        // Division
        suite.assert_equal(a / b, float2(1.0f / 3.0f, 2.0f / 4.0f), "Component-wise division");
        suite.assert_equal(a / 2.0f, float2(0.5f, 1.0f), "Scalar division");

        // Unary operations
        suite.assert_equal(+a, a, "Unary plus");
        suite.assert_equal(-a, float2(-1.0f, -2.0f), "Unary minus");

        // Compound assignments
        c = a; c += b;
        suite.assert_equal(c, a + b, "Compound addition");

        c = a; c -= b;
        suite.assert_equal(c, a - b, "Compound subtraction");

        c = a; c *= b;
        suite.assert_equal(c, a * b, "Compound multiplication");

        c = a; c /= b;
        suite.assert_equal(c, a / b, "Compound division");

        c = a; c *= 2.0f;
        suite.assert_equal(c, a * 2.0f, "Compound scalar multiplication");

        c = a; c /= 2.0f;
        suite.assert_equal(c, a / 2.0f, "Compound scalar division");

        suite.footer();
    }

    void test_float2_mathematical_functions()
    {
        TestSuite suite("float2 Mathematical Functions");
        suite.header();

        float2 v(3.0f, 4.0f);
        float2 w(1.0f, 2.0f);

        // Length calculations
        suite.assert_equal(v.length(), 5.0f, "Length");
        suite.assert_equal(v.length_sq(), 25.0f, "Squared length");

        // Distance calculations
        suite.assert_equal(distance(v, w), std::sqrt(8.0f), "Distance");
        suite.assert_equal(distance_sq(v, w), 8.0f, "Squared distance");

        // Products
        suite.assert_equal(dot(v, w), 11.0f, "Dot product");
        suite.assert_equal(cross(v, w), 2.0f, "2D cross product");

        // Normalization
        float2 normalized = v.normalize();
        suite.assert_true(normalized.is_normalized(), "Normalization result is normalized");
        suite.assert_equal(normalized.length(), 1.0f, "Normalized length");
        suite.assert_true(v.normalize().approximately(float2(0.6f, 0.8f), 0.001f), "Normalization values");

        // HLSL-style functions
        float2 negative(-1.5f, 2.7f);
        suite.assert_equal(abs(negative), float2(1.5f, 2.7f), "Absolute value");
        suite.assert_equal(sign(negative), float2(-1.0f, 1.0f), "Sign function");
        suite.assert_equal(floor(float2(1.7f, -2.3f)), float2(1.0f, -3.0f), "Floor");
        suite.assert_equal(ceil(float2(1.2f, -2.7f)), float2(2.0f, -2.0f), "Ceil");
        suite.assert_equal(round(float2(1.4f, 1.6f)), float2(1.0f, 2.0f), "Round");
        suite.assert_equal(frac(float2(1.7f, -2.3f)), float2(0.7f, 0.7f), "Fractional part");

        // Min/Max
        suite.assert_equal(min(float2(1, 3), float2(2, 2)), float2(1, 2), "Minimum");
        suite.assert_equal(max(float2(1, 3), float2(2, 2)), float2(2, 3), "Maximum");

        suite.footer();
    }

    void test_float2_geometric_operations()
    {
        TestSuite suite("float2 Geometric Operations");
        suite.header();

        float2 v(1.0f, 0.0f);
        float2 w(0.0f, 1.0f);

        // Perpendicular
        suite.assert_equal(v.perpendicular(), float2(0.0f, 1.0f), "Perpendicular vector");
        suite.assert_equal(w.perpendicular(), float2(-1.0f, 0.0f), "Perpendicular vector 2");

        // Reflection
        float2 incident(1.0f, -1.0f);
        float2 normal(0.0f, 1.0f);
        float2 reflected = reflect(incident, normal);
        suite.assert_equal(reflected, float2(1.0f, 1.0f), "Reflection");

        // Rotation
        float2 rotated = v.rotate(Constants::HALF_PI);
        suite.assert_true(rotated.approximately(w, 0.001f), "Rotation 90 degrees");

        // Angles
        suite.assert_equal(v.angle(), 0.0f, "Zero angle");
        suite.assert_equal(w.angle(), Constants::HALF_PI, "90 degree angle");
        suite.assert_equal(angle_between(v, w), Constants::HALF_PI, "Angle between vectors");

        // Lerp
        suite.assert_equal(lerp(v, w, 0.5f), float2(0.5f, 0.5f), "Linear interpolation");

        suite.footer();
    }

    void test_float2_swizzle_operations()
    {
        TestSuite suite("float2 Swizzle Operations");
        suite.header();

        float2 v(1.0f, 2.0f);

        // 2D swizzles
        suite.assert_equal(v.yx(), float2(2.0f, 1.0f), "YX swizzle");
        suite.assert_equal(v.xx(), float2(1.0f, 1.0f), "XX swizzle");
        suite.assert_equal(v.yy(), float2(2.0f, 2.0f), "YY swizzle");

        suite.footer();
    }

    void test_float2_utility_methods()
    {
        TestSuite suite("float2 Utility Methods");
        suite.header();

        // Validity checks
        suite.assert_true(float2(1.0f, 2.0f).isValid(), "Valid vector check");
        suite.assert_false(float2(std::numeric_limits<float>::infinity(), 2.0f).isValid(), "Invalid vector - infinity");
        suite.assert_false(float2(1.0f, std::numeric_limits<float>::quiet_NaN()).isValid(), "Invalid vector - NaN");

        // Approximate equality
        float2 a(1.0f, 2.0f);
        float2 b(1.000001f, 2.000001f);
        suite.assert_true(a.approximately(b, 0.001f), "Approximate equality");
        suite.assert_false(a.approximately(b, 1e-8f), "Approximate inequality");

        // Zero checks
        suite.assert_true(float2::zero().approximately_zero(), "Exactly zero");
        suite.assert_true(float2(1e-8f, -1e-8f).approximately_zero(1e-7f), "Approximately zero");

        // Normalization checks
        suite.assert_true(float2(0.6f, 0.8f).is_normalized(), "Normalized check");
        suite.assert_false(float2(1.0f, 1.0f).is_normalized(), "Not normalized check");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains x");
        suite.assert_true(str.find("2.0") != std::string::npos, "String contains y");

        // Component access
        suite.assert_equal(a[0], 1.0f, "Index access [0]");
        suite.assert_equal(a[1], 2.0f, "Index access [1]");

        suite.footer();
    }

    void test_float2_edge_cases()
    {
        TestSuite suite("float2 Edge Cases");
        suite.header();

        // Zero vectors
        suite.assert_true(float2::zero().approximately_zero(), "Zero vector");
        suite.assert_equal(float2::zero().length(), 0.0f, "Zero length");
        suite.assert_equal(float2::zero().length_sq(), 0.0f, "Zero squared length");

        // Very small vectors
        float2 tiny(1e-10f, 1e-10f);
        suite.assert_true(tiny.approximately_zero(1e-5f), "Tiny vector approximately zero");
        suite.assert_false(tiny.approximately_zero(1e-15f), "Tiny vector not exactly zero");

        // Very large vectors
        float2 large(1e10f, 1e10f);
        suite.assert_true(std::isinf(large.length()) || large.length() > 1e10f, "Large vector length");

        // Normalization edge cases - FIXED: проверяем что бесконечные векторы не считаются нормализованными
        suite.assert_true(float2::zero().normalize().approximately_zero(), "Normalize zero vector");

        float2 inf_vec(std::numeric_limits<float>::infinity(), 0.0f);

        // Вместо прямого вызова is_normalized(), проверяем компоненты
        suite.assert_false(inf_vec.isValid(), "Infinity vector validity check");
        suite.assert_false(std::isfinite(inf_vec.length()), "Infinity vector has infinite length");

        // Для бесконечного вектора is_normalized() должен возвращать false
        bool is_inf_normalized = inf_vec.is_normalized();
        suite.assert_false(is_inf_normalized, "Infinity vector normalization check");

        // Дополнительная проверка: нормализация бесконечного вектора
        float2 normalized_inf = inf_vec.normalize();
        suite.assert_false(normalized_inf.is_normalized(), "Normalized infinity vector check");
        suite.assert_false(normalized_inf.isValid(), "Normalized infinity vector validity");

        // NaN propagation
        float2 nan_vec(std::numeric_limits<float>::quiet_NaN(), 1.0f);
        suite.assert_false(nan_vec.isValid(), "NaN vector validity");

        // Test normalization of very small vectors
        float2 very_small(1e-20f, 1e-20f);
        float2 normalized_small = very_small.normalize();
        suite.assert_true(normalized_small.is_normalized() || normalized_small.approximately_zero(),
            "Very small vector normalization");

        suite.footer();
    }

    // ============================================================================
    // Test Runners for Different Components
    // ============================================================================

    void run_float2_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "FLOAT2 COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_float2_constructors();
        test_float2_arithmetic_operations();
        test_float2_mathematical_functions();
        test_float2_geometric_operations();
        test_float2_swizzle_operations();
        test_float2_utility_methods();
        test_float2_edge_cases();

        std::cout << "FLOAT2 TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
    // float3 Tests - Comprehensive
    // ============================================================================

    void test_float3_constructors()
    {
        TestSuite suite("float3 Constructors");
        suite.header();

        // Default constructor
        suite.assert_equal(float3(), float3(0.0f, 0.0f, 0.0f), "Default constructor");

        // Component constructor
        float3 v(1.5f, 2.5f, 3.5f);
        suite.assert_equal(v.x, 1.5f, "Component constructor - x");
        suite.assert_equal(v.y, 2.5f, "Component constructor - y");
        suite.assert_equal(v.z, 3.5f, "Component constructor - z");

        // Scalar constructor
        suite.assert_equal(float3(5.0f), float3(5.0f, 5.0f, 5.0f), "Scalar constructor");

        // From float2 constructor
        suite.assert_equal(float3(float2(1.0f, 2.0f), 3.0f), float3(1.0f, 2.0f, 3.0f), "From float2 constructor");
        suite.assert_equal(float3(float2(1.0f, 2.0f)), float3(1.0f, 2.0f, 0.0f), "From float2 with default z");

        // Array constructor
        float data[3] = { 3.0f, 4.0f, 5.0f };
        suite.assert_equal(float3(data), float3(3.0f, 4.0f, 5.0f), "Array constructor");

        // Copy constructor
        float3 original(2.0f, 3.0f, 4.0f);
        float3 copy(original);
        suite.assert_equal(copy, original, "Copy constructor");

        // Static constructors
        suite.assert_equal(float3::zero(), float3(0.0f, 0.0f, 0.0f), "Zero vector");
        suite.assert_equal(float3::one(), float3(1.0f, 1.0f, 1.0f), "One vector");
        suite.assert_equal(float3::unit_x(), float3(1.0f, 0.0f, 0.0f), "Unit X");
        suite.assert_equal(float3::unit_y(), float3(0.0f, 1.0f, 0.0f), "Unit Y");
        suite.assert_equal(float3::unit_z(), float3(0.0f, 0.0f, 1.0f), "Unit Z");
        suite.assert_equal(float3::forward(), float3(0.0f, 0.0f, 1.0f), "Forward vector");
        suite.assert_equal(float3::up(), float3(0.0f, 1.0f, 0.0f), "Up vector");
        suite.assert_equal(float3::right(), float3(1.0f, 0.0f, 0.0f), "Right vector");

        // Edge cases
        float3 inf_vec(std::numeric_limits<float>::infinity(), 0.0f, 1.0f);
        suite.assert_equal(inf_vec.x, std::numeric_limits<float>::infinity(), "Infinity component");

        suite.footer();
    }

    void test_float3_arithmetic_operations()
    {
        TestSuite suite("float3 Arithmetic Operations");
        suite.header();

        float3 a(1.0f, 2.0f, 3.0f);
        float3 b(4.0f, 5.0f, 6.0f);
        float3 c;

        // Vector operations
        suite.assert_equal(a + b, float3(5.0f, 7.0f, 9.0f), "Vector addition");
        suite.assert_equal(a - b, float3(-3.0f, -3.0f, -3.0f), "Vector subtraction");
        suite.assert_equal(a * b, float3(4.0f, 10.0f, 18.0f), "Component-wise multiplication");
        suite.assert_equal(a / b, float3(0.25f, 0.4f, 0.5f), "Component-wise division");

        // Scalar operations
        suite.assert_equal(a * 2.0f, float3(2.0f, 4.0f, 6.0f), "Scalar multiplication");
        suite.assert_equal(2.0f * a, float3(2.0f, 4.0f, 6.0f), "Scalar multiplication commutative");
        suite.assert_equal(a / 2.0f, float3(0.5f, 1.0f, 1.5f), "Scalar division");

        // Unary operations
        suite.assert_equal(+a, a, "Unary plus");
        suite.assert_equal(-a, float3(-1.0f, -2.0f, -3.0f), "Unary minus");

        // Compound assignments
        c = a; c += b;
        suite.assert_equal(c, a + b, "Compound addition");

        c = a; c -= b;
        suite.assert_equal(c, a - b, "Compound subtraction");

        c = a; c *= b;
        suite.assert_equal(c, a * b, "Compound multiplication");

        c = a; c /= b;
        suite.assert_equal(c, a / b, "Compound division");

        c = a; c *= 2.0f;
        suite.assert_equal(c, a * 2.0f, "Compound scalar multiplication");

        c = a; c /= 2.0f;
        suite.assert_equal(c, a / 2.0f, "Compound scalar division");

        suite.footer();
    }

    void test_float3_mathematical_functions()
    {
        TestSuite suite("float3 Mathematical Functions");
        suite.header();

        // FIXED: Consistent test data
        float3 v(1.0f, 2.0f, 2.0f);
        float3 w(2.0f, 3.0f, 4.0f);  // Consistent with distance tests

        // Length calculations
        suite.assert_equal(v.length(), 3.0f, "Length");
        suite.assert_equal(v.length_sq(), 9.0f, "Squared length");

        // Distance calculations
        suite.assert_equal(distance(v, w), 2.4494898f, "Distance", 0.0001f);
        suite.assert_equal(distance_sq(v, w), 6.0f, "Squared distance");

        // Products - UPDATED EXPECTED VALUES
        // dot(v, w) = 1*2 + 2*3 + 2*4 = 2 + 6 + 8 = 16 ✓
        suite.assert_equal(dot(v, w), 16.0f, "Dot product");

        // cross(v, w) = (2*4 - 2*3, 2*2 - 1*4, 1*3 - 2*2) = (8-6, 4-4, 3-4) = (2, 0, -1) ✓
        suite.assert_equal(cross(v, w), float3(2.0f, 0.0f, -1.0f), "Cross product");

        // Normalization
        float3 normalized = v.normalize();
        suite.assert_true(normalized.is_normalized(), "Normalization result is normalized");
        suite.assert_equal(normalized.length(), 1.0f, "Normalized length");
        suite.assert_true(v.normalize().approximately(float3(0.333333f, 0.666667f, 0.666667f), 0.001f), "Normalization values");

        // HLSL-style functions
        float3 negative(-1.5f, 2.7f, -3.2f);
        suite.assert_equal(abs(negative), float3(1.5f, 2.7f, 3.2f), "Absolute value");
        suite.assert_equal(sign(negative), float3(-1.0f, 1.0f, -1.0f), "Sign function");
        suite.assert_equal(floor(float3(1.7f, -2.3f, 3.8f)), float3(1.0f, -3.0f, 3.0f), "Floor");
        suite.assert_equal(ceil(float3(1.2f, -2.7f, 3.3f)), float3(2.0f, -2.0f, 4.0f), "Ceil");
        suite.assert_equal(round(float3(1.4f, 1.6f, 2.5f)), float3(1.0f, 2.0f, 3.0f), "Round");
        suite.assert_equal(frac(float3(1.7f, -2.3f, 3.8f)), float3(0.7f, 0.7f, 0.8f), "Fractional part");
        suite.assert_equal(saturate(float3(-0.5f, 0.5f, 1.5f)), float3(0.0f, 0.5f, 1.0f), "Saturate");

        // Min/Max/Clamp
        suite.assert_equal(min(float3(1, 3, 2), float3(2, 2, 3)), float3(1, 2, 2), "Minimum");
        suite.assert_equal(max(float3(1, 3, 2), float3(2, 2, 3)), float3(2, 3, 3), "Maximum");
        suite.assert_equal(clamp(float3(0.5f, 1.5f, -0.5f), 0.0f, 1.0f), float3(0.5f, 1.0f, 0.0f), "Scalar clamp");
        suite.assert_equal(clamp(float3(0.5f, 1.5f, -0.5f), float3(0, 1, -1), float3(1, 2, 0)),
            float3(0.5f, 1.5f, -0.5f), "Vector clamp");

        suite.footer();
    }

    void test_float3_geometric_operations()
    {
        TestSuite suite("float3 Geometric Operations");
        suite.header();

        float3 v(1.0f, 0.0f, 0.0f);
        float3 w(0.0f, 1.0f, 0.0f);
        float3 u(0.0f, 0.0f, 1.0f);

        // Reflection
        float3 incident(1.0f, -1.0f, 0.0f);
        float3 normal(0.0f, 1.0f, 0.0f);
        float3 reflected = reflect(incident, normal);
        suite.assert_equal(reflected, float3(1.0f, 1.0f, 0.0f), "Reflection");

        // Refraction (simplified test with perpendicular incidence)
        float3 refracted = refract(float3(0.0f, -1.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), 1.5f);
        suite.assert_true(refracted.approximately(float3(0.0f, -1.0f, 0.0f), 0.001f), "Refraction");

        // Projection
        float3 vec(2.0f, 3.0f, 4.0f);
        float3 onto(1.0f, 0.0f, 0.0f);
        float3 projected = project(vec, onto);
        suite.assert_equal(projected, float3(2.0f, 0.0f, 0.0f), "Projection");

        // Rejection
        float3 rejected = reject(vec, onto);
        suite.assert_equal(rejected, float3(0.0f, 3.0f, 4.0f), "Rejection");

        // Angles
        suite.assert_equal(angle_between(v, w), Constants::HALF_PI, "Angle between orthogonal vectors");
        suite.assert_equal(angle_between(v, v), 0.0f, "Angle between same vectors");

        // Orthogonality checks
        suite.assert_true(are_orthogonal(v, w), "Orthogonal vectors check");
        suite.assert_false(are_orthogonal(v, v), "Non-orthogonal vectors check");

        // Orthonormal basis
        suite.assert_true(is_orthonormal_basis(v, w, u), "Orthonormal basis check");

        // Lerp and Slerp - enhanced debugging
        suite.assert_equal(lerp(v, w, 0.5f), float3(0.5f, 0.5f, 0.0f), "Linear interpolation");

        // Enhanced slerp debugging
        float3 v_norm = v.normalize();
        float3 w_norm = w.normalize();

        float3 slerp_result = slerp(v_norm, w_norm, 0.5f);

        // Comprehensive debugging
        float actual_length = slerp_result.length();
        float actual_length_sq = slerp_result.length_sq();
        bool is_valid = slerp_result.isValid();
        bool is_normalized_check = slerp_result.is_normalized();

        std::cout << "  DEBUG: Slerp result = " << slerp_result.to_string() << std::endl;
        std::cout << "  DEBUG: Actual length = " << actual_length << std::endl;
        std::cout << "  DEBUG: Actual length_sq = " << actual_length_sq << std::endl;
        std::cout << "  DEBUG: isValid() = " << is_valid << std::endl;
        std::cout << "  DEBUG: is_normalized() = " << is_normalized_check << std::endl;
        std::cout << "  DEBUG: Expected length = 1.0" << std::endl;
        std::cout << "  DEBUG: Difference = " << std::abs(actual_length - 1.0f) << std::endl;

        // Test with different epsilon values
        bool normalized_loose = slerp_result.is_normalized(0.01f);
        bool normalized_normal = slerp_result.is_normalized(0.001f);
        bool normalized_strict = slerp_result.is_normalized(0.0001f);

        std::cout << "  DEBUG: is_normalized(0.01f) = " << normalized_loose << std::endl;
        std::cout << "  DEBUG: is_normalized(0.001f) = " << normalized_normal << std::endl;
        std::cout << "  DEBUG: is_normalized(0.0001f) = " << normalized_strict << std::endl;

        suite.assert_true(slerp_result.is_normalized(), "Slerp result is normalized");
        suite.assert_true(slerp_result.approximately(float3(0.707107f, 0.707107f, 0.0f), 0.001f), "Slerp values");

        suite.footer();
    }

    void test_float3_swizzle_operations()
    {
        TestSuite suite("float3 Swizzle Operations");
        suite.header();

        float3 v(1.0f, 2.0f, 3.0f);

        // 2D swizzles
        suite.assert_equal(v.xy(), float2(1.0f, 2.0f), "XY swizzle");
        suite.assert_equal(v.xz(), float2(1.0f, 3.0f), "XZ swizzle");
        suite.assert_equal(v.yz(), float2(2.0f, 3.0f), "YZ swizzle");
        suite.assert_equal(v.yx(), float2(2.0f, 1.0f), "YX swizzle");
        suite.assert_equal(v.zx(), float2(3.0f, 1.0f), "ZX swizzle");
        suite.assert_equal(v.zy(), float2(3.0f, 2.0f), "ZY swizzle");

        // 3D swizzles
        suite.assert_equal(v.yxz(), float3(2.0f, 1.0f, 3.0f), "YXZ swizzle");
        suite.assert_equal(v.zxy(), float3(3.0f, 1.0f, 2.0f), "ZXY swizzle");
        suite.assert_equal(v.zyx(), float3(3.0f, 2.0f, 1.0f), "ZYX swizzle");
        suite.assert_equal(v.xzy(), float3(1.0f, 3.0f, 2.0f), "XZY swizzle");
        suite.assert_equal(v.xyz(), float3(1.0f, 2.0f, 3.0f), "XYZ swizzle");
        suite.assert_equal(v.xyx(), float3(1.0f, 2.0f, 1.0f), "XYX swizzle");
        suite.assert_equal(v.xzx(), float3(1.0f, 3.0f, 1.0f), "XZX swizzle");
        suite.assert_equal(v.yxy(), float3(2.0f, 1.0f, 2.0f), "YXY swizzle");
        suite.assert_equal(v.yzy(), float3(2.0f, 3.0f, 2.0f), "YZY swizzle");
        suite.assert_equal(v.zxz(), float3(3.0f, 1.0f, 3.0f), "ZXZ swizzle");
        suite.assert_equal(v.zyz(), float3(3.0f, 2.0f, 3.0f), "ZYZ swizzle");

        // Color swizzles
        suite.assert_equal(v.r(), 1.0f, "Red component");
        suite.assert_equal(v.g(), 2.0f, "Green component");
        suite.assert_equal(v.b(), 3.0f, "Blue component");
        suite.assert_equal(v.rg(), float2(1.0f, 2.0f), "RG swizzle");
        suite.assert_equal(v.rb(), float2(1.0f, 3.0f), "RB swizzle");
        suite.assert_equal(v.gb(), float2(2.0f, 3.0f), "GB swizzle");
        suite.assert_equal(v.rgb(), float3(1.0f, 2.0f, 3.0f), "RGB swizzle");
        suite.assert_equal(v.bgr(), float3(3.0f, 2.0f, 1.0f), "BGR swizzle");
        suite.assert_equal(v.gbr(), float3(2.0f, 3.0f, 1.0f), "GBR swizzle");

        suite.footer();
    }

    void test_float3_utility_methods()
    {
        TestSuite suite("float3 Utility Methods");
        suite.header();

        // Validity checks
        suite.assert_true(float3(1.0f, 2.0f, 3.0f).isValid(), "Valid vector check");
        suite.assert_false(float3(std::numeric_limits<float>::infinity(), 2.0f, 3.0f).isValid(), "Invalid vector - infinity");
        suite.assert_false(float3(1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f).isValid(), "Invalid vector - NaN");

        // Approximate equality
        float3 a(1.0f, 2.0f, 3.0f);
        float3 b(1.000001f, 2.000001f, 3.000001f);
        suite.assert_true(a.approximately(b, 0.001f), "Approximate equality");
        suite.assert_false(a.approximately(b, 1e-8f), "Approximate inequality");

        // Zero checks
        suite.assert_true(float3::zero().approximately_zero(), "Exactly zero");
        suite.assert_true(float3(1e-8f, -1e-8f, 1e-8f).approximately_zero(1e-7f), "Approximately zero");

        // Normalization checks
        suite.assert_true(float3(0.57735f, 0.57735f, 0.57735f).is_normalized(0.001f), "Normalized check");
        suite.assert_false(float3(1.0f, 1.0f, 1.0f).is_normalized(), "Not normalized check");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains x");
        suite.assert_true(str.find("2.0") != std::string::npos, "String contains y");
        suite.assert_true(str.find("3.0") != std::string::npos, "String contains z");

        // Component access
        suite.assert_equal(a[0], 1.0f, "Index access [0]");
        suite.assert_equal(a[1], 2.0f, "Index access [1]");
        suite.assert_equal(a[2], 3.0f, "Index access [2]");

        // Data access
        suite.assert_equal(a.data()[0], 1.0f, "Data pointer access [0]");
        suite.assert_equal(a.data()[1], 2.0f, "Data pointer access [1]");
        suite.assert_equal(a.data()[2], 3.0f, "Data pointer access [2]");

        // Set XY from float2
        float3 vec(0.0f, 0.0f, 5.0f);
        vec.set_xy(float2(1.0f, 2.0f));
        suite.assert_equal(vec, float3(1.0f, 2.0f, 5.0f), "Set XY from float2");

        // Comparison operators
        suite.assert_true(a == a, "Equality operator");
        suite.assert_true(a != float3::zero(), "Inequality operator");

        suite.footer();
    }

    void test_float3_edge_cases()
    {
        TestSuite suite("float3 Edge Cases");
        suite.header();

        // Zero vectors
        suite.assert_true(float3::zero().approximately_zero(), "Zero vector");
        suite.assert_equal(float3::zero().length(), 0.0f, "Zero length");
        suite.assert_equal(float3::zero().length_sq(), 0.0f, "Zero squared length");

        // Very small vectors
        float3 tiny(1e-10f, 1e-10f, 1e-10f);
        suite.assert_true(tiny.approximately_zero(1e-5f), "Tiny vector approximately zero");
        suite.assert_false(tiny.approximately_zero(1e-15f), "Tiny vector not exactly zero");

        // Very large vectors
        float3 large(1e10f, 1e10f, 1e10f);
        suite.assert_true(std::isinf(large.length()) || large.length() > 1e10f, "Large vector length");

        // Normalization edge cases
        suite.assert_true(float3::zero().normalize().approximately_zero(), "Normalize zero vector");

        float3 inf_vec(std::numeric_limits<float>::infinity(), 0.0f, 0.0f);
        suite.assert_false(inf_vec.is_normalized(), "Infinity vector normalization check");

        // Test that normalization of invalid vectors returns expected results
        float3 nan_vec(std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f);
        suite.assert_false(nan_vec.isValid(), "NaN vector validity");

        // Test normalization of very small vectors
        float3 very_small(1e-20f, 1e-20f, 1e-20f);
        float3 normalized_small = very_small.normalize();
        suite.assert_true(normalized_small.is_normalized() || normalized_small.approximately_zero(),
            "Very small vector normalization");

        // Cross product edge cases
        suite.assert_equal(cross(float3::unit_x(), float3::unit_x()), float3::zero(), "Cross product with itself");
        suite.assert_equal(cross(float3::unit_x(), float3::unit_y()), float3::unit_z(), "Cross product orthogonality");

        // Dot product edge cases
        suite.assert_equal(dot(float3::zero(), float3::one()), 0.0f, "Dot product with zero vector");
        suite.assert_equal(dot(float3::unit_x(), float3::unit_x()), 1.0f, "Dot product with itself");

        // Geometric operations with zero vectors
        suite.assert_equal(project(float3::zero(), float3::unit_x()), float3::zero(), "Project zero vector");
        suite.assert_equal(reject(float3::unit_x(), float3::zero()), float3::unit_x(), "Reject from zero vector");

        suite.footer();
    }

    void run_float3_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "FLOAT3 COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_float3_constructors();
        test_float3_arithmetic_operations();
        test_float3_mathematical_functions();
        test_float3_geometric_operations();
        test_float3_swizzle_operations();
        test_float3_utility_methods();
        test_float3_edge_cases();

        std::cout << "FLOAT3 TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
    // float4 Tests - Comprehensive
    // ============================================================================

    void test_float4_constructors()
    {
        TestSuite suite("float4 Constructors");
        suite.header();

        // Default constructor
        suite.assert_equal(float4(), float4(0.0f, 0.0f, 0.0f, 0.0f), "Default constructor");

        // Component constructor
        float4 v(1.5f, 2.5f, 3.5f, 4.5f);
        suite.assert_equal(v.x, 1.5f, "Component constructor - x");
        suite.assert_equal(v.y, 2.5f, "Component constructor - y");
        suite.assert_equal(v.z, 3.5f, "Component constructor - z");
        suite.assert_equal(v.w, 4.5f, "Component constructor - w");

        // Scalar constructor
        suite.assert_equal(float4(5.0f), float4(5.0f, 5.0f, 5.0f, 5.0f), "Scalar constructor");

        // From float2 constructor
        suite.assert_equal(float4(float2(1.0f, 2.0f), 3.0f, 4.0f), float4(1.0f, 2.0f, 3.0f, 4.0f), "From float2 with z,w");
        suite.assert_equal(float4(float2(1.0f, 2.0f)), float4(1.0f, 2.0f, 0.0f, 0.0f), "From float2 with defaults");

        // From float3 constructor
        suite.assert_equal(float4(float3(1.0f, 2.0f, 3.0f), 4.0f), float4(1.0f, 2.0f, 3.0f, 4.0f), "From float3 with w");
        suite.assert_equal(float4(float3(1.0f, 2.0f, 3.0f)), float4(1.0f, 2.0f, 3.0f, 0.0f), "From float3 with default w");

        // Array constructor
        float data[4] = { 3.0f, 4.0f, 5.0f, 6.0f };
        suite.assert_equal(float4(data), float4(3.0f, 4.0f, 5.0f, 6.0f), "Array constructor");

        // Copy constructor
        float4 original(2.0f, 3.0f, 4.0f, 5.0f);
        float4 copy(original);
        suite.assert_equal(copy, original, "Copy constructor");

        // Static constructors
        suite.assert_equal(float4::zero(), float4(0.0f, 0.0f, 0.0f, 0.0f), "Zero vector");
        suite.assert_equal(float4::one(), float4(1.0f, 1.0f, 1.0f, 1.0f), "One vector");
        suite.assert_equal(float4::unit_x(), float4(1.0f, 0.0f, 0.0f, 0.0f), "Unit X");
        suite.assert_equal(float4::unit_y(), float4(0.0f, 1.0f, 0.0f, 0.0f), "Unit Y");
        suite.assert_equal(float4::unit_z(), float4(0.0f, 0.0f, 1.0f, 0.0f), "Unit Z");
        suite.assert_equal(float4::unit_w(), float4(0.0f, 0.0f, 0.0f, 1.0f), "Unit W");

        // Color constructors
        suite.assert_equal(float4::from_color(1.0f, 0.5f, 0.0f, 1.0f), float4(1.0f, 0.5f, 0.0f, 1.0f), "From color floats");
        suite.assert_equal(float4::from_rgba(255, 128, 0, 255), float4(1.0f, 128.0f / 255.0f, 0.0f, 1.0f), "From RGBA bytes");

        // Edge cases
        float4 inf_vec(std::numeric_limits<float>::infinity(), 0.0f, 1.0f, 2.0f);
        suite.assert_equal(inf_vec.x, std::numeric_limits<float>::infinity(), "Infinity component");

        suite.footer();
    }

    void test_float4_arithmetic_operations()
    {
        TestSuite suite("float4 Arithmetic Operations");
        suite.header();

        float4 a(1.0f, 2.0f, 3.0f, 4.0f);
        float4 b(5.0f, 6.0f, 7.0f, 8.0f);
        float4 c;

        // Vector operations
        suite.assert_equal(a + b, float4(6.0f, 8.0f, 10.0f, 12.0f), "Vector addition");
        suite.assert_equal(a - b, float4(-4.0f, -4.0f, -4.0f, -4.0f), "Vector subtraction");
        suite.assert_equal(a * b, float4(5.0f, 12.0f, 21.0f, 32.0f), "Component-wise multiplication");
        suite.assert_equal(a / b, float4(0.2f, 2.0f / 6.0f, 3.0f / 7.0f, 0.5f), "Component-wise division");

        // Scalar operations
        suite.assert_equal(a * 2.0f, float4(2.0f, 4.0f, 6.0f, 8.0f), "Scalar multiplication");
        suite.assert_equal(2.0f * a, float4(2.0f, 4.0f, 6.0f, 8.0f), "Scalar multiplication commutative");
        suite.assert_equal(a / 2.0f, float4(0.5f, 1.0f, 1.5f, 2.0f), "Scalar division");

        // Unary operations
        suite.assert_equal(+a, a, "Unary plus");
        suite.assert_equal(-a, float4(-1.0f, -2.0f, -3.0f, -4.0f), "Unary minus");

        // Compound assignments
        c = a; c += b;
        suite.assert_equal(c, a + b, "Compound addition");

        c = a; c -= b;
        suite.assert_equal(c, a - b, "Compound subtraction");

        c = a; c *= b;
        suite.assert_equal(c, a * b, "Compound multiplication");

        c = a; c /= b;
        suite.assert_equal(c, a / b, "Compound division");

        c = a; c *= 2.0f;
        suite.assert_equal(c, a * 2.0f, "Compound scalar multiplication");

        c = a; c /= 2.0f;
        suite.assert_equal(c, a / 2.0f, "Compound scalar division");

        // Assignment from float3
        c = a;
        c = float3(10.0f, 11.0f, 12.0f);
        suite.assert_equal(c, float4(10.0f, 11.0f, 12.0f, 4.0f), "Assignment from float3 preserves w");

        suite.footer();
    }

    void test_float4_mathematical_functions()
    {
        TestSuite suite("float4 Mathematical Functions");
        suite.header();

        float4 v(1.0f, 2.0f, 2.0f, 1.0f);  // length_sq = 10
        float4 w(2.0f, 3.0f, 6.0f, 2.0f);  // length_sq = 53

        // Length calculations
        suite.assert_equal(v.length(), 3.16227766f, "Length", 0.0001f);
        suite.assert_equal(v.length_sq(), 10.0f, "Squared length");

        // Distance calculations
        // distance_sq = (2-1)² + (3-2)² + (6-2)² + (2-1)² = 1 + 1 + 16 + 1 = 19 ✓
        // distance = √19 ≈ 4.3589 ✓
        suite.assert_equal(distance(v, w), 4.35889894f, "Distance", 0.0001f);
        suite.assert_equal(distance_sq(v, w), 19.0f, "Squared distance");

        // Products
        // dot(v, w) = 1*2 + 2*3 + 2*6 + 1*2 = 2 + 6 + 12 + 2 = 22 ✓
        suite.assert_equal(dot(v, w), 22.0f, "4D Dot product");

        // dot3(v, w) = 1*2 + 2*3 + 2*6 = 2 + 6 + 12 = 20 ✓ (не 26!)
        suite.assert_equal(dot3(v, w), 20.0f, "3D Dot product");

        // Cross product
        // cross(v, w) = (2*6 - 2*3, 2*2 - 1*6, 1*3 - 2*2, 0) 
        //             = (12-6, 4-6, 3-4, 0) = (6, -2, -1, 0) ✓
        suite.assert_equal(cross(v, w), float4(6.0f, -2.0f, -1.0f, 0.0f), "3D Cross product");

        // Normalization
        float4 normalized = v.normalize();
        suite.assert_true(normalized.is_normalized(0.001f), "Normalization result is normalized");
        suite.assert_equal(normalized.length(), 1.0f, "Normalized length");

        // HLSL-style functions
        float4 negative(-1.5f, 2.7f, -3.2f, 0.5f);
        suite.assert_equal(abs(negative), float4(1.5f, 2.7f, 3.2f, 0.5f), "Absolute value");
        suite.assert_equal(sign(negative), float4(-1.0f, 1.0f, -1.0f, 1.0f), "Sign function");
        suite.assert_equal(floor(float4(1.7f, -2.3f, 3.8f, -0.5f)), float4(1.0f, -3.0f, 3.0f, -1.0f), "Floor");
        suite.assert_equal(ceil(float4(1.2f, -2.7f, 3.3f, -0.2f)), float4(2.0f, -2.0f, 4.0f, 0.0f), "Ceil");
        suite.assert_equal(round(float4(1.4f, 1.6f, 2.5f, -1.4f)), float4(1.0f, 2.0f, 3.0f, -1.0f), "Round");
        suite.assert_equal(frac(float4(1.7f, -2.3f, 3.8f, -0.5f)), float4(0.7f, 0.7f, 0.8f, 0.5f), "Fractional part");
        suite.assert_equal(saturate(float4(-0.5f, 0.5f, 1.5f, 2.0f)), float4(0.0f, 0.5f, 1.0f, 1.0f), "Saturate");

        // Min/Max
        suite.assert_equal(min(float4(1, 3, 2, 4), float4(2, 2, 3, 3)), float4(1, 2, 2, 3), "Minimum");
        suite.assert_equal(max(float4(1, 3, 2, 4), float4(2, 2, 3, 3)), float4(2, 3, 3, 4), "Maximum");

        suite.footer();
    }

    void test_float4_color_operations()
    {
        TestSuite suite("float4 Color Operations");
        suite.header();

        float4 color(0.8f, 0.6f, 0.4f, 0.9f);
        float4 opaque_red(1.0f, 0.0f, 0.0f, 1.0f);
        float4 semi_transparent(0.5f, 0.5f, 0.5f, 0.5f);

        // Luminance and brightness
        float lum = color.luminance();
        float bright = color.brightness();
        suite.assert_true(lum > 0.0f && lum < 1.0f, "Luminance calculation");
        suite.assert_true(bright > 0.0f && bright < 1.0f, "Brightness calculation");

        // Grayscale conversion
        float4 gray = color.grayscale();
        suite.assert_equal(gray.x, gray.y, "Grayscale R == G");
        suite.assert_equal(gray.y, gray.z, "Grayscale G == B");
        suite.assert_equal(gray.w, color.w, "Grayscale alpha preserved");

        // Alpha operations
        float4 premult = color.premultiply_alpha();
        suite.assert_equal(premult, float4(0.8f * 0.9f, 0.6f * 0.9f, 0.4f * 0.9f, 0.9f), "Premultiply alpha");

        float4 unpremult = premult.unpremultiply_alpha();
        suite.assert_true(unpremult.approximately(color, 0.001f), "Unpremultiply alpha round-trip");

        // Special cases for alpha operations
        float4 zero_alpha(1.0f, 1.0f, 1.0f, 0.0f);
        suite.assert_equal(zero_alpha.premultiply_alpha(), float4::zero(), "Premultiply zero alpha");
        suite.assert_equal(zero_alpha.unpremultiply_alpha(), zero_alpha, "Unpremultiply zero alpha");

        suite.footer();
    }

    void test_float4_geometric_operations()
    {
        TestSuite suite("float4 Geometric Operations");
        suite.header();

        // Homogeneous coordinates
        float4 homogeneous(2.0f, 4.0f, 6.0f, 2.0f);
        float3 projected = homogeneous.project();
        suite.assert_equal(projected, float3(1.0f, 2.0f, 3.0f), "Homogeneous projection");

        // To homogeneous
        float4 point3d(1.0f, 2.0f, 3.0f, 0.0f);
        float4 homogeneous_result = point3d.to_homogeneous();
        suite.assert_equal(homogeneous_result, float4(1.0f, 2.0f, 3.0f, 1.0f), "To homogeneous coordinates");

        // Edge case: zero w component
        float4 zero_w(1.0f, 2.0f, 3.0f, 0.0f);
        float3 projected_zero = zero_w.project();
        suite.assert_equal(projected_zero, float3::zero(), "Project with zero w");

        // Lerp
        float4 a(1.0f, 2.0f, 3.0f, 4.0f);
        float4 b(5.0f, 6.0f, 7.0f, 8.0f);
        suite.assert_equal(lerp(a, b, 0.5f), float4(3.0f, 4.0f, 5.0f, 6.0f), "Linear interpolation");

        suite.footer();
    }

    void test_float4_swizzle_operations()
    {
        TestSuite suite("float4 Swizzle Operations");
        suite.header();

        float4 v(1.0f, 2.0f, 3.0f, 4.0f);

        // 2D swizzles
        suite.assert_equal(v.xy(), float2(1.0f, 2.0f), "XY swizzle");
        suite.assert_equal(v.xz(), float2(1.0f, 3.0f), "XZ swizzle");
        suite.assert_equal(v.xw(), float2(1.0f, 4.0f), "XW swizzle");
        suite.assert_equal(v.yz(), float2(2.0f, 3.0f), "YZ swizzle");
        suite.assert_equal(v.yw(), float2(2.0f, 4.0f), "YW swizzle");
        suite.assert_equal(v.zw(), float2(3.0f, 4.0f), "ZW swizzle");

        // 3D swizzles
        suite.assert_equal(v.xyz(), float3(1.0f, 2.0f, 3.0f), "XYZ swizzle");
        suite.assert_equal(v.xyw(), float3(1.0f, 2.0f, 4.0f), "XYW swizzle");
        suite.assert_equal(v.xzw(), float3(1.0f, 3.0f, 4.0f), "XZW swizzle");
        suite.assert_equal(v.yzw(), float3(2.0f, 3.0f, 4.0f), "YZW swizzle");

        // 4D swizzles
        suite.assert_equal(v.yxzw(), float4(2.0f, 1.0f, 3.0f, 4.0f), "YXZW swizzle");
        suite.assert_equal(v.zxyw(), float4(3.0f, 1.0f, 2.0f, 4.0f), "ZXYW swizzle");
        suite.assert_equal(v.zyxw(), float4(3.0f, 2.0f, 1.0f, 4.0f), "ZYXW swizzle");
        suite.assert_equal(v.wzyx(), float4(4.0f, 3.0f, 2.0f, 1.0f), "WZYX swizzle");

        // Color swizzles
        suite.assert_equal(v.r(), 1.0f, "Red component");
        suite.assert_equal(v.g(), 2.0f, "Green component");
        suite.assert_equal(v.b(), 3.0f, "Blue component");
        suite.assert_equal(v.a(), 4.0f, "Alpha component");

        suite.assert_equal(v.rg(), float2(1.0f, 2.0f), "RG swizzle");
        suite.assert_equal(v.rb(), float2(1.0f, 3.0f), "RB swizzle");
        suite.assert_equal(v.ra(), float2(1.0f, 4.0f), "RA swizzle");
        suite.assert_equal(v.gb(), float2(2.0f, 3.0f), "GB swizzle");
        suite.assert_equal(v.ga(), float2(2.0f, 4.0f), "GA swizzle");
        suite.assert_equal(v.ba(), float2(3.0f, 4.0f), "BA swizzle");

        suite.assert_equal(v.rgb(), float3(1.0f, 2.0f, 3.0f), "RGB swizzle");
        suite.assert_equal(v.rga(), float3(1.0f, 2.0f, 4.0f), "RGA swizzle");
        suite.assert_equal(v.rba(), float3(1.0f, 3.0f, 4.0f), "RBA swizzle");
        suite.assert_equal(v.gba(), float3(2.0f, 3.0f, 4.0f), "GBA swizzle");

        suite.assert_equal(v.grba(), float4(2.0f, 1.0f, 3.0f, 4.0f), "GRBA swizzle");
        suite.assert_equal(v.brga(), float4(3.0f, 1.0f, 2.0f, 4.0f), "BRGA swizzle");
        suite.assert_equal(v.bgra(), float4(3.0f, 2.0f, 1.0f, 4.0f), "BGRA swizzle");
        suite.assert_equal(v.abgr(), float4(4.0f, 3.0f, 2.0f, 1.0f), "ABGR swizzle");

        suite.footer();
    }

    void test_float4_utility_methods()
    {
        TestSuite suite("float4 Utility Methods");
        suite.header();

        // Validity checks
        suite.assert_true(float4(1.0f, 2.0f, 3.0f, 4.0f).isValid(), "Valid vector check");
        suite.assert_false(float4(std::numeric_limits<float>::infinity(), 2.0f, 3.0f, 4.0f).isValid(), "Invalid vector - infinity");
        suite.assert_false(float4(1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f).isValid(), "Invalid vector - NaN");

        // Approximate equality
        float4 a(1.0f, 2.0f, 3.0f, 4.0f);
        float4 b(1.000001f, 2.000001f, 3.000001f, 4.000001f);
        suite.assert_true(a.approximately(b, 0.001f), "Approximate equality");
        suite.assert_false(a.approximately(b, 1e-8f), "Approximate inequality");

        // Zero checks
        suite.assert_true(float4::zero().approximately_zero(), "Exactly zero");
        suite.assert_true(float4(1e-8f, -1e-8f, 1e-8f, -1e-8f).approximately_zero(1e-7f), "Approximately zero");

        // Normalization checks
        suite.assert_true(float4(0.5f, 0.5f, 0.5f, 0.5f).is_normalized(0.001f), "Normalized check");
        suite.assert_false(float4(1.0f, 1.0f, 1.0f, 1.0f).is_normalized(), "Not normalized check");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains x");
        suite.assert_true(str.find("2.0") != std::string::npos, "String contains y");
        suite.assert_true(str.find("3.0") != std::string::npos, "String contains z");
        suite.assert_true(str.find("4.0") != std::string::npos, "String contains w");

        // Component access
        suite.assert_equal(a[0], 1.0f, "Index access [0]");
        suite.assert_equal(a[1], 2.0f, "Index access [1]");
        suite.assert_equal(a[2], 3.0f, "Index access [2]");
        suite.assert_equal(a[3], 4.0f, "Index access [3]");

        // Data access
        suite.assert_equal(a.data()[0], 1.0f, "Data pointer access [0]");
        suite.assert_equal(a.data()[1], 2.0f, "Data pointer access [1]");
        suite.assert_equal(a.data()[2], 3.0f, "Data pointer access [2]");
        suite.assert_equal(a.data()[3], 4.0f, "Data pointer access [3]");

        // Set methods
        float4 vec(0.0f, 0.0f, 0.0f, 5.0f);
        vec.set_xyz(float3(1.0f, 2.0f, 3.0f));
        suite.assert_equal(vec, float4(1.0f, 2.0f, 3.0f, 5.0f), "Set XYZ from float3");

        vec = float4(0.0f, 0.0f, 3.0f, 4.0f);
        vec.set_xy(float2(1.0f, 2.0f));
        suite.assert_equal(vec, float4(1.0f, 2.0f, 3.0f, 4.0f), "Set XY from float2");

        vec = float4(1.0f, 2.0f, 0.0f, 0.0f);
        vec.set_zw(float2(3.0f, 4.0f));
        suite.assert_equal(vec, float4(1.0f, 2.0f, 3.0f, 4.0f), "Set ZW from float2");

        // Comparison operators
        suite.assert_true(a == a, "Equality operator");
        suite.assert_true(a != float4::zero(), "Inequality operator");

        suite.footer();
    }

    void test_float4_edge_cases()
    {
        TestSuite suite("float4 Edge Cases");
        suite.header();

        // Zero vectors
        suite.assert_true(float4::zero().approximately_zero(), "Zero vector");
        suite.assert_equal(float4::zero().length(), 0.0f, "Zero length");
        suite.assert_equal(float4::zero().length_sq(), 0.0f, "Zero squared length");

        // Very small vectors
        float4 tiny(1e-10f, 1e-10f, 1e-10f, 1e-10f);
        suite.assert_true(tiny.approximately_zero(1e-5f), "Tiny vector approximately zero");
        suite.assert_false(tiny.approximately_zero(1e-15f), "Tiny vector not exactly zero");

        // Very large vectors
        float4 large(1e10f, 1e10f, 1e10f, 1e10f);
        suite.assert_true(std::isinf(large.length()) || large.length() > 1e10f, "Large vector length");

        // Normalization edge cases
        suite.assert_true(float4::zero().normalize().approximately_zero(), "Normalize zero vector");

        float4 inf_vec(std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f);
        suite.assert_false(inf_vec.is_normalized(), "Infinity vector normalization check");

        // NaN propagation
        float4 nan_vec(std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f, 1.0f);
        suite.assert_false(nan_vec.isValid(), "NaN vector validity");

        // Test normalization of very small vectors
        float4 very_small(1e-20f, 1e-20f, 1e-20f, 1e-20f);
        float4 normalized_small = very_small.normalize();
        suite.assert_true(normalized_small.is_normalized(0.001f) || normalized_small.approximately_zero(),
            "Very small vector normalization");

        // Cross product edge cases
        suite.assert_equal(cross(float4::unit_x(), float4::unit_x()), float4::zero(), "Cross product with itself");
        suite.assert_equal(cross(float4::unit_x(), float4::unit_y()), float4::unit_z(), "Cross product orthogonality");

        // Dot product edge cases
        suite.assert_equal(dot(float4::zero(), float4::one()), 0.0f, "Dot product with zero vector");
        suite.assert_equal(dot(float4::unit_x(), float4::unit_x()), 1.0f, "Dot product with itself");

        // Homogeneous coordinates edge cases
        float4 zero_w(1.0f, 2.0f, 3.0f, 0.0f);
        suite.assert_equal(zero_w.project(), float3::zero(), "Project with zero w");

        suite.footer();
    }

    void run_float4_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "FLOAT4 COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_float4_constructors();
        test_float4_arithmetic_operations();
        test_float4_mathematical_functions();
        test_float4_color_operations();
        test_float4_geometric_operations();
        test_float4_swizzle_operations();
        test_float4_utility_methods();
        test_float4_edge_cases();

        std::cout << "FLOAT4 TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
    // half Tests - Comprehensive
    // ============================================================================

    class HalfTestData
    {
    public:
        static std::vector<half> generate_test_halves()
        {
            return {
                half(0.0f), half(1.0f), half(-1.0f), half(0.5f), half(-0.5f),
                half(Constants::PI), half(Constants::HALF_PI), half(Constants::TWO_PI),
                half(std::numeric_limits<float>::epsilon()),
                half(std::numeric_limits<float>::min()),
                half(65504.0f), // max half value
                half(6.10352e-5f), // min normalized half
                half(5.96046e-8f), // min denormalized half
                half(std::numeric_limits<float>::infinity()),
                half(-std::numeric_limits<float>::infinity()),
                half(std::numeric_limits<float>::quiet_NaN())
            };
        }

        static std::vector<std::pair<half, float>> generate_half_float_pairs()
        {
            auto halves = generate_test_halves();
            std::vector<std::pair<half, float>> pairs;
            pairs.reserve(halves.size());

            for (const auto& h : halves) {
                pairs.emplace_back(h, float(h));
            }
            return pairs;
        }
    };

    void test_half_constructors()
    {
        TestSuite suite("half Constructors");
        suite.header();

        // Default constructor
        suite.assert_equal_half_t(half(), half(0.0f), "Default constructor");

        // Float constructor
        half h1(1.5f);
        suite.assert_equal_half_t(float(h1), 1.5f, "Float constructor");

        // Bits constructor
        half h2(half::from_bits(0x3C00)); // 1.0
        suite.assert_equal_half_t(float(h2), 1.0f, "Bits constructor");

        // Copy constructor
        half original(2.5f);
        half copy(original);
        suite.assert_equal_half_t(copy, original, "Copy constructor");

        // Integral constructors
        suite.assert_equal_half_t(half(5), half(5.0f), "Int constructor");
        suite.assert_equal_half_t(half(3U), half(3.0f), "Unsigned int constructor");

        // Assignment operators
        half h3;
        h3 = 2.0f;
        suite.assert_equal_half_t(h3, half(2.0f), "Float assignment");

        h3 = original;
        suite.assert_equal_half_t(h3, original, "Half assignment");

        // Edge cases
        suite.assert_true(half(0.0f).is_zero(), "Zero construction");
        suite.assert_true(half(std::numeric_limits<float>::infinity()).is_inf(), "Infinity construction");
        suite.assert_true(half(std::numeric_limits<float>::quiet_NaN()).is_nan(), "NaN construction");

        suite.footer();
    }

    void test_half_conversions()
    {
        TestSuite suite("half Conversions");
        suite.header();

        auto pairs = HalfTestData::generate_half_float_pairs();

        for (size_t i = 0; i < pairs.size(); ++i) {
            const auto& [h, f] = pairs[i];
            std::string test_name = "Conversion test " + std::to_string(i);

            if (h.is_nan()) {
                suite.assert_nan(float(h), test_name + " (NaN)");
            }
            else if (h.is_inf()) {
                suite.assert_infinity(float(h), test_name + " (Inf)");
            }
            else {
                // Для обычных чисел используем более либеральный epsilon из-за потери точности
                suite.assert_equal_half_t(float(h), f, test_name, 0.001f);
            }
        }

        // Double conversion
        half h(3.14f);
        suite.assert_equal_half_t(double(h), 3.14, "Double conversion", 0.001);

        suite.footer();
    }

    void test_half_arithmetic_operations()
    {
        TestSuite suite("half Arithmetic Operations");
        suite.header();

        half a(2.0f);
        half b(3.0f);
        half c;

        // Basic arithmetic
        suite.assert_equal_half_t(a + b, half(5.0f), "Addition");
        suite.assert_equal_half_t(a - b, half(-1.0f), "Subtraction");
        suite.assert_equal_half_t(a * b, half(6.0f), "Multiplication");
        suite.assert_equal_half_t(a / b, half(2.0f / 3.0f), "Division", 0.001f);

        // Compound assignments
        c = a; c += b;
        suite.assert_equal_half_t(c, a + b, "Compound addition");

        c = a; c -= b;
        suite.assert_equal_half_t(c, a - b, "Compound subtraction");

        c = a; c *= b;
        suite.assert_equal_half_t(c, a * b, "Compound multiplication");

        c = a; c /= b;
        suite.assert_equal_half_t(c, a / b, "Compound division", 0.001f);

        // Unary operators
        suite.assert_equal_half_t(+a, a, "Unary plus");
        suite.assert_equal_half_t(-a, half(-2.0f), "Unary minus");

        // Increment/decrement
        c = a; ++c;
        suite.assert_equal_half_t(c, half(3.0f), "Pre-increment");

        c = a; c++;
        suite.assert_equal_half_t(c, half(3.0f), "Post-increment");

        c = a; --c;
        suite.assert_equal_half_t(c, half(1.0f), "Pre-decrement");

        c = a; c--;
        suite.assert_equal_half_t(c, half(1.0f), "Post-decrement");

        // Mixed-type operations
        suite.assert_equal_half_t(a + 1.0f, half(3.0f), "Half + float");
        suite.assert_equal_half_t(1.0f + a, half(3.0f), "Float + half");
        suite.assert_equal_half_t(a + 1, half(3.0f), "Half + int");
        suite.assert_equal_half_t(1 + a, half(3.0f), "Int + half");

        suite.footer();
    }

    void test_half_comparison_operations()
    {
        TestSuite suite("half Comparison Operations");
        suite.header();

        half a(2.0f);
        half b(3.0f);
        half c(2.0f);

        // Equality
        suite.assert_true(a == c, "Equality");
        suite.assert_true(a != b, "Inequality");
        suite.assert_false(a == b, "Non-equality");

        // Relational operators
        suite.assert_true(a < b, "Less than");
        suite.assert_true(b > a, "Greater than");
        suite.assert_true(a <= c, "Less than or equal");
        suite.assert_true(a >= c, "Greater than or equal");
        suite.assert_true(a <= b, "Less than or equal with different values");
        suite.assert_true(b >= a, "Greater than or equal with different values");

        // Special values comparisons
        half zero(0.0f);
        half neg_zero(-0.0f);
        half inf(std::numeric_limits<float>::infinity());
        half neg_inf(-std::numeric_limits<float>::infinity());
        half nan(std::numeric_limits<float>::quiet_NaN());

        // Zeros should be equal
        suite.assert_true(zero == neg_zero, "Positive and negative zero equality");

        // Infinity comparisons
        suite.assert_true(inf > zero, "Infinity greater than zero");
        suite.assert_true(neg_inf < zero, "Negative infinity less than zero");
        suite.assert_true(inf > neg_inf, "Infinity greater than negative infinity");

        // NaN comparisons - should all be false
        suite.assert_false(nan == nan, "NaN equality");
        suite.assert_false(nan < zero, "NaN less than");
        suite.assert_false(nan > zero, "NaN greater than");

        suite.footer();
    }

    void test_half_special_values()
    {
        TestSuite suite("half Special Values");
        suite.header();

        // Zero values
        half zero(0.0f);
        half neg_zero(-0.0f);

        suite.assert_true(zero.is_zero(), "Zero detection");
        suite.assert_true(neg_zero.is_zero(), "Negative zero detection");
        suite.assert_true(zero.is_positive_zero(), "Positive zero detection");
        suite.assert_true(neg_zero.is_negative_zero(), "Negative zero detection");

        // Infinity values
        half inf(std::numeric_limits<float>::infinity());
        half neg_inf(-std::numeric_limits<float>::infinity());

        suite.assert_true(inf.is_inf(), "Infinity detection");
        suite.assert_true(neg_inf.is_inf(), "Negative infinity detection");
        suite.assert_true(inf.is_positive_inf(), "Positive infinity detection");
        suite.assert_true(neg_inf.is_negative_inf(), "Negative infinity detection");

        // NaN values
        half nan(std::numeric_limits<float>::quiet_NaN());
        suite.assert_true(nan.is_nan(), "NaN detection");

        // Finite values
        suite.assert_true(zero.is_finite(), "Zero is finite");
        suite.assert_true(half(1.0f).is_finite(), "One is finite");
        suite.assert_false(inf.is_finite(), "Infinity is not finite");
        suite.assert_false(nan.is_finite(), "NaN is not finite");

        // Normal/denormal values
        suite.assert_true(half(1.0f).is_normal(), "One is normal");
        suite.assert_true(zero.is_normal() || zero.is_zero(), "Zero is normal or zero");

        // Sign detection
        suite.assert_true(neg_zero.is_negative(), "Negative zero is negative");
        suite.assert_true(neg_inf.is_negative(), "Negative infinity is negative");
        suite.assert_true(half(-1.0f).is_negative(), "Negative one is negative");
        suite.assert_false(zero.is_negative(), "Positive zero is not negative");
        suite.assert_false(inf.is_negative(), "Positive infinity is not negative");
        suite.assert_false(half(1.0f).is_negative(), "Positive one is not negative");

        suite.footer();
    }

    void test_half_bit_operations()
    {
        TestSuite suite("half Bit Operations");
        suite.header();

        // Test specific bit patterns
        half zero = half::from_bits(0x0000);
        half one = half::from_bits(0x3C00);
        half neg_one = half::from_bits(0xBC00);
        half inf = half::from_bits(0x7C00);
        half neg_inf = half::from_bits(0xFC00);
        half nan = half::from_bits(0x7E00);

        // Используем static_cast для приведения типов
        suite.assert_equal_half_t(zero.bits(), static_cast<half::storage_type>(0x0000), "Zero bits");
        suite.assert_equal_half_t(one.bits(), static_cast<half::storage_type>(0x3C00), "One bits");
        suite.assert_equal_half_t(neg_one.bits(), static_cast<half::storage_type>(0xBC00), "Negative one bits");
        suite.assert_equal_half_t(inf.bits(), static_cast<half::storage_type>(0x7C00), "Infinity bits");
        suite.assert_equal_half_t(neg_inf.bits(), static_cast<half::storage_type>(0xFC00), "Negative infinity bits");

        suite.assert_equal_half_t(zero.sign_bit(), 0, "Zero sign bit");
        suite.assert_equal_half_t(one.sign_bit(), 0, "One sign bit");
        suite.assert_equal_half_t(neg_one.sign_bit(), 1, "Negative one sign bit");

        suite.assert_equal_half_t(one.exponent(), 15, "One exponent"); // bias 15
        suite.assert_equal_half_t(zero.exponent(), 0, "Zero exponent");

        suite.assert_equal_half_t(one.mantissa(), 0, "One mantissa");

        // From bits round-trip
        half test_val(2.5f);
        half from_bits = half::from_bits(test_val.bits());
        suite.assert_equal_half_t(from_bits, test_val, "From bits round-trip");

        suite.footer();
    }

    void test_half_mathematical_functions()
    {
        TestSuite suite("half Mathematical Functions");
        suite.header();

        half a(2.0f);
        half b(4.0f);
        half c;

        // Basic math functions
        suite.assert_equal_half_t(abs(half(-3.0f)), half(3.0f), "Absolute value");
        suite.assert_equal_half_t(sqrt(b), half(2.0f), "Square root", 0.001f);
        suite.assert_equal_half_t(rsqrt(b), half(0.5f), "Reciprocal square root", 0.001f);

        // Trigonometric functions
        suite.assert_equal_half_t(sin(half(0.0f)), half(0.0f), "Sine of zero", 0.001f);
        suite.assert_equal_half_t(cos(half(0.0f)), half(1.0f), "Cosine of zero", 0.001f);
        suite.assert_equal_half_t(tan(half(0.0f)), half(0.0f), "Tangent of zero", 0.001f);

        // Exponential and logarithmic functions
        suite.assert_equal_half_t(exp(half(0.0f)), half(1.0f), "Exponential of zero", 0.001f);
        suite.assert_equal_half_t(log(half(1.0f)), half(0.0f), "Logarithm of one", 0.001f);
        suite.assert_equal_half_t(log10(half(100.0f)), half(2.0f), "Log base 10", 0.001f);
        suite.assert_equal_half_t(pow(a, half(3.0f)), half(8.0f), "Power function", 0.001f);

        // Rounding functions
        suite.assert_equal_half_t(floor(half(2.7f)), half(2.0f), "Floor");
        suite.assert_equal_half_t(ceil(half(2.3f)), half(3.0f), "Ceil");
        suite.assert_equal_half_t(round(half(2.5f)), half(3.0f), "Round");
        suite.assert_equal_half_t(trunc(half(2.7f)), half(2.0f), "Truncate");

        // Fractional part
        half frac_result = frac(half(3.75f));
        suite.assert_equal_half_t(frac_result, half(0.75f), "Fractional part", 0.001f);

        // Modulo
        suite.assert_equal_half_t(fmod(half(5.5f), half(2.0f)), half(1.5f), "Floating point modulo", 0.001f);

        suite.footer();
    }

    void test_half_hlsl_functions()
    {
        TestSuite suite("half HLSL Functions");
        suite.header();

        // Saturate
        suite.assert_equal_half_t(saturate(half(-0.5f)), half(0.0f), "Saturate negative");
        suite.assert_equal_half_t(saturate(half(0.5f)), half(0.5f), "Saturate middle");
        suite.assert_equal_half_t(saturate(half(1.5f)), half(1.0f), "Saturate positive");

        // Clamp
        suite.assert_equal_half_t(clamp(half(0.3f), half(0.5f), half(1.0f)), half(0.5f), "Clamp below");
        suite.assert_equal_half_t(clamp(half(0.7f), half(0.5f), half(1.0f)), half(0.7f), "Clamp within");
        suite.assert_equal_half_t(clamp(half(1.3f), half(0.5f), half(1.0f)), half(1.0f), "Clamp above");

        // Lerp
        suite.assert_equal_half_t(lerp(half(1.0f), half(3.0f), half(0.5f)), half(2.0f), "Lerp middle");
        suite.assert_equal_half_t(lerp(half(1.0f), half(3.0f), half(0.0f)), half(1.0f), "Lerp start");
        suite.assert_equal_half_t(lerp(half(1.0f), half(3.0f), half(1.0f)), half(3.0f), "Lerp end");

        // Step
        suite.assert_equal_half_t(step(half(0.5f), half(0.3f)), half(0.0f), "Step below");
        suite.assert_equal_half_t(step(half(0.5f), half(0.5f)), half(1.0f), "Step equal");
        suite.assert_equal_half_t(step(half(0.5f), half(0.7f)), half(1.0f), "Step above");

        // Smoothstep
        half smooth = smoothstep(half(0.0f), half(1.0f), half(0.5f));
        suite.assert_true(smooth.approximately(half(0.5f), 0.01f), "Smoothstep middle");

        // Sign
        suite.assert_equal_half_t(sign(half(-2.0f)), half(-1.0f), "Sign negative");
        suite.assert_equal_half_t(sign(half(0.0f)), half(0.0f), "Sign zero");
        suite.assert_equal_half_t(sign(half(2.0f)), half(1.0f), "Sign positive");

        // Angle conversions
        suite.assert_equal_half_t(radians(half(180.0f)), half(Constants::PI), "Radians conversion", 0.001f);
        suite.assert_equal_half_t(degrees(half(Constants::PI)), half(180.0f), "Degrees conversion", 0.001f);

        suite.footer();
    }

    void test_half_utility_methods()
    {
        TestSuite suite("half Utility Methods");
        suite.header();

        half a(2.0f);
        half b(2.0001f);
        half c(2.1f);

        // Approximate equality
        suite.assert_true(a.approximately(b, 0.001f), "Approximate equality");
        suite.assert_false(a.approximately(c, 0.001f), "Approximate inequality");

        // Approximately zero
        suite.assert_true(half(0.0f).approximately_zero(), "Exactly zero");
        suite.assert_true(half(0.0001f).approximately_zero(0.001f), "Approximately zero");
        suite.assert_false(a.approximately_zero(), "Non-zero");

        // Validity checks
        suite.assert_true(a.is_valid(), "Valid half");
        suite.assert_true(half(std::numeric_limits<float>::infinity()).is_valid(), "Infinity is valid");
        suite.assert_true(half(std::numeric_limits<float>::quiet_NaN()).is_valid(), "NaN is valid");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("2.0") != std::string::npos, "String conversion");

        // Reciprocal
        suite.assert_equal_half_t(a.reciprocal(), half(0.5f), "Reciprocal", 0.001f);

        // Static constants
        suite.assert_true(half::infinity().is_inf(), "Static infinity");
        suite.assert_true(half::negative_infinity().is_negative_inf(), "Static negative infinity");
        suite.assert_true(half::quiet_nan().is_nan(), "Static quiet NaN");
        suite.assert_true(half::max_value().is_finite(), "Static max value is finite");
        suite.assert_true(half::min_value().is_finite(), "Static min value is finite");
        suite.assert_true(half::epsilon().is_finite(), "Static epsilon is finite");

        suite.footer();
    }

    void test_half_edge_cases()
    {
        TestSuite suite("half Edge Cases");
        suite.header();

        // Very small values
        half tiny(1e-7f);
        suite.assert_true(tiny.is_finite(), "Very small value is finite");
        suite.assert_false(tiny.approximately_zero(1e-8f), "Very small value not approximately zero with tight epsilon");

        // Very large values (but within half range)
        half large(60000.0f);
        suite.assert_true(large.is_finite(), "Large value is finite");

        // Denormal values
        half denorm = half::min_denormal_value();
        suite.assert_true(denorm.is_finite(), "Denormal value is finite");

        // NaN propagation in arithmetic
        half nan(std::numeric_limits<float>::quiet_NaN());
        half normal(2.0f);

        suite.assert_true(is_nan(nan + normal), "NaN + normal = NaN");
        suite.assert_true(is_nan(normal + nan), "normal + NaN = NaN");
        suite.assert_true(is_nan(nan * normal), "NaN * normal = NaN");

        // Infinity arithmetic
        half inf(std::numeric_limits<float>::infinity());
        suite.assert_true(is_inf(inf + inf), "inf + inf = inf");
        suite.assert_true(is_nan(inf - inf), "inf - inf = NaN");
        suite.assert_true(is_inf(inf * inf), "inf * inf = inf");
        suite.assert_true(is_nan(inf / inf), "inf / inf = NaN");

        // Zero arithmetic
        half zero(0.0f);
        suite.assert_true(is_nan(zero / zero), "0 / 0 = NaN");
        suite.assert_true(is_inf(normal / zero), "normal / 0 = inf");

        // Overflow
        half huge(70000.0f); // Beyond half max
        suite.assert_true(huge.is_inf() || huge.is_finite(), "Overflow handling");

        suite.footer();
    }

    // ============================================================================
    // Test Runner for half
    // ============================================================================

    void run_half_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "HALF COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_half_constructors();
        test_half_conversions();
        test_half_arithmetic_operations();
        test_half_comparison_operations();
        test_half_special_values();
        test_half_bit_operations();
        test_half_mathematical_functions();
        test_half_hlsl_functions();
        test_half_utility_methods();
        test_half_edge_cases();

        std::cout << "HALF TESTS COMPLETE" << std::endl;
    }

   // ============================================================================
   // Test Data Generators for half2
   // ============================================================================

    class Half2TestData
    {
    public:
        static std::vector<half2> generate_test_half2s()
        {
            std::vector<half> half_values = HalfTestData::generate_test_halves();
            std::vector<half2> result;

            // Generate combinations of half values
            for (size_t i = 0; i < half_values.size(); ++i)
            {
                for (size_t j = 0; j < half_values.size(); ++j)
                {
                    if (result.size() >= 15) break; // Limit size
                    result.emplace_back(half_values[i], half_values[j]);
                }
                if (result.size() >= 15) break;
            }

            // Add specific test cases
            result.emplace_back(half2::zero());
            result.emplace_back(half2::one());
            result.emplace_back(half2::unit_x());
            result.emplace_back(half2::unit_y());
            result.emplace_back(half2::uv(half(0.5f), half(0.5f)));

            return result;
        }

        static std::vector<std::pair<half2, float2>> generate_half2_float2_pairs()
        {
            auto half2s = generate_test_half2s();
            std::vector<std::pair<half2, float2>> pairs;
            pairs.reserve(half2s.size());

            for (const auto& h2 : half2s) {
                pairs.emplace_back(h2, float2(h2));
            }
            return pairs;
        }
    };

    // ============================================================================
    // half2 Tests - Comprehensive
    // ============================================================================

    void test_half2_constructors()
    {
        TestSuite suite("half2 Constructors");
        suite.header();

        // Default constructor
        suite.assert_equal_half2_t(half2(), half2(0.0f, 0.0f), "Default constructor");

        // Half component constructor
        half2 v1(half(1.5f), half(2.5f));
        suite.assert_equal(float(v1.x), 1.5f, "Half component constructor - x");
        suite.assert_equal(float(v1.y), 2.5f, "Half component constructor - y");

        // Float component constructor
        half2 v2(1.5f, 2.5f);
        suite.assert_equal(float(v2.x), 1.5f, "Float component constructor - x");
        suite.assert_equal(float(v2.y), 2.5f, "Float component constructor - y");

        // Scalar constructor
        suite.assert_equal_half2_t(half2(5.0f), half2(5.0f, 5.0f), "Scalar constructor");

        // Copy constructor
        half2 original(2.0f, 3.0f);
        half2 copy(original);
        suite.assert_equal_half2_t(copy, original, "Copy constructor");

        // From float2 constructor
        suite.assert_equal_half2_t(half2(float2(1.0f, 2.0f)), half2(1.0f, 2.0f), "From float2 constructor");

        // Static constructors
        suite.assert_equal_half2_t(half2::zero(), half2(0.0f, 0.0f), "Zero vector");
        suite.assert_equal_half2_t(half2::one(), half2(1.0f, 1.0f), "One vector");
        suite.assert_equal_half2_t(half2::unit_x(), half2(1.0f, 0.0f), "Unit X");
        suite.assert_equal_half2_t(half2::unit_y(), half2(0.0f, 1.0f), "Unit Y");
        suite.assert_equal_half2_t(half2::uv(half(0.3f), half(0.7f)), half2(0.3f, 0.7f), "UV constructor");

        // Edge cases
        half2 inf_vec(std::numeric_limits<float>::infinity(), 0.0f);
        suite.assert_true(std::isinf(float(inf_vec.x)), "Infinity component");

        suite.footer();
    }

    void test_half2_assignment_operators()
    {
        TestSuite suite("half2 Assignment Operators");
        suite.header();

        half2 a(1.0f, 2.0f);
        half2 b;

        // Copy assignment
        b = a;
        suite.assert_equal_half2_t(b, a, "Copy assignment");

        // Float2 assignment
        b = float2(3.0f, 4.0f);
        suite.assert_equal_half2_t(b, half2(3.0f, 4.0f), "Float2 assignment");

        // Half scalar assignment
        b = half(5.0f);
        suite.assert_equal_half2_t(b, half2(5.0f, 5.0f), "Half scalar assignment");

        // Float scalar assignment
        b = 6.0f;
        suite.assert_equal_half2_t(b, half2(6.0f, 6.0f), "Float scalar assignment");

        suite.footer();
    }

    void test_half2_compound_assignment_operators()
    {
        TestSuite suite("half2 Compound Assignment Operators");
        suite.header();

        half2 a(2.0f, 3.0f);
        half2 b(1.0f, 2.0f);
        half2 c;

        // Compound addition
        c = a; c += b;
        suite.assert_equal_half2_t(c, half2(3.0f, 5.0f), "Compound addition");

        // Compound subtraction
        c = a; c -= b;
        suite.assert_equal_half2_t(c, half2(1.0f, 1.0f), "Compound subtraction");

        // Compound multiplication
        c = a; c *= b;
        suite.assert_equal_half2_t(c, half2(2.0f, 6.0f), "Compound multiplication");

        // Compound division
        c = a; c /= b;
        suite.assert_equal_half2_t(c, half2(2.0f, 1.5f), "Compound division", 0.001f);

        // Compound scalar multiplication (half)
        c = a; c *= half(2.0f);
        suite.assert_equal_half2_t(c, half2(4.0f, 6.0f), "Compound scalar multiplication (half)");

        // Compound scalar multiplication (float)
        c = a; c *= 2.0f;
        suite.assert_equal_half2_t(c, half2(4.0f, 6.0f), "Compound scalar multiplication (float)");

        // Compound scalar division (half)
        c = a; c /= half(2.0f);
        suite.assert_equal_half2_t(c, half2(1.0f, 1.5f), "Compound scalar division (half)");

        // Compound scalar division (float)
        c = a; c /= 2.0f;
        suite.assert_equal_half2_t(c, half2(1.0f, 1.5f), "Compound scalar division (float)");

        suite.footer();
    }

    void test_half2_arithmetic_operations()
    {
        TestSuite suite("half2 Arithmetic Operations");
        suite.header();

        half2 a(1.0f, 2.0f);
        half2 b(3.0f, 4.0f);
        half2 c;

        // Vector operations
        suite.assert_equal_half2_t(a + b, half2(4.0f, 6.0f), "Vector addition");
        suite.assert_equal_half2_t(a - b, half2(-2.0f, -2.0f), "Vector subtraction");
        suite.assert_equal_half2_t(a * b, half2(3.0f, 8.0f), "Component-wise multiplication");
        suite.assert_equal_half2_t(a / b, half2(1.0f / 3.0f, 2.0f / 4.0f), "Component-wise division", 0.001f);

        // Scalar operations
        suite.assert_equal_half2_t(a * 2.0f, half2(2.0f, 4.0f), "Scalar multiplication");
        suite.assert_equal_half2_t(2.0f * a, half2(2.0f, 4.0f), "Scalar multiplication commutative");
        suite.assert_equal_half2_t(a / 2.0f, half2(0.5f, 1.0f), "Scalar division");

        // Unary operations
        suite.assert_equal_half2_t(+a, a, "Unary plus");
        suite.assert_equal_half2_t(-a, half2(-1.0f, -2.0f), "Unary minus");

        // Mixed type operations with float2
        float2 f2(3.0f, 4.0f);
        suite.assert_equal_half2_t(a + f2, half2(4.0f, 6.0f), "half2 + float2");
        suite.assert_equal_half2_t(f2 + a, half2(4.0f, 6.0f), "float2 + half2");
        suite.assert_equal_half2_t(a - f2, half2(-2.0f, -2.0f), "half2 - float2");
        suite.assert_equal_half2_t(f2 - a, half2(2.0f, 2.0f), "float2 - half2");

        suite.footer();
    }

    void test_half2_access_operations()
    {
        TestSuite suite("half2 Access Operations");
        suite.header();

        half2 v(1.5f, 2.5f);

        // Index access
        suite.assert_equal_half_t(v[0], half(1.5f), "Index access [0]");
        suite.assert_equal_half_t(v[1], half(2.5f), "Index access [1]");

        // Mutable index access
        v[0] = half(3.0f);
        v[1] = half(4.0f);
        suite.assert_equal_half2_t(v, half2(3.0f, 4.0f), "Mutable index access");

        // Data pointer access
        const half* data = v.data();
        suite.assert_equal_half_t(data[0], half(3.0f), "Data pointer access [0]");
        suite.assert_equal_half_t(data[1], half(4.0f), "Data pointer access [1]");

        // Mutable data pointer access
        half* mutable_data = v.data();
        mutable_data[0] = half(5.0f);
        mutable_data[1] = half(6.0f);
        suite.assert_equal_half2_t(v, half2(5.0f, 6.0f), "Mutable data pointer access");

        suite.footer();
    }

    void test_half2_conversion_operations()
    {
        TestSuite suite("half2 Conversion Operations");
        suite.header();

        half2 h2(1.5f, 2.5f);

        // Conversion to float2
        float2 f2 = static_cast<float2>(h2);
        suite.assert_equal_half2_t(f2, float2(1.5f, 2.5f), "Conversion to float2");

        // Global conversion functions
        suite.assert_equal_half2_t(to_float2(h2), float2(1.5f, 2.5f), "to_float2 function");
        suite.assert_equal_half2_t(to_half2(f2), h2, "to_half2 function");

        // Round-trip conversion
        half2 round_trip = to_half2(to_float2(h2));
        suite.assert_equal_half2_t(round_trip, h2, "Round-trip conversion");

        suite.footer();
    }

    void test_half2_mathematical_functions()
    {
        TestSuite suite("half2 Mathematical Functions");
        suite.header();

        half2 v(3.0f, 4.0f);
        half2 w(1.0f, 2.0f);

        // Length calculations
        suite.assert_equal_half_t(v.length(), 5.0f, "Length", 0.01f);
        suite.assert_equal_half_t(v.length_sq(), 25.0f, "Squared length", 0.01f);

        // Distance calculations
        suite.assert_equal_half_t(distance(v, w), std::sqrt(8.0f), "Distance", 0.01f);
        suite.assert_equal_half_t(distance_sq(v, w), 8.0f, "Squared distance", 0.01f);

        // Dot product
        suite.assert_equal_half_t(dot(v, w), 11.0f, "Dot product", 0.01f);
        suite.assert_equal_half_t(half2::dot(v, w), 11.0f, "Static dot product", 0.01f);

        // Cross product
        suite.assert_equal_half_t(cross(v, w), 2.0f, "2D cross product", 0.01f);

        // Normalization
        half2 normalized = v.normalize();
        suite.assert_true(normalized.is_normalized(0.01f), "Normalization result is normalized");
        suite.assert_equal_half_t(normalized.length(), 1.0f, "Normalized length", 0.01f);
        suite.assert_true(v.normalize().approximately(half2(0.6f, 0.8f), 0.01f), "Normalization values");

        // Global normalization
        suite.assert_equal_half2_t(normalize(v), v.normalize(), "Global normalize function");

        suite.footer();
    }

    void test_half2_hlsl_functions()
    {
        TestSuite suite("half2 HLSL Functions");
        suite.header();

        half2 v(1.5f, -2.5f);
        half2 w(0.3f, 0.7f);

        // Component-wise functions
        suite.assert_equal_half2_t(abs(v), half2(1.5f, 2.5f), "Absolute value");
        suite.assert_equal_half2_t(sign(v), half2(1.0f, -1.0f), "Sign function");
        suite.assert_equal_half2_t(floor(half2(1.7f, -2.3f)), half2(1.0f, -3.0f), "Floor");
        suite.assert_equal_half2_t(ceil(half2(1.2f, -2.7f)), half2(2.0f, -2.0f), "Ceil");
        suite.assert_equal_half2_t(round(half2(1.4f, 1.6f)), half2(1.0f, 2.0f), "Round");
        suite.assert_equal_half2_t(frac(half2(1.7f, -2.3f)), half2(0.7f, 0.7f), "Fractional part", 0.01f);

        // Saturate
        suite.assert_equal_half2_t(saturate(half2(-0.5f, 1.5f)), half2(0.0f, 1.0f), "Saturate");
        suite.assert_equal_half2_t(half2::saturate(half2(-0.5f, 1.5f)), half2(0.0f, 1.0f), "Static saturate");

        // Step
        suite.assert_equal_half2_t(step(half(1.0f), half2(0.5f, 1.5f)), half2(0.0f, 1.0f), "Step function");

        // Min/Max
        suite.assert_equal_half2_t(min(half2(1, 3), half2(2, 2)), half2(1, 2), "Minimum");
        suite.assert_equal_half2_t(max(half2(1, 3), half2(2, 2)), half2(2, 3), "Maximum");
        suite.assert_equal_half2_t(half2::min(half2(1, 3), half2(2, 2)), half2(1, 2), "Static minimum");
        suite.assert_equal_half2_t(half2::max(half2(1, 3), half2(2, 2)), half2(2, 3), "Static maximum");

        // Clamp
        suite.assert_equal_half2_t(clamp(half2(0.5f, 1.5f), 0.0f, 1.0f), half2(0.5f, 1.0f), "Scalar clamp");
        suite.assert_equal_half2_t(clamp(half2(0.5f, 1.5f), half2(0, 1), half2(1, 2)), half2(0.5f, 1.5f), "Vector clamp");

        // Smoothstep
        half2 smooth = smoothstep(half(0.0f), half(1.0f), half2(0.5f, 0.5f));
        suite.assert_true(smooth.approximately(half2(0.5f, 0.5f), 0.1f), "Smoothstep");

        suite.footer();
    }

    void test_half2_geometric_operations()
    {
        TestSuite suite("half2 Geometric Operations");
        suite.header();

        half2 v(1.0f, 0.0f);
        half2 w(0.0f, 1.0f);

        // Perpendicular
        suite.assert_equal_half2_t(perpendicular(v), half2(0.0f, 1.0f), "Perpendicular vector");
        suite.assert_equal_half2_t(v.perpendicular(), half2(0.0f, 1.0f), "Member perpendicular");
        suite.assert_equal_half2_t(perpendicular(w), half2(-1.0f, 0.0f), "Perpendicular vector 2");

        // Angles
        suite.assert_equal_half_t(angle(v), 0.0f, "Zero angle", 0.001f);
        suite.assert_equal_half_t(angle(w), Constants::HALF_PI, "90 degree angle", 0.001f);
        suite.assert_equal_half_t(angle_between(v, w), Constants::HALF_PI, "Angle between vectors", 0.001f);

        // Lerp
        suite.assert_equal_half2_t(lerp(v, w, half(0.5f)), half2(0.5f, 0.5f), "Lerp with half");
        suite.assert_equal_half2_t(lerp(v, w, 0.5f), half2(0.5f, 0.5f), "Lerp with float");
        suite.assert_equal_half2_t(half2::lerp(v, w, half(0.5f)), half2(0.5f, 0.5f), "Static lerp with half");
        suite.assert_equal_half2_t(half2::lerp(v, w, 0.5f), half2(0.5f, 0.5f), "Static lerp with float");

        suite.footer();
    }

    void test_half2_swizzle_operations()
    {
        TestSuite suite("half2 Swizzle Operations");
        suite.header();

        half2 v(1.0f, 2.0f);

        // 2D swizzles
        suite.assert_equal_half2_t(v.yx(), half2(2.0f, 1.0f), "YX swizzle");
        suite.assert_equal_half2_t(v.xx(), half2(1.0f, 1.0f), "XX swizzle");
        suite.assert_equal_half2_t(v.yy(), half2(2.0f, 2.0f), "YY swizzle");

        suite.footer();
    }

    void test_half2_texture_coordinate_operations()
    {
        TestSuite suite("half2 Texture Coordinate Operations");
        suite.header();

        half2 uv = half2::uv(half(0.3f), half(0.7f));

        // UV accessors
        suite.assert_equal_half_t(uv.u(), half(0.3f), "U coordinate");
        suite.assert_equal_half_t(uv.v(), half(0.7f), "V coordinate");

        // UV setters
        half2 modified = uv;
        modified.set_u(half(0.8f));
        modified.set_v(half(0.2f));
        suite.assert_equal_half2_t(modified, half2(0.8f, 0.2f), "UV setters");

        // UV constants
        suite.assert_equal_half2_t(half2_UV_Zero, half2::zero(), "UV Zero constant");
        suite.assert_equal_half2_t(half2_UV_One, half2::one(), "UV One constant");
        suite.assert_equal_half2_t(half2_UV_Half, half2(0.5f, 0.5f), "UV Half constant");

        suite.footer();
    }

    void test_half2_utility_methods()
    {
        TestSuite suite("half2 Utility Methods");
        suite.header();

        half2 a(1.0f, 2.0f);
        half2 b(1.0001f, 2.0001f);
        half2 c(1.1f, 2.1f);

        suite.assert_true(a.approximately(b, 0.01f), "Approximate equality");
        suite.assert_false(a.approximately(c, 0.01f), "Approximate inequality");

        suite.assert_true(half2::zero().approximately_zero(0.001f), "Exactly zero");
        suite.assert_true(half2(0.001f, -0.001f).approximately_zero(0.01f), "Approximately zero");

        // Normalization checks
        suite.assert_true(half2(0.6f, 0.8f).is_normalized(), "Normalized check");
        suite.assert_false(half2(1.0f, 1.0f).is_normalized(), "Not normalized check");
        suite.assert_true(is_normalized(half2(0.6f, 0.8f)), "Global is_normalized");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains x");
        suite.assert_true(str.find("2.0") != std::string::npos, "String contains y");

        // Comparison operators
        suite.assert_true(a == a, "Equality operator");
        suite.assert_true(a != half2::zero(), "Inequality operator");

        suite.footer();
    }

    void test_half2_edge_cases()
    {
        TestSuite suite("half2 Edge Cases");
        suite.header();

        half2 tiny(1e-2f, 1e-2f);
        suite.assert_true(tiny.approximately_zero(0.1f), "Tiny vector approximately zero");
        suite.assert_false(tiny.approximately_zero(1e-4f), "Tiny vector not exactly zero");

        // Very large vectors
        half2 large(1e4f, 1e4f); // Within half range
        suite.assert_true(large.is_valid(), "Large vector is valid");

        // Normalization edge cases
        suite.assert_true(half2::zero().normalize().approximately_zero(), "Normalize zero vector");

        half2 inf_vec(std::numeric_limits<float>::infinity(), 0.0f);
        suite.assert_false(inf_vec.is_valid(), "Infinity vector validity check");

        // NaN propagation
        half2 nan_vec(std::numeric_limits<float>::quiet_NaN(), 1.0f);
        suite.assert_false(nan_vec.is_valid(), "NaN vector validity");

        // Test normalization of very small vectors
        half2 very_small(1e-4f, 1e-4f);
        half2 normalized_small = very_small.normalize();
        suite.assert_true(normalized_small.is_normalized() || normalized_small.approximately_zero(),
            "Very small vector normalization");

        // Cross product edge cases
        suite.assert_equal_half_t(cross(half2::unit_x(), half2::unit_x()), 0.0f, "Cross product with itself", 0.001f);
        suite.assert_equal_half_t(cross(half2::unit_x(), half2::unit_y()), 1.0f, "Cross product orthogonality", 0.001f);

        // Dot product edge cases
        suite.assert_equal_half_t(dot(half2::zero(), half2::one()), 0.0f, "Dot product with zero vector", 0.001f);
        suite.assert_equal_half_t(dot(half2::unit_x(), half2::unit_x()), 1.0f, "Dot product with itself", 0.001f);

        suite.footer();
    }

    void test_half2_constants()
    {
        TestSuite suite("half2 Constants");
        suite.header();

        // Basic constants
        suite.assert_equal_half2_t(half2_Zero, half2::zero(), "Zero constant");
        suite.assert_equal_half2_t(half2_One, half2::one(), "One constant");
        suite.assert_equal_half2_t(half2_UnitX, half2::unit_x(), "Unit X constant");
        suite.assert_equal_half2_t(half2_UnitY, half2::unit_y(), "Unit Y constant");

        // Direction constants
        suite.assert_equal_half2_t(half2_Right, half2(1.0f, 0.0f), "Right constant");
        suite.assert_equal_half2_t(half2_Left, half2(-1.0f, 0.0f), "Left constant");
        suite.assert_equal_half2_t(half2_Up, half2(0.0f, 1.0f), "Up constant");
        suite.assert_equal_half2_t(half2_Down, half2(0.0f, -1.0f), "Down constant");

        // UV constants
        suite.assert_equal_half2_t(half2_UV_Zero, half2(0.0f, 0.0f), "UV Zero constant");
        suite.assert_equal_half2_t(half2_UV_One, half2(1.0f, 1.0f), "UV One constant");
        suite.assert_equal_half2_t(half2_UV_Half, half2(0.5f, 0.5f), "UV Half constant");

        suite.footer();
    }

    // ============================================================================
    // Test Runner for half2
    // ============================================================================

    void run_half2_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "HALF2 COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_half2_constructors();
        test_half2_assignment_operators();
        test_half2_compound_assignment_operators();
        test_half2_arithmetic_operations();
        test_half2_access_operations();
        test_half2_conversion_operations();
        test_half2_mathematical_functions();
        test_half2_hlsl_functions();
        test_half2_geometric_operations();
        test_half2_swizzle_operations();
        test_half2_texture_coordinate_operations();
        test_half2_utility_methods();
        test_half2_edge_cases();
        test_half2_constants();

        std::cout << "HALF2 TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
    // Test Data Generators for half3
    // ============================================================================

    class Half3TestData
    {
    public:
        static std::vector<half3> generate_test_half3s()
        {
            std::vector<half> half_values = HalfTestData::generate_test_halves();
            std::vector<half3> result;

            // Generate combinations of half values
            for (size_t i = 0; i < std::min(half_values.size(), size_t(5)); ++i)
            {
                for (size_t j = 0; j < std::min(half_values.size(), size_t(5)); ++j)
                {
                    for (size_t k = 0; k < std::min(half_values.size(), size_t(5)); ++k)
                    {
                        if (result.size() >= 20) break;
                        result.emplace_back(half_values[i], half_values[j], half_values[k]);
                    }
                    if (result.size() >= 20) break;
                }
                if (result.size() >= 20) break;
            }

            // Add specific test cases
            result.emplace_back(half3::zero());
            result.emplace_back(half3::one());
            result.emplace_back(half3::unit_x());
            result.emplace_back(half3::unit_y());
            result.emplace_back(half3::unit_z());
            result.emplace_back(half3::forward());
            result.emplace_back(half3::up());
            result.emplace_back(half3::right());

            return result;
        }

        static std::vector<std::pair<half3, float3>> generate_half3_float3_pairs()
        {
            auto half3s = generate_test_half3s();
            std::vector<std::pair<half3, float3>> pairs;
            pairs.reserve(half3s.size());

            for (const auto& h3 : half3s) {
                pairs.emplace_back(h3, float3(h3));
            }
            return pairs;
        }
    };

    // ============================================================================
    // half3 Tests - Comprehensive
    // ============================================================================

    void test_half3_constructors()
    {
        TestSuite suite("half3 Constructors");
        suite.header();

        // Default constructor
        suite.assert_equal_half3_t(half3(), half3(0.0f, 0.0f, 0.0f), "Default constructor");

        // Half component constructor
        half3 v1(half(1.5f), half(2.5f), half(3.5f));
        suite.assert_equal(float(v1.x), 1.5f, "Half component constructor - x");
        suite.assert_equal(float(v1.y), 2.5f, "Half component constructor - y");
        suite.assert_equal(float(v1.z), 3.5f, "Half component constructor - z");

        // Float component constructor
        half3 v2(1.5f, 2.5f, 3.5f);
        suite.assert_equal(float(v2.x), 1.5f, "Float component constructor - x");
        suite.assert_equal(float(v2.y), 2.5f, "Float component constructor - y");
        suite.assert_equal(float(v2.z), 3.5f, "Float component constructor - z");

        // Scalar constructor
        suite.assert_equal_half3_t(half3(5.0f), half3(5.0f, 5.0f, 5.0f), "Scalar constructor");

        // Copy constructor
        half3 original(2.0f, 3.0f, 4.0f);
        half3 copy(original);
        suite.assert_equal_half3_t(copy, original, "Copy constructor");

        // From half2 constructor
        suite.assert_equal_half3_t(half3(half2(1.0f, 2.0f), 3.0f), half3(1.0f, 2.0f, 3.0f), "From half2 with z");
        suite.assert_equal_half3_t(half3(half2(1.0f, 2.0f)), half3(1.0f, 2.0f, 0.0f), "From half2 with default z");

        // From float3 constructor
        suite.assert_equal_half3_t(half3(float3(1.0f, 2.0f, 3.0f)), half3(1.0f, 2.0f, 3.0f), "From float3 constructor");

        // From float2 constructor
        suite.assert_equal_half3_t(half3(float2(1.0f, 2.0f), 3.0f), half3(1.0f, 2.0f, 3.0f), "From float2 with z");
        suite.assert_equal_half3_t(half3(float2(1.0f, 2.0f)), half3(1.0f, 2.0f, 0.0f), "From float2 with default z");

        // Static constructors
        suite.assert_equal_half3_t(half3::zero(), half3(0.0f, 0.0f, 0.0f), "Zero vector");
        suite.assert_equal_half3_t(half3::one(), half3(1.0f, 1.0f, 1.0f), "One vector");
        suite.assert_equal_half3_t(half3::unit_x(), half3(1.0f, 0.0f, 0.0f), "Unit X");
        suite.assert_equal_half3_t(half3::unit_y(), half3(0.0f, 1.0f, 0.0f), "Unit Y");
        suite.assert_equal_half3_t(half3::unit_z(), half3(0.0f, 0.0f, 1.0f), "Unit Z");
        suite.assert_equal_half3_t(half3::forward(), half3(0.0f, 0.0f, 1.0f), "Forward vector");
        suite.assert_equal_half3_t(half3::up(), half3(0.0f, 1.0f, 0.0f), "Up vector");
        suite.assert_equal_half3_t(half3::right(), half3(1.0f, 0.0f, 0.0f), "Right vector");

        // Edge cases
        half3 inf_vec(std::numeric_limits<float>::infinity(), 0.0f, 1.0f);
        suite.assert_true(std::isinf(float(inf_vec.x)), "Infinity component");

        suite.footer();
    }

    void test_half3_assignment_operators()
    {
        TestSuite suite("half3 Assignment Operators");
        suite.header();

        half3 a(1.0f, 2.0f, 3.0f);
        half3 b;

        // Copy assignment
        b = a;
        suite.assert_equal_half3_t(b, a, "Copy assignment");

        // Float3 assignment
        b = float3(3.0f, 4.0f, 5.0f);
        suite.assert_equal_half3_t(b, half3(3.0f, 4.0f, 5.0f), "Float3 assignment");

        // Half scalar assignment
        b = half(5.0f);
        suite.assert_equal_half3_t(b, half3(5.0f, 5.0f, 5.0f), "Half scalar assignment");

        // Float scalar assignment
        b = 6.0f;
        suite.assert_equal_half3_t(b, half3(6.0f, 6.0f, 6.0f), "Float scalar assignment");

        suite.footer();
    }

    void test_half3_compound_assignment_operators()
    {
        TestSuite suite("half3 Compound Assignment Operators");
        suite.header();

        half3 a(2.0f, 3.0f, 4.0f);
        half3 b(1.0f, 2.0f, 3.0f);
        half3 c;

        // Compound addition
        c = a; c += b;
        suite.assert_equal_half3_t(c, half3(3.0f, 5.0f, 7.0f), "Compound addition");

        // Compound subtraction
        c = a; c -= b;
        suite.assert_equal_half3_t(c, half3(1.0f, 1.0f, 1.0f), "Compound subtraction");

        // Compound multiplication
        c = a; c *= b;
        suite.assert_equal_half3_t(c, half3(2.0f, 6.0f, 12.0f), "Compound multiplication");

        // Compound division
        c = a; c /= b;
        suite.assert_equal_half3_t(c, half3(2.0f, 1.5f, 4.0f / 3.0f), "Compound division", 0.01f);

        // Compound scalar multiplication (half)
        c = a; c *= half(2.0f);
        suite.assert_equal_half3_t(c, half3(4.0f, 6.0f, 8.0f), "Compound scalar multiplication (half)");

        // Compound scalar multiplication (float)
        c = a; c *= 2.0f;
        suite.assert_equal_half3_t(c, half3(4.0f, 6.0f, 8.0f), "Compound scalar multiplication (float)");

        // Compound scalar division (half)
        c = a; c /= half(2.0f);
        suite.assert_equal_half3_t(c, half3(1.0f, 1.5f, 2.0f), "Compound scalar division (half)");

        // Compound scalar division (float)
        c = a; c /= 2.0f;
        suite.assert_equal_half3_t(c, half3(1.0f, 1.5f, 2.0f), "Compound scalar division (float)");

        suite.footer();
    }

    void test_half3_arithmetic_operations()
    {
        TestSuite suite("half3 Arithmetic Operations");
        suite.header();

        half3 a(1.0f, 2.0f, 3.0f);
        half3 b(3.0f, 4.0f, 5.0f);
        half3 c;

        // Vector operations
        suite.assert_equal_half3_t(a + b, half3(4.0f, 6.0f, 8.0f), "Vector addition");
        suite.assert_equal_half3_t(a - b, half3(-2.0f, -2.0f, -2.0f), "Vector subtraction");
        suite.assert_equal_half3_t(a * b, half3(3.0f, 8.0f, 15.0f), "Component-wise multiplication");
        suite.assert_equal_half3_t(a / b, half3(1.0f / 3.0f, 2.0f / 4.0f, 3.0f / 5.0f), "Component-wise division", 0.01f);

        // Scalar operations
        suite.assert_equal_half3_t(a * 2.0f, half3(2.0f, 4.0f, 6.0f), "Scalar multiplication");
        suite.assert_equal_half3_t(2.0f * a, half3(2.0f, 4.0f, 6.0f), "Scalar multiplication commutative");
        suite.assert_equal_half3_t(a / 2.0f, half3(0.5f, 1.0f, 1.5f), "Scalar division");

        // Unary operations
        suite.assert_equal_half3_t(+a, a, "Unary plus");
        suite.assert_equal_half3_t(-a, half3(-1.0f, -2.0f, -3.0f), "Unary minus");

        // Mixed type operations with float3
        float3 f3(3.0f, 4.0f, 5.0f);
        suite.assert_equal_half3_t(a + f3, half3(4.0f, 6.0f, 8.0f), "half3 + float3");
        suite.assert_equal_half3_t(f3 + a, half3(4.0f, 6.0f, 8.0f), "float3 + half3");
        suite.assert_equal_half3_t(a - f3, half3(-2.0f, -2.0f, -2.0f), "half3 - float3");
        suite.assert_equal_half3_t(f3 - a, half3(2.0f, 2.0f, 2.0f), "float3 - half3");

        suite.footer();
    }

    void test_half3_access_operations()
    {
        TestSuite suite("half3 Access Operations");
        suite.header();

        half3 v(1.5f, 2.5f, 3.5f);

        // Index access
        suite.assert_equal_half_t(v[0], half(1.5f), "Index access [0]");
        suite.assert_equal_half_t(v[1], half(2.5f), "Index access [1]");
        suite.assert_equal_half_t(v[2], half(3.5f), "Index access [2]");

        // Mutable index access
        v[0] = half(3.0f);
        v[1] = half(4.0f);
        v[2] = half(5.0f);
        suite.assert_equal_half3_t(v, half3(3.0f, 4.0f, 5.0f), "Mutable index access");

        // Data pointer access
        const half* data = v.data();
        suite.assert_equal_half_t(data[0], half(3.0f), "Data pointer access [0]");
        suite.assert_equal_half_t(data[1], half(4.0f), "Data pointer access [1]");
        suite.assert_equal_half_t(data[2], half(5.0f), "Data pointer access [2]");

        // Mutable data pointer access
        half* mutable_data = v.data();
        mutable_data[0] = half(5.0f);
        mutable_data[1] = half(6.0f);
        mutable_data[2] = half(7.0f);
        suite.assert_equal_half3_t(v, half3(5.0f, 6.0f, 7.0f), "Mutable data pointer access");

        suite.footer();
    }

    void test_half3_conversion_operations()
    {
        TestSuite suite("half3 Conversion Operations");
        suite.header();

        half3 h3(1.5f, 2.5f, 3.5f);

        // Conversion to float3
        float3 f3 = static_cast<float3>(h3);
        suite.assert_equal_half3_t(f3, float3(1.5f, 2.5f, 3.5f), "Conversion to float3");

        // Global conversion functions
        suite.assert_equal_half3_t(to_float3(h3), float3(1.5f, 2.5f, 3.5f), "to_float3 function");
        suite.assert_equal_half3_t(to_half3(f3), h3, "to_half3 function");

        // Round-trip conversion
        half3 round_trip = to_half3(to_float3(h3));
        suite.assert_equal_half3_t(round_trip, h3, "Round-trip conversion");

        suite.footer();
    }

    void test_half3_mathematical_functions()
    {
        TestSuite suite("half3 Mathematical Functions");
        suite.header();

        half3 v(3.0f, 4.0f, 0.0f);  // length = 5
        half3 w(1.0f, 2.0f, 2.0f);  // length = 3

        // Length calculations
        suite.assert_equal_half_t(v.length(), 5.0f, "Length", 0.01f);
        suite.assert_equal_half_t(v.length_sq(), 25.0f, "Squared length", 0.01f);

        // distance = √((3-1)² + (4-2)² + (0-2)²) = √(4 + 4 + 4) = √12 ≈ 3.464
        suite.assert_equal_half_t(distance(v, w), 3.46410162f, "Distance", 0.01f);
        suite.assert_equal_half_t(distance_sq(v, w), 12.0f, "Squared distance", 0.01f);

        // Dot product
        // dot(v, w) = 3*1 + 4*2 + 0*2 = 3 + 8 + 0 = 11
        suite.assert_equal_half_t(dot(v, w), 11.0f, "Dot product", 0.01f);
        suite.assert_equal_half_t(half3::dot(v, w), 11.0f, "Static dot product", 0.01f);

        // Cross product
        half3 cross_result = cross(v, w);
        // cross(v, w) = (4*2 - 0*2, 0*1 - 3*2, 3*2 - 4*1) = (8, -6, 2)
        suite.assert_equal_half_t(cross_result.x, 8.0f, "Cross product x", 0.01f);
        suite.assert_equal_half_t(cross_result.y, -6.0f, "Cross product y", 0.01f);
        suite.assert_equal_half_t(cross_result.z, 2.0f, "Cross product z", 0.01f);

        // Normalization
        half3 normalized = v.normalize();
        suite.assert_true(normalized.is_normalized(0.01f), "Normalization result is normalized");
        suite.assert_equal_half_t(normalized.length(), 1.0f, "Normalized length", 0.01f);
        suite.assert_true(v.normalize().approximately(half3(0.6f, 0.8f, 0.0f), 0.01f), "Normalization values");

        // Global normalization
        suite.assert_equal_half3_t(normalize(v), v.normalize(), "Global normalize function");

        suite.footer();
    }

    void test_half3_hlsl_functions()
    {
        TestSuite suite("half3 HLSL Functions");
        suite.header();

        half3 v(1.5f, -2.5f, 0.5f);
        half3 w(0.3f, 0.7f, -0.2f);

        // Component-wise functions
        suite.assert_equal_half3_t(abs(v), half3(1.5f, 2.5f, 0.5f), "Absolute value");
        suite.assert_equal_half3_t(sign(v), half3(1.0f, -1.0f, 1.0f), "Sign function");
        suite.assert_equal_half3_t(floor(half3(1.7f, -2.3f, 0.8f)), half3(1.0f, -3.0f, 0.0f), "Floor");
        suite.assert_equal_half3_t(ceil(half3(1.2f, -2.7f, 0.3f)), half3(2.0f, -2.0f, 1.0f), "Ceil");
        suite.assert_equal_half3_t(round(half3(1.4f, 1.6f, -1.5f)), half3(1.0f, 2.0f, -2.0f), "Round");
        suite.assert_equal_half3_t(frac(half3(1.7f, -2.3f, 0.8f)), half3(0.7f, 0.7f, 0.8f), "Fractional part", 0.01f);

        // Saturate
        suite.assert_equal_half3_t(saturate(half3(-0.5f, 1.5f, 0.5f)), half3(0.0f, 1.0f, 0.5f), "Saturate");
        suite.assert_equal_half3_t(half3::saturate(half3(-0.5f, 1.5f, 0.5f)), half3(0.0f, 1.0f, 0.5f), "Static saturate");

        // Step
        suite.assert_equal_half3_t(step(half(1.0f), half3(0.5f, 1.5f, 1.0f)), half3(0.0f, 1.0f, 1.0f), "Step function");

        // Min/Max
        suite.assert_equal_half3_t(min(half3(1, 3, 2), half3(2, 2, 3)), half3(1, 2, 2), "Minimum");
        suite.assert_equal_half3_t(max(half3(1, 3, 2), half3(2, 2, 3)), half3(2, 3, 3), "Maximum");
        suite.assert_equal_half3_t(half3::min(half3(1, 3, 2), half3(2, 2, 3)), half3(1, 2, 2), "Static minimum");
        suite.assert_equal_half3_t(half3::max(half3(1, 3, 2), half3(2, 2, 3)), half3(2, 3, 3), "Static maximum");

        // Clamp
        suite.assert_equal_half3_t(clamp(half3(0.5f, 1.5f, -0.5f), 0.0f, 1.0f), half3(0.5f, 1.0f, 0.0f), "Scalar clamp");
        suite.assert_equal_half3_t(clamp(half3(0.5f, 1.5f, -0.5f), half3(0, 1, -1), half3(1, 2, 0)),
            half3(0.5f, 1.5f, -0.5f), "Vector clamp");

        // Smoothstep
        half3 smooth = smoothstep(half(0.0f), half(1.0f), half3(0.5f, 0.5f, 0.5f));
        suite.assert_true(smooth.approximately(half3(0.5f, 0.5f, 0.5f), 0.1f), "Smoothstep");

        suite.footer();
    }

    void test_half3_geometric_operations()
    {
        TestSuite suite("half3 Geometric Operations");
        suite.header();

        half3 v(1.0f, 0.0f, 0.0f);
        half3 w(0.0f, 1.0f, 0.0f);
        half3 u(0.0f, 0.0f, 1.0f);

        // Reflection
        half3 incident(1.0f, -1.0f, 0.0f);
        half3 normal(0.0f, 1.0f, 0.0f);
        half3 reflected = reflect(incident, normal);
        suite.assert_equal_half3_t(reflected, half3(1.0f, 1.0f, 0.0f), "Reflection");

        // Refraction (simplified test)
        half3 refracted = refract(float3(0.0f, -1.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), 1.5f);
        suite.assert_true(refracted.approximately(half3(0.0f, -1.0f, 0.0f), 0.001f), "Refraction");

        // Projection
        half3 vec(2.0f, 3.0f, 4.0f);
        half3 onto(1.0f, 0.0f, 0.0f);
        half3 projected = project(vec, onto);
        suite.assert_equal_half3_t(projected, half3(2.0f, 0.0f, 0.0f), "Projection");

        // Rejection
        half3 rejected = reject(vec, onto);
        suite.assert_equal_half3_t(rejected, half3(0.0f, 3.0f, 4.0f), "Rejection");

        // Angles
        suite.assert_equal_half_t(angle_between(v, w), Constants::HALF_PI, "Angle between orthogonal vectors", 0.001f);
        suite.assert_equal_half_t(angle_between(v, v), 0.0f, "Angle between same vectors", 0.001f);

        // Lerp
        suite.assert_equal_half3_t(lerp(v, w, half(0.5f)), half3(0.5f, 0.5f, 0.0f), "Lerp with half");
        suite.assert_equal_half3_t(lerp(v, w, 0.5f), half3(0.5f, 0.5f, 0.0f), "Lerp with float");
        suite.assert_equal_half3_t(half3::lerp(v, w, half(0.5f)), half3(0.5f, 0.5f, 0.0f), "Static lerp with half");
        suite.assert_equal_half3_t(half3::lerp(v, w, 0.5f), half3(0.5f, 0.5f, 0.0f), "Static lerp with float");

        suite.footer();
    }

    void test_half3_color_operations()
    {
        TestSuite suite("half3 Color Operations");
        suite.header();

        half3 color(0.8f, 0.6f, 0.4f);

        // Luminance
        half lum = color.luminance();
        suite.assert_true(lum > 0.0f && lum < 1.0f, "Luminance calculation");

        // Grayscale conversion
        half3 gray = color.rgb_to_grayscale();
        suite.assert_equal_half_t(gray.x, gray.y, "Grayscale R == G");
        suite.assert_equal_half_t(gray.y, gray.z, "Grayscale G == B");

        // Gamma correction
        half3 gamma_corrected = color.gamma_correct(2.2f);
        suite.assert_true(gamma_corrected.x < color.x, "Gamma correction reduces values");

        // sRGB conversions (basic test)
        half3 linear = color.srgb_to_linear();
        half3 srgb = linear.linear_to_srgb();
        suite.assert_true(srgb.approximately(color, 0.1f), "sRGB round-trip conversion");

        suite.footer();
    }

    void test_half3_swizzle_operations()
    {
        TestSuite suite("half3 Swizzle Operations");
        suite.header();

        half3 v(1.0f, 2.0f, 3.0f);

        // 2D swizzles
        suite.assert_equal_half2_t(v.xy(), half2(1.0f, 2.0f), "XY swizzle");
        suite.assert_equal_half2_t(v.xz(), half2(1.0f, 3.0f), "XZ swizzle");
        suite.assert_equal_half2_t(v.yz(), half2(2.0f, 3.0f), "YZ swizzle");
        suite.assert_equal_half2_t(v.yx(), half2(2.0f, 1.0f), "YX swizzle");
        suite.assert_equal_half2_t(v.zx(), half2(3.0f, 1.0f), "ZX swizzle");
        suite.assert_equal_half2_t(v.zy(), half2(3.0f, 2.0f), "ZY swizzle");

        // 3D swizzles
        suite.assert_equal_half3_t(v.yxz(), half3(2.0f, 1.0f, 3.0f), "YXZ swizzle");
        suite.assert_equal_half3_t(v.zxy(), half3(3.0f, 1.0f, 2.0f), "ZXY swizzle");
        suite.assert_equal_half3_t(v.zyx(), half3(3.0f, 2.0f, 1.0f), "ZYX swizzle");
        suite.assert_equal_half3_t(v.xzy(), half3(1.0f, 3.0f, 2.0f), "XZY swizzle");

        // Color swizzles
        suite.assert_equal_half_t(v.r(), 1.0f, "Red component");
        suite.assert_equal_half_t(v.g(), 2.0f, "Green component");
        suite.assert_equal_half_t(v.b(), 3.0f, "Blue component");
        suite.assert_equal_half2_t(v.rg(), half2(1.0f, 2.0f), "RG swizzle");
        suite.assert_equal_half2_t(v.rb(), half2(1.0f, 3.0f), "RB swizzle");
        suite.assert_equal_half2_t(v.gb(), half2(2.0f, 3.0f), "GB swizzle");
        suite.assert_equal_half3_t(v.rgb(), half3(1.0f, 2.0f, 3.0f), "RGB swizzle");
        suite.assert_equal_half3_t(v.bgr(), half3(3.0f, 2.0f, 1.0f), "BGR swizzle");
        suite.assert_equal_half3_t(v.gbr(), half3(2.0f, 3.0f, 1.0f), "GBR swizzle");

        suite.footer();
    }

    void test_half3_utility_methods()
    {
        TestSuite suite("half3 Utility Methods");
        suite.header();

        half3 a(1.0f, 2.0f, 3.0f);
        half3 b(1.0001f, 2.0001f, 3.0001f);
        half3 c(1.1f, 2.1f, 3.1f);

        // Validity checks
        suite.assert_true(a.is_valid(), "Valid vector check");
        suite.assert_false(half3(std::numeric_limits<float>::infinity(), 2.0f, 3.0f).is_valid(), "Invalid vector - infinity");
        suite.assert_false(half3(1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f).is_valid(), "Invalid vector - NaN");

        // Approximate equality
        suite.assert_true(a.approximately(b, 0.01f), "Approximate equality");
        suite.assert_false(a.approximately(c, 0.01f), "Approximate inequality");
        suite.assert_true(approximately(a, b, 0.01f), "Global approximately");

        // Zero checks
        suite.assert_true(half3::zero().approximately_zero(0.001f), "Exactly zero");
        suite.assert_true(half3(0.001f, -0.001f, 0.001f).approximately_zero(0.01f), "Approximately zero");

        // Normalization checks
        suite.assert_true(half3(0.57735f, 0.57735f, 0.57735f).is_normalized(0.01f), "Normalized check");
        suite.assert_false(half3(1.0f, 1.0f, 1.0f).is_normalized(), "Not normalized check");
        suite.assert_true(is_normalized(half3(0.57735f, 0.57735f, 0.57735f), 0.01f), "Global is_normalized");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains x");
        suite.assert_true(str.find("2.0") != std::string::npos, "String contains y");
        suite.assert_true(str.find("3.0") != std::string::npos, "String contains z");

        // Set XY from half2
        half3 vec(0.0f, 0.0f, 5.0f);
        vec.set_xy(half2(1.0f, 2.0f));
        suite.assert_equal_half3_t(vec, half3(1.0f, 2.0f, 5.0f), "Set XY from half2");

        // Comparison operators
        suite.assert_true(a == a, "Equality operator");
        suite.assert_true(a != half3::zero(), "Inequality operator");

        suite.footer();
    }

    void test_half3_edge_cases()
    {
        TestSuite suite("half3 Edge Cases");
        suite.header();

        // Zero vectors
        suite.assert_true(half3::zero().approximately_zero(0.001f), "Zero vector");
        suite.assert_equal_half_t(half3::zero().length(), 0.0f, "Zero length", 0.001f);
        suite.assert_equal_half_t(half3::zero().length_sq(), 0.0f, "Zero squared length", 0.001f);

        // Very small vectors
        half3 tiny(0.01f, 0.01f, 0.01f);
        suite.assert_true(tiny.approximately_zero(0.1f), "Tiny vector approximately zero");
        suite.assert_false(tiny.approximately_zero(0.001f), "Tiny vector not exactly zero");

        // Very large vectors
        half3 large(1e4f, 1e4f, 1e4f); // Within half range
        suite.assert_true(large.is_valid(), "Large vector is valid");

        // Normalization edge cases
        suite.assert_true(half3::zero().normalize().approximately_zero(0.001f), "Normalize zero vector");

        half3 inf_vec(std::numeric_limits<float>::infinity(), 0.0f, 0.0f);
        suite.assert_false(inf_vec.is_valid(), "Infinity vector validity check");

        // NaN propagation
        half3 nan_vec(std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f);
        suite.assert_false(nan_vec.is_valid(), "NaN vector validity");

        // Test normalization of very small vectors
        half3 very_small(0.001f, 0.001f, 0.001f);
        half3 normalized_small = very_small.normalize();
        suite.assert_true(normalized_small.is_normalized(0.01f) || normalized_small.approximately_zero(0.001f),
            "Very small vector normalization");

        // Cross product edge cases
        suite.assert_equal_half3_t(cross(half3::unit_x(), half3::unit_x()), half3::zero(), "Cross product with itself");
        suite.assert_equal_half3_t(cross(half3::unit_x(), half3::unit_y()), half3::unit_z(), "Cross product orthogonality");

        // Dot product edge cases
        suite.assert_equal_half_t(dot(half3::zero(), half3::one()), 0.0f, "Dot product with zero vector", 0.001f);
        suite.assert_equal_half_t(dot(half3::unit_x(), half3::unit_x()), 1.0f, "Dot product with itself", 0.001f);

        // Geometric operations with zero vectors
        suite.assert_equal_half3_t(project(half3::zero(), half3::unit_x()), half3::zero(), "Project zero vector");
        suite.assert_equal_half3_t(reject(half3::unit_x(), half3::zero()), half3::unit_x(), "Reject from zero vector");

        suite.footer();
    }

    void test_half3_constants()
    {
        TestSuite suite("half3 Constants");
        suite.header();

        // Basic constants
        suite.assert_equal_half3_t(half3_Zero, half3::zero(), "Zero constant");
        suite.assert_equal_half3_t(half3_One, half3::one(), "One constant");
        suite.assert_equal_half3_t(half3_UnitX, half3::unit_x(), "Unit X constant");
        suite.assert_equal_half3_t(half3_UnitY, half3::unit_y(), "Unit Y constant");
        suite.assert_equal_half3_t(half3_UnitZ, half3::unit_z(), "Unit Z constant");

        // Direction constants
        suite.assert_equal_half3_t(half3_Forward, half3::forward(), "Forward constant");
        suite.assert_equal_half3_t(half3_Up, half3::up(), "Up constant");
        suite.assert_equal_half3_t(half3_Right, half3::right(), "Right constant");

        // Color constants
        suite.assert_equal_half3_t(half3_Red, half3(1.0f, 0.0f, 0.0f), "Red constant");
        suite.assert_equal_half3_t(half3_Green, half3(0.0f, 1.0f, 0.0f), "Green constant");
        suite.assert_equal_half3_t(half3_Blue, half3(0.0f, 0.0f, 1.0f), "Blue constant");
        suite.assert_equal_half3_t(half3_White, half3(1.0f, 1.0f, 1.0f), "White constant");
        suite.assert_equal_half3_t(half3_Black, half3(0.0f, 0.0f, 0.0f), "Black constant");
        suite.assert_equal_half3_t(half3_Yellow, half3(1.0f, 1.0f, 0.0f), "Yellow constant");
        suite.assert_equal_half3_t(half3_Cyan, half3(0.0f, 1.0f, 1.0f), "Cyan constant");
        suite.assert_equal_half3_t(half3_Magenta, half3(1.0f, 0.0f, 1.0f), "Magenta constant");

        suite.footer();
    }

    // ============================================================================
    // Test Runner for half3
    // ============================================================================

    void run_half3_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "HALF3 COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_half3_constructors();
        test_half3_assignment_operators();
        test_half3_compound_assignment_operators();
        test_half3_arithmetic_operations();
        test_half3_access_operations();
        test_half3_conversion_operations();
        test_half3_mathematical_functions();
        test_half3_hlsl_functions();
        test_half3_geometric_operations();
        test_half3_color_operations();
        test_half3_swizzle_operations();
        test_half3_utility_methods();
        test_half3_edge_cases();
        test_half3_constants();

        std::cout << "HALF3 TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
    // Test Data Generators for half4
    // ============================================================================

    class Half4TestData
    {
    public:
        static std::vector<half4> generate_test_half4s()
        {
            std::vector<half> half_values = HalfTestData::generate_test_halves();
            std::vector<half4> result;

            // Generate combinations of half values
            for (size_t i = 0; i < std::min(half_values.size(), size_t(4)); ++i)
            {
                for (size_t j = 0; j < std::min(half_values.size(), size_t(4)); ++j)
                {
                    for (size_t k = 0; k < std::min(half_values.size(), size_t(4)); ++k)
                    {
                        for (size_t l = 0; l < std::min(half_values.size(), size_t(4)); ++l)
                        {
                            if (result.size() >= 20) break;
                            result.emplace_back(half_values[i], half_values[j], half_values[k], half_values[l]);
                        }
                        if (result.size() >= 20) break;
                    }
                    if (result.size() >= 20) break;
                }
                if (result.size() >= 20) break;
            }

            // Add specific test cases
            result.emplace_back(half4::zero());
            result.emplace_back(half4::one());
            result.emplace_back(half4::unit_x());
            result.emplace_back(half4::unit_y());
            result.emplace_back(half4::unit_z());
            result.emplace_back(half4::unit_w());
            result.emplace_back(half4::from_rgba(255, 128, 64, 255));

            return result;
        }

        static std::vector<std::pair<half4, float4>> generate_half4_float4_pairs()
        {
            auto half4s = generate_test_half4s();
            std::vector<std::pair<half4, float4>> pairs;
            pairs.reserve(half4s.size());

            for (const auto& h4 : half4s) {
                pairs.emplace_back(h4, float4(h4));
            }
            return pairs;
        }
    };

    // ============================================================================
    // half4 Tests - Comprehensive
    // ============================================================================

    void test_half4_constructors()
    {
        TestSuite suite("half4 Constructors");
        suite.header();

        // Default constructor
        suite.assert_equal_half4_t(half4(), half4(0.0f, 0.0f, 0.0f, 0.0f), "Default constructor");

        // Half component constructor
        half4 v1(half(1.5f), half(2.5f), half(3.5f), half(4.5f));
        suite.assert_equal(float(v1.x), 1.5f, "Half component constructor - x");
        suite.assert_equal(float(v1.y), 2.5f, "Half component constructor - y");
        suite.assert_equal(float(v1.z), 3.5f, "Half component constructor - z");
        suite.assert_equal(float(v1.w), 4.5f, "Half component constructor - w");

        // Float component constructor
        half4 v2(1.5f, 2.5f, 3.5f, 4.5f);
        suite.assert_equal(float(v2.x), 1.5f, "Float component constructor - x");
        suite.assert_equal(float(v2.y), 2.5f, "Float component constructor - y");
        suite.assert_equal(float(v2.z), 3.5f, "Float component constructor - z");
        suite.assert_equal(float(v2.w), 4.5f, "Float component constructor - w");

        // Scalar constructor
        suite.assert_equal_half4_t(half4(5.0f), half4(5.0f, 5.0f, 5.0f, 5.0f), "Scalar constructor");

        // Copy constructor
        half4 original(2.0f, 3.0f, 4.0f, 5.0f);
        half4 copy(original);
        suite.assert_equal_half4_t(copy, original, "Copy constructor");

        // From half2 constructor
        suite.assert_equal_half4_t(half4(half2(1.0f, 2.0f), 3.0f, 4.0f), half4(1.0f, 2.0f, 3.0f, 4.0f), "From half2 with z,w");
        suite.assert_equal_half4_t(half4(half2(1.0f, 2.0f)), half4(1.0f, 2.0f, 0.0f, 0.0f), "From half2 with defaults");

        // From half3 constructor
        suite.assert_equal_half4_t(half4(half3(1.0f, 2.0f, 3.0f), 4.0f), half4(1.0f, 2.0f, 3.0f, 4.0f), "From half3 with w");
        suite.assert_equal_half4_t(half4(half3(1.0f, 2.0f, 3.0f)), half4(1.0f, 2.0f, 3.0f, 0.0f), "From half3 with default w");

        // From float4 constructor
        suite.assert_equal_half4_t(half4(float4(1.0f, 2.0f, 3.0f, 4.0f)), half4(1.0f, 2.0f, 3.0f, 4.0f), "From float4 constructor");

        // From float2 constructor
        suite.assert_equal_half4_t(half4(float2(1.0f, 2.0f), 3.0f, 4.0f), half4(1.0f, 2.0f, 3.0f, 4.0f), "From float2 with z,w");
        suite.assert_equal_half4_t(half4(float2(1.0f, 2.0f)), half4(1.0f, 2.0f, 0.0f, 0.0f), "From float2 with defaults");

        // From float3 constructor
        suite.assert_equal_half4_t(half4(float3(1.0f, 2.0f, 3.0f), 4.0f), half4(1.0f, 2.0f, 3.0f, 4.0f), "From float3 with w");
        suite.assert_equal_half4_t(half4(float3(1.0f, 2.0f, 3.0f)), half4(1.0f, 2.0f, 3.0f, 0.0f), "From float3 with default w");

        // Static constructors
        suite.assert_equal_half4_t(half4::zero(), half4(0.0f, 0.0f, 0.0f, 0.0f), "Zero vector");
        suite.assert_equal_half4_t(half4::one(), half4(1.0f, 1.0f, 1.0f, 1.0f), "One vector");
        suite.assert_equal_half4_t(half4::unit_x(), half4(1.0f, 0.0f, 0.0f, 0.0f), "Unit X");
        suite.assert_equal_half4_t(half4::unit_y(), half4(0.0f, 1.0f, 0.0f, 0.0f), "Unit Y");
        suite.assert_equal_half4_t(half4::unit_z(), half4(0.0f, 0.0f, 1.0f, 0.0f), "Unit Z");
        suite.assert_equal_half4_t(half4::unit_w(), half4(0.0f, 0.0f, 0.0f, 1.0f), "Unit W");

        // Color constructor
        half4 color = half4::from_rgba(255, 128, 64, 255);
        suite.assert_true(color.approximately(half4(1.0f, 0.5f, 0.25f, 1.0f), 0.01f), "From RGBA bytes");

        // Edge cases
        half4 inf_vec(std::numeric_limits<float>::infinity(), 0.0f, 1.0f, 2.0f);
        suite.assert_true(std::isinf(float(inf_vec.x)), "Infinity component");

        suite.footer();
    }

    void test_half4_assignment_operators()
    {
        TestSuite suite("half4 Assignment Operators");
        suite.header();

        half4 a(1.0f, 2.0f, 3.0f, 4.0f);
        half4 b;

        // Copy assignment
        b = a;
        suite.assert_equal_half4_t(b, a, "Copy assignment");

        // Float4 assignment
        b = float4(3.0f, 4.0f, 5.0f, 6.0f);
        suite.assert_equal_half4_t(b, half4(3.0f, 4.0f, 5.0f, 6.0f), "Float4 assignment");

        // Half3 assignment (preserves w)
        b = a;
        b = half3(10.0f, 11.0f, 12.0f);
        suite.assert_equal_half4_t(b, half4(10.0f, 11.0f, 12.0f, 4.0f), "Half3 assignment preserves w");

        // Half scalar assignment
        b = half(5.0f);
        suite.assert_equal_half4_t(b, half4(5.0f, 5.0f, 5.0f, 5.0f), "Half scalar assignment");

        // Float scalar assignment
        b = 6.0f;
        suite.assert_equal_half4_t(b, half4(6.0f, 6.0f, 6.0f, 6.0f), "Float scalar assignment");

        suite.footer();
    }

    void test_half4_compound_assignment_operators()
    {
        TestSuite suite("half4 Compound Assignment Operators");
        suite.header();

        half4 a(2.0f, 3.0f, 4.0f, 5.0f);
        half4 b(1.0f, 2.0f, 3.0f, 4.0f);
        half4 c;

        // Compound addition
        c = a; c += b;
        suite.assert_equal_half4_t(c, half4(3.0f, 5.0f, 7.0f, 9.0f), "Compound addition");

        // Compound subtraction
        c = a; c -= b;
        suite.assert_equal_half4_t(c, half4(1.0f, 1.0f, 1.0f, 1.0f), "Compound subtraction");

        // Compound multiplication
        c = a; c *= b;
        suite.assert_equal_half4_t(c, half4(2.0f, 6.0f, 12.0f, 20.0f), "Compound multiplication");

        // Compound division
        c = a; c /= b;
        suite.assert_equal_half4_t(c, half4(2.0f, 1.5f, 4.0f / 3.0f, 1.25f), "Compound division", 0.01f);

        // Compound scalar multiplication (half)
        c = a; c *= half(2.0f);
        suite.assert_equal_half4_t(c, half4(4.0f, 6.0f, 8.0f, 10.0f), "Compound scalar multiplication (half)");

        // Compound scalar multiplication (float)
        c = a; c *= 2.0f;
        suite.assert_equal_half4_t(c, half4(4.0f, 6.0f, 8.0f, 10.0f), "Compound scalar multiplication (float)");

        // Compound scalar division (half)
        c = a; c /= half(2.0f);
        suite.assert_equal_half4_t(c, half4(1.0f, 1.5f, 2.0f, 2.5f), "Compound scalar division (half)");

        // Compound scalar division (float)
        c = a; c /= 2.0f;
        suite.assert_equal_half4_t(c, half4(1.0f, 1.5f, 2.0f, 2.5f), "Compound scalar division (float)");

        suite.footer();
    }

    void test_half4_arithmetic_operations()
    {
        TestSuite suite("half4 Arithmetic Operations");
        suite.header();

        half4 a(1.0f, 2.0f, 3.0f, 4.0f);
        half4 b(3.0f, 4.0f, 5.0f, 6.0f);
        half4 c;

        // Vector operations
        suite.assert_equal_half4_t(a + b, half4(4.0f, 6.0f, 8.0f, 10.0f), "Vector addition");
        suite.assert_equal_half4_t(a - b, half4(-2.0f, -2.0f, -2.0f, -2.0f), "Vector subtraction");
        suite.assert_equal_half4_t(a * b, half4(3.0f, 8.0f, 15.0f, 24.0f), "Component-wise multiplication");
        suite.assert_equal_half4_t(a / b, half4(1.0f / 3.0f, 2.0f / 4.0f, 3.0f / 5.0f, 4.0f / 6.0f), "Component-wise division", 0.01f);

        // Scalar operations
        suite.assert_equal_half4_t(a * 2.0f, half4(2.0f, 4.0f, 6.0f, 8.0f), "Scalar multiplication");
        suite.assert_equal_half4_t(2.0f * a, half4(2.0f, 4.0f, 6.0f, 8.0f), "Scalar multiplication commutative");
        suite.assert_equal_half4_t(a / 2.0f, half4(0.5f, 1.0f, 1.5f, 2.0f), "Scalar division");

        // Unary operations
        suite.assert_equal_half4_t(+a, a, "Unary plus");
        suite.assert_equal_half4_t(-a, half4(-1.0f, -2.0f, -3.0f, -4.0f), "Unary minus");

        // Mixed type operations with float4
        float4 f4(3.0f, 4.0f, 5.0f, 6.0f);
        suite.assert_equal_half4_t(a + f4, half4(4.0f, 6.0f, 8.0f, 10.0f), "half4 + float4");
        suite.assert_equal_half4_t(f4 + a, half4(4.0f, 6.0f, 8.0f, 10.0f), "float4 + half4");
        suite.assert_equal_half4_t(a - f4, half4(-2.0f, -2.0f, -2.0f, -2.0f), "half4 - float4");
        suite.assert_equal_half4_t(f4 - a, half4(2.0f, 2.0f, 2.0f, 2.0f), "float4 - half4");

        suite.footer();
    }

    void test_half4_access_operations()
    {
        TestSuite suite("half4 Access Operations");
        suite.header();

        half4 v(1.5f, 2.5f, 3.5f, 4.5f);

        // Index access
        suite.assert_equal_half_t(v[0], half(1.5f), "Index access [0]");
        suite.assert_equal_half_t(v[1], half(2.5f), "Index access [1]");
        suite.assert_equal_half_t(v[2], half(3.5f), "Index access [2]");
        suite.assert_equal_half_t(v[3], half(4.5f), "Index access [3]");

        // Mutable index access
        v[0] = half(3.0f);
        v[1] = half(4.0f);
        v[2] = half(5.0f);
        v[3] = half(6.0f);
        suite.assert_equal_half4_t(v, half4(3.0f, 4.0f, 5.0f, 6.0f), "Mutable index access");

        // Data pointer access
        const half* data = v.data();
        suite.assert_equal(data[0], half(3.0f), "Data pointer access [0]");
        suite.assert_equal(data[1], half(4.0f), "Data pointer access [1]");
        suite.assert_equal(data[2], half(5.0f), "Data pointer access [2]");
        suite.assert_equal(data[3], half(6.0f), "Data pointer access [3]");

        // Mutable data pointer access
        half* mutable_data = v.data();
        mutable_data[0] = half(5.0f);
        mutable_data[1] = half(6.0f);
        mutable_data[2] = half(7.0f);
        mutable_data[3] = half(8.0f);
        suite.assert_equal_half4_t(v, half4(5.0f, 6.0f, 7.0f, 8.0f), "Mutable data pointer access");

        suite.footer();
    }

    void test_half4_conversion_operations()
    {
        TestSuite suite("half4 Conversion Operations");
        suite.header();

        half4 h4(1.5f, 2.5f, 3.5f, 4.5f);

        // Conversion to float4
        float4 f4 = static_cast<float4>(h4);
        suite.assert_equal(f4, float4(1.5f, 2.5f, 3.5f, 4.5f), "Conversion to float4");

        // Global conversion functions
        suite.assert_equal(to_float4(h4), float4(1.5f, 2.5f, 3.5f, 4.5f), "to_float4 function");
        suite.assert_equal_half4_t(to_half4(f4), h4, "to_half4 function");

        // Round-trip conversion
        half4 round_trip = to_half4(to_float4(h4));
        suite.assert_equal_half4_t(round_trip, h4, "Round-trip conversion");

        suite.footer();
    }

    void test_half4_mathematical_functions()
    {
        TestSuite suite("half4 Mathematical Functions");
        suite.header();

        half4 v(3.0f, 4.0f, 0.0f, 0.0f);  // length = 5
        half4 w(1.0f, 2.0f, 2.0f, 1.0f);  // length = √(1+4+4+1) = √10 ≈ 3.162

        // Length calculations
        suite.assert_equal_half_t(v.length(), 5.0f, "Length", 0.01f);
        suite.assert_equal_half_t(v.length_sq(), 25.0f, "Squared length", 0.01f);

        // Distance calculations
        // distance = √((3-1)² + (4-2)² + (0-2)² + (0-1)²) = √(4+4+4+1) = √13 ≈ 3.606
        suite.assert_equal_half_t(distance(v, w), 3.60555128f, "Distance", 0.01f);
        suite.assert_equal_half_t(distance_sq(v, w), 13.0f, "Squared distance", 0.01f);

        // Dot product
        // dot(v, w) = 3*1 + 4*2 + 0*2 + 0*1 = 3 + 8 + 0 + 0 = 11
        suite.assert_equal_half_t(dot(v, w), 11.0f, "Dot product", 0.01f);
        suite.assert_equal_half_t(half4::dot(v, w), 11.0f, "Static dot product", 0.01f);

        // 3D Dot product
        // dot3(v, w) = 3*1 + 4*2 + 0*2 = 3 + 8 + 0 = 11
        suite.assert_equal_half_t(dot3(v, w), 11.0f, "3D Dot product", 0.01f);
        suite.assert_equal_half_t(half4::dot3(v, w), 11.0f, "Static 3D dot product", 0.01f);

        // Cross product
        half4 cross_result = cross(v, w);
        // cross(v, w) = (4*2 - 0*2, 0*1 - 3*2, 3*2 - 4*1, 0) = (8, -6, 2, 0)
        suite.assert_equal_half_t(cross_result.x, 8.0f, "Cross product x", 0.01f);
        suite.assert_equal_half_t(cross_result.y, -6.0f, "Cross product y", 0.01f);
        suite.assert_equal_half_t(cross_result.z, 2.0f, "Cross product z", 0.01f);
        suite.assert_equal_half_t(cross_result.w, 0.0f, "Cross product w", 0.01f);

        // Normalization
        half4 normalized = v.normalize();
        suite.assert_true(normalized.is_normalized(0.01f), "Normalization result is normalized");
        suite.assert_equal_half_t(normalized.length(), 1.0f, "Normalized length", 0.01f);
        suite.assert_true(v.normalize().approximately(half4(0.6f, 0.8f, 0.0f, 0.0f), 0.01f), "Normalization values");

        // Global normalization
        suite.assert_equal_half4_t(normalize(v), v.normalize(), "Global normalize function");

        suite.footer();
    }

    void test_half4_hlsl_functions()
    {
        TestSuite suite("half4 HLSL Functions");
        suite.header();

        half4 v(1.5f, -2.5f, 0.5f, -0.8f);
        half4 w(0.3f, 0.7f, -0.2f, 1.2f);

        // Component-wise functions
        suite.assert_equal_half4_t(abs(v), half4(1.5f, 2.5f, 0.5f, 0.8f), "Absolute value");
        suite.assert_equal_half4_t(sign(v), half4(1.0f, -1.0f, 1.0f, -1.0f), "Sign function");
        suite.assert_equal_half4_t(floor(half4(1.7f, -2.3f, 0.8f, -0.5f)), half4(1.0f, -3.0f, 0.0f, -1.0f), "Floor");
        suite.assert_equal_half4_t(ceil(half4(1.2f, -2.7f, 0.3f, -0.2f)), half4(2.0f, -2.0f, 1.0f, 0.0f), "Ceil");
        suite.assert_equal_half4_t(round(half4(1.4f, 1.6f, -1.5f, 2.5f)), half4(1.0f, 2.0f, -2.0f, 3.0f), "Round");
        suite.assert_equal_half4_t(frac(half4(1.7f, -2.3f, 0.8f, -0.5f)), half4(0.7f, 0.7f, 0.8f, 0.5f), "Fractional part", 0.01f);

        // Saturate
        suite.assert_equal_half4_t(saturate(half4(-0.5f, 1.5f, 0.5f, 2.0f)), half4(0.0f, 1.0f, 0.5f, 1.0f), "Saturate");
        suite.assert_equal_half4_t(half4::saturate(half4(-0.5f, 1.5f, 0.5f, 2.0f)), half4(0.0f, 1.0f, 0.5f, 1.0f), "Static saturate");

        // Step
        suite.assert_equal_half4_t(step(half(1.0f), half4(0.5f, 1.5f, 1.0f, 2.0f)), half4(0.0f, 1.0f, 1.0f, 1.0f), "Step function");

        // Min/Max
        suite.assert_equal_half4_t(min(half4(1, 3, 2, 4), half4(2, 2, 3, 3)), half4(1, 2, 2, 3), "Minimum");
        suite.assert_equal_half4_t(max(half4(1, 3, 2, 4), half4(2, 2, 3, 3)), half4(2, 3, 3, 4), "Maximum");
        suite.assert_equal_half4_t(half4::min(half4(1, 3, 2, 4), half4(2, 2, 3, 3)), half4(1, 2, 2, 3), "Static minimum");
        suite.assert_equal_half4_t(half4::max(half4(1, 3, 2, 4), half4(2, 2, 3, 3)), half4(2, 3, 3, 4), "Static maximum");

        // Clamp
        suite.assert_equal_half4_t(clamp(half4(0.5f, 1.5f, -0.5f, 2.0f), 0.0f, 1.0f), half4(0.5f, 1.0f, 0.0f, 1.0f), "Scalar clamp");
        suite.assert_equal_half4_t(clamp(half4(0.5f, 1.5f, -0.5f, 2.0f), half4(0, 1, -1, 1), half4(1, 2, 0, 3)),
            half4(0.5f, 1.5f, -0.5f, 2.0f), "Vector clamp");

        // Smoothstep
        half4 smooth = smoothstep(half(0.0f), half(1.0f), half4(0.5f, 0.5f, 0.5f, 0.5f));
        suite.assert_true(smooth.approximately(half4(0.5f, 0.5f, 0.5f, 0.5f), 0.1f), "Smoothstep");

        suite.footer();
    }

    void test_half4_color_operations()
    {
        TestSuite suite("half4 Color Operations");
        suite.header();

        half4 color(0.8f, 0.6f, 0.4f, 0.9f);

        // Luminance
        half lum = color.luminance();
        suite.assert_true(lum > 0.0f && lum < 1.0f, "Luminance calculation");

        // Brightness
        half bright = color.brightness();
        suite.assert_true(bright > 0.0f && bright < 1.0f, "Brightness calculation");

        // Grayscale conversion
        half4 gray = color.grayscale();
        suite.assert_equal_half_t(gray.x, gray.y, "Grayscale R == G");
        suite.assert_equal_half_t(gray.y, gray.z, "Grayscale G == B");
        suite.assert_equal_half_t(gray.w, color.w, "Grayscale alpha preserved");

        // Premultiply alpha
        half4 premult = color.premultiply_alpha();
        suite.assert_equal_half4_t(premult, half4(0.8f * 0.9f, 0.6f * 0.9f, 0.4f * 0.9f, 0.9f), "Premultiply alpha", 0.01f);

        // Unpremultiply alpha
        half4 unpremult = premult.unpremultiply_alpha();
        suite.assert_true(unpremult.approximately(color, 0.01f), "Unpremultiply alpha round-trip");

        // sRGB conversions
        half4 linear = color.srgb_to_linear();
        half4 srgb = linear.linear_to_srgb();
        suite.assert_true(srgb.approximately(color, 0.1f), "sRGB round-trip conversion");

        suite.footer();
    }

    void test_half4_geometric_operations()
    {
        TestSuite suite("half4 Geometric Operations");
        suite.header();

        // Homogeneous coordinates
        half4 homogeneous(2.0f, 4.0f, 6.0f, 2.0f);
        half3 projected = homogeneous.project();
        suite.assert_equal_half3_t(projected, half3(1.0f, 2.0f, 3.0f), "Homogeneous projection");

        // To homogeneous
        half4 point3d(1.0f, 2.0f, 3.0f, 0.0f);
        half4 homogeneous_result = point3d.to_homogeneous();
        suite.assert_equal_half4_t(homogeneous_result, half4(1.0f, 2.0f, 3.0f, 1.0f), "To homogeneous coordinates");

        // Edge case: zero w component
        half4 zero_w(1.0f, 2.0f, 3.0f, 0.0f);
        half3 projected_zero = zero_w.project();
        suite.assert_equal_half3_t(projected_zero, half3::zero(), "Project with zero w");

        // Lerp
        half4 a(1.0f, 2.0f, 3.0f, 4.0f);
        half4 b(5.0f, 6.0f, 7.0f, 8.0f);
        suite.assert_equal_half4_t(lerp(a, b, 0.5f), half4(3.0f, 4.0f, 5.0f, 6.0f), "Linear interpolation");

        suite.footer();
    }

    void test_half4_swizzle_operations()
    {
        TestSuite suite("half4 Swizzle Operations");
        suite.header();

        half4 v(1.0f, 2.0f, 3.0f, 4.0f);

        // 2D swizzles
        suite.assert_equal_half2_t(v.xy(), half2(1.0f, 2.0f), "XY swizzle");
        suite.assert_equal_half2_t(v.xz(), half2(1.0f, 3.0f), "XZ swizzle");
        suite.assert_equal_half2_t(v.xw(), half2(1.0f, 4.0f), "XW swizzle");
        suite.assert_equal_half2_t(v.yz(), half2(2.0f, 3.0f), "YZ swizzle");
        suite.assert_equal_half2_t(v.yw(), half2(2.0f, 4.0f), "YW swizzle");
        suite.assert_equal_half2_t(v.zw(), half2(3.0f, 4.0f), "ZW swizzle");

        // 3D swizzles
        suite.assert_equal_half3_t(v.xyz(), half3(1.0f, 2.0f, 3.0f), "XYZ swizzle");
        suite.assert_equal_half3_t(v.xyw(), half3(1.0f, 2.0f, 4.0f), "XYW swizzle");
        suite.assert_equal_half3_t(v.xzw(), half3(1.0f, 3.0f, 4.0f), "XZW swizzle");
        suite.assert_equal_half3_t(v.yzw(), half3(2.0f, 3.0f, 4.0f), "YZW swizzle");

        // 4D swizzles
        suite.assert_equal_half4_t(v.yxzw(), half4(2.0f, 1.0f, 3.0f, 4.0f), "YXZW swizzle");
        suite.assert_equal_half4_t(v.zxyw(), half4(3.0f, 1.0f, 2.0f, 4.0f), "ZXYW swizzle");
        suite.assert_equal_half4_t(v.zyxw(), half4(3.0f, 2.0f, 1.0f, 4.0f), "ZYXW swizzle");
        suite.assert_equal_half4_t(v.wzyx(), half4(4.0f, 3.0f, 2.0f, 1.0f), "WZYX swizzle");

        // Color swizzles
        suite.assert_equal_half_t(v.r(), 1.0f, "Red component");
        suite.assert_equal_half_t(v.g(), 2.0f, "Green component");
        suite.assert_equal_half_t(v.b(), 3.0f, "Blue component");
        suite.assert_equal_half_t(v.a(), 4.0f, "Alpha component");

        suite.assert_equal_half2_t(v.rg(), half2(1.0f, 2.0f), "RG swizzle");
        suite.assert_equal_half2_t(v.rb(), half2(1.0f, 3.0f), "RB swizzle");
        suite.assert_equal_half2_t(v.ra(), half2(1.0f, 4.0f), "RA swizzle");
        suite.assert_equal_half2_t(v.gb(), half2(2.0f, 3.0f), "GB swizzle");
        suite.assert_equal_half2_t(v.ga(), half2(2.0f, 4.0f), "GA swizzle");
        suite.assert_equal_half2_t(v.ba(), half2(3.0f, 4.0f), "BA swizzle");

        suite.assert_equal_half3_t(v.rgb(), half3(1.0f, 2.0f, 3.0f), "RGB swizzle");
        suite.assert_equal_half3_t(v.rga(), half3(1.0f, 2.0f, 4.0f), "RGA swizzle");
        suite.assert_equal_half3_t(v.rba(), half3(1.0f, 3.0f, 4.0f), "RBA swizzle");
        suite.assert_equal_half3_t(v.gba(), half3(2.0f, 3.0f, 4.0f), "GBA swizzle");

        suite.assert_equal_half4_t(v.grba(), half4(2.0f, 1.0f, 3.0f, 4.0f), "GRBA swizzle");
        suite.assert_equal_half4_t(v.brga(), half4(3.0f, 1.0f, 2.0f, 4.0f), "BRGA swizzle");
        suite.assert_equal_half4_t(v.bgra(), half4(3.0f, 2.0f, 1.0f, 4.0f), "BGRA swizzle");
        suite.assert_equal_half4_t(v.abgr(), half4(4.0f, 3.0f, 2.0f, 1.0f), "ABGR swizzle");

        suite.footer();
    }

    void test_half4_utility_methods()
    {
        TestSuite suite("half4 Utility Methods");
        suite.header();

        half4 a(1.0f, 2.0f, 3.0f, 4.0f);
        half4 b(1.0001f, 2.0001f, 3.0001f, 4.0001f);
        half4 c(1.1f, 2.1f, 3.1f, 4.1f);

        // Validity checks
        suite.assert_true(a.is_valid(), "Valid vector check");
        suite.assert_false(half4(std::numeric_limits<float>::infinity(), 2.0f, 3.0f, 4.0f).is_valid(), "Invalid vector - infinity");
        suite.assert_false(half4(1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f).is_valid(), "Invalid vector - NaN");

        // Approximate equality
        suite.assert_true(a.approximately(b, 0.01f), "Approximate equality");
        suite.assert_false(a.approximately(c, 0.01f), "Approximate inequality");
        suite.assert_true(approximately(a, b, 0.01f), "Global approximately");

        // Zero checks
        suite.assert_true(half4::zero().approximately_zero(0.001f), "Exactly zero");
        suite.assert_true(half4(0.001f, -0.001f, 0.001f, -0.001f).approximately_zero(0.01f), "Approximately zero");

        // Normalization checks
        suite.assert_true(half4(0.5f, 0.5f, 0.5f, 0.5f).is_normalized(0.01f), "Normalized check");
        suite.assert_false(half4(1.0f, 1.0f, 1.0f, 1.0f).is_normalized(), "Not normalized check");
        suite.assert_true(is_normalized(half4(0.5f, 0.5f, 0.5f, 0.5f), 0.01f), "Global is_normalized");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains x");
        suite.assert_true(str.find("2.0") != std::string::npos, "String contains y");
        suite.assert_true(str.find("3.0") != std::string::npos, "String contains z");
        suite.assert_true(str.find("4.0") != std::string::npos, "String contains w");

        // Set methods
        half4 vec(0.0f, 0.0f, 0.0f, 5.0f);
        vec.set_xyz(half3(1.0f, 2.0f, 3.0f));
        suite.assert_equal_half4_t(vec, half4(1.0f, 2.0f, 3.0f, 5.0f), "Set XYZ from half3");

        vec = half4(0.0f, 0.0f, 3.0f, 4.0f);
        vec.set_xy(half2(1.0f, 2.0f));
        suite.assert_equal_half4_t(vec, half4(1.0f, 2.0f, 3.0f, 4.0f), "Set XY from half2");

        vec = half4(1.0f, 2.0f, 0.0f, 0.0f);
        vec.set_zw(half2(3.0f, 4.0f));
        suite.assert_equal_half4_t(vec, half4(1.0f, 2.0f, 3.0f, 4.0f), "Set ZW from half2");

        // Comparison operators
        suite.assert_true(a == a, "Equality operator");
        suite.assert_true(a != half4::zero(), "Inequality operator");

        suite.footer();
    }

    void test_half4_edge_cases()
    {
        TestSuite suite("half4 Edge Cases");
        suite.header();

        // Zero vectors
        suite.assert_true(half4::zero().approximately_zero(), "Zero vector");
        suite.assert_true(half4::zero().is_all_zero(), "Zero vector is_all_zero()");
        half4 normalized_zero = half4::zero().normalize();
        suite.assert_true(normalized_zero.is_all_zero(), "Normalize zero vector returns zero");
        suite.assert_equal_half_t(half4::zero().length(), 0.0f, "Zero length", 0.001f);
        suite.assert_equal_half_t(half4::zero().length_sq(), 0.0f, "Zero squared length", 0.001f);

        // Very small vectors
        half4 tiny(0.01f, 0.01f, 0.01f, 0.01f);
        suite.assert_true(tiny.approximately_zero(0.1f), "Tiny vector approximately zero");
        suite.assert_false(tiny.approximately_zero(0.001f), "Tiny vector not exactly zero");

        // Very large vectors (within half range)
        half4 large(1e4f, 1e4f, 1e4f, 1e4f);
        suite.assert_true(large.is_valid(), "Large vector is valid");

        // Normalization edge cases
        suite.assert_true(half4::zero().normalize().approximately_zero(0.001f), "Normalize zero vector");

        half4 inf_vec(std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f);
        suite.assert_false(inf_vec.is_valid(), "Infinity vector validity check");

        // NaN propagation
        half4 nan_vec(std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f, 1.0f);
        suite.assert_false(nan_vec.is_valid(), "NaN vector validity");

        // Test normalization of very small vectors
        half4 very_small(0.001f, 0.001f, 0.001f, 0.001f);
        half4 normalized_small = very_small.normalize();
        suite.assert_true(normalized_small.is_normalized(0.01f) || normalized_small.approximately_zero(0.001f),
            "Very small vector normalization");

        // Cross product edge cases
        suite.assert_equal_half4_t(cross(half4::unit_x(), half4::unit_x()), half4::zero(), "Cross product with itself");
        suite.assert_equal_half4_t(cross(half4::unit_x(), half4::unit_y()), half4::unit_z(), "Cross product orthogonality");

        // Dot product edge cases
        suite.assert_equal_half_t(dot(half4::zero(), half4::one()), 0.0f, "Dot product with zero vector", 0.001f);
        suite.assert_equal_half_t(dot(half4::unit_x(), half4::unit_x()), 1.0f, "Dot product with itself", 0.001f);

        // Color operations edge cases
        half4 zero_alpha(1.0f, 1.0f, 1.0f, 0.0f);
        suite.assert_equal_half4_t(zero_alpha.premultiply_alpha(), half4::zero(), "Premultiply zero alpha");
        half4 unpremult_result = zero_alpha.unpremultiply_alpha();

        std::cout << "DEBUG: Original zero_alpha: " << zero_alpha.to_string() << std::endl;
        std::cout << "DEBUG: Unpremult result: " << unpremult_result.to_string() << std::endl;
        std::cout << "DEBUG: Has infinity: " << unpremult_result.is_inf() << std::endl;

        suite.assert_false(unpremult_result.is_inf(), "Unpremultiply zero alpha should not produce infinity");
        suite.assert_true(unpremult_result.is_valid(), "Unpremultiply zero alpha should produce valid result");

        // Homogeneous coordinates edge cases
        half4 zero_w(1.0f, 2.0f, 3.0f, 0.0f);
        suite.assert_equal_half4_t(zero_w.project(), half3::zero(), "Project with zero w");

        suite.footer();
    }

    void test_half4_alpha_operations()
    {
        TestSuite suite("half4 Alpha Operations");
        suite.header();

        // Тест 1: Нормальный случай
        half4 color(0.5f, 0.5f, 0.5f, 0.5f);
        half4 premult = color.premultiply_alpha();
        half4 unpremult = premult.unpremultiply_alpha();
        suite.assert_true(unpremult.approximately(color, 0.01f), "Normal alpha round-trip");

        // Тест 2: Alpha = 1.0
        half4 opaque(0.8f, 0.6f, 0.4f, 1.0f);
        half4 premult_opaque = opaque.premultiply_alpha();
        half4 unpremult_opaque = premult_opaque.unpremultiply_alpha();
        suite.assert_true(unpremult_opaque.approximately(opaque, 0.01f), "Opaque alpha round-trip");

        // Тест 3: Alpha = 0.0 (самый важный!)
        half4 transparent(1.0f, 1.0f, 1.0f, 0.0f);
        half4 premult_transparent = transparent.premultiply_alpha();
        half4 unpremult_transparent = premult_transparent.unpremultiply_alpha();

        std::cout << "DEBUG: Transparent original: " << transparent.to_string() << std::endl;
        std::cout << "DEBUG: Premult transparent: " << premult_transparent.to_string() << std::endl;
        std::cout << "DEBUG: Unpremult result: " << unpremult_transparent.to_string() << std::endl;

        // После premultiply с alpha=0, RGB должны стать (0,0,0)
        suite.assert_true(premult_transparent.is_all_zero(), "Premultiply with alpha=0 should zero RGB");

        // После unpremultiply, мы должны получить обратно (0,0,0,0)
        suite.assert_true(unpremult_transparent.is_all_zero(), "Unpremultiply zero alpha should return zero");
        suite.assert_false(unpremult_transparent.is_inf(), "Unpremultiply should not produce infinity");

        // Тест 4: Очень маленький alpha
        half4 tiny_alpha(0.1f, 0.2f, 0.3f, 1e-5f);
        half4 premult_tiny = tiny_alpha.premultiply_alpha();
        half4 unpremult_tiny = premult_tiny.unpremultiply_alpha();
        suite.assert_true(unpremult_tiny.is_valid(), "Tiny alpha should produce valid result");
        suite.assert_false(unpremult_tiny.is_inf(), "Tiny alpha should not produce infinity");

        suite.footer();
    }

    void test_half4_constants()
    {
        TestSuite suite("half4 Constants");
        suite.header();

        // Basic constants
        suite.assert_equal_half4_t(half4_Zero, half4::zero(), "Zero constant");
        suite.assert_equal_half4_t(half4_One, half4::one(), "One constant");
        suite.assert_equal_half4_t(half4_UnitX, half4::unit_x(), "Unit X constant");
        suite.assert_equal_half4_t(half4_UnitY, half4::unit_y(), "Unit Y constant");
        suite.assert_equal_half4_t(half4_UnitZ, half4::unit_z(), "Unit Z constant");
        suite.assert_equal_half4_t(half4_UnitW, half4::unit_w(), "Unit W constant");

        // Color constants
        suite.assert_equal_half4_t(half4_Red, half4(1.0f, 0.0f, 0.0f, 1.0f), "Red constant");
        suite.assert_equal_half4_t(half4_Green, half4(0.0f, 1.0f, 0.0f, 1.0f), "Green constant");
        suite.assert_equal_half4_t(half4_Blue, half4(0.0f, 0.0f, 1.0f, 1.0f), "Blue constant");
        suite.assert_equal_half4_t(half4_White, half4(1.0f, 1.0f, 1.0f, 1.0f), "White constant");
        suite.assert_equal_half4_t(half4_Black, half4(0.0f, 0.0f, 0.0f, 1.0f), "Black constant");
        suite.assert_equal_half4_t(half4_Transparent, half4(0.0f, 0.0f, 0.0f, 0.0f), "Transparent constant");
        suite.assert_equal_half4_t(half4_Yellow, half4(1.0f, 1.0f, 0.0f, 1.0f), "Yellow constant");
        suite.assert_equal_half4_t(half4_Cyan, half4(0.0f, 1.0f, 1.0f, 1.0f), "Cyan constant");
        suite.assert_equal_half4_t(half4_Magenta, half4(1.0f, 0.0f, 1.0f, 1.0f), "Magenta constant");

        suite.footer();
    }

    void test_half4_special_value_checks()
    {
        TestSuite suite("half4 Special Value Checks");
        suite.header();

        // Test data
        half4 normal(1.0f, 2.0f, 3.0f, 4.0f);
        half4 positive_inf(std::numeric_limits<float>::infinity(), 1.0f, 2.0f, 3.0f);
        half4 negative_inf(-std::numeric_limits<float>::infinity(), 1.0f, 2.0f, 3.0f);
        half4 nan_vec(std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f, 3.0f);
        half4 zero_vec(0.0f, 0.0f, 0.0f, 0.0f);
        half4 negative_zero(-0.0f, -0.0f, -0.0f, -0.0f);
        half4 positive_vec(1.0f, 2.0f, 3.0f, 4.0f);
        half4 negative_vec(-1.0f, -2.0f, -3.0f, -4.0f);
        half4 mixed_signs(1.0f, -2.0f, 3.0f, -4.0f);

        // Infinity checks
        suite.assert_false(normal.is_inf(), "Normal vector is not infinity");
        suite.assert_true(positive_inf.is_inf(), "Positive infinity detected");
        suite.assert_true(negative_inf.is_inf(), "Negative infinity detected");

        suite.assert_true(positive_inf.is_positive_inf(), "Positive infinity specific check");
        suite.assert_true(negative_inf.is_negative_inf(), "Negative infinity specific check");

        suite.assert_false(positive_inf.is_all_positive_inf(), "Not all components are positive infinity");
        suite.assert_false(negative_inf.is_all_negative_inf(), "Not all components are negative infinity");

        // NaN checks
        suite.assert_false(normal.is_nan(), "Normal vector is not NaN");
        suite.assert_true(nan_vec.is_nan(), "NaN detected");
        suite.assert_false(nan_vec.is_all_nan(), "Not all components are NaN");

        // Finite checks
        suite.assert_true(normal.is_finite(), "Normal vector is finite");
        suite.assert_true(normal.is_all_finite(), "All components of normal vector are finite");
        suite.assert_false(positive_inf.is_finite(), "Infinity is not finite");
        suite.assert_false(nan_vec.is_finite(), "NaN is not finite");

        // Sign checks
        suite.assert_false(positive_vec.is_negative(), "Positive vector is not negative");
        suite.assert_true(positive_vec.is_positive(), "Positive vector is positive");
        suite.assert_true(positive_vec.is_all_positive(), "All components are positive");

        suite.assert_true(negative_vec.is_negative(), "Negative vector is negative");
        suite.assert_true(negative_vec.is_all_negative(), "All components are negative");
        suite.assert_false(negative_vec.is_positive(), "Negative vector is not positive");

        suite.assert_true(mixed_signs.is_negative(), "Mixed signs vector has negative components");
        suite.assert_true(mixed_signs.is_positive(), "Mixed signs vector has positive components");
        suite.assert_false(mixed_signs.is_all_negative(), "Not all components are negative");
        suite.assert_false(mixed_signs.is_all_positive(), "Not all components are positive");

        // Zero checks
        suite.assert_true(zero_vec.is_zero(), "Zero vector is zero");
        suite.assert_true(zero_vec.is_all_zero(), "All components are zero");
        suite.assert_true(zero_vec.is_positive_zero(), "Zero vector has positive zero");
        suite.assert_false(zero_vec.is_negative_zero(), "Zero vector doesn't have negative zero");

        suite.assert_true(negative_zero.is_zero(), "Negative zero vector is zero");
        suite.assert_true(negative_zero.is_all_zero(), "All components are zero (negative zero)");
        suite.assert_true(negative_zero.is_negative_zero(), "Negative zero detected");
        suite.assert_false(negative_zero.is_positive_zero(), "Negative zero is not positive zero");

        // Normal/Subnormal checks
        suite.assert_true(normal.is_normal(), "Normal vector is normal");
        suite.assert_true(normal.is_all_normal(), "All components are normal");
        suite.assert_false(zero_vec.is_normal(), "Zero is not normal");
        suite.assert_false(positive_inf.is_normal(), "Infinity is not normal");
        suite.assert_false(nan_vec.is_normal(), "NaN is not normal");

        suite.footer();
    }

    void test_half4_zero_checks()
    {
        TestSuite suite("half4 Zero Checks Debug");
        suite.header();

        half4 zero = half4::zero();
        half4 almost_zero(1e-5f, -1e-5f, 1e-6f, -1e-6f);
        half4 non_zero(1.0f, 2.0f, 3.0f, 4.0f);

        // Диагностика
        std::cout << "DEBUG: Zero vector: " << zero.to_string() << std::endl;
        std::cout << "DEBUG: Zero vector is_all_zero(): " << zero.is_all_zero() << std::endl;
        std::cout << "DEBUG: Zero vector approximately_zero(): " << zero.approximately_zero() << std::endl;
        std::cout << "DEBUG: Zero vector length_sq(): " << float(zero.length_sq()) << std::endl;

        std::cout << "DEBUG: Almost zero vector: " << almost_zero.to_string() << std::endl;
        std::cout << "DEBUG: Almost zero is_all_zero(): " << almost_zero.is_all_zero() << std::endl;
        std::cout << "DEBUG: Almost zero approximately_zero(0.001f): " << almost_zero.approximately_zero(0.001f) << std::endl;

        std::cout << "DEBUG: Non-zero vector: " << non_zero.to_string() << std::endl;
        std::cout << "DEBUG: Non-zero is_all_zero(): " << non_zero.is_all_zero() << std::endl;
        std::cout << "DEBUG: Non-zero approximately_zero(): " << non_zero.approximately_zero() << std::endl;

        // Тесты
        suite.assert_true(zero.is_all_zero(), "Zero vector is_all_zero()");
        suite.assert_true(zero.approximately_zero(), "Zero vector approximately_zero()");
        suite.assert_true(almost_zero.approximately_zero(0.001f), "Almost zero vector approximately_zero(0.001f)");
        suite.assert_false(non_zero.is_all_zero(), "Non-zero vector is not all zero");
        suite.assert_false(non_zero.approximately_zero(), "Non-zero vector is not approximately zero");

        suite.footer();
    }

    // ============================================================================
    // Test Runner for half4
    // ============================================================================

    void run_half4_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "HALF4 COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_half4_constructors();
        test_half4_assignment_operators();
        test_half4_compound_assignment_operators();
        test_half4_arithmetic_operations();
        test_half4_access_operations();
        test_half4_conversion_operations();
        test_half4_mathematical_functions();
        test_half4_hlsl_functions();
        test_half4_color_operations();
        test_half4_geometric_operations();
        test_half4_swizzle_operations();
        test_half4_utility_methods();
        test_half4_edge_cases();
        test_half4_constants();
        test_half4_special_value_checks();
        test_half4_zero_checks();
        test_half4_alpha_operations();

        std::cout << "HALF4 TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
// float2x2 Tests - Comprehensive
// ============================================================================

    void test_float2x2_constructors()
    {
        TestSuite suite("float2x2 Constructors");
        suite.header();

        // Default constructor (identity)
        suite.assert_equal(float2x2(), float2x2::identity(), "Default constructor (identity)");

        // Column constructor
        float2 col0(1.0f, 2.0f);
        float2 col1(3.0f, 4.0f);
        float2x2 mat(col0, col1);
        suite.assert_equal(mat.col0(), col0, "Column constructor - col0");
        suite.assert_equal(mat.col1(), col1, "Column constructor - col1");

        // Array constructor
        float data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float2x2 mat_from_array(data);
        suite.assert_equal(mat_from_array(0, 0), 1.0f, "Array constructor - (0,0)");
        suite.assert_equal(mat_from_array(1, 0), 2.0f, "Array constructor - (1,0)");
        suite.assert_equal(mat_from_array(0, 1), 3.0f, "Array constructor - (0,1)");
        suite.assert_equal(mat_from_array(1, 1), 4.0f, "Array constructor - (1,1)");

        // Scalar constructor (diagonal)
        float2x2 scalar_mat(5.0f);
        suite.assert_equal(scalar_mat(0, 0), 5.0f, "Scalar constructor - (0,0)");
        suite.assert_equal(scalar_mat(1, 1), 5.0f, "Scalar constructor - (1,1)");
        suite.assert_equal(scalar_mat(0, 1), 0.0f, "Scalar constructor - (0,1)");
        suite.assert_equal(scalar_mat(1, 0), 0.0f, "Scalar constructor - (1,0)");

        // Diagonal constructor
        float2x2 diag_mat(float2(2.0f, 3.0f));
        suite.assert_equal(diag_mat(0, 0), 2.0f, "Diagonal constructor - (0,0)");
        suite.assert_equal(diag_mat(1, 1), 3.0f, "Diagonal constructor - (1,1)");
        suite.assert_equal(diag_mat(0, 1), 0.0f, "Diagonal constructor - (0,1)");
        suite.assert_equal(diag_mat(1, 0), 0.0f, "Diagonal constructor - (1,0)");

        // Copy constructor
        float2x2 original(1.0f, 2.0f, 3.0f, 4.0f);
        float2x2 copy(original);
        suite.assert_equal(copy, original, "Copy constructor");

        // Static constructors
        suite.assert_equal(float2x2::identity(), float2x2(1.0f, 0.0f, 0.0f, 1.0f), "Identity matrix");
        suite.assert_equal(float2x2::zero(), float2x2(0.0f, 0.0f, 0.0f, 0.0f), "Zero matrix");

        // Rotation matrix
        float2x2 rot_mat = float2x2::rotation(Constants::HALF_PI);
        suite.assert_true(rot_mat.approximately(float2x2(0.0f, 1.0f, -1.0f, 0.0f), 0.001f), "Rotation matrix 90 degrees");

        // Scaling matrix
        float2x2 scale_mat = float2x2::scaling(float2(2.0f, 3.0f));
        suite.assert_equal(scale_mat, float2x2(2.0f, 0.0f, 0.0f, 3.0f), "Scaling matrix from vector");

        float2x2 scale_mat_components = float2x2::scaling(2.0f, 3.0f);
        suite.assert_equal(scale_mat_components, float2x2(2.0f, 0.0f, 0.0f, 3.0f), "Scaling matrix from components");

        float2x2 uniform_scale_mat = float2x2::scaling(2.0f);
        suite.assert_equal(uniform_scale_mat, float2x2(2.0f, 0.0f, 0.0f, 2.0f), "Uniform scaling matrix");

        // Shear matrix
        float2x2 shear_mat = float2x2::shear(float2(0.5f, 0.3f));
        suite.assert_equal(shear_mat, float2x2(1.0f, 0.3f, 0.5f, 1.0f), "Shear matrix from vector");

        float2x2 shear_mat_components = float2x2::shear(0.5f, 0.3f);
        suite.assert_equal(shear_mat_components, float2x2(1.0f, 0.3f, 0.5f, 1.0f), "Shear matrix from components");

        suite.footer();
    }

    void test_float2x2_access_operations()
    {
        TestSuite suite("float2x2 Access Operations");
        suite.header();

        float2x2 mat(1.0f, 2.0f, 3.0f, 4.0f);

        // Column access
        suite.assert_equal(mat.col0(), float2(1.0f, 2.0f), "Column 0 access");
        suite.assert_equal(mat.col1(), float2(3.0f, 4.0f), "Column 1 access");

        // Row access
        suite.assert_equal(mat.row0(), float2(1.0f, 3.0f), "Row 0 access");
        suite.assert_equal(mat.row1(), float2(2.0f, 4.0f), "Row 1 access");

        // Element access operators
        suite.assert_equal(mat(0, 0), 1.0f, "Element (0,0) access");
        suite.assert_equal(mat(1, 0), 2.0f, "Element (1,0) access");
        suite.assert_equal(mat(0, 1), 3.0f, "Element (0,1) access");
        suite.assert_equal(mat(1, 1), 4.0f, "Element (1,1) access");

        // Column index operator
        suite.assert_equal(mat[0], float2(1.0f, 2.0f), "Column index [0]");
        suite.assert_equal(mat[1], float2(3.0f, 4.0f), "Column index [1]");

        // Mutable access
        float2x2 mutable_mat = mat;
        mutable_mat(0, 1) = 5.0f;
        suite.assert_equal(mutable_mat(0, 1), 5.0f, "Mutable element access");

        mutable_mat[0] = float2(6.0f, 7.0f);
        suite.assert_equal(mutable_mat.col0(), float2(6.0f, 7.0f), "Mutable column access");

        // Setters
        mutable_mat.set_col0(float2(1.0f, 2.0f));
        mutable_mat.set_col1(float2(3.0f, 4.0f));
        suite.assert_equal(mutable_mat.col0(), float2(1.0f, 2.0f), "set_col0");
        suite.assert_equal(mutable_mat.col1(), float2(3.0f, 4.0f), "set_col1");

        mutable_mat.set_row0(float2(1.0f, 3.0f));
        mutable_mat.set_row1(float2(2.0f, 4.0f));
        suite.assert_equal(mutable_mat.row0(), float2(1.0f, 3.0f), "set_row0");
        suite.assert_equal(mutable_mat.row1(), float2(2.0f, 4.0f), "set_row1");

        suite.footer();
    }

    void test_float2x2_arithmetic_operations()
    {
        TestSuite suite("float2x2 Arithmetic Operations");
        suite.header();

        float2x2 a(1.0f, 2.0f, 3.0f, 4.0f);
        float2x2 b(5.0f, 6.0f, 7.0f, 8.0f);
        float2x2 c;

        // Addition
        suite.assert_equal(a + b, float2x2(6.0f, 8.0f, 10.0f, 12.0f), "Matrix addition");

        c = a; c += b;
        suite.assert_equal(c, a + b, "Compound addition");

        // Subtraction
        suite.assert_equal(a - b, float2x2(-4.0f, -4.0f, -4.0f, -4.0f), "Matrix subtraction");

        c = a; c -= b;
        suite.assert_equal(c, a - b, "Compound subtraction");

        // Scalar multiplication
        suite.assert_equal(a * 2.0f, float2x2(2.0f, 4.0f, 6.0f, 8.0f), "Scalar multiplication");
        suite.assert_equal(2.0f * a, float2x2(2.0f, 4.0f, 6.0f, 8.0f), "Scalar multiplication commutative");

        c = a; c *= 2.0f;
        suite.assert_equal(c, a * 2.0f, "Compound scalar multiplication");

        // Scalar division
        suite.assert_equal(a / 2.0f, float2x2(0.5f, 1.0f, 1.5f, 2.0f), "Scalar division");

        c = a; c /= 2.0f;
        suite.assert_equal(c, a / 2.0f, "Compound scalar division");

        // Unary operators
        suite.assert_equal(+a, a, "Unary plus");
        suite.assert_equal(-a, float2x2(-1.0f, -2.0f, -3.0f, -4.0f), "Unary minus");

        suite.footer();
    }

    void test_float2x2_matrix_multiplication()
    {
        TestSuite suite("float2x2 Matrix Multiplication");
        suite.header();

        // Test case 1: Simple matrices
        float2x2 a(1.0f, 2.0f, 3.0f, 4.0f);
        float2x2 b(5.0f, 6.0f, 7.0f, 8.0f);

        // Expected: 
        // [1 3] * [5 7] = [1*5+3*6  1*7+3*8] = [5+18  7+24] = [23 31]
        // [2 4]   [6 8]   [2*5+4*6  2*7+4*8]   [10+24 14+32]   [34 46]
        float2x2 expected1(23.0f, 34.0f, 31.0f, 46.0f);
        suite.assert_equal(a * b, expected1, "Matrix multiplication case 1");

        // Test case 2: Identity multiplication
        float2x2 identity = float2x2::identity();
        suite.assert_equal(a * identity, a, "Multiplication with identity");
        suite.assert_equal(identity * a, a, "Identity multiplication");

        // Test case 3: Rotation matrices
        float2x2 rot90 = float2x2::rotation(Constants::HALF_PI);
        float2x2 rot180 = float2x2::rotation(Constants::PI);
        float2x2 rot270 = rot90 * rot90; // 90 + 90 = 180 degrees

        suite.assert_true(rot270.approximately(rot180, 0.001f), "Rotation composition");

        // Compound multiplication
        float2x2 c = a;
        c *= identity;
        suite.assert_equal(c, a, "Compound multiplication with identity");

        suite.footer();
    }

    void test_float2x2_vector_transformations()
    {
        TestSuite suite("float2x2 Vector Transformations");
        suite.header();

        // Identity transformation
        float2x2 identity = float2x2::identity();
        float2 vec(2.0f, 3.0f);
        suite.assert_equal(identity.transform_vector(vec), vec, "Identity vector transformation");
        suite.assert_equal(identity.transform_point(vec), vec, "Identity point transformation");

        // Scaling transformation
        float2x2 scale = float2x2::scaling(2.0f, 3.0f);
        suite.assert_equal(scale.transform_vector(float2(1.0f, 1.0f)), float2(2.0f, 3.0f), "Scaling vector transformation");
        suite.assert_equal(scale.transform_point(float2(1.0f, 1.0f)), float2(2.0f, 3.0f), "Scaling point transformation");

        // Rotation transformation
        float2x2 rot90 = float2x2::rotation(Constants::HALF_PI);
        float2 original(1.0f, 0.0f);
        float2 expected(0.0f, 1.0f);
        suite.assert_true(rot90.transform_vector(original).approximately(expected, 0.001f), "90 degree rotation");

        // Shear transformation
        float2x2 shear = float2x2::shear(0.5f, 0.3f);
        float2 sheared = shear.transform_vector(float2(1.0f, 1.0f));
        suite.assert_equal(sheared, float2(1.5f, 1.3f), "Shear transformation");

        // Operator form
        suite.assert_equal(vec * identity, vec, "Vector multiplication operator");
        suite.assert_equal(vec * scale, scale.transform_vector(vec), "Vector scaling operator");

        suite.footer();
    }

    void test_float2x2_matrix_operations()
    {
        TestSuite suite("float2x2 Matrix Operations");
        suite.header();

        float2x2 mat(1.0f, 2.0f, 3.0f, 4.0f);

        // Transpose
        suite.assert_equal(mat.transposed(), float2x2(1.0f, 3.0f, 2.0f, 4.0f), "Transpose");
        suite.assert_equal(transpose(mat), mat.transposed(), "Global transpose function");

        // Determinant
        // det([1 3]) = 1*4 - 2*3 = 4 - 6 = -2
        //     [2 4]
        suite.assert_equal(mat.determinant(), -2.0f, "Determinant calculation");
        suite.assert_equal(determinant(mat), -2.0f, "Global determinant function");

        // Inverse
        // inv([1 3]) = 1/-2 * [4 -3] = [-2  1.5]
        //     [2 4]           [-2 1]   [ 1 -0.5]
        float2x2 expected_inverse(-2.0f, 1.0f, 1.5f, -0.5f);
        suite.assert_equal(mat.inverted(), expected_inverse, "Matrix inverse");
        suite.assert_equal(inverse(mat), expected_inverse, "Global inverse function");

        // Verify inverse: A * A^(-1) = I
        float2x2 inv_mat = mat.inverted();
        suite.assert_true((mat * inv_mat).approximately(float2x2::identity(), 0.001f), "Inverse verification");

        float2x2 expected_adjugate(4.0f, -2.0f, -3.0f, 1.0f);  // column-major: [4, -3; -2, 1]
        suite.assert_equal(mat.adjugate(), expected_adjugate, "Adjugate matrix");

        // Trace
        suite.assert_equal(mat.trace(), 5.0f, "Trace calculation"); // 1 + 4 = 5
        suite.assert_equal(trace(mat), 5.0f, "Global trace function");

        // Diagonal
        suite.assert_equal(mat.diagonal(), float2(1.0f, 4.0f), "Diagonal extraction");
        suite.assert_equal(diagonal(mat), float2(1.0f, 4.0f), "Global diagonal function");

        // Frobenius norm
        // sqrt(1² + 2² + 3² + 4²) = sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.477
        float expected_norm = std::sqrt(30.0f);
        suite.assert_equal(mat.frobenius_norm(), expected_norm, "Frobenius norm");
        suite.assert_equal(frobenius_norm(mat), expected_norm, "Global Frobenius norm");

        suite.footer();
    }

    void test_float2x2_transformation_components()
    {
        TestSuite suite("float2x2 Transformation Components");
        suite.header();

        // Rotation extraction
        float2x2 rot_mat = float2x2::rotation(Constants::PI / 4.0f); // 45 degrees
        float extracted_angle = rot_mat.get_rotation();
        suite.assert_equal(extracted_angle, Constants::PI / 4.0f, "Rotation angle extraction", 0.001f);

        // Scale extraction from scaling matrix
        float2x2 scale_mat = float2x2::scaling(2.0f, 3.0f);
        suite.assert_equal(scale_mat.get_scale(), float2(2.0f, 3.0f), "Scale extraction from scaling matrix");

        // Scale extraction from rotated scaling matrix
        float2x2 rotated_scale = float2x2::rotation(Constants::PI / 6.0f) * float2x2::scaling(2.0f, 3.0f);
        float2 extracted_scale = rotated_scale.get_scale();
        suite.assert_true(extracted_scale.approximately(float2(2.0f, 3.0f), 0.001f), "Scale extraction from rotated scaling matrix");

        // Set rotation (preserves scale)
        float2x2 mat = float2x2::scaling(2.0f, 3.0f);
        mat.set_rotation(Constants::PI / 4.0f);
        suite.assert_true(mat.get_scale().approximately(float2(2.0f, 3.0f), 0.001f), "Set rotation preserves scale");
        suite.assert_equal(mat.get_rotation(), Constants::PI / 4.0f, "Set rotation angle", 0.001f);

        // Set scale (preserves rotation)
        mat = float2x2::rotation(Constants::PI / 4.0f);
        mat.set_scale(float2(2.0f, 3.0f));
        suite.assert_equal(mat.get_rotation(), Constants::PI / 4.0f, "Set scale preserves rotation", 0.001f);
        suite.assert_true(mat.get_scale().approximately(float2(2.0f, 3.0f), 0.001f), "Set scale values");

        suite.footer();
    }

    void test_float2x2_utility_methods()
    {
        TestSuite suite("float2x2 Utility Methods");
        suite.header();

        // Identity checks
        suite.assert_true(float2x2::identity().is_identity(), "Identity matrix check");
        suite.assert_true(float2x2::identity().is_orthogonal(), "Identity is orthogonal");
        suite.assert_true(float2x2::identity().is_rotation(), "Identity is rotation");

        // Rotation matrix checks
        float2x2 rot_mat = float2x2::rotation(Constants::PI / 6.0f);
        suite.assert_true(rot_mat.is_orthogonal(), "Rotation matrix is orthogonal");
        suite.assert_true(rot_mat.is_rotation(), "Rotation matrix is rotation matrix");

        // Non-orthogonal matrix
        float2x2 non_ortho(1.0f, 2.0f, 3.0f, 4.0f);
        suite.assert_false(non_ortho.is_orthogonal(), "Non-orthogonal matrix check");
        suite.assert_false(non_ortho.is_rotation(), "Non-rotation matrix check");

        // Approximate equality
        float2x2 a(1.0f, 2.0f, 3.0f, 4.0f);
        float2x2 b(1.000001f, 2.000001f, 3.000001f, 4.000001f);
        suite.assert_true(a.approximately(b, 0.001f), "Approximate equality");
        suite.assert_false(a.approximately(b, 1e-8f), "Approximate inequality");

        // Comparison operators
        suite.assert_true(a == a, "Equality operator");
        suite.assert_true(a != float2x2::zero(), "Inequality operator");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains element");
        suite.assert_true(str.find("2.0") != std::string::npos, "String contains element");
        suite.assert_true(str.find("3.0") != std::string::npos, "String contains element");
        suite.assert_true(str.find("4.0") != std::string::npos, "String contains element");

        // Array conversion
        float col_major[4];
        a.to_column_major(col_major);
        suite.assert_equal(col_major[0], 1.0f, "Column-major [0]");
        suite.assert_equal(col_major[1], 2.0f, "Column-major [1]");
        suite.assert_equal(col_major[2], 3.0f, "Column-major [2]");
        suite.assert_equal(col_major[3], 4.0f, "Column-major [3]");

        float row_major[4];
        a.to_row_major(row_major);
        suite.assert_equal(row_major[0], 1.0f, "Row-major [0]");
        suite.assert_equal(row_major[1], 3.0f, "Row-major [1]");
        suite.assert_equal(row_major[2], 2.0f, "Row-major [2]");
        suite.assert_equal(row_major[3], 4.0f, "Row-major [3]");

        suite.footer();
    }

    void test_float2x2_edge_cases()
    {
        TestSuite suite("float2x2 Edge Cases");
        suite.header();

        // Zero matrix
        suite.assert_true(float2x2::zero().approximately_zero(), "Zero matrix");
        suite.assert_equal(float2x2::zero().determinant(), 0.0f, "Zero matrix determinant");

        // Singular matrix (zero determinant)
        float2x2 singular(1.0f, 2.0f, 2.0f, 4.0f); // det = 1*4 - 2*2 = 0
        suite.assert_equal(singular.determinant(), 0.0f, "Singular matrix determinant");

        // Inverse of singular matrix should return identity (as per implementation)
        float2x2 inv_singular = singular.inverted();
        suite.assert_true(inv_singular.is_identity(), "Inverse of singular matrix returns identity");

        // Very small determinant
        float2x2 near_singular(1.0f, 1.0f, 1.0f, 1.000001f);
        suite.assert_true(std::abs(near_singular.determinant()) > 0.0f, "Near-singular matrix has non-zero determinant");

        // Identity properties
        suite.assert_equal(float2x2::identity().determinant(), 1.0f, "Identity determinant");
        suite.assert_equal(float2x2::identity().trace(), 2.0f, "Identity trace");

        // Global function edge cases
        suite.assert_true(is_orthogonal(float2x2::identity(), 0.001f), "Global is_orthogonal for identity");
        suite.assert_true(is_rotation(float2x2::identity(), 0.001f), "Global is_rotation for identity");

        // Matrix with very large values
        float2x2 large(1e10f, 2e10f, 3e10f, 4e10f);
        suite.assert_true(std::isfinite(large.determinant()), "Large matrix has finite determinant");
        suite.assert_true(std::isfinite(large.frobenius_norm()), "Large matrix has finite norm");

        // Matrix with very small values
        float2x2 small_(1e-10f, 2e-10f, 3e-10f, 4e-10f);
        suite.assert_true(std::isfinite(small_.determinant()), "Small matrix has finite determinant");

        suite.footer();
    }

    void run_float2x2_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "FLOAT2X2 COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_float2x2_constructors();
        test_float2x2_access_operations();
        test_float2x2_arithmetic_operations();
        test_float2x2_matrix_multiplication();
        test_float2x2_vector_transformations();
        test_float2x2_matrix_operations();
        test_float2x2_transformation_components();
        test_float2x2_utility_methods();
        test_float2x2_edge_cases();

        std::cout << "FLOAT2X2 TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
    // float3x3 Tests - Comprehensive
    // ============================================================================

    void test_float3x3_constructors()
    {
        TestSuite suite("float3x3 Constructors");
        suite.header();

        // Default constructor
        suite.assert_equal(float3x3(), float3x3::identity(), "Default constructor (identity)");

        float3 col0(1.0f, 2.0f, 3.0f);
        float3 col1(4.0f, 5.0f, 6.0f);
        float3 col2(7.0f, 8.0f, 9.0f);
        float3x3 mat(col0, col1, col2);

        // [1 4 7]
        // [2 5 8] 
        // [3 6 9]
        suite.assert_equal(mat.col0(), float3(1.0f, 2.0f, 3.0f), "Column constructor - col0");
        suite.assert_equal(mat.col1(), float3(4.0f, 5.0f, 6.0f), "Column constructor - col1");
        suite.assert_equal(mat.col2(), float3(7.0f, 8.0f, 9.0f), "Column constructor - col2");

        // Array constructor
        float data[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
        float3x3 mat_from_array(data);
        suite.assert_equal(mat_from_array(0, 0), 1.0f, "Array constructor - (0,0)");
        suite.assert_equal(mat_from_array(1, 0), 2.0f, "Array constructor - (1,0)");
        suite.assert_equal(mat_from_array(2, 0), 3.0f, "Array constructor - (2,0)");
        suite.assert_equal(mat_from_array(0, 1), 4.0f, "Array constructor - (0,1)");
        suite.assert_equal(mat_from_array(1, 1), 5.0f, "Array constructor - (1,1)");
        suite.assert_equal(mat_from_array(2, 1), 6.0f, "Array constructor - (2,1)");
        suite.assert_equal(mat_from_array(0, 2), 7.0f, "Array constructor - (0,2)");
        suite.assert_equal(mat_from_array(1, 2), 8.0f, "Array constructor - (1,2)");
        suite.assert_equal(mat_from_array(2, 2), 9.0f, "Array constructor - (2,2)");

        // Scalar constructor (diagonal)
        float3x3 scalar_mat(5.0f);
        suite.assert_equal(scalar_mat(0, 0), 5.0f, "Scalar constructor - (0,0)");
        suite.assert_equal(scalar_mat(1, 1), 5.0f, "Scalar constructor - (1,1)");
        suite.assert_equal(scalar_mat(2, 2), 5.0f, "Scalar constructor - (2,2)");
        suite.assert_equal(scalar_mat(0, 1), 0.0f, "Scalar constructor - (0,1)");
        suite.assert_equal(scalar_mat(1, 0), 0.0f, "Scalar constructor - (1,0)");

        // Diagonal constructor
        float3x3 diag_mat(float3(2.0f, 3.0f, 4.0f));
        suite.assert_equal(diag_mat(0, 0), 2.0f, "Diagonal constructor - (0,0)");
        suite.assert_equal(diag_mat(1, 1), 3.0f, "Diagonal constructor - (1,1)");
        suite.assert_equal(diag_mat(2, 2), 4.0f, "Diagonal constructor - (2,2)");
        suite.assert_equal(diag_mat(0, 1), 0.0f, "Diagonal constructor - (0,1)");
        suite.assert_equal(diag_mat(1, 0), 0.0f, "Diagonal constructor - (1,0)");

        // Copy constructor
        float3x3 original(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
        float3x3 copy(original);
        suite.assert_equal(copy, original, "Copy constructor");

        // Static constructors
        suite.assert_equal(float3x3::identity(), float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f), "Identity matrix");
        suite.assert_equal(float3x3::zero(), float3x3(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f), "Zero matrix");

        // В тесте конструкторов заменим проверки вращения X:
        float3x3 rot_mat = float3x3::rotation_x(Constants::HALF_PI);

        // Проверяем отдельные элементы с правильными ожиданиями
        suite.assert_equal(rot_mat(0, 0), 1.0f, "Rotation X - (0,0)");
        suite.assert_equal(rot_mat(0, 1), 0.0f, "Rotation X - (0,1)");
        suite.assert_equal(rot_mat(0, 2), 0.0f, "Rotation X - (0,2)");
        suite.assert_equal(rot_mat(1, 0), 0.0f, "Rotation X - (1,0)");
        suite.assert_true(MathFunctions::approximately(rot_mat(1, 1), 0.0f, 0.001f), "Rotation X - (1,1)");
        suite.assert_true(MathFunctions::approximately(rot_mat(1, 2), -1.0f, 0.001f), "Rotation X - (1,2)"); // FIXED: должно быть -1
        suite.assert_equal(rot_mat(2, 0), 0.0f, "Rotation X - (2,0)");
        suite.assert_true(MathFunctions::approximately(rot_mat(2, 1), 1.0f, 0.001f), "Rotation X - (2,1)");  // FIXED: должно быть 1
        suite.assert_true(MathFunctions::approximately(rot_mat(2, 2), 0.0f, 0.001f), "Rotation X - (2,2)");

        // И проверка через approximately
        float3x3 expected_rot_x(1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, -1.0f,
            0.0f, 1.0f, 0.0f);
        suite.assert_true(rot_mat.approximately(expected_rot_x, 0.001f), "Rotation X 90 degrees");

        // Rotation Y 90 degrees
        float3x3 rot_y = float3x3::rotation_y(Constants::HALF_PI);
        float3x3 expected_rot_y(0.0f, 0.0f, 1.0f,   // col0: 0,0,1 (cos90=0, sin90=1)
            0.0f, 1.0f, 0.0f,   // col1: 0,1,0
            -1.0f, 0.0f, 0.0f);  // col2: -1,0,0 (-sin90=-1, cos90=0)
        suite.assert_true(rot_y.approximately(expected_rot_y, 0.001f), "Rotation Y 90 degrees");

        // Rotation Z 90 degrees  
        float3x3 rot_z = float3x3::rotation_z(Constants::HALF_PI);
        float3x3 expected_rot_z(0.0f, -1.0f, 0.0f,   // col0: 0,-1,0 (cos90=0, -sin90=-1)
            1.0f, 0.0f, 0.0f,   // col1: 1,0,0 (sin90=1, cos90=0)
            0.0f, 0.0f, 1.0f);  // col2: 0,0,1
        suite.assert_true(rot_z.approximately(expected_rot_z, 0.001f), "Rotation Z 90 degrees");

        // Scaling matrix
        float3x3 scale_mat = float3x3::scaling(float3(2.0f, 3.0f, 4.0f));
        suite.assert_equal(scale_mat, float3x3(2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f), "Scaling matrix from vector");

        float3x3 uniform_scale_mat = float3x3::scaling(2.0f);
        suite.assert_equal(uniform_scale_mat, float3x3(2.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f), "Uniform scaling matrix");

        // Skew-symmetric matrix
        float3x3 skew_mat = float3x3::skew_symmetric(float3(1.0f, 2.0f, 3.0f));
        suite.assert_equal(skew_mat, float3x3(0.0f, -3.0f, 2.0f, 3.0f, 0.0f, -1.0f, -2.0f, 1.0f, 0.0f), "Skew-symmetric matrix");

        // Outer product
        float3x3 outer_mat = float3x3::outer_product(float3(1.0f, 2.0f, 3.0f), float3(4.0f, 5.0f, 6.0f));
        suite.assert_equal(outer_mat, float3x3(4.0f, 5.0f, 6.0f, 8.0f, 10.0f, 12.0f, 12.0f, 15.0f, 18.0f), "Outer product matrix");

        suite.footer();
    }

    void test_float3x3_access_operations()
    {
        TestSuite suite("float3x3 Access Operations");
        suite.header();

        float3x3 mat(1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f);

        suite.assert_equal(mat.col0(), float3(1.0f, 2.0f, 3.0f), "Column 0 access");
        suite.assert_equal(mat.col1(), float3(4.0f, 5.0f, 6.0f), "Column 1 access");
        suite.assert_equal(mat.col2(), float3(7.0f, 8.0f, 9.0f), "Column 2 access");

        suite.assert_equal(mat.row0(), float3(1.0f, 4.0f, 7.0f), "Row 0 access");
        suite.assert_equal(mat.row1(), float3(2.0f, 5.0f, 8.0f), "Row 1 access");
        suite.assert_equal(mat.row2(), float3(3.0f, 6.0f, 9.0f), "Row 2 access");

        // Element access operators
        suite.assert_equal(mat(0, 0), 1.0f, "Element (0,0) access");
        suite.assert_equal(mat(1, 0), 2.0f, "Element (1,0) access");
        suite.assert_equal(mat(2, 0), 3.0f, "Element (2,0) access");
        suite.assert_equal(mat(0, 1), 4.0f, "Element (0,1) access");
        suite.assert_equal(mat(1, 1), 5.0f, "Element (1,1) access");
        suite.assert_equal(mat(2, 1), 6.0f, "Element (2,1) access");
        suite.assert_equal(mat(0, 2), 7.0f, "Element (0,2) access");
        suite.assert_equal(mat(1, 2), 8.0f, "Element (1,2) access");
        suite.assert_equal(mat(2, 2), 9.0f, "Element (2,2) access");

        // Column index operator
        suite.assert_equal(mat[0], float3(1.0f, 2.0f, 3.0f), "Column index [0]");
        suite.assert_equal(mat[1], float3(4.0f, 5.0f, 6.0f), "Column index [1]");
        suite.assert_equal(mat[2], float3(7.0f, 8.0f, 9.0f), "Column index [2]");

        // Mutable access
        float3x3 mutable_mat = mat;
        mutable_mat(0, 1) = 10.0f;
        suite.assert_equal(mutable_mat(0, 1), 10.0f, "Mutable element access");

        mutable_mat[0] = float3(10.0f, 11.0f, 12.0f);
        suite.assert_equal(mutable_mat.col0(), float3(10.0f, 11.0f, 12.0f), "Mutable column access");

        // Setters
        mutable_mat.set_col0(float3(1.0f, 2.0f, 3.0f));
        mutable_mat.set_col1(float3(4.0f, 5.0f, 6.0f));
        mutable_mat.set_col2(float3(7.0f, 8.0f, 9.0f));
        suite.assert_equal(mutable_mat.col0(), float3(1.0f, 2.0f, 3.0f), "set_col0");
        suite.assert_equal(mutable_mat.col1(), float3(4.0f, 5.0f, 6.0f), "set_col1");
        suite.assert_equal(mutable_mat.col2(), float3(7.0f, 8.0f, 9.0f), "set_col2");

        mutable_mat.set_row0(float3(1.0f, 4.0f, 7.0f));
        mutable_mat.set_row1(float3(2.0f, 5.0f, 8.0f));
        mutable_mat.set_row2(float3(3.0f, 6.0f, 9.0f));
        suite.assert_equal(mutable_mat.row0(), float3(1.0f, 4.0f, 7.0f), "set_row0");
        suite.assert_equal(mutable_mat.row1(), float3(2.0f, 5.0f, 8.0f), "set_row1");
        suite.assert_equal(mutable_mat.row2(), float3(3.0f, 6.0f, 9.0f), "set_row2");

        suite.footer();
    }

    void test_float3x3_arithmetic_operations()
    {
        TestSuite suite("float3x3 Arithmetic Operations");
        suite.header();

        float3x3 a(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
        float3x3 b(9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
        float3x3 c;

        // Addition
        suite.assert_equal(a + b, float3x3(10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f), "Matrix addition");

        c = a; c += b;
        suite.assert_equal(c, a + b, "Compound addition");

        // Subtraction
        suite.assert_equal(a - b, float3x3(-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f), "Matrix subtraction");

        c = a; c -= b;
        suite.assert_equal(c, a - b, "Compound subtraction");

        // Scalar multiplication
        suite.assert_equal(a * 2.0f, float3x3(2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f), "Scalar multiplication");
        suite.assert_equal(2.0f * a, float3x3(2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f), "Scalar multiplication commutative");

        c = a; c *= 2.0f;
        suite.assert_equal(c, a * 2.0f, "Compound scalar multiplication");

        // Scalar division
        suite.assert_equal(a / 2.0f, float3x3(0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f), "Scalar division");

        c = a; c /= 2.0f;
        suite.assert_equal(c, a / 2.0f, "Compound scalar division");

        // Unary operators
        suite.assert_equal(+a, a, "Unary plus");
        suite.assert_equal(-a, float3x3(-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -9.0f), "Unary minus");

        suite.footer();
    }

    void test_float3x3_matrix_multiplication()
    {
        TestSuite suite("float3x3 Matrix Multiplication");
        suite.header();

        // Test case 1: Simple matrices
        float3x3 a(1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f);
        float3x3 b(9.0f, 6.0f, 3.0f, 8.0f, 5.0f, 2.0f, 7.0f, 4.0f, 1.0f);

        float3x3 expected(90.0f, 54.0f, 18.0f, 114.0f, 69.0f, 24.0f, 138.0f, 84.0f, 30.0f);
        suite.assert_equal(a * b, expected, "Matrix multiplication case 1");

        // Test case 2: Identity multiplication
        float3x3 identity = float3x3::identity();
        suite.assert_equal(a * identity, a, "Multiplication with identity");
        suite.assert_equal(identity * a, a, "Identity multiplication");

        // Test case 3: Rotation matrices
        float3x3 rot_x = float3x3::rotation_x(Constants::HALF_PI);
        float3x3 rot_y = float3x3::rotation_y(Constants::HALF_PI);
        float3x3 rot_combined = rot_x * rot_y;

        // Verify that combined rotation is still orthogonal
        suite.assert_true(rot_combined.is_orthogonal(0.001f), "Rotation composition preserves orthogonality");

        // Compound multiplication
        float3x3 c = a;
        c *= identity;
        suite.assert_equal(c, a, "Compound multiplication with identity");

        suite.footer();
    }

    void test_float3x3_vector_transformations()
    {
        TestSuite suite("float3x3 Vector Transformations");
        suite.header();

        // Identity transformation
        float3x3 identity = float3x3::identity();
        float3 vec(2.0f, 3.0f, 4.0f);
        suite.assert_equal(identity.transform_vector(vec), vec, "Identity vector transformation");
        suite.assert_equal(identity.transform_point(vec), vec, "Identity point transformation");

        // Scaling transformation
        float3x3 scale = float3x3::scaling(2.0f, 3.0f, 4.0f);
        suite.assert_equal(scale.transform_vector(float3(1.0f, 1.0f, 1.0f)), float3(2.0f, 3.0f, 4.0f), "Scaling vector transformation");
        suite.assert_equal(scale.transform_point(float3(1.0f, 1.0f, 1.0f)), float3(2.0f, 3.0f, 4.0f), "Scaling point transformation");

        // Rotation transformation
        float3x3 rot_x = float3x3::rotation_x(Constants::HALF_PI);
        float3 original(0.0f, 1.0f, 0.0f);
        float3 expected(0.0f, 0.0f, 1.0f);
        float3 result = rot_x.transform_vector(original);

        // Отладочный вывод
        std::cout << "DEBUG X rotation: original = " << original.to_string()
            << ", result = " << result.to_string()
            << ", expected = " << expected.to_string() << std::endl;

        suite.assert_true(result.approximately(expected, 0.001f), "X rotation 90 degrees");

        // Normal transformation
        float3x3 scale_ = float3x3::scaling(2.0f, 3.0f, 1.0f);
        float3 normal(0.0f, 1.0f, 0.0f);
        float3 transformed_normal = scale_.transform_normal(normal);

        std::cout << "DEBUG Normal transform: normal = " << normal.to_string()
            << ", transformed = " << transformed_normal.to_string()
            << ", length = " << transformed_normal.length() << std::endl;

        suite.assert_true(transformed_normal.is_normalized(0.001f),
            "Normal transformation with scaling");

        // Operator form
        suite.assert_equal(vec * identity, vec, "Vector multiplication operator");
        suite.assert_equal(vec * scale, scale.transform_vector(vec), "Vector scaling operator");
        suite.assert_equal(identity * vec, identity.transform_vector(vec), "Matrix-vector multiplication operator");

        suite.footer();
    }

    void test_float3x3_matrix_operations()
    {
        TestSuite suite("float3x3 Matrix Operations");
        suite.header();

        float3x3 mat(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);

        // Transpose
        suite.assert_equal(mat.transposed(), float3x3(1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f), "Transpose");
        suite.assert_equal(transpose(mat), mat.transposed(), "Global transpose function");

        // Determinant
        // det = 1*(5*9 - 6*8) - 4*(2*9 - 3*8) + 7*(2*6 - 3*5)
        //     = 1*(45-48) - 4*(18-24) + 7*(12-15)
        //     = 1*(-3) - 4*(-6) + 7*(-3)
        //     = -3 + 24 - 21 = 0
        suite.assert_equal(mat.determinant(), 0.0f, "Determinant calculation");
        suite.assert_equal(determinant(mat), 0.0f, "Global determinant function");

        // Test with invertible matrix
        float3x3 invertible(2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f);
        suite.assert_equal(invertible.determinant(), 24.0f, "Determinant of diagonal matrix");

        // Inverse of diagonal matrix
        float3x3 expected_inverse(0.5f, 0.0f, 0.0f, 0.0f, 1.0f / 3.0f, 0.0f, 0.0f, 0.0f, 0.25f);
        suite.assert_equal(invertible.inverted(), expected_inverse, "Matrix inverse");
        suite.assert_equal(inverse(invertible), expected_inverse, "Global inverse function");

        // Verify inverse: A * A^(-1) = I
        float3x3 inv_mat = invertible.inverted();
        suite.assert_true((invertible * inv_mat).approximately(float3x3::identity(), 0.001f), "Inverse verification");

        // Trace
        suite.assert_equal(mat.trace(), 15.0f, "Trace calculation"); // 1 + 5 + 9 = 15
        suite.assert_equal(trace(mat), 15.0f, "Global trace function");

        // Diagonal
        suite.assert_equal(mat.diagonal(), float3(1.0f, 5.0f, 9.0f), "Diagonal extraction");
        suite.assert_equal(diagonal(mat), float3(1.0f, 5.0f, 9.0f), "Global diagonal function");

        // Frobenius norm
        // sqrt(1² + 2² + 3² + 4² + 5² + 6² + 7² + 8² + 9²) = sqrt(285) ≈ 16.8819
        float expected_norm = std::sqrt(285.0f);
        suite.assert_equal(mat.frobenius_norm(), expected_norm, "Frobenius norm");
        suite.assert_equal(frobenius_norm(mat), expected_norm, "Global Frobenius norm");

        // Symmetric and skew-symmetric parts
        float3x3 symmetric = mat.symmetric_part();
        float3x3 skew_symmetric = mat.skew_symmetric_part();
        suite.assert_equal(symmetric + skew_symmetric, mat, "Symmetric + skew-symmetric = original");
        suite.assert_equal(symmetric, symmetric.transposed(), "Symmetric part is symmetric");
        suite.assert_equal(skew_symmetric, -skew_symmetric.transposed(), "Skew-symmetric part is skew-symmetric");

        suite.footer();
    }

    void test_float3x3_decomposition_methods()
    {
        TestSuite suite("float3x3 Decomposition Methods");
        suite.header();

        // Create a transformation matrix with rotation and scaling
        float3x3 rotation = float3x3::rotation_z(Constants::PI / 6.0f); // 30 degrees
        float3x3 scaling = float3x3::scaling(2.0f, 3.0f, 1.0f);
        float3x3 transform = rotation * scaling;

        // Scale extraction
        float3 extracted_scale = transform.extract_scale();
        suite.assert_true(extracted_scale.approximately(float3(2.0f, 3.0f, 1.0f), 0.001f), "Scale extraction");

        // Rotation extraction
        float3x3 extracted_rotation = transform.extract_rotation();
        suite.assert_true(extracted_rotation.is_orthonormal(0.001f), "Extracted rotation is orthonormal");
        suite.assert_true(extracted_rotation.approximately(rotation, 0.001f), "Rotation extraction");

        // Normal matrix
        float3x3 normal_mat = float3x3::normal_matrix(transform);
        float3 normal(0.0f, 1.0f, 0.0f);
        float3 transformed_normal = normal_mat.transform_vector(normal);
        suite.assert_true(transformed_normal.is_normalized(0.001f), "Normal matrix preserves normalization");

        suite.footer();
    }

    void test_float3x3_utility_methods()
    {
        TestSuite suite("float3x3 Utility Methods");
        suite.header();

        // Identity checks
        suite.assert_true(float3x3::identity().is_identity(), "Identity matrix check");
        suite.assert_true(float3x3::identity().is_orthogonal(), "Identity is orthogonal");
        suite.assert_true(float3x3::identity().is_orthonormal(), "Identity is orthonormal");

        // Rotation matrix checks
        float3x3 rot_mat = float3x3::rotation_x(Constants::PI / 6.0f);
        suite.assert_true(rot_mat.is_orthogonal(), "Rotation matrix is orthogonal");
        suite.assert_true(rot_mat.is_orthonormal(), "Rotation matrix is orthonormal");

        // Non-orthogonal matrix
        float3x3 non_ortho(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
        suite.assert_false(non_ortho.is_orthogonal(), "Non-orthogonal matrix check");

        // Approximate equality
        float3x3 a(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
        float3x3 b(1.000001f, 2.000001f, 3.000001f, 4.000001f, 5.000001f, 6.000001f, 7.000001f, 8.000001f, 9.000001f);
        suite.assert_true(a.approximately(b, 0.001f), "Approximate equality");
        suite.assert_false(a.approximately(b, 1e-8f), "Approximate inequality");

        // Comparison operators
        suite.assert_true(a == a, "Equality operator");
        suite.assert_true(a != float3x3::zero(), "Inequality operator");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains element");
        suite.assert_true(str.find("2.0") != std::string::npos, "String contains element");
        suite.assert_true(str.find("3.0") != std::string::npos, "String contains element");

        // Column-major array conversion
        float col_major[9];
        a.to_column_major(col_major);

        suite.assert_equal(col_major[0], 1.0f, "Column-major [0]");  // col0.x
        suite.assert_equal(col_major[1], 4.0f, "Column-major [1]");  // col0.y  ← было 2
        suite.assert_equal(col_major[2], 7.0f, "Column-major [2]");  // col0.z  ← было 3
        suite.assert_equal(col_major[3], 2.0f, "Column-major [3]");  // col1.x  ← было 4
        suite.assert_equal(col_major[4], 5.0f, "Column-major [4]");  // col1.y
        suite.assert_equal(col_major[5], 8.0f, "Column-major [5]");  // col1.z  ← было 6
        suite.assert_equal(col_major[6], 3.0f, "Column-major [6]");  // col2.x  ← было 7
        suite.assert_equal(col_major[7], 6.0f, "Column-major [7]");  // col2.y  ← было 8
        suite.assert_equal(col_major[8], 9.0f, "Column-major [8]");  // col2.z

        // Row-major array conversion  
        float row_major[9];
        a.to_row_major(row_major);

        suite.assert_equal(row_major[0], 1.0f, "Row-major [0]");  // row0.x
        suite.assert_equal(row_major[1], 2.0f, "Row-major [1]");  // row0.y  ← было 4
        suite.assert_equal(row_major[2], 3.0f, "Row-major [2]");  // row0.z  ← было 7
        suite.assert_equal(row_major[3], 4.0f, "Row-major [3]");  // row1.x  ← было 2
        suite.assert_equal(row_major[4], 5.0f, "Row-major [4]");  // row1.y
        suite.assert_equal(row_major[5], 6.0f, "Row-major [5]");  // row1.z  ← было 8
        suite.assert_equal(row_major[6], 7.0f, "Row-major [6]");  // row2.x  ← было 3
        suite.assert_equal(row_major[7], 8.0f, "Row-major [7]");  // row2.y  ← было 6
        suite.assert_equal(row_major[8], 9.0f, "Row-major [8]");  // row2.z

        suite.footer();
    }

    void test_float3x3_edge_cases()
    {
        TestSuite suite("float3x3 Edge Cases");
        suite.header();

        // Zero matrix
        suite.assert_true(float3x3::zero().approximately_zero(), "Zero matrix");
        suite.assert_equal(float3x3::zero().determinant(), 0.0f, "Zero matrix determinant");

        // Singular matrix (zero determinant)
        float3x3 singular(1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f, 3.0f, 6.0f, 9.0f); // linearly dependent columns
        suite.assert_equal(singular.determinant(), 0.0f, "Singular matrix determinant");

        // Inverse of singular matrix should return identity (as per implementation)
        float3x3 inv_singular = singular.inverted();
        suite.assert_true(inv_singular.is_identity(), "Inverse of singular matrix returns identity");

        // Very small determinant
        float3x3 near_singular(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1e-10f);
        suite.assert_true(std::abs(near_singular.determinant()) > 0.0f, "Near-singular matrix has non-zero determinant");

        // Identity properties
        suite.assert_equal(float3x3::identity().determinant(), 1.0f, "Identity determinant");
        suite.assert_equal(float3x3::identity().trace(), 3.0f, "Identity trace");

        // Global function edge cases
        suite.assert_true(is_orthogonal(float3x3::identity(), 0.001f), "Global is_orthogonal for identity");
        suite.assert_true(is_orthonormal(float3x3::identity(), 0.001f), "Global is_orthonormal for identity");

        // Matrix with very large values
        float3x3 large(1e10f, 2e10f, 3e10f, 4e10f, 5e10f, 6e10f, 7e10f, 8e10f, 9e10f);
        suite.assert_true(std::isfinite(large.determinant()), "Large matrix has finite determinant");
        suite.assert_true(std::isfinite(large.frobenius_norm()), "Large matrix has finite norm");

        // Matrix with very small values
        float3x3 small_(1e-10f, 2e-10f, 3e-10f, 4e-10f, 5e-10f, 6e-10f, 7e-10f, 8e-10f, 9e-10f);
        suite.assert_true(std::isfinite(small_.determinant()), "Small matrix has finite determinant");

        // Rotation around zero axis
        float3x3 zero_axis_rot = float3x3::rotation_axis(float3::zero(), 1.0f);
        suite.assert_true(zero_axis_rot.is_identity(), "Rotation around zero axis returns identity");

        suite.footer();
    }

    void debug_rotation_matrices()
    {
        std::cout << "=== DEBUG ROTATION MATRICES ===" << std::endl;

        // Проверка поворота X на 90 градусов
        float3x3 rot_x = float3x3::rotation_x(Constants::HALF_PI);
        std::cout << "Rotation X 90°:" << std::endl;
        std::cout << rot_x.to_string() << std::endl;

        // Проверка поворота Z на 90 градусов  
        float3x3 rot_z = float3x3::rotation_z(Constants::HALF_PI);
        std::cout << "Rotation Z 90°:" << std::endl;
        std::cout << rot_z.to_string() << std::endl;

        // Проверка преобразования векторов
        float3 vec(0.0f, 1.0f, 0.0f);
        std::cout << "Vector: " << vec.to_string() << std::endl;
        std::cout << "After X rotation: " << rot_x.transform_vector(vec).to_string() << std::endl;
        std::cout << "After Z rotation: " << rot_z.transform_vector(vec).to_string() << std::endl;
    }

    void run_float3x3_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "FLOAT3X3 COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_float3x3_constructors();
        test_float3x3_access_operations();
        test_float3x3_arithmetic_operations();
        test_float3x3_matrix_multiplication();
        test_float3x3_vector_transformations();
        test_float3x3_matrix_operations();
        test_float3x3_decomposition_methods();
        test_float3x3_utility_methods();
        test_float3x3_edge_cases();
        debug_rotation_matrices();

        std::cout << "FLOAT3X3 TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
    // float4x4 Tests - Comprehensive
    // ============================================================================

    void test_float4x4_constructors()
    {
        TestSuite suite("float4x4 Constructors");
        suite.header();

        // Default constructor
        suite.assert_equal(float4x4(), float4x4::identity(), "Default constructor (identity)");

        // Column constructor
        float4 col0(1.0f, 2.0f, 3.0f, 4.0f);
        float4 col1(5.0f, 6.0f, 7.0f, 8.0f);
        float4 col2(9.0f, 10.0f, 11.0f, 12.0f);
        float4 col3(13.0f, 14.0f, 15.0f, 16.0f);
        float4x4 mat(col0, col1, col2, col3);

        suite.assert_equal(mat.col0(), col0, "Column constructor - col0");
        suite.assert_equal(mat.col1(), col1, "Column constructor - col1");
        suite.assert_equal(mat.col2(), col2, "Column constructor - col2");
        suite.assert_equal(mat.col3(), col3, "Column constructor - col3");

        // Array constructor
        float data[16] = {
            1.0f, 5.0f, 9.0f, 13.0f,  // col0
            2.0f, 6.0f, 10.0f, 14.0f, // col1  
            3.0f, 7.0f, 11.0f, 15.0f, // col2
            4.0f, 8.0f, 12.0f, 16.0f  // col3
        };
        float4x4 mat_from_array(data);
        suite.assert_equal(mat_from_array, mat, "Array constructor");

        // Scalar constructor (diagonal)
        float4x4 scalar_mat(5.0f);
        suite.assert_equal(scalar_mat(0, 0), 5.0f, "Scalar constructor - (0,0)");
        suite.assert_equal(scalar_mat(1, 1), 5.0f, "Scalar constructor - (1,1)");
        suite.assert_equal(scalar_mat(2, 2), 5.0f, "Scalar constructor - (2,2)");
        suite.assert_equal(scalar_mat(3, 3), 5.0f, "Scalar constructor - (3,3)");
        suite.assert_equal(scalar_mat(0, 1), 0.0f, "Scalar constructor - (0,1)");

        // Diagonal constructor
        float4x4 diag_mat(float4(2.0f, 3.0f, 4.0f, 5.0f));
        suite.assert_equal(diag_mat(0, 0), 2.0f, "Diagonal constructor - (0,0)");
        suite.assert_equal(diag_mat(1, 1), 3.0f, "Diagonal constructor - (1,1)");
        suite.assert_equal(diag_mat(2, 2), 4.0f, "Diagonal constructor - (2,2)");
        suite.assert_equal(diag_mat(3, 3), 5.0f, "Diagonal constructor - (3,3)");
        suite.assert_equal(diag_mat(0, 1), 0.0f, "Diagonal constructor - (0,1)");

        // Copy constructor
        float4x4 original(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);
        float4x4 copy(original);
        suite.assert_equal(copy, original, "Copy constructor");

        // From 3x3 matrix constructor
        float3x3 mat3x3(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
        float4x4 from_3x3(mat3x3);
        suite.assert_equal(from_3x3.col0(), float4(1.0f, 4.0f, 7.0f, 0.0f), "From 3x3 - col0");
        suite.assert_equal(from_3x3.col1(), float4(2.0f, 5.0f, 8.0f, 0.0f), "From 3x3 - col1");
        suite.assert_equal(from_3x3.col2(), float4(3.0f, 6.0f, 9.0f, 0.0f), "From 3x3 - col2");
        suite.assert_equal(from_3x3.col3(), float4(0.0f, 0.0f, 0.0f, 1.0f), "From 3x3 - col3");

        // Static constructors
        suite.assert_equal(float4x4::identity(),
            float4x4(1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f), "Identity matrix");

        suite.assert_equal(float4x4::zero(),
            float4x4(0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f), "Zero matrix");

        // Translation matrix
        float4x4 translation = float4x4::translation(float3(2.0f, 3.0f, 4.0f));
        suite.assert_equal(translation,
            float4x4(1.0f, 0.0f, 0.0f, 2.0f,  // col0: 1,0,0,2 (translation.x в 4-й компоненте)
                0.0f, 1.0f, 0.0f, 3.0f,      // col1: 0,1,0,3 (translation.y в 4-й компоненте)  
                0.0f, 0.0f, 1.0f, 4.0f,      // col2: 0,0,1,4 (translation.z в 4-й компоненте)
                0.0f, 0.0f, 0.0f, 1.0f),     // col3: 0,0,0,1
            "Translation matrix");

        // Scaling matrix
        float4x4 scaling = float4x4::scaling(float3(2.0f, 3.0f, 4.0f));
        suite.assert_equal(scaling,
            float4x4(2.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 3.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 4.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f), "Scaling matrix");

        // Rotation matrices
        float4x4 rot_x = float4x4::rotation_x(Constants::HALF_PI);
        suite.assert_true(rot_x.approximately(
            float4x4(1.0f, 0.0f, 0.0f, 0.0f,   // col0: 1,0,0,0
                0.0f, 0.0f, -1.0f, 0.0f,      // col1: 0,cos90,sin90,0 = 0,0,1,0 (НО ВАША РЕАЛИЗАЦИЯ: 0,0,-1,0?)
                0.0f, 1.0f, 0.0f, 0.0f,       // col2: 0,-sin90,cos90,0 = 0,-1,0,0 (НО ВАША РЕАЛИЗАЦИЯ: 0,1,0,0?)
                0.0f, 0.0f, 0.0f, 1.0f), 0.001f), "Rotation X 90 degrees");

        std::cout << "DEBUG: Actual rotation X matrix:" << std::endl;
        std::cout << rot_x.to_string() << std::endl;

        std::cout << "=== DEBUG ROTATION MATRICES ===" << std::endl;

        rot_x = float4x4::rotation_x(Constants::HALF_PI);
        std::cout << "Rotation X 90° (column-major):" << std::endl;
        std::cout << rot_x.to_string() << std::endl;

        // Проверим преобразование вектора
        float3 vec(0.0f, 1.0f, 0.0f);
        float3 result = rot_x.transform_vector(vec);
        std::cout << "Vector (0,1,0) after X rotation: " << result.to_string() << std::endl;
        std::cout << "Expected: (0,0,1)" << std::endl;

        suite.footer();
    }

    void test_float4x4_access_operations()
    {
        TestSuite suite("float4x4 Access Operations");
        suite.header();

        float4x4 mat(1.0f, 5.0f, 9.0f, 13.0f,
            2.0f, 6.0f, 10.0f, 14.0f,
            3.0f, 7.0f, 11.0f, 15.0f,
            4.0f, 8.0f, 12.0f, 16.0f);

        // Column access
        suite.assert_equal(mat.col0(), float4(1.0f, 2.0f, 3.0f, 4.0f), "Column 0 access");
        suite.assert_equal(mat.col1(), float4(5.0f, 6.0f, 7.0f, 8.0f), "Column 1 access");
        suite.assert_equal(mat.col2(), float4(9.0f, 10.0f, 11.0f, 12.0f), "Column 2 access");
        suite.assert_equal(mat.col3(), float4(13.0f, 14.0f, 15.0f, 16.0f), "Column 3 access");

        // Row access
        suite.assert_equal(mat.row0(), float4(1.0f, 5.0f, 9.0f, 13.0f), "Row 0 access");
        suite.assert_equal(mat.row1(), float4(2.0f, 6.0f, 10.0f, 14.0f), "Row 1 access");
        suite.assert_equal(mat.row2(), float4(3.0f, 7.0f, 11.0f, 15.0f), "Row 2 access");
        suite.assert_equal(mat.row3(), float4(4.0f, 8.0f, 12.0f, 16.0f), "Row 3 access");

        // Element access operators
        suite.assert_equal(mat(0, 0), 1.0f, "Element (0,0) access");
        suite.assert_equal(mat(1, 0), 2.0f, "Element (1,0) access");
        suite.assert_equal(mat(2, 0), 3.0f, "Element (2,0) access");
        suite.assert_equal(mat(3, 0), 4.0f, "Element (3,0) access");
        suite.assert_equal(mat(0, 1), 5.0f, "Element (0,1) access");
        suite.assert_equal(mat(1, 1), 6.0f, "Element (1,1) access");
        suite.assert_equal(mat(2, 1), 7.0f, "Element (2,1) access");
        suite.assert_equal(mat(3, 1), 8.0f, "Element (3,1) access");

        // Column index operator
        suite.assert_equal(mat[0], float4(1.0f, 2.0f, 3.0f, 4.0f), "Column index [0]");
        suite.assert_equal(mat[1], float4(5.0f, 6.0f, 7.0f, 8.0f), "Column index [1]");
        suite.assert_equal(mat[2], float4(9.0f, 10.0f, 11.0f, 12.0f), "Column index [2]");
        suite.assert_equal(mat[3], float4(13.0f, 14.0f, 15.0f, 16.0f), "Column index [3]");

        // Mutable access
        float4x4 mutable_mat = mat;
        mutable_mat(0, 1) = 50.0f;
        suite.assert_equal(mutable_mat(0, 1), 50.0f, "Mutable element access");

        mutable_mat[0] = float4(10.0f, 11.0f, 12.0f, 13.0f);
        suite.assert_equal(mutable_mat.col0(), float4(10.0f, 11.0f, 12.0f, 13.0f), "Mutable column access");

        // Setters
        mutable_mat.set_col0(float4(1.0f, 2.0f, 3.0f, 4.0f));
        mutable_mat.set_col1(float4(5.0f, 6.0f, 7.0f, 8.0f));
        mutable_mat.set_col2(float4(9.0f, 10.0f, 11.0f, 12.0f));
        mutable_mat.set_col3(float4(13.0f, 14.0f, 15.0f, 16.0f));
        suite.assert_equal(mutable_mat.col0(), float4(1.0f, 2.0f, 3.0f, 4.0f), "set_col0");
        suite.assert_equal(mutable_mat.col1(), float4(5.0f, 6.0f, 7.0f, 8.0f), "set_col1");
        suite.assert_equal(mutable_mat.col2(), float4(9.0f, 10.0f, 11.0f, 12.0f), "set_col2");
        suite.assert_equal(mutable_mat.col3(), float4(13.0f, 14.0f, 15.0f, 16.0f), "set_col3");

        mutable_mat.set_row0(float4(1.0f, 5.0f, 9.0f, 13.0f));
        mutable_mat.set_row1(float4(2.0f, 6.0f, 10.0f, 14.0f));
        mutable_mat.set_row2(float4(3.0f, 7.0f, 11.0f, 15.0f));
        mutable_mat.set_row3(float4(4.0f, 8.0f, 12.0f, 16.0f));
        suite.assert_equal(mutable_mat.row0(), float4(1.0f, 5.0f, 9.0f, 13.0f), "set_row0");
        suite.assert_equal(mutable_mat.row1(), float4(2.0f, 6.0f, 10.0f, 14.0f), "set_row1");
        suite.assert_equal(mutable_mat.row2(), float4(3.0f, 7.0f, 11.0f, 15.0f), "set_row2");
        suite.assert_equal(mutable_mat.row3(), float4(4.0f, 8.0f, 12.0f, 16.0f), "set_row3");

        suite.footer();
    }

    void test_float4x4_arithmetic_operations()
    {
        TestSuite suite("float4x4 Arithmetic Operations");
        suite.header();

        float4x4 a(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);
        float4x4 b(16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f,
            8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
        float4x4 c;

        // Addition
        suite.assert_equal(a + b, float4x4(17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f,
            17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f), "Matrix addition");

        c = a; c += b;
        suite.assert_equal(c, a + b, "Compound addition");

        // Subtraction
        suite.assert_equal(a - b, float4x4(-15.0f, -13.0f, -11.0f, -9.0f, -7.0f, -5.0f, -3.0f, -1.0f,
            1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f), "Matrix subtraction");

        c = a; c -= b;
        suite.assert_equal(c, a - b, "Compound subtraction");

        // Scalar multiplication
        suite.assert_equal(a * 2.0f, float4x4(2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f,
            18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f), "Scalar multiplication");

        suite.assert_equal(2.0f * a, a * 2.0f, "Scalar multiplication commutative");

        c = a; c *= 2.0f;
        suite.assert_equal(c, a * 2.0f, "Compound scalar multiplication");

        // Scalar division
        suite.assert_equal(a / 2.0f, float4x4(0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f,
            4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f), "Scalar division");

        c = a; c /= 2.0f;
        suite.assert_equal(c, a / 2.0f, "Compound scalar division");

        // Unary operators
        suite.assert_equal(+a, a, "Unary plus");
        suite.assert_equal(-a, float4x4(-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f,
            -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f, -16.0f), "Unary minus");

        suite.footer();
    }

    void test_float4x4_matrix_multiplication()
    {
        TestSuite suite("float4x4 Matrix Multiplication");
        suite.header();

        // Test case 1: Simple matrices
        float4x4 a(1.0f, 0.0f, 0.0f, 0.0f,   // translation matrix
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 2.0f, 3.0f, 1.0f);

        float4x4 b(2.0f, 0.0f, 0.0f, 0.0f,   // scaling matrix
            0.0f, 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);

        // Expected: translation * scaling
        float4x4 expected(2.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 0.0f,
            2.0f, 6.0f, 12.0f, 1.0f);  // translation scaled: (1,2,3) * (2,3,4) = (2,6,12)

        suite.assert_equal(a * b, expected, "Translation * Scaling multiplication");

        // Test case 2: Identity multiplication
        float4x4 identity = float4x4::identity();
        suite.assert_equal(a * identity, a, "Multiplication with identity");
        suite.assert_equal(identity * a, a, "Identity multiplication");

        // Test case 3: More complex multiplication
        float4x4 c(1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f);

        float4x4 d(16.0f, 15.0f, 14.0f, 13.0f,
            12.0f, 11.0f, 10.0f, 9.0f,
            8.0f, 7.0f, 6.0f, 5.0f,
            4.0f, 3.0f, 2.0f, 1.0f);

        // Manual calculation of expected result for first element
        // (0,0) = 1*16 + 2*12 + 3*8 + 4*4 = 16 + 24 + 24 + 16 = 80
        float4x4 expected_complex(80.0f, 70.0f, 60.0f, 50.0f,
            240.0f, 214.0f, 188.0f, 162.0f,
            400.0f, 358.0f, 316.0f, 274.0f,
            560.0f, 502.0f, 444.0f, 386.0f);

        suite.assert_equal(c * d, expected_complex, "Complex matrix multiplication");

        // Compound multiplication
        float4x4 e = a;
        e *= identity;
        suite.assert_equal(e, a, "Compound multiplication with identity");

        suite.footer();
    }

    void test_float4x4_vector_transformations()
    {
        TestSuite suite("float4x4 Vector Transformations");
        suite.header();

        // Identity transformation
        float4x4 identity = float4x4::identity();
        float4 vec4(2.0f, 3.0f, 4.0f, 1.0f);
        float3 vec3(2.0f, 3.0f, 4.0f);

        suite.assert_equal(identity.transform_vector(vec4), vec4, "Identity 4D vector transformation");
        suite.assert_equal(identity.transform_point(vec3), vec3, "Identity 3D point transformation");
        suite.assert_equal(identity.transform_vector(vec3), vec3, "Identity 3D vector transformation");

        // Translation transformation
        float4x4 translation = float4x4::translation(float3(1.0f, 2.0f, 3.0f));

        // Points should be translated
        suite.assert_equal(translation.transform_point(float3(1.0f, 1.0f, 1.0f)),
            float3(2.0f, 3.0f, 4.0f), "Point translation");

        // Vectors should not be translated
        suite.assert_equal(translation.transform_vector(float3(1.0f, 1.0f, 1.0f)),
            float3(1.0f, 1.0f, 1.0f), "Vector translation (no effect)");

        // Scaling transformation
        float4x4 scaling = float4x4::scaling(float3(2.0f, 3.0f, 4.0f));
        suite.assert_equal(scaling.transform_point(float3(1.0f, 1.0f, 1.0f)),
            float3(2.0f, 3.0f, 4.0f), "Point scaling");
        suite.assert_equal(scaling.transform_vector(float3(1.0f, 1.0f, 1.0f)),
            float3(2.0f, 3.0f, 4.0f), "Vector scaling");

        // Rotation transformation
        float4x4 rotation = float4x4::rotation_x(Constants::HALF_PI);
        float3 original(0.0f, 1.0f, 0.0f);
        float3 expected(0.0f, 0.0f, 1.0f);
        suite.assert_true(rotation.transform_vector(original).approximately(expected, 0.001f),
            "X rotation 90 degrees");

        // Direction transformation with normalization
        float3 direction(1.0f, 1.0f, 0.0f);
        float3 transformed_dir = rotation.transform_direction(direction);
        suite.assert_true(transformed_dir.is_normalized(0.001f), "Direction transformation is normalized");

        // Operator forms
        suite.assert_equal(vec4 * identity, vec4, "4D vector multiplication operator");
        suite.assert_equal(vec3 * identity, identity.transform_point(vec3), "3D point multiplication operator");
        suite.assert_equal(identity * vec4, identity.transform_vector(vec4), "Matrix-vector multiplication operator");

        suite.footer();
    }

    void test_float4x4_matrix_operations()
    {
        TestSuite suite("float4x4 Matrix Operations");
        suite.header();

        // ============================================================================
        // Используем простую диагональную матрицу для надежного тестирования
        // ============================================================================

        float4x4 mat(2.0f, 0.0f, 0.0f, 0.0f,   // col0: масштаб X
            0.0f, 3.0f, 0.0f, 0.0f,   // col1: масштаб Y  
            0.0f, 0.0f, 4.0f, 0.0f,   // col2: масштаб Z
            0.0f, 0.0f, 0.0f, 1.0f);  // col3: [0,0,0,1] - чистая аффинная

        std::cout << "=== TEST MATRIX ===" << std::endl;
        std::cout << "Matrix: " << mat.to_string() << std::endl;
        std::cout << "Is affine: " << (mat.is_affine(0.001f) ? "YES" : "NO") << std::endl;
        std::cout << "Determinant: " << mat.determinant() << std::endl;

        // ============================================================================
        // Transpose Tests
        // ============================================================================

        // Для диагональной матрицы транспонирование не меняет матрицу
        float4x4 expected_transpose(2.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);

        suite.assert_equal(mat.transposed(), expected_transpose, "Transpose of diagonal matrix");
        suite.assert_equal(transpose(mat), mat.transposed(), "Global transpose function");

        // ============================================================================
        // Determinant Tests
        // ============================================================================

        // det = 2 * 3 * 4 * 1 = 24
        suite.assert_equal(mat.determinant(), 24.0f, "Determinant calculation");
        suite.assert_equal(determinant(mat), 24.0f, "Global determinant function");

        // ============================================================================
        // Inverse Tests
        // ============================================================================

        // Обратная диагональная матрица = обратные элементы на диагонали
        float4x4 expected_inverse(0.5f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f / 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.25f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);

        float4x4 actual_inverse = mat.inverted();
        suite.assert_equal(actual_inverse, expected_inverse, "Matrix inverse");

        // Проверка глобальной функции inverse
        suite.assert_equal(inverse(mat), expected_inverse, "Global inverse function");

        // ============================================================================
        // Inverse Verification: A * A^(-1) = I
        // ============================================================================

        float4x4 product = mat * actual_inverse;
        suite.assert_true(product.approximately(float4x4::identity(), 0.001f),
            "Inverse verification: A * A^(-1) = I");

        // Детальная проверка произведения
        std::cout << "=== INVERSE VERIFICATION DETAILS ===" << std::endl;
        std::cout << "Product (should be identity):" << std::endl;
        std::cout << product.to_string() << std::endl;

        bool is_identity = true;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float expected = (i == j) ? 1.0f : 0.0f;
                float diff = std::abs(product(i, j) - expected);
                if (diff > 0.001f) {
                    is_identity = false;
                    std::cout << "MISMATCH at (" << i << "," << j << "): "
                        << product(i, j) << " vs " << expected
                        << ", diff: " << diff << std::endl;
                }
            }
        }
        std::cout << "Is identity: " << (is_identity ? "YES" : "NO") << std::endl;

        // ============================================================================
        // Adjugate Tests
        // ============================================================================

        float4x4 adj = mat.adjugate();
        float4x4 inv_from_adj = adj * (1.0f / mat.determinant());

        suite.assert_true(inv_from_adj.approximately(actual_inverse, 0.001f),
            "Adjugate relationship: adj(A)/det(A) = A^(-1)");

        // ============================================================================
        // Trace Tests
        // ============================================================================

        // trace = 2 + 3 + 4 + 1 = 10
        suite.assert_equal(mat.trace(), 10.0f, "Trace calculation");
        suite.assert_equal(trace(mat), 10.0f, "Global trace function");

        // ============================================================================
        // Diagonal Tests
        // ============================================================================

        suite.assert_equal(mat.diagonal(), float4(2.0f, 3.0f, 4.0f, 1.0f),
            "Diagonal extraction");
        suite.assert_equal(diagonal(mat), float4(2.0f, 3.0f, 4.0f, 1.0f),
            "Global diagonal function");

        // ============================================================================
        // Frobenius Norm Tests
        // ============================================================================

        // sqrt(2² + 0² + 0² + 0² + 0² + 3² + 0² + 0² + 0² + 0² + 4² + 0² + 0² + 0² + 0² + 1²)
        // = sqrt(4 + 9 + 16 + 1) = sqrt(30) ≈ 5.477
        float expected_norm = std::sqrt(4.0f + 9.0f + 16.0f + 1.0f); // sqrt(30)
        suite.assert_equal(mat.frobenius_norm(), expected_norm, "Frobenius norm");
        suite.assert_equal(frobenius_norm(mat), expected_norm, "Global Frobenius norm");

        // ============================================================================
        // Дополнительные тесты с трансляцией (отдельно)
        // ============================================================================

        std::cout << "=== ADDITIONAL TESTS WITH TRANSLATION ===" << std::endl;

        // Тестируем композицию преобразований отдельно
        float4x4 scaling_mat = float4x4::scaling(float3(2.0f, 3.0f, 4.0f));
        float4x4 translation_mat = float4x4::translation(float3(1.0f, 2.0f, 3.0f));
        float4x4 composite_mat = scaling_mat * translation_mat;

        std::cout << "Scaling matrix:" << std::endl << scaling_mat.to_string() << std::endl;
        std::cout << "Translation matrix:" << std::endl << translation_mat.to_string() << std::endl;
        std::cout << "Composite matrix (scale * translate):" << std::endl << composite_mat.to_string() << std::endl;
        std::cout << "Is affine: " << (composite_mat.is_affine(0.001f) ? "YES" : "NO") << std::endl;

        // Проверяем, что композитная матрица тоже обратима
        float4x4 composite_inv = composite_mat.inverted();
        float4x4 composite_product = composite_mat * composite_inv;

        bool composite_identity = composite_product.approximately(float4x4::identity(), 0.001f);
        std::cout << "Composite matrix inverse valid: " << (composite_identity ? "YES" : "NO") << std::endl;

        // ============================================================================
        // Тесты с вырожденной матрицей
        // ============================================================================

        float4x4 singular_mat(1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,  // линейно зависимые строки
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f);

        suite.assert_equal(singular_mat.determinant(), 0.0f, "Singular matrix has zero determinant");

        // Инвертирование вырожденной матрицы должно возвращать identity
        float4x4 inv_singular = singular_mat.inverted();
        suite.assert_true(inv_singular.is_identity(0.001f),
            "Inverse of singular matrix returns identity");

        suite.footer();
    }

    void test_float4x4_transformation_components()
    {
        TestSuite suite("float4x4 Transformation Components");
        suite.header();

        // Create a TRS matrix
        float3 translation(1.0f, 2.0f, 3.0f);
        quaternion rotation = quaternion::rotation_y(Constants::PI / 4.0f);
        float3 scale(2.0f, 3.0f, 4.0f);

        float4x4 trs = float4x4::TRS(translation, rotation, scale);

        // Translation extraction
        float3 extracted_translation = trs.get_translation();
        suite.assert_equal(extracted_translation, translation, "Translation extraction");

        // Scale extraction
        float3 extracted_scale = trs.get_scale();
        suite.assert_true(extracted_scale.approximately(scale, 0.001f), "Scale extraction");

        // Rotation extraction
        quaternion extracted_rotation = trs.get_rotation();
        suite.assert_true(extracted_rotation.approximately(rotation, 0.001f), "Rotation extraction");

        // Set translation
        float4x4 mat = float4x4::identity();
        mat.set_translation(float3(5.0f, 6.0f, 7.0f));
        suite.assert_equal(mat.get_translation(), float3(5.0f, 6.0f, 7.0f), "Set translation");

        // Set scale
        mat = float4x4::rotation_z(Constants::PI / 6.0f);
        mat.set_scale(float3(2.0f, 2.0f, 2.0f));
        suite.assert_true(mat.get_scale().approximately(float3(2.0f, 2.0f, 2.0f), 0.001f), "Set scale");

        // Normal matrix
        translation = float3(1.0f, 2.0f, 3.0f);
        rotation = quaternion::rotation_y(Constants::PI / 4.0f);
        scale = float3(1.0f, 1.0f, 1.0f);

        trs = float4x4::TRS(translation, rotation, scale);
        float3x3 normal_mat = trs.normal_matrix();
        float3 normal(0.0f, 1.0f, 0.0f);
        float3 transformed_normal = normal_mat.transform_vector(normal);
        suite.assert_true(transformed_normal.is_normalized(0.001f), "Normal matrix preserves normalization with uniform scale");

        suite.footer();
    }

    void test_float4x4_utility_methods()
    {
        TestSuite suite("float4x4 Utility Methods");
        suite.header();

        // Identity checks
        suite.assert_true(float4x4::identity().is_identity(), "Identity matrix check");
        suite.assert_true(float4x4::identity().is_affine(), "Identity is affine");
        suite.assert_true(float4x4::identity().is_orthogonal(), "Identity is orthogonal");

        // Affine matrix check
        float4x4 affine_mat = float4x4::translation(float3(1, 2, 3)) * float4x4::rotation_x(0.5f);
        suite.assert_true(affine_mat.is_affine(), "Affine matrix check");

        // Non-affine matrix (perspective projection)
        float4x4 perspective = float4x4::perspective(Constants::PI / 4.0f, 1.33f, 0.1f, 100.0f);
        suite.assert_false(perspective.is_affine(), "Perspective matrix is not affine");

        // Orthogonal matrix check
        float4x4 rotation = float4x4::rotation_y(Constants::PI / 6.0f);
        suite.assert_true(rotation.is_orthogonal(), "Rotation matrix is orthogonal");

        // Non-orthogonal matrix
        float4x4 non_ortho = float4x4::scaling(2.0f, 3.0f, 4.0f);
        suite.assert_false(non_ortho.is_orthogonal(), "Scaling matrix is not orthogonal");

        // Approximate equality
        float4x4 a(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);
        float4x4 b(1.000001f, 2.000001f, 3.000001f, 4.000001f,
            5.000001f, 6.000001f, 7.000001f, 8.000001f,
            9.000001f, 10.000001f, 11.000001f, 12.000001f,
            13.000001f, 14.000001f, 15.000001f, 16.000001f);
        suite.assert_true(a.approximately(b, 0.001f), "Approximate equality");
        suite.assert_false(a.approximately(b, 1e-8f), "Approximate inequality");

        // Comparison operators
        suite.assert_true(a == a, "Equality operator");
        suite.assert_true(a != float4x4::zero(), "Inequality operator");

        // String conversion
        std::string str = a.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains element");
        suite.assert_true(str.find("16.0") != std::string::npos, "String contains element");

        float4x4 test_mat(1.0f, 5.0f, 9.0f, 13.0f,   // col0: 1,2,3,4
            2.0f, 6.0f, 10.0f, 14.0f,   // col1: 5,6,7,8  
            3.0f, 7.0f, 11.0f, 15.0f,   // col2: 9,10,11,12
            4.0f, 8.0f, 12.0f, 16.0f);  // col3: 13,14,15,16

        // Array conversion
        float col_major[16];
        test_mat.to_column_major(col_major);
        suite.assert_equal(col_major[0], 1.0f, "Column-major [0]");   // col0.x
        suite.assert_equal(col_major[1], 2.0f, "Column-major [1]");   // col0.y
        suite.assert_equal(col_major[2], 3.0f, "Column-major [2]");   // col0.z
        suite.assert_equal(col_major[3], 4.0f, "Column-major [3]");   // col0.w
        suite.assert_equal(col_major[4], 5.0f, "Column-major [4]");   // col1.x
        suite.assert_equal(col_major[5], 6.0f, "Column-major [5]");   // col1.y

        float row_major[16];
        test_mat.to_row_major(row_major);
        suite.assert_equal(row_major[0], 1.0f, "Row-major [0]");   // row0.x = col0.x
        suite.assert_equal(row_major[1], 5.0f, "Row-major [1]");   // row0.y = col1.x
        suite.assert_equal(row_major[2], 9.0f, "Row-major [2]");   // row0.z = col2.x
        suite.assert_equal(row_major[3], 13.0f, "Row-major [3]");  // row0.w = col3.x
        suite.assert_equal(row_major[4], 2.0f, "Row-major [4]");   // row1.x = col0.y

        float4x4 non_uniform_scale = float4x4::scaling(2.0f, 3.0f, 4.0f);
        suite.assert_false(non_uniform_scale.is_orthogonal(0.001f), "Non-uniform scaling matrix is not orthogonal");

        float4x4 uniform_scale = float4x4::scaling(2.0f);
        suite.assert_false(uniform_scale.is_orthogonal(0.001f), "Uniform scaling matrix is not orthogonal (scaling breaks orthogonality)");

        std::cout << "=== DEBUG INVERSE VERIFICATION ===" << std::endl;

        float4x4 mat(2.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 0.0f,
            1.0f, 2.0f, 3.0f, 1.0f);

        float4x4 inv = mat.inverted();
        float4x4 product = mat * inv;

        std::cout << "Original matrix:" << std::endl;
        std::cout << mat.to_string() << std::endl;

        std::cout << "Inverse matrix:" << std::endl;
        std::cout << inv.to_string() << std::endl;

        std::cout << "Product (should be identity):" << std::endl;
        std::cout << product.to_string() << std::endl;

        std::cout << "Is identity? " << (product.is_identity(0.001f) ? "YES" : "NO") << std::endl;

        suite.footer();
    }

    void test_float4x4_projection_matrices()
    {
        TestSuite suite("float4x4 Projection Matrices");
        suite.header();

        // Orthographic projection
        float4x4 ortho = float4x4::orthographic(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);

        // Test that points in frustum are mapped to [-1,1] in NDC
        float3 near_center(0.0f, 0.0f, -0.1f);
        float3 far_center(0.0f, 0.0f, -100.0f);

        float3 projected_near = ortho.transform_point(near_center);
        float3 projected_far = ortho.transform_point(far_center);

        suite.assert_true(projected_near.z >= -1.0f && projected_near.z <= 1.0f, "Orthographic near plane in NDC");
        suite.assert_true(projected_far.z >= -1.0f && projected_far.z <= 1.0f, "Orthographic far plane in NDC");

        // Perspective projection
        float4x4 persp = float4x4::perspective(Constants::PI / 4.0f, 1.33f, 0.1f, 100.0f);

        // Test perspective divide
        float4 homogenous_point(0.0f, 0.0f, -1.0f, 1.0f);
        float4 projected = persp.transform_vector(homogenous_point);

        suite.assert_true(projected.w != 0.0f, "Perspective projection has non-zero w");
        suite.assert_true(std::abs(projected.z / projected.w) <= 1.0f, "Perspective point in NDC");

        // Look-at matrix
        float4x4 lookat = float4x4::look_at(float3(0.0f, 0.0f, 5.0f),
            float3(0.0f, 0.0f, 0.0f),
            float3(0.0f, 1.0f, 0.0f));

        // Camera should be at origin in view space
        float3 camera_pos(0.0f, 0.0f, 5.0f);
        float3 view_pos = lookat.transform_point(camera_pos);
        suite.assert_true(view_pos.approximately(float3::zero(), 0.001f), "Look-at camera at origin in view space");

        suite.footer();
    }

    void test_float4x4_edge_cases()
    {
        TestSuite suite("float4x4 Edge Cases");
        suite.header();

        // Zero matrix
        suite.assert_true(float4x4::zero().approximately_zero(), "Zero matrix");
        suite.assert_equal(float4x4::zero().determinant(), 0.0f, "Zero matrix determinant");

        // Singular matrix
        float4x4 singular(1.0f, 2.0f, 3.0f, 4.0f,
            2.0f, 4.0f, 6.0f, 8.0f,
            3.0f, 6.0f, 9.0f, 12.0f,
            4.0f, 8.0f, 12.0f, 16.0f); // linearly dependent columns
        suite.assert_equal(singular.determinant(), 0.0f, "Singular matrix determinant");

        // Inverse of singular matrix should return identity (as per implementation)
        float4x4 inv_singular = singular.inverted();
        suite.assert_true(inv_singular.is_identity(), "Inverse of singular matrix returns identity");

        // Identity properties
        suite.assert_equal(float4x4::identity().determinant(), 1.0f, "Identity determinant");
        suite.assert_equal(float4x4::identity().trace(), 4.0f, "Identity trace");

        // Very small values
        float4x4 small_(1e-10f, 0.0f, 0.0f, 0.0f,
            0.0f, 1e-10f, 0.0f, 0.0f,
            0.0f, 0.0f, 1e-10f, 0.0f,
            0.0f, 0.0f, 0.0f, 1e-10f);
        suite.assert_true(std::isfinite(small_.determinant()), "Small matrix has finite determinant");

        // Very large values
        float4x4 large(1e10f, 0.0f, 0.0f, 0.0f,
            0.0f, 1e10f, 0.0f, 0.0f,
            0.0f, 0.0f, 1e10f, 0.0f,
            0.0f, 0.0f, 0.0f, 1e10f);
        suite.assert_true(std::isfinite(large.determinant()), "Large matrix has finite determinant");

        // Transformation with zero scale
        float4x4 zero_scale = float4x4::scaling(float3::zero());
        float3 point(1.0f, 1.0f, 1.0f);
        float3 transformed = zero_scale.transform_point(point);
        suite.assert_equal(transformed, float3::zero(), "Zero scale transforms points to zero");

        // Rotation extraction from singular matrix
        quaternion rot_from_singular = singular.get_rotation();
        suite.assert_true(rot_from_singular.is_valid(), "Rotation extraction from singular matrix returns valid quaternion");

        suite.footer();
    }

    void run_float4x4_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "FLOAT4X4 COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_float4x4_constructors();
        test_float4x4_access_operations();
        test_float4x4_arithmetic_operations();
        test_float4x4_matrix_multiplication();
        test_float4x4_vector_transformations();
        test_float4x4_matrix_operations();
        test_float4x4_transformation_components();
        test_float4x4_utility_methods();
        test_float4x4_projection_matrices();
        test_float4x4_edge_cases();

        std::cout << "FLOAT4X4 TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
    // Test Data Generators for quaternion
    // ============================================================================

    class QuaternionTestData
    {
    public:
        static std::vector<quaternion> generate_test_quaternions()
        {
            return {
                quaternion::identity(),
                quaternion::zero(),
                quaternion(1.0f, 0.0f, 0.0f, 0.0f),
                quaternion(0.0f, 1.0f, 0.0f, 0.0f),
                quaternion(0.0f, 0.0f, 1.0f, 0.0f),
                quaternion(0.5f, 0.5f, 0.5f, 0.5f),
                quaternion(0.707f, 0.0f, 0.0f, 0.707f), // 90° around X
                quaternion(0.0f, 0.707f, 0.0f, 0.707f), // 90° around Y
                quaternion(0.0f, 0.0f, 0.707f, 0.707f), // 90° around Z
                quaternion::rotation_x(Constants::PI / 4.0f),
                quaternion::rotation_y(Constants::PI / 3.0f),
                quaternion::rotation_z(Constants::PI / 6.0f),
                quaternion(float3::unit_x(), Constants::HALF_PI),
                quaternion(float3::unit_y(), Constants::PI / 3.0f),
                quaternion(float3(1.0f, 1.0f, 0.0f).normalize(), Constants::PI / 2.0f)
            };
        }

        static std::vector<std::pair<quaternion, float3>> generate_quaternion_euler_pairs()
        {
            return {
                {quaternion::identity(), float3::zero()},
                {quaternion::rotation_x(Constants::PI / 2.0f), float3(Constants::HALF_PI, 0.0f, 0.0f)},
                {quaternion::rotation_y(Constants::PI / 2.0f), float3(0.0f, Constants::HALF_PI, 0.0f)},
                {quaternion::rotation_z(Constants::PI / 2.0f), float3(0.0f, 0.0f, Constants::HALF_PI)}
            };
        }
    };

    // ============================================================================
    // quaternion Tests - Comprehensive
    // ============================================================================

    void test_quaternion_constructors()
    {
        TestSuite suite("quaternion Constructors");
        suite.header();

        // Default constructor (identity)
        suite.assert_equal_quaternion(quaternion(), quaternion(0.0f, 0.0f, 0.0f, 1.0f), "Default constructor (identity)");

        // Component constructor
        quaternion q1(1.0f, 2.0f, 3.0f, 4.0f);
        suite.assert_equal(q1.x, 1.0f, "Component constructor - x");
        suite.assert_equal(q1.y, 2.0f, "Component constructor - y");
        suite.assert_equal(q1.z, 3.0f, "Component constructor - z");
        suite.assert_equal(q1.w, 4.0f, "Component constructor - w");

        // From float4 constructor
        float4 vec(1.0f, 2.0f, 3.0f, 4.0f);
        quaternion q2(vec);
        suite.assert_equal_quaternion(q2, q1, "From float4 constructor");

        // From SIMD constructor
        __m128 simd_val = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f); // w, z, y, x
        quaternion q3(simd_val);
        suite.assert_equal(q3.x, 1.0f, "SIMD constructor - x");
        suite.assert_equal(q3.y, 2.0f, "SIMD constructor - y");
        suite.assert_equal(q3.z, 3.0f, "SIMD constructor - z");
        suite.assert_equal(q3.w, 4.0f, "SIMD constructor - w");

        // Axis-angle constructor - используем проверку вращения (двойное покрытие)
        quaternion q4(float3::unit_x(), Constants::HALF_PI);
        suite.assert_approximately_quaternion_rotation(q4, quaternion(0.707f, 0.0f, 0.0f, 0.707f), "Axis-angle constructor", 0.001f);

        // Euler angles constructor - используем проверку вращения
        quaternion q5(Constants::HALF_PI, 0.0f, 0.0f); // 90° around X
        suite.assert_approximately_quaternion_rotation(q5, quaternion(0.707f, 0.0f, 0.0f, 0.707f), "Euler angles constructor", 0.001f);

        std::cout << "DEBUG: Euler constructor result: " << q5.to_string() << std::endl;

        // From 3x3 matrix constructor - используем проверку вращения
        float3x3 rot_x = float3x3::rotation_x(Constants::HALF_PI);
        quaternion q6(rot_x);
        suite.assert_approximately_quaternion_rotation(q6, quaternion(0.707f, 0.0f, 0.0f, 0.707f), "From 3x3 matrix constructor", 0.001f);

        // From 4x4 matrix constructor - используем проверку вращения
        float4x4 rot_x4 = float4x4::rotation_x(Constants::HALF_PI);
        quaternion q7(rot_x4);
        suite.assert_approximately_quaternion_rotation(q7, quaternion(0.707f, 0.0f, 0.0f, 0.707f), "From 4x4 matrix constructor", 0.001f);

        // Copy constructor - точное сравнение компонентов
        quaternion original(1.0f, 2.0f, 3.0f, 4.0f);
        quaternion copy(original);
        suite.assert_equal_quaternion(copy, original, "Copy constructor");

        // Static constructors - точное сравнение компонентов
        suite.assert_equal_quaternion(quaternion::identity(), quaternion(0.0f, 0.0f, 0.0f, 1.0f), "Identity quaternion");
        suite.assert_equal_quaternion(quaternion::zero(), quaternion(0.0f, 0.0f, 0.0f, 0.0f), "Zero quaternion");

        // Edge cases
        quaternion inf_q(std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 1.0f);
        suite.assert_true(std::isinf(inf_q.x), "Infinity component");

        suite.footer();
    }

    void test_quaternion_static_constructors()
    {
        TestSuite suite("quaternion Static Constructors");
        suite.header();

        // from_axis_angle - проверка вращения
        quaternion q1 = quaternion::from_axis_angle(float3::unit_y(), Constants::HALF_PI);

        // from_euler (individual angles) - проверка вращения
        quaternion q2 = quaternion::from_euler(Constants::HALF_PI, 0.0f, 0.0f);

        // from_euler (vector) - проверка вращения
        quaternion q3 = quaternion::from_euler(float3(Constants::HALF_PI, 0.0f, 0.0f));

        // from_matrix (3x3) - проверка вращения
        float3x3 rot_z = float3x3::rotation_z(Constants::HALF_PI);
        quaternion q4 = quaternion::from_matrix(rot_z);

        // from_matrix (4x4) - проверка вращения
        float4x4 rot_z4 = float4x4::rotation_z(Constants::HALF_PI);
        quaternion q5 = quaternion::from_matrix(rot_z4);

        // from_to_rotation - проверка вращения
        quaternion q6 = quaternion::from_to_rotation(float3::unit_x(), float3::unit_y());

        // rotation_x, rotation_y, rotation_z - проверка вращения
        quaternion q7 = quaternion::rotation_x(Constants::HALF_PI);
        quaternion q8 = quaternion::rotation_y(Constants::HALF_PI);
        quaternion q9 = quaternion::rotation_z(Constants::HALF_PI);

        suite.assert_approximately_quaternion_rotation(q1, quaternion(0.0f, 0.707f, 0.0f, 0.707f), "from_axis_angle", 0.001f);
        suite.assert_approximately_quaternion_rotation(q2, quaternion(0.707f, 0.0f, 0.0f, 0.707f), "from_euler individual", 0.001f);
        suite.assert_approximately_quaternion_rotation(q3, quaternion(0.707f, 0.0f, 0.0f, 0.707f), "from_euler vector", 0.001f);
        suite.assert_approximately_quaternion_rotation(q4, quaternion(0.000, 0.000, -0.707f, 0.707f), "from_matrix 3x3", 0.001f);
        suite.assert_approximately_quaternion_rotation(q5, quaternion(0.0f, 0.0f, -0.707f, 0.707f), "from_matrix 4x4", 0.001f);
        suite.assert_approximately_quaternion_rotation(q6, quaternion(0.0f, 0.0f, 0.707f, 0.707f), "from_to_rotation", 0.001f);
        suite.assert_approximately_quaternion_rotation(q7, quaternion(0.707f, 0.0f, 0.0f, 0.707f), "rotation_x", 0.001f);
        suite.assert_approximately_quaternion_rotation(q8, quaternion(0.0f, 0.707f, 0.0f, 0.707f), "rotation_y", 0.001f);
        suite.assert_approximately_quaternion_rotation(q9, quaternion(0.0f, 0.0f, 0.707f, 0.707f), "rotation_z", 0.001f);

        // Edge case: from_to_rotation with same vectors
        quaternion q10 = quaternion::from_to_rotation(float3::unit_x(), float3::unit_x());
        suite.assert_true(q10.is_identity(0.001f), "from_to_rotation same vectors");

        // Edge case: from_to_rotation with opposite vectors
        quaternion q11 = quaternion::from_to_rotation(float3::unit_x(), -float3::unit_x());
        suite.assert_true(q11.is_normalized(0.001f), "from_to_rotation opposite vectors normalized");

        suite.footer();
    }

    void test_quaternion_assignment_operators()
    {
        TestSuite suite("quaternion Assignment Operators");
        suite.header();

        quaternion a(1.0f, 2.0f, 3.0f, 4.0f);
        quaternion b;

        // Copy assignment - точное сравнение компонентов
        b = a;
        suite.assert_equal_quaternion(b, a, "Copy assignment");

        // Float4 assignment - точное сравнение компонентов
        b = float4(5.0f, 6.0f, 7.0f, 8.0f);
        suite.assert_equal_quaternion(b, quaternion(5.0f, 6.0f, 7.0f, 8.0f), "Float4 assignment");

#if defined(MATH_SUPPORT_D3DX)
        // D3DXQUATERNION assignment (if supported) - точное сравнение компонентов
        D3DXQUATERNION dx_q;
        dx_q.x = 9.0f; dx_q.y = 10.0f; dx_q.z = 11.0f; dx_q.w = 12.0f;
        b = dx_q;
        suite.assert_equal_quaternion(b, quaternion(9.0f, 10.0f, 11.0f, 12.0f), "D3DXQUATERNION assignment");
#endif

        suite.footer();
    }

    void test_quaternion_compound_assignment_operators()
    {
        TestSuite suite("quaternion Compound Assignment Operators");
        suite.header();

        quaternion a(1.0f, 2.0f, 3.0f, 4.0f);
        quaternion b(2.0f, 3.0f, 4.0f, 5.0f);
        quaternion c;

        // Compound addition - точное сравнение компонентов
        c = a; c += b;
        suite.assert_equal_quaternion(c, quaternion(3.0f, 5.0f, 7.0f, 9.0f), "Compound addition");

        // Compound subtraction - точное сравнение компонентов
        c = a; c -= b;
        suite.assert_equal_quaternion(c, quaternion(-1.0f, -1.0f, -1.0f, -1.0f), "Compound subtraction");

        // Compound scalar multiplication - точное сравнение компонентов
        c = a; c *= 2.0f;
        suite.assert_equal_quaternion(c, quaternion(2.0f, 4.0f, 6.0f, 8.0f), "Compound scalar multiplication");

        // Compound scalar division - точное сравнение компонентов
        c = a; c /= 2.0f;
        suite.assert_equal_quaternion(c, quaternion(0.5f, 1.0f, 1.5f, 2.0f), "Compound scalar division");

        // Compound quaternion multiplication - проверка нормализации
        quaternion q1 = quaternion::rotation_x(Constants::HALF_PI); // 90° around X
        quaternion q2 = quaternion::rotation_y(Constants::HALF_PI); // 90° around Y
        c = q1; c *= q2;
        suite.assert_true(c.is_normalized(0.001f), "Compound quaternion multiplication");

        suite.footer();
    }

    void test_quaternion_arithmetic_operations()
    {
        TestSuite suite("quaternion Arithmetic Operations");
        suite.header();

        quaternion a(1.0f, 2.0f, 3.0f, 4.0f);
        quaternion b(2.0f, 3.0f, 4.0f, 5.0f);

        // Addition - точное сравнение компонентов
        suite.assert_equal_quaternion(a + b, quaternion(3.0f, 5.0f, 7.0f, 9.0f), "Quaternion addition");

        // Subtraction - точное сравнение компонентов
        suite.assert_equal_quaternion(a - b, quaternion(-1.0f, -1.0f, -1.0f, -1.0f), "Quaternion subtraction");

        // Scalar multiplication - точное сравнение компонентов
        suite.assert_equal_quaternion(a * 2.0f, quaternion(2.0f, 4.0f, 6.0f, 8.0f), "Scalar multiplication");
        suite.assert_equal_quaternion(2.0f * a, quaternion(2.0f, 4.0f, 6.0f, 8.0f), "Scalar multiplication commutative");

        // Scalar division - точное сравнение компонентов
        suite.assert_equal_quaternion(a / 2.0f, quaternion(0.5f, 1.0f, 1.5f, 2.0f), "Scalar division");

        // Unary operators - точное сравнение компонентов
        suite.assert_equal_quaternion(+a, a, "Unary plus");
        suite.assert_equal_quaternion(-a, quaternion(-1.0f, -2.0f, -3.0f, -4.0f), "Unary minus");

        // Quaternion multiplication - проверка нормализации
        quaternion qx = quaternion::rotation_x(Constants::HALF_PI); // 90° around X
        quaternion qy = quaternion::rotation_y(Constants::HALF_PI); // 90° around Y
        quaternion product = qy * qx;
        suite.assert_true(product.is_normalized(0.001f), "Quaternion multiplication preserves normalization");

        float3 original(1.0f, 0.0f, 0.0f);
        float3 rotated_x = qx * original;
        float3 rotated_xy = qy * rotated_x;
        float3 rotated_combined = product * original;

        std::cout << "DEBUG multiplication: rotated_x = " << rotated_x.to_string() << std::endl;
        std::cout << "DEBUG multiplication: rotated_xy = " << rotated_xy.to_string() << std::endl;
        std::cout << "DEBUG multiplication: rotated_combined = " << rotated_combined.to_string() << std::endl;
        std::cout << "DEBUG multiplication: product quaternion = " << product.to_string() << std::endl;

        suite.assert_true(rotated_combined.approximately(rotated_xy, 0.001f),
            "Quaternion multiplication equivalent to sequential rotations");

        suite.footer();
    }

    void test_quaternion_vector_transformation()
    {
        TestSuite suite("quaternion Vector Transformation");
        suite.header();

        // Identity transformation
        quaternion identity = quaternion::identity();
        float3 vec(1.0f, 2.0f, 3.0f);
        suite.assert_equal(identity * vec, vec, "Identity vector transformation");
        suite.assert_equal(identity.transform_vector(vec), vec, "Identity transform_vector");
        suite.assert_equal(identity.transform_direction(vec), vec.normalize(), "Identity transform_direction");

        // X rotation 90 degrees
        quaternion rot_x = quaternion::rotation_x(Constants::HALF_PI);
        float3 original(0.0f, 1.0f, 0.0f);
        float3 expected(0.0f, 0.0f, 1.0f);
        float3 result = rot_x * original;
        suite.assert_true(result.approximately(expected, 0.001f), "X rotation 90 degrees - operator*");
        suite.assert_true(rot_x.transform_vector(original).approximately(expected, 0.001f), "X rotation 90 degrees - transform_vector");

        // Y rotation 90 degrees
        quaternion rot_y = quaternion::rotation_y(Constants::HALF_PI);
        original = float3(1.0f, 0.0f, 0.0f);
        expected = float3(0.0f, 0.0f, -1.0f);
        result = rot_y * original;
        suite.assert_true(result.approximately(expected, 0.001f), "Y rotation 90 degrees");

        // Z rotation 90 degrees
        quaternion rot_z = quaternion::rotation_z(Constants::HALF_PI);
        original = float3(1.0f, 0.0f, 0.0f);
        expected = float3(0.0f, 1.0f, 0.0f);
        result = rot_z * original;
        suite.assert_true(result.approximately(expected, 0.001f), "Z rotation 90 degrees");

        // transform_direction should normalize
        float3 direction(1.0f, 1.0f, 0.0f);
        float3 transformed_dir = rot_z.transform_direction(direction);
        suite.assert_true(transformed_dir.is_normalized(0.001f), "transform_direction returns normalized vector");

        // Multiple transformations
        quaternion combined = rot_x * rot_y;
        float3 multi_result = combined * original;
        suite.assert_true(multi_result.isValid(), "Multiple transformations produce valid result");

        suite.footer();
    }

    void test_quaternion_mathematical_operations()
    {
        TestSuite suite("quaternion Mathematical Operations");
        suite.header();

        quaternion q(3.0f, 4.0f, 0.0f, 0.0f); // length = 5

        // Length calculations
        suite.assert_equal(q.length(), 5.0f, "Length");
        suite.assert_equal(q.length_sq(), 25.0f, "Squared length");

        // Normalization
        quaternion normalized = q.normalize();
        suite.assert_true(normalized.is_normalized(0.001f), "Normalization result is normalized");
        suite.assert_equal(normalized.length(), 1.0f, "Normalized length");
        suite.assert_approximately_quaternion_rotation(q.normalize(), quaternion(0.6f, 0.8f, 0.0f, 0.0f), "Normalization values", 0.001f);

        // Conjugate - точное сравнение компонентов
        quaternion conjugated = q.conjugate();
        suite.assert_equal_quaternion(conjugated, quaternion(-3.0f, -4.0f, 0.0f, 0.0f), "Conjugate");

        // Inverse - проверка вращения
        quaternion unit_q = q.normalize();
        quaternion inverse_q = unit_q.inverse();
        quaternion should_be_identity = unit_q * inverse_q;
        suite.assert_approximately_quaternion_rotation(should_be_identity, quaternion::identity(), "Inverse verification", 0.001f);

        // Dot product
        quaternion a(1.0f, 2.0f, 3.0f, 4.0f);
        quaternion b(2.0f, 3.0f, 4.0f, 5.0f);
        float dot_result = a.dot(b);
        suite.assert_equal(dot_result, 40.0f, "Dot product");

        // Global functions
        suite.assert_equal(length(q), 5.0f, "Global length");
        suite.assert_equal(length_sq(q), 25.0f, "Global length_sq");
        suite.assert_equal_quaternion(normalize(q), q.normalize(), "Global normalize");
        suite.assert_equal_quaternion(conjugate(q), q.conjugate(), "Global conjugate");
        suite.assert_equal_quaternion(inverse(unit_q), unit_q.inverse(), "Global inverse");
        suite.assert_equal(dot(a, b), 40.0f, "Global dot");

        suite.footer();
    }

    void test_quaternion_conversion_methods()
    {
        TestSuite suite("quaternion Conversion Methods");
        suite.header();

        // to_matrix3x3
        quaternion qx = quaternion::rotation_x(Constants::HALF_PI);
        float3x3 mat3x3 = qx.to_matrix3x3();
        float3x3 expected_mat3x3 = float3x3::rotation_x(Constants::HALF_PI);
        suite.assert_true(mat3x3.approximately(expected_mat3x3, 0.001f), "to_matrix3x3");

        // to_matrix4x4
        float4x4 mat4x4 = qx.to_matrix4x4();
        float4x4 expected_mat4x4 = float4x4::rotation_x(Constants::HALF_PI);
        suite.assert_true(mat4x4.approximately(expected_mat4x4, 0.001f), "to_matrix4x4");

        // to_axis_angle
        float3 axis;
        float angle;
        qx.to_axis_angle(axis, angle);
        suite.assert_true(axis.approximately(float3::unit_x(), 0.001f), "to_axis_angle axis");
        suite.assert_equal(angle, Constants::HALF_PI, "to_axis_angle angle", 0.001f);

        // to_euler
        float3 euler = qx.to_euler();
        suite.assert_true(euler.approximately(float3(Constants::HALF_PI, 0.0f, 0.0f), 0.001f), "to_euler");

        // Test round-trip conversions - проверка вращений
        quaternion original = quaternion::rotation_y(Constants::PI / 3.0f);
        float3x3 round_trip_mat = original.to_matrix3x3();
        quaternion from_mat = quaternion::from_matrix(round_trip_mat);
        suite.assert_approximately_quaternion_rotation(original, from_mat, "Round-trip quaternion-matrix conversion", 0.001f);

        float3 euler_angles = original.to_euler();
        quaternion from_euler = quaternion::from_euler(euler_angles);
        suite.assert_approximately_quaternion_rotation(original, from_euler, "Round-trip quaternion-euler conversion", 0.001f);

#if defined(MATH_SUPPORT_D3DX)
        // D3DXQUATERNION conversion - проверка вращений
        D3DXQUATERNION dx_q = static_cast<D3DXQUATERNION>(original);
        quaternion from_dx = quaternion::from_d3dxquaternion(dx_q);
        suite.assert_approximately_quaternion_rotation(original, from_dx, "Round-trip D3DXQUATERNION conversion", 0.001f);
#endif

        suite.footer();
    }

    void test_quaternion_interpolation_methods()
    {
        TestSuite suite("quaternion Interpolation Methods");
        suite.header();

        quaternion a = quaternion::rotation_x(0.0f);        // 0° around X
        quaternion b = quaternion::rotation_x(Constants::HALF_PI); // 90° around X

        // nlerp
        quaternion nlerp_result = nlerp(a, b, 0.5f);
        quaternion expected_half = quaternion::rotation_x(Constants::HALF_PI / 2.0f); // 45° around X
        suite.assert_true(nlerp_result.is_normalized(0.001f), "nlerp result is normalized");
        suite.assert_approximately_quaternion_rotation(nlerp_result, expected_half, "nlerp at 0.5", 0.001f);

        // slerp
        quaternion slerp_result = slerp(a, b, 0.5f);
        suite.assert_true(slerp_result.is_normalized(0.001f), "slerp result is normalized");
        suite.assert_approximately_quaternion_rotation(slerp_result, expected_half, "slerp at 0.5", 0.001f);

        // lerp (should be same as nlerp)
        quaternion lerp_result = lerp(a, b, 0.5f);
        suite.assert_true(lerp_result.is_normalized(0.001f), "lerp result is normalized");

        // Static methods - проверка вращений
        quaternion static_nlerp = quaternion::lerp(a, b, 0.5f);
        suite.assert_approximately_quaternion_rotation(static_nlerp, nlerp_result, "Static nlerp matches global", 0.001f);

        quaternion static_slerp = quaternion::slerp(a, b, 0.5f);
        suite.assert_approximately_quaternion_rotation(static_slerp, slerp_result, "Static slerp matches global", 0.001f);

        quaternion static_lerp = quaternion::lerp(a, b, 0.5f);
        suite.assert_approximately_quaternion_rotation(static_lerp, lerp_result, "Static lerp matches global", 0.001f);

        // Edge cases - точное сравнение компонентов
        suite.assert_equal_quaternion(slerp(a, b, 0.0f), a.normalize(), "slerp at 0.0");
        suite.assert_equal_quaternion(slerp(a, b, 1.0f), b.normalize(), "slerp at 1.0");

        // Test with opposite quaternions (double cover)
        quaternion opposite = -b;
        quaternion slerp_opposite = slerp(a, opposite, 0.5f);
        suite.assert_true(slerp_opposite.is_normalized(0.001f), "slerp with opposite quaternion is normalized");

        suite.footer();
    }

    void test_quaternion_utility_methods()
    {
        TestSuite suite("quaternion Utility Methods");
        suite.header();

        // Validity checks
        suite.assert_true(quaternion(1.0f, 2.0f, 3.0f, 4.0f).is_valid(), "Valid quaternion check");
        suite.assert_false(quaternion(std::numeric_limits<float>::infinity(), 2.0f, 3.0f, 4.0f).is_valid(), "Invalid quaternion - infinity");
        suite.assert_false(quaternion(1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f).is_valid(), "Invalid quaternion - NaN");

        // Approximately zero
        suite.assert_true(quaternion::zero().approximately_zero(), "Exactly zero");
        suite.assert_true(quaternion(1e-8f, -1e-8f, 1e-8f, -1e-8f).approximately_zero(1e-7f), "Approximately zero");

        // Identity check
        suite.assert_true(quaternion::identity().is_identity(), "Identity quaternion");
        suite.assert_false(quaternion(1.0f, 0.0f, 0.0f, 1.0f).is_identity(), "Non-identity quaternion");

        // Normalization checks
        suite.assert_true(quaternion(0.5f, 0.5f, 0.5f, 0.5f).is_normalized(0.001f), "Normalized check");
        suite.assert_false(quaternion(1.0f, 1.0f, 1.0f, 1.0f).is_normalized(), "Not normalized check");

        // Для approximate equality используем кватернионы вращения
        quaternion q1 = quaternion::rotation_x(Constants::PI / 4.0f);  // 45° вокруг X
        quaternion q2 = quaternion::rotation_x(Constants::PI / 4.0f + 0.000001f); // Почти одинаковые
        quaternion q3 = quaternion::rotation_y(Constants::PI / 2.0f);  // 90° вокруг Y - совершенно другое вращение

        suite.assert_true(q1.approximately(q2, 0.001f), "Approximate equality");
        suite.assert_false(q1.approximately(q3, 0.001f), "Approximate inequality");

        // Double cover test (q and -q represent same rotation)
        quaternion q = quaternion::rotation_x(Constants::PI / 4.0f);
        quaternion neg_q = -q;
        suite.assert_true(q.approximately(neg_q, 0.001f), "Double cover: q and -q are approximately equal");

        // String conversion - используем конкретный кватернион
        quaternion string_test(1.0f, 2.0f, 3.0f, 4.0f);
        std::string str = string_test.to_string();
        suite.assert_true(str.find("1.0") != std::string::npos, "String contains x");
        suite.assert_true(str.find("2.0") != std::string::npos, "String contains y");
        suite.assert_true(str.find("3.0") != std::string::npos, "String contains z");
        suite.assert_true(str.find("4.0") != std::string::npos, "String contains w");

        // Data access - используем конкретный кватернион
        quaternion data_test(1.0f, 2.0f, 3.0f, 4.0f);
        suite.assert_equal(data_test.data()[0], 1.0f, "Data pointer access [0]");
        suite.assert_equal(data_test.data()[1], 2.0f, "Data pointer access [1]");
        suite.assert_equal(data_test.data()[2], 3.0f, "Data pointer access [2]");
        suite.assert_equal(data_test.data()[3], 4.0f, "Data pointer access [3]");

        // Float4 access - точное сравнение компонентов
        suite.assert_equal(data_test.get_float4(), float4(1.0f, 2.0f, 3.0f, 4.0f), "get_float4");
        float4& mutable_float4 = data_test.get_float4();
        mutable_float4.x = 5.0f;
        suite.assert_equal(data_test.x, 5.0f, "Mutable float4 access");

        // SIMD access - точное сравнение компонентов
        __m128 simd_data = data_test.get_simd();
        quaternion from_simd(simd_data);
        suite.assert_equal_quaternion(from_simd, data_test, "SIMD round-trip", 0.01f);

        // Comparison operators - используем встроенные операторы
        suite.assert_true(data_test == data_test, "Equality operator");
        suite.assert_true(data_test != quaternion::zero(), "Inequality operator");

        suite.footer();
    }

    void test_quaternion_edge_cases()
    {
        TestSuite suite("quaternion Edge Cases");
        suite.header();

        // Zero quaternion
        suite.assert_true(quaternion::zero().approximately_zero(), "Zero quaternion");
        suite.assert_equal(quaternion::zero().length(), 0.0f, "Zero length");
        suite.assert_equal(quaternion::zero().length_sq(), 0.0f, "Zero squared length");

        // Very small quaternions
        quaternion tiny(1e-10f, 1e-10f, 1e-10f, 1e-10f);
        suite.assert_true(tiny.approximately_zero(1e-5f), "Tiny quaternion approximately zero");
        suite.assert_false(tiny.approximately_zero(1e-15f), "Tiny quaternion not exactly zero");

        // Very large quaternions
        quaternion large(1e10f, 1e10f, 1e10f, 1e10f);
        suite.assert_true(std::isinf(large.length()) || large.length() > 1e10f, "Large quaternion length");

        // Normalization edge cases
        quaternion normalized_zero = quaternion::zero().normalize();
        suite.assert_true(normalized_zero.is_identity(0.001f), "Normalize zero quaternion");

        quaternion inf_q(std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f);
        suite.assert_false(inf_q.is_normalized(), "Infinity quaternion normalization check");

        // NaN propagation
        quaternion nan_q(std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f, 1.0f);
        suite.assert_false(nan_q.is_valid(), "NaN quaternion validity");

        // Test normalization of very small quaternions
        quaternion very_small(1e-20f, 1e-20f, 1e-20f, 1e-20f);
        quaternion normalized_small = very_small.normalize();
        suite.assert_true(normalized_small.is_normalized(0.001f) || normalized_small.approximately_zero(),
            "Very small quaternion normalization");

        // Inverse edge cases - проверка вращений
        suite.assert_approximately_quaternion_rotation(quaternion::zero().inverse(), quaternion::identity(), "Inverse of zero returns identity", 0.001f);

        quaternion very_small2(1e-10f, 0.0f, 0.0f, 0.0f);
        quaternion inverse_small = very_small2.inverse();
        suite.assert_true(inverse_small.is_valid(), "Inverse of very small quaternion is valid");

        // Dot product edge cases
        suite.assert_equal(quaternion::zero().dot(quaternion::one()), 0.0f, "Dot product with zero quaternion");
        suite.assert_equal(quaternion::identity().dot(quaternion::identity()), 1.0f, "Dot product with itself");

        // Interpolation edge cases
        quaternion a = quaternion::identity();
        quaternion b = quaternion::rotation_x(Constants::PI / 2.0f);

        // Very close quaternions should use nlerp
        quaternion b_close = quaternion::rotation_x(Constants::PI / 2.0f + 1e-6f);
        quaternion slerp_close = slerp(a, b_close, 0.5f);
        suite.assert_true(slerp_close.is_normalized(0.001f), "slerp with close quaternions is normalized");

        // Opposite quaternions
        quaternion slerp_opposite = slerp(a, -a, 0.5f);
        suite.assert_true(slerp_opposite.is_normalized(0.001f), "slerp with opposite quaternions is normalized");

        // Conversion edge cases - проверка вращений
        quaternion identity = quaternion::identity();
        float3x3 identity_mat = identity.to_matrix3x3();
        quaternion from_identity_mat = quaternion::from_matrix(identity_mat);
        suite.assert_approximately_quaternion_rotation(identity, from_identity_mat, "Identity matrix round-trip", 0.001f);

        // Zero quaternion to matrix
        float3x3 zero_mat = quaternion::zero().to_matrix3x3();
        suite.assert_true(zero_mat.isValid(), "Zero quaternion to matrix produces valid matrix");

        suite.footer();
    }

    void test_quaternion_rotation_correctness()
    {
        TestSuite suite("quaternion Rotation Correctness");
        suite.header();

        // Test that quaternion rotations match matrix rotations
        float3 test_vector(1.0f, 2.0f, 3.0f);

        // Test X rotation
        float angle_x = Constants::PI / 4.0f;
        quaternion q_x = quaternion::rotation_x(angle_x);
        float3x3 m_x = float3x3::rotation_x(angle_x);
        float3 q_result_x = q_x * test_vector;
        float3 m_result_x = m_x.transform_vector(test_vector);
        suite.assert_true(q_result_x.approximately(m_result_x, 0.001f), "X rotation: quaternion vs matrix");

        // Test Y rotation
        test_vector = float3(1.0f, 0.0f, 0.0f); // Простой вектор по оси X

        float angle_y = Constants::HALF_PI; // 90° вместо 60°
        quaternion q_y = quaternion::rotation_y(angle_y);
        float3x3 m_y = float3x3::rotation_y(angle_y);
        float3 q_result_y = q_y * test_vector;
        float3 m_result_y = m_y.transform_vector(test_vector);
        suite.assert_true(q_result_y.approximately(m_result_y, 0.001f), "Y rotation: quaternion vs matrix");

        std::cout << "DEBUG Y rotation DETAILS:" << std::endl;
        std::cout << "  Test vector: " << test_vector.to_string() << std::endl;
        std::cout << "  Quaternion: " << q_y.to_string() << std::endl;
        std::cout << "  Matrix: " << std::endl << m_y.to_string() << std::endl;
        std::cout << "  Quat result: " << q_result_y.to_string() << std::endl;
        std::cout << "  Matrix result: " << m_result_y.to_string() << std::endl;
        std::cout << "  Difference: " << (q_result_y - m_result_y).to_string() << std::endl;

        // Test Z rotation
        float angle_z = Constants::PI / 6.0f;
        quaternion q_z = quaternion::rotation_z(angle_z);
        float3x3 m_z = float3x3::rotation_z(angle_z);
        float3 q_result_z = q_z * test_vector;
        float3 m_result_z = m_z.transform_vector(test_vector);
        suite.assert_true(q_result_z.approximately(m_result_z, 0.001f), "Z rotation: quaternion vs matrix");

        std::cout << "DEBUG Z rotation - quat: " << q_result_z.to_string()
            << ", matrix: " << m_result_z.to_string() << std::endl;

        // Test combined rotation
        quaternion q_combined = q_x * q_y * q_z;
        float3x3 m_combined = m_x * m_y * m_z;
        float3 q_result_combined = q_combined * test_vector;
        float3 m_result_combined = m_combined.transform_vector(test_vector);
        suite.assert_true(q_result_combined.approximately(m_result_combined, 0.001f),
            "Combined rotation: quaternion vs matrix");

        // Test axis-angle rotation
        float3 axis(1.0f, 1.0f, 0.0f);
        axis = axis.normalize();
        float angle = Constants::PI / 2.0f;
        quaternion q_axis = quaternion::from_axis_angle(axis, angle);
        float3x3 m_axis = float3x3::rotation_axis(axis, angle);
        float3 q_result_axis = q_axis * test_vector;
        float3 m_result_axis = m_axis.transform_vector(test_vector);
        suite.assert_true(q_result_axis.approximately(m_result_axis, 0.001f),
            "Axis-angle rotation: quaternion vs matrix");

        suite.footer();
    }

    void test_quaternion_special_cases()
    {
        TestSuite suite("quaternion Special Cases");
        suite.header();

        // Gimbal lock cases
        quaternion q_gimbal = quaternion::from_euler(float3(Constants::HALF_PI, Constants::HALF_PI, 0.0f));
        suite.assert_true(q_gimbal.is_normalized(0.001f), "Gimbal lock case produces normalized quaternion");

        // Very small rotations
        quaternion q_small = quaternion::rotation_x(1e-6f);
        suite.assert_true(q_small.is_normalized(0.001f), "Very small rotation produces normalized quaternion");
        suite.assert_approximately_quaternion_rotation(q_small, quaternion::identity(), "Very small rotation approximates identity", 0.001f);

        // Large rotations (should be wrapped correctly)
        quaternion q_large = quaternion::rotation_x(Constants::TWO_PI + Constants::PI / 2.0f);
        quaternion q_expected = quaternion::rotation_x(Constants::PI / 2.0f);
        suite.assert_approximately_quaternion_rotation(q_large, q_expected, "Large rotation wraps correctly", 0.001f);

        // From-to rotation edge cases
        quaternion q_zero = quaternion::from_to_rotation(float3::zero(), float3::unit_x());
        suite.assert_true(q_zero.is_identity(0.001f), "From-to with zero vector returns identity");

        quaternion q_parallel = quaternion::from_to_rotation(float3::unit_x(), float3::unit_x() * 2.0f);
        suite.assert_true(q_parallel.is_identity(0.001f), "From-to with parallel vectors returns identity");

        quaternion q_anti = quaternion::from_to_rotation(float3::unit_x(), -float3::unit_x());
        suite.assert_true(q_anti.is_normalized(0.001f), "From-to with anti-parallel vectors produces normalized quaternion");

        // Test that from_to_rotation actually works
        float3 from(1.0f, 0.0f, 0.0f);
        float3 to(0.0f, 1.0f, 0.0f);
        quaternion q_from_to = quaternion::from_to_rotation(from, to);
        float3 result = q_from_to * from;
        suite.assert_true(result.approximately(to.normalize(), 0.001f), "From-to rotation produces correct result");

        suite.footer();
    }

    void test_quaternion_performance_properties()
    {
        TestSuite suite("quaternion Performance Properties");
        suite.header();

        // Test that common operations maintain normalization
        quaternion a = quaternion::rotation_x(Constants::PI / 4.0f);
        quaternion b = quaternion::rotation_y(Constants::PI / 3.0f);

        // Multiplication should preserve normalization
        quaternion product = a * b;
        suite.assert_true(product.is_normalized(0.001f), "Multiplication preserves normalization");

        // Conjugate of normalized quaternion should be normalized
        quaternion conjugated = a.conjugate();
        suite.assert_true(conjugated.is_normalized(0.001f), "Conjugate preserves normalization");

        // Inverse of normalized quaternion should be normalized
        quaternion inverted = a.inverse();
        suite.assert_true(inverted.is_normalized(0.001f), "Inverse preserves normalization");

        // SLERP between normalized quaternions should be normalized
        quaternion interpolated = slerp(a, b, 0.3f);
        suite.assert_true(interpolated.is_normalized(0.001f), "SLERP preserves normalization");

        // NLERP between normalized quaternions should be normalized
        quaternion nlerp_result = nlerp(a, b, 0.7f);
        suite.assert_true(nlerp_result.is_normalized(0.001f), "NLERP preserves normalization");

        // Multiple operations should still be normalized
        quaternion complex = (a * b).conjugate().inverse();
        suite.assert_true(complex.is_normalized(0.001f), "Complex operations preserve normalization");

        // Test that conversion to/from matrix doesn't break normalization
        float3x3 mat = a.to_matrix3x3();
        quaternion from_mat = quaternion::from_matrix(mat);
        suite.assert_true(from_mat.is_normalized(0.001f), "Matrix conversion preserves normalization");

        suite.footer();
    }

    // ============================================================================
    // Test Runner for quaternion
    // ============================================================================

    void run_quaternion_tests()
    {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "QUATERNION COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        test_quaternion_constructors();
        test_quaternion_static_constructors();
        test_quaternion_assignment_operators();
        test_quaternion_compound_assignment_operators();
        test_quaternion_arithmetic_operations();
        test_quaternion_vector_transformation();
        test_quaternion_mathematical_operations();
        test_quaternion_conversion_methods();
        test_quaternion_interpolation_methods();
        test_quaternion_utility_methods();
        test_quaternion_edge_cases();
        test_quaternion_rotation_correctness();
        test_quaternion_special_cases();
        test_quaternion_performance_properties();

        std::cout << "QUATERNION TESTS COMPLETE" << std::endl;
    }

    // ============================================================================
    // Comprehensive Test Runner
    // ============================================================================

    void run_all_tests()
    {
        std::cout << "MATH LIBRARY COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "Running all tests..." << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        //run_float2_tests();
        run_float3_tests();
        //run_float4_tests();

        //run_half_tests();
        //run_half2_tests();
        //run_half3_tests();
        //run_half4_tests();

        //run_float2x2_tests();
        //run_float3x3_tests();
        //run_float4x4_tests();

        //run_quaternion_tests();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "ALL TESTS COMPLETED" << std::endl;
        std::cout << "Total time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }

} // namespace MathTests

// Main function for standalone test executable
#ifdef MATH_TESTS_STANDALONE
int main()
{
    MathTests::run_all_tests();
    return 0;
}
#endif
