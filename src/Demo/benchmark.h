// Description: Comprehensive benchmarks for Math Library
// Author: DeepSeek, NS_Deathman  
// Version: 2.0 - Complete rewrite with unified style and detailed analysis
#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <array> 
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

// Include the math library
#include "../MathAPI/MathAPI.h"

namespace MathBenchmarks
{
    using namespace Math;

    // ============================================================================
    // Benchmark Configuration
    // ============================================================================

    struct BenchmarkConfig
    {
        static constexpr int WARMUP_ITERATIONS = 1000;
        static constexpr int DEFAULT_ITERATIONS = 1000000;
        static constexpr int MEMORY_ITERATIONS = 100000;
        static constexpr int MATRIX_ITERATIONS = 100000;
        static constexpr bool VERBOSE_OUTPUT = true;
        static constexpr int PRECISION = 3;
    };

    // ============================================================================
    // Benchmark Utilities
    // ============================================================================

    class HighPrecisionTimer
    {
    private:
        std::chrono::high_resolution_clock::time_point start_time;

    public:
        void start()
        {
            start_time = std::chrono::high_resolution_clock::now();
        }

        double elapsed_milliseconds() const
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }

        double elapsed_nanoseconds() const
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::nano>(end_time - start_time).count();
        }
    };

    class BenchmarkResult
    {
    public:
        std::string name;
        double time_ms;
        double operations_per_ms;
        double operations_per_ns;
        int iterations;

        BenchmarkResult(const std::string& n, double t, int iter)
            : name(n), time_ms(t), iterations(iter)
        {
            operations_per_ms = iterations / std::max(time_ms, 0.001);
            operations_per_ns = (iterations * 1000.0) / std::max(time_ms, 0.001);
        }
    };

    class BenchmarkSuite
    {
    private:
        std::string suite_name;
        int default_iterations;
        bool verbose;
        std::vector<BenchmarkResult> results;
        HighPrecisionTimer suite_timer;

    public:
        BenchmarkSuite(const std::string& name, int iter = BenchmarkConfig::DEFAULT_ITERATIONS,
            bool verb = BenchmarkConfig::VERBOSE_OUTPUT)
            : suite_name(name), default_iterations(iter), verbose(verb)
        {
            suite_timer.start();
        }

        template<typename Func>
        BenchmarkResult measure(const std::string& test_name, Func&& func, int iterations = -1)
        {
            if (iterations == -1) iterations = default_iterations;

            // Warmup phase
            for (int i = 0; i < BenchmarkConfig::WARMUP_ITERATIONS; ++i)
            {
                func();
            }

            // Measurement phase
            HighPrecisionTimer timer;
            timer.start();

            for (int i = 0; i < iterations; ++i)
            {
                func();
            }

            double elapsed = timer.elapsed_milliseconds();
            BenchmarkResult result(test_name, elapsed, iterations);
            results.push_back(result);

            if (verbose)
            {
                std::cout << "  " << std::left << std::setw(50) << test_name
                    << ": " << std::fixed << std::setprecision(BenchmarkConfig::PRECISION)
                    << elapsed << " ms"
                    << " (" << std::setprecision(1) << result.operations_per_ms << " ops/ms)"
                    << " (" << std::setprecision(0) << result.operations_per_ns << " ns/op)"
                    << std::endl;
            }

            return result;
        }

        void header()
        {
            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "BENCHMARK SUITE: " << suite_name << std::endl;
            std::cout << "Default Iterations: " << default_iterations << std::endl;
            std::cout << std::string(80, '=') << std::endl;
        }

        void footer()
        {
            double suite_time = suite_timer.elapsed_milliseconds();

            std::cout << std::string(80, '-') << std::endl;
            std::cout << "SUMMARY: " << suite_name << std::endl;
            std::cout << "Total tests: " << results.size() << std::endl;
            std::cout << "Suite duration: " << suite_time << " ms" << std::endl;

            if (!results.empty())
            {
                auto fastest = *std::min_element(results.begin(), results.end(),
                    [](const BenchmarkResult& a, const BenchmarkResult& b) {
                        return a.operations_per_ns > b.operations_per_ns;
                    });

                auto slowest = *std::max_element(results.begin(), results.end(),
                    [](const BenchmarkResult& a, const BenchmarkResult& b) {
                        return a.operations_per_ns > b.operations_per_ns;
                    });

                std::cout << "Fastest: " << fastest.name << " (" << fastest.operations_per_ns << " ns/op)" << std::endl;
                std::cout << "Slowest: " << slowest.name << " (" << slowest.operations_per_ns << " ns/op)" << std::endl;
            }

            std::cout << std::string(80, '=') << std::endl;
        }

        const std::vector<BenchmarkResult>& get_results() const { return results; }
    };

    class ComparisonBenchmark
    {
    private:
        std::string comparison_name;

    public:
        ComparisonBenchmark(const std::string& name) : comparison_name(name) {}

        struct ComparisonResult
        {
            std::string test_name;
            double library_time;
            double native_time;
            double speedup;
            double library_ns_per_op;
            double native_ns_per_op;
        };

        ComparisonResult compare(const std::string& test_name,
            double lib_time, double native_time, int iterations)
        {
            double speedup = native_time / lib_time;
            double lib_ns_per_op = (lib_time * 1000000.0) / iterations;
            double native_ns_per_op = (native_time * 1000000.0) / iterations;

            ComparisonResult result{ test_name, lib_time, native_time, speedup,
                                   lib_ns_per_op, native_ns_per_op };

            std::cout << "  " << std::left << std::setw(40) << test_name
                << ": Library=" << std::fixed << std::setprecision(3) << lib_time << "ms"
                << ", Native=" << native_time << "ms"
                << ", Speedup=" << std::setprecision(2) << speedup << "x"
                << " (Lib: " << std::setprecision(1) << lib_ns_per_op << "ns/op"
                << ", Native: " << native_ns_per_op << "ns/op)"
                << std::endl;

            return result;
        }
    };

    // ============================================================================
    // Benchmark Data Generators
    // ============================================================================

    class BenchmarkData
    {
    public:
        static std::vector<float2> generate_float2_data(int count)
        {
            std::vector<float2> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x = static_cast<float>(std::sin(i * 0.1f));
                float y = static_cast<float>(std::cos(i * 0.1f));
                data.emplace_back(x, y);
            }
            return data;
        }

        static std::vector<float3> generate_float3_data(int count)
        {
            std::vector<float3> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x = static_cast<float>(std::sin(i * 0.1f));
                float y = static_cast<float>(std::cos(i * 0.1f));
                float z = static_cast<float>(std::sin(i * 0.2f));
                data.emplace_back(x, y, z);
            }
            return data;
        }

        static std::vector<float4> generate_float4_data(int count)
        {
            std::vector<float4> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x = static_cast<float>(std::sin(i * 0.1f));
                float y = static_cast<float>(std::cos(i * 0.1f));
                float z = static_cast<float>(std::sin(i * 0.2f));
                float w = static_cast<float>(std::cos(i * 0.2f));
                data.emplace_back(x, y, z, w);
            }
            return data;
        }
    };

    // ============================================================================
    // float2 Benchmarks - Comprehensive
    // ============================================================================

    void benchmark_float2_operations()
    {
        BenchmarkSuite suite("float2 Operations", 2000000, true);
        suite.header();
        ComparisonBenchmark comparator("float2 Operations");

        // Test data
        float2 a(1.5f, 2.5f);
        float2 b(3.0f, 4.0f);
        float2 c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison
        struct NativeFloat2 { float x, y; };
        NativeFloat2 na{ 1.5f, 2.5f };
        NativeFloat2 nb{ 3.0f, 4.0f };
        NativeFloat2 nc{ 0.0f, 0.0f };

        // 1. Addition
        auto lib_time = suite.measure("float2 Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += c.x + c.y;
            }).time_ms;

        auto native_time = suite.measure("float2 Addition (Native)", [&]() {
            nc.x = na.x + nb.x;
            nc.y = na.y + nb.y;
            result_accumulator += nc.x + nc.y;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 2000000);

        // 2. Subtraction
        lib_time = suite.measure("float2 Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += c.x + c.y;
            }).time_ms;

        native_time = suite.measure("float2 Subtraction (Native)", [&]() {
            nc.x = na.x - nb.x;
            nc.y = na.y - nb.y;
            result_accumulator += nc.x + nc.y;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 2000000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("float2 Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += c.x + c.y;
            }).time_ms;

        native_time = suite.measure("float2 Scalar Multiplication (Native)", [&]() {
            nc.x = na.x * scalar;
            nc.y = na.y * scalar;
            result_accumulator += nc.x + nc.y;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 2000000);

        // 4. Dot Product
        lib_time = suite.measure("float2 Dot Product (Library)", [&]() {
            float result = dot(a, b);
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float2 Dot Product (Native)", [&]() {
            float result = na.x * nb.x + na.y * nb.y;
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Dot Product", lib_time, native_time, 2000000);

        // 5. Length
        lib_time = suite.measure("float2 Length (Library)", [&]() {
            float result = a.length();
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float2 Length (Native)", [&]() {
            float result = std::sqrt(na.x * na.x + na.y * na.y);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Length", lib_time, native_time, 2000000);

        // 6. Normalization
        lib_time = suite.measure("float2 Normalize (Library)", [&]() {
            c = a.normalize();
            result_accumulator += c.x + c.y;
            }).time_ms;

        native_time = suite.measure("float2 Normalize (Native)", [&]() {
            float len = std::sqrt(na.x * na.x + na.y * na.y);
            if (len > 0.0f) {
                nc.x = na.x / len;
                nc.y = na.y / len;
            }
            result_accumulator += nc.x + nc.y;
            }).time_ms;

        comparator.compare("Normalize", lib_time, native_time, 2000000);

        // 7. Cross Product
        lib_time = suite.measure("float2 Cross Product (Library)", [&]() {
            float result = cross(a, b);
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float2 Cross Product (Native)", [&]() {
            float result = na.x * nb.y - na.y * nb.x;
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Cross Product", lib_time, native_time, 2000000);

        // 8. Component-wise Operations
        lib_time = suite.measure("float2 Component-wise Multiply (Library)", [&]() {
            c = a * b;
            result_accumulator += c.x + c.y;
            }).time_ms;

        native_time = suite.measure("float2 Component-wise Multiply (Native)", [&]() {
            nc.x = na.x * nb.x;
            nc.y = na.y * nb.y;
            result_accumulator += nc.x + nc.y;
            }).time_ms;

        comparator.compare("Component-wise Multiply", lib_time, native_time, 2000000);

        suite.footer();
    }

    void benchmark_float2_advanced_operations()
    {
        BenchmarkSuite suite("float2 Advanced Operations", 1000000, true);
        suite.header();

        float2 a(1.5f, 2.5f);
        float2 b(3.0f, 4.0f);
        float2 c;
        volatile float result_accumulator = 0.0f;

        // Geometric operations
        suite.measure("float2 Distance", [&]() {
            float result = distance(a, b);
            result_accumulator += result;
            });

        suite.measure("float2 Distance Squared", [&]() {
            float result = distance_sq(a, b);
            result_accumulator += result;
            });

        suite.measure("float2 Lerp", [&]() {
            c = lerp(a, b, 0.5f);
            result_accumulator += c.x + c.y;
            });

        suite.measure("float2 Reflect", [&]() {
            c = reflect(a, b.normalize());
            result_accumulator += c.x + c.y;
            });

        suite.measure("float2 Rotate", [&]() {
            c = a.rotate(Constants::PI / 4.0f);
            result_accumulator += c.x + c.y;
            });

        // HLSL-style functions
        suite.measure("float2 Abs", [&]() {
            c = abs(float2(-1.5f, 2.5f));
            result_accumulator += c.x + c.y;
            });

        suite.measure("float2 Floor", [&]() {
            c = floor(float2(1.7f, -2.3f));
            result_accumulator += c.x + c.y;
            });

        suite.measure("float2 Ceil", [&]() {
            c = ceil(float2(1.2f, -2.7f));
            result_accumulator += c.x + c.y;
            });

        suite.measure("float2 Round", [&]() {
            c = round(float2(1.4f, 1.6f));
            result_accumulator += c.x + c.y;
            });

        suite.measure("float2 Frac", [&]() {
            c = frac(float2(1.7f, -2.3f));
            result_accumulator += c.x + c.y;
            });

        suite.footer();
    }

    void benchmark_float2_memory_patterns()
    {
        BenchmarkSuite suite("float2 Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = BenchmarkData::generate_float2_data(ARRAY_SIZE);
        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("float2 Sequential Sum", [&]() {
            float2 sum = float2::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += sum.x + sum.y;
            });

        suite.measure("float2 Sequential Dot Products", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(data[i], data[i + 1]);
            }
            result_accumulator += sum;
            });

        suite.measure("float2 Sequential Normalization", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float2 normalized = data[i].normalize();
                result_accumulator += normalized.x + normalized.y;
            }
            });

        // Random access pattern (simulated)
        suite.measure("float2 Strided Access", [&]() {
            float2 sum = float2::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += sum.x + sum.y;
            });

        suite.footer();
    }

    // ============================================================================
    // float3 Benchmarks - Comprehensive
    // ============================================================================

    void benchmark_float3_operations()
    {
        BenchmarkSuite suite("float3 Operations", 1500000, true);
        suite.header();
        ComparisonBenchmark comparator("float3 Operations");

        // Test data
        float3 a(1.5f, 2.5f, 3.5f);
        float3 b(4.0f, 5.0f, 6.0f);
        float3 c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison
        struct NativeFloat3 { float x, y, z; };
        NativeFloat3 na{ 1.5f, 2.5f, 3.5f };
        NativeFloat3 nb{ 4.0f, 5.0f, 6.0f };
        NativeFloat3 nc{ 0.0f, 0.0f, 0.0f };

        // 1. Addition
        auto lib_time = suite.measure("float3 Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += c.x + c.y + c.z;
            }).time_ms;

        auto native_time = suite.measure("float3 Addition (Native)", [&]() {
            nc.x = na.x + nb.x;
            nc.y = na.y + nb.y;
            nc.z = na.z + nb.z;
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 1500000);

        // 2. Subtraction
        lib_time = suite.measure("float3 Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += c.x + c.y + c.z;
            }).time_ms;

        native_time = suite.measure("float3 Subtraction (Native)", [&]() {
            nc.x = na.x - nb.x;
            nc.y = na.y - nb.y;
            nc.z = na.z - nb.z;
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 1500000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("float3 Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += c.x + c.y + c.z;
            }).time_ms;

        native_time = suite.measure("float3 Scalar Multiplication (Native)", [&]() {
            nc.x = na.x * scalar;
            nc.y = na.y * scalar;
            nc.z = na.z * scalar;
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 1500000);

        // 4. Dot Product
        lib_time = suite.measure("float3 Dot Product (Library)", [&]() {
            float result = dot(a, b);
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float3 Dot Product (Native)", [&]() {
            float result = na.x * nb.x + na.y * nb.y + na.z * nb.z;
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Dot Product", lib_time, native_time, 1500000);

        // 5. Cross Product
        lib_time = suite.measure("float3 Cross Product (Library)", [&]() {
            c = cross(a, b);
            result_accumulator += c.x + c.y + c.z;
            }).time_ms;

        native_time = suite.measure("float3 Cross Product (Native)", [&]() {
            nc.x = na.y * nb.z - na.z * nb.y;
            nc.y = na.z * nb.x - na.x * nb.z;
            nc.z = na.x * nb.y - na.y * nb.x;
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Cross Product", lib_time, native_time, 1500000);

        // 6. Length
        lib_time = suite.measure("float3 Length (Library)", [&]() {
            float result = a.length();
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float3 Length (Native)", [&]() {
            float result = std::sqrt(na.x * na.x + na.y * na.y + na.z * na.z);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Length", lib_time, native_time, 1500000);

        // 7. Normalization
        lib_time = suite.measure("float3 Normalize (Library)", [&]() {
            c = a.normalize();
            result_accumulator += c.x + c.y + c.z;
            }).time_ms;

        native_time = suite.measure("float3 Normalize (Native)", [&]() {
            float len = std::sqrt(na.x * na.x + na.y * na.y + na.z * na.z);
            if (len > 0.0f) {
                nc.x = na.x / len;
                nc.y = na.y / len;
                nc.z = na.z / len;
            }
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Normalize", lib_time, native_time, 1500000);

        // 8. Distance
        lib_time = suite.measure("float3 Distance (Library)", [&]() {
            float result = distance(a, b);
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float3 Distance (Native)", [&]() {
            float dx = na.x - nb.x;
            float dy = na.y - nb.y;
            float dz = na.z - nb.z;
            float result = std::sqrt(dx * dx + dy * dy + dz * dz);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Distance", lib_time, native_time, 1500000);

        suite.footer();
    }

    void benchmark_float3_advanced_operations()
    {
        BenchmarkSuite suite("float3 Advanced Operations", 1000000, true);
        suite.header();

        float3 a(1.5f, 2.5f, 3.5f);
        float3 b(4.0f, 5.0f, 6.0f);
        float3 c;
        volatile float result_accumulator = 0.0f;

        // Geometric operations
        suite.measure("float3 Distance Squared", [&]() {
            float result = distance_sq(a, b);
            result_accumulator += result;
            });

        suite.measure("float3 Lerp", [&]() {
            c = lerp(a, b, 0.5f);
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Slerp", [&]() {
            c = slerp(a.normalize(), b.normalize(), 0.5f);
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Reflect", [&]() {
            c = reflect(a, b.normalize());
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Project", [&]() {
            c = project(a, b);
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Reject", [&]() {
            c = reject(a, b);
            result_accumulator += c.x + c.y + c.z;
            });

        // HLSL-style functions
        suite.measure("float3 Abs", [&]() {
            c = abs(float3(-1.5f, 2.5f, -3.5f));
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Floor", [&]() {
            c = floor(float3(1.7f, -2.3f, 3.8f));
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Ceil", [&]() {
            c = ceil(float3(1.2f, -2.7f, 3.3f));
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Round", [&]() {
            c = round(float3(1.4f, 1.6f, 2.5f));
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Frac", [&]() {
            c = frac(float3(1.7f, -2.3f, 3.8f));
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Saturate", [&]() {
            c = saturate(float3(-0.5f, 0.5f, 1.5f));
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Min", [&]() {
            c = min(a, b);
            result_accumulator += c.x + c.y + c.z;
            });

        suite.measure("float3 Max", [&]() {
            c = max(a, b);
            result_accumulator += c.x + c.y + c.z;
            });

        suite.footer();
    }

    void benchmark_float3_swizzle_operations()
    {
        BenchmarkSuite suite("float3 Swizzle Operations", 2000000, true);
        suite.header();

        float3 v(1.0f, 2.0f, 3.0f);
        volatile float result_accumulator = 0.0f;

        // 2D swizzles
        suite.measure("float3 XY Swizzle", [&]() {
            auto result = v.xy();
            result_accumulator += result.x + result.y;
            });

        suite.measure("float3 XZ Swizzle", [&]() {
            auto result = v.xz();
            result_accumulator += result.x + result.y;
            });

        suite.measure("float3 YZ Swizzle", [&]() {
            auto result = v.yz();
            result_accumulator += result.x + result.y;
            });

        // 3D swizzles
        suite.measure("float3 YXZ Swizzle", [&]() {
            auto result = v.yxz();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float3 ZXY Swizzle", [&]() {
            auto result = v.zxy();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float3 ZYX Swizzle", [&]() {
            auto result = v.zyx();
            result_accumulator += result.x + result.y + result.z;
            });

        // Color swizzles
        suite.measure("float3 RGB Swizzle", [&]() {
            auto result = v.rgb();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float3 BGR Swizzle", [&]() {
            auto result = v.bgr();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.footer();
    }

    void benchmark_float3_memory_patterns()
    {
        BenchmarkSuite suite("float3 Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = BenchmarkData::generate_float3_data(ARRAY_SIZE);
        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("float3 Sequential Sum", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        suite.measure("float3 Sequential Dot Products", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(data[i], data[i + 1]);
            }
            result_accumulator += sum;
            });

        suite.measure("float3 Sequential Cross Products", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += cross(data[i], data[i + 1]);
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        suite.measure("float3 Sequential Normalization", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float3 normalized = data[i].normalize();
                result_accumulator += normalized.x + normalized.y + normalized.z;
            }
            });

        // Random access pattern (simulated)
        suite.measure("float3 Strided Access", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        suite.footer();
    }

    // ============================================================================
    // float4 Benchmarks - Comprehensive
    // ============================================================================

    void benchmark_float4_operations()
    {
        BenchmarkSuite suite("float4 Operations", 1000000, true);
        suite.header();
        ComparisonBenchmark comparator("float4 Operations");

        // Test data
        float4 a(1.5f, 2.5f, 3.5f, 4.5f);
        float4 b(5.0f, 6.0f, 7.0f, 8.0f);
        float4 c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison
        struct NativeFloat4 { float x, y, z, w; };
        NativeFloat4 na{ 1.5f, 2.5f, 3.5f, 4.5f };
        NativeFloat4 nb{ 5.0f, 6.0f, 7.0f, 8.0f };
        NativeFloat4 nc{ 0.0f, 0.0f, 0.0f, 0.0f };

        // 1. Addition
        auto lib_time = suite.measure("float4 Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        auto native_time = suite.measure("float4 Addition (Native)", [&]() {
            nc.x = na.x + nb.x;
            nc.y = na.y + nb.y;
            nc.z = na.z + nb.z;
            nc.w = na.w + nb.w;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 1000000);

        // 2. Subtraction
        lib_time = suite.measure("float4 Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        native_time = suite.measure("float4 Subtraction (Native)", [&]() {
            nc.x = na.x - nb.x;
            nc.y = na.y - nb.y;
            nc.z = na.z - nb.z;
            nc.w = na.w - nb.w;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 1000000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("float4 Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        native_time = suite.measure("float4 Scalar Multiplication (Native)", [&]() {
            nc.x = na.x * scalar;
            nc.y = na.y * scalar;
            nc.z = na.z * scalar;
            nc.w = na.w * scalar;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 1000000);

        // 4. Dot Product
        lib_time = suite.measure("float4 Dot Product (Library)", [&]() {
            float result = dot(a, b);
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float4 Dot Product (Native)", [&]() {
            float result = na.x * nb.x + na.y * nb.y + na.z * nb.z + na.w * nb.w;
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Dot Product", lib_time, native_time, 1000000);

        // 5. 3D Dot Product
        lib_time = suite.measure("float4 3D Dot Product (Library)", [&]() {
            float result = dot3(a, b);
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float4 3D Dot Product (Native)", [&]() {
            float result = na.x * nb.x + na.y * nb.y + na.z * nb.z;
            result_accumulator += result;
            }).time_ms;

        comparator.compare("3D Dot Product", lib_time, native_time, 1000000);

        // 6. Cross Product
        lib_time = suite.measure("float4 Cross Product (Library)", [&]() {
            c = cross(a, b);
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        native_time = suite.measure("float4 Cross Product (Native)", [&]() {
            nc.x = na.y * nb.z - na.z * nb.y;
            nc.y = na.z * nb.x - na.x * nb.z;
            nc.z = na.x * nb.y - na.y * nb.x;
            nc.w = 0.0f;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Cross Product", lib_time, native_time, 1000000);

        // 7. Length
        lib_time = suite.measure("float4 Length (Library)", [&]() {
            float result = a.length();
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float4 Length (Native)", [&]() {
            float result = std::sqrt(na.x * na.x + na.y * na.y + na.z * na.z + na.w * na.w);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Length", lib_time, native_time, 1000000);

        // 8. Normalization
        lib_time = suite.measure("float4 Normalize (Library)", [&]() {
            c = a.normalize();
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        native_time = suite.measure("float4 Normalize (Native)", [&]() {
            float len = std::sqrt(na.x * na.x + na.y * na.y + na.z * na.z + na.w * na.w);
            if (len > 0.0f) {
                nc.x = na.x / len;
                nc.y = na.y / len;
                nc.z = na.z / len;
                nc.w = na.w / len;
            }
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Normalize", lib_time, native_time, 1000000);

        suite.footer();
    }

    void benchmark_float4_advanced_operations()
    {
        BenchmarkSuite suite("float4 Advanced Operations", 1000000, true);
        suite.header();

        float4 a(1.5f, 2.5f, 3.5f, 4.5f);
        float4 b(5.0f, 6.0f, 7.0f, 8.0f);
        float4 c;
        volatile float result_accumulator = 0.0f;

        // Mathematical operations
        suite.measure("float4 Distance", [&]() {
            float result = distance(a, b);
            result_accumulator += result;
            });

        suite.measure("float4 Distance Squared", [&]() {
            float result = distance_sq(a, b);
            result_accumulator += result;
            });

        suite.measure("float4 Lerp", [&]() {
            c = lerp(a, b, 0.5f);
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        // HLSL-style functions
        suite.measure("float4 Abs", [&]() {
            c = abs(float4(-1.5f, 2.5f, -3.5f, -4.5f));
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        suite.measure("float4 Floor", [&]() {
            c = floor(float4(1.7f, -2.3f, 3.8f, -0.5f));
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        suite.measure("float4 Ceil", [&]() {
            c = ceil(float4(1.2f, -2.7f, 3.3f, -0.2f));
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        suite.measure("float4 Round", [&]() {
            c = round(float4(1.4f, 1.6f, 2.5f, -1.4f));
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        suite.measure("float4 Frac", [&]() {
            c = frac(float4(1.7f, -2.3f, 3.8f, -0.5f));
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        suite.measure("float4 Saturate", [&]() {
            c = saturate(float4(-0.5f, 0.5f, 1.5f, 2.0f));
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        suite.measure("float4 Min", [&]() {
            c = min(a, b);
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        suite.measure("float4 Max", [&]() {
            c = max(a, b);
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        // Color operations
        suite.measure("float4 Luminance", [&]() {
            float result = a.luminance();
            result_accumulator += result;
            });

        suite.measure("float4 Brightness", [&]() {
            float result = a.brightness();
            result_accumulator += result;
            });

        suite.measure("float4 Grayscale", [&]() {
            c = a.grayscale();
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        suite.measure("float4 Premultiply Alpha", [&]() {
            c = a.premultiply_alpha();
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        // Geometric operations
        suite.measure("float4 Project", [&]() {
            float3 result = a.project();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float4 To Homogeneous", [&]() {
            c = a.to_homogeneous();
            result_accumulator += c.x + c.y + c.z + c.w;
            });

        suite.footer();
    }

    void benchmark_float4_swizzle_operations()
    {
        BenchmarkSuite suite("float4 Swizzle Operations", 2000000, true);
        suite.header();

        float4 v(1.0f, 2.0f, 3.0f, 4.0f);
        volatile float result_accumulator = 0.0f;

        // 2D swizzles
        suite.measure("float4 XY Swizzle", [&]() {
            auto result = v.xy();
            result_accumulator += result.x + result.y;
            });

        suite.measure("float4 XZ Swizzle", [&]() {
            auto result = v.xz();
            result_accumulator += result.x + result.y;
            });

        suite.measure("float4 XW Swizzle", [&]() {
            auto result = v.xw();
            result_accumulator += result.x + result.y;
            });

        // 3D swizzles
        suite.measure("float4 XYZ Swizzle", [&]() {
            auto result = v.xyz();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float4 XYW Swizzle", [&]() {
            auto result = v.xyw();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float4 YZW Swizzle", [&]() {
            auto result = v.yzw();
            result_accumulator += result.x + result.y + result.z;
            });

        // 4D swizzles
        suite.measure("float4 YXZW Swizzle", [&]() {
            auto result = v.yxzw();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("float4 WZYX Swizzle", [&]() {
            auto result = v.wzyx();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        // Color swizzles
        suite.measure("float4 RGB Swizzle", [&]() {
            auto result = v.rgb();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float4 RGBA Swizzle", [&]() {
            auto result = v; // implicit
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("float4 BGRA Swizzle", [&]() {
            auto result = v.bgra();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("float4 ABGR Swizzle", [&]() {
            auto result = v.abgr();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.footer();
    }

    void benchmark_float4_memory_patterns()
    {
        BenchmarkSuite suite("float4 Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = BenchmarkData::generate_float4_data(ARRAY_SIZE);
        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("float4 Sequential Sum", [&]() {
            float4 sum = float4::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        suite.measure("float4 Sequential Dot Products", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(data[i], data[i + 1]);
            }
            result_accumulator += sum;
            });

        suite.measure("float4 Sequential 3D Dot Products", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot3(data[i], data[i + 1]);
            }
            result_accumulator += sum;
            });

        suite.measure("float4 Sequential Cross Products", [&]() {
            float4 sum = float4::zero();
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += cross(data[i], data[i + 1]);
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        suite.measure("float4 Sequential Normalization", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float4 normalized = data[i].normalize();
                result_accumulator += normalized.x + normalized.y + normalized.z + normalized.w;
            }
            });

        // Color operations pattern
        suite.measure("float4 Sequential Grayscale", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float4 gray = data[i].grayscale();
                result_accumulator += gray.x + gray.y + gray.z + gray.w;
            }
            });

        // Random access pattern (simulated)
        suite.measure("float4 Strided Access", [&]() {
            float4 sum = float4::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        suite.footer();
    }

    // ============================================================================
    // half Benchmarks - Comprehensive
    // ============================================================================

    class HalfBenchmarkData
    {
    public:
        static std::vector<half> generate_half_data(int count)
        {
            std::vector<half> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x = static_cast<float>(std::sin(i * 0.1f) * 10.0f); // Keep in reasonable range
                data.emplace_back(x);
            }
            return data;
        }

        static std::vector<std::pair<half, half>> generate_half_pairs(int count)
        {
            std::vector<std::pair<half, half>> pairs;
            pairs.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x = static_cast<float>(std::sin(i * 0.1f) * 10.0f);
                float y = static_cast<float>(std::cos(i * 0.1f) * 10.0f);
                pairs.emplace_back(half(x), half(y));
            }
            return pairs;
        }
    };

    void benchmark_half_operations()
    {
        BenchmarkSuite suite("half Operations", 2000000, true);
        suite.header();
        ComparisonBenchmark comparator("half Operations");

        // Test data
        half a(1.5f);
        half b(3.0f);
        half c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison (using float)
        float na = 1.5f;
        float nb = 3.0f;
        float nc = 0.0f;

        // 1. Addition
        auto lib_time = suite.measure("half Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += float(c);
            }).time_ms;

        auto native_time = suite.measure("half Addition (Native Float)", [&]() {
            nc = na + nb;
            result_accumulator += nc;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 2000000);

        // 2. Subtraction
        lib_time = suite.measure("half Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += float(c);
            }).time_ms;

        native_time = suite.measure("half Subtraction (Native Float)", [&]() {
            nc = na - nb;
            result_accumulator += nc;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 2000000);

        // 3. Multiplication
        lib_time = suite.measure("half Multiplication (Library)", [&]() {
            c = a * b;
            result_accumulator += float(c);
            }).time_ms;

        native_time = suite.measure("half Multiplication (Native Float)", [&]() {
            nc = na * nb;
            result_accumulator += nc;
            }).time_ms;

        comparator.compare("Multiplication", lib_time, native_time, 2000000);

        // 4. Division
        lib_time = suite.measure("half Division (Library)", [&]() {
            c = a / b;
            result_accumulator += float(c);
            }).time_ms;

        native_time = suite.measure("half Division (Native Float)", [&]() {
            nc = na / nb;
            result_accumulator += nc;
            }).time_ms;

        comparator.compare("Division", lib_time, native_time, 2000000);

        // 5. Conversion overhead
        lib_time = suite.measure("half Float Conversion (To Float)", [&]() {
            float result = float(a);
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("half Float Conversion (From Float)", [&]() {
            half result = half(na);
            result_accumulator += float(result);
            }).time_ms;

        comparator.compare("Float Conversion", lib_time, native_time, 2000000);

        suite.footer();
    }

    void benchmark_half_mathematical_functions()
    {
        BenchmarkSuite suite("half Mathematical Functions", 1000000, true);
        suite.header();

        half a(2.0f);
        half b(4.0f);
        volatile float result_accumulator = 0.0f;

        // Basic math functions
        suite.measure("half Abs", [&]() {
            half result = abs(half(-3.0f));
            result_accumulator += float(result);
            });

        suite.measure("half Sqrt", [&]() {
            half result = sqrt(b);
            result_accumulator += float(result);
            });

        suite.measure("half Rsqrt", [&]() {
            half result = rsqrt(b);
            result_accumulator += float(result);
            });

        // Trigonometric functions
        suite.measure("half Sin", [&]() {
            half result = sin(a);
            result_accumulator += float(result);
            });

        suite.measure("half Cos", [&]() {
            half result = cos(a);
            result_accumulator += float(result);
            });

        suite.measure("half Tan", [&]() {
            half result = tan(a);
            result_accumulator += float(result);
            });

        // Exponential and logarithmic functions
        suite.measure("half Exp", [&]() {
            half result = exp(a);
            result_accumulator += float(result);
            });

        suite.measure("half Log", [&]() {
            half result = log(b);
            result_accumulator += float(result);
            });

        suite.measure("half Log2", [&]() {
            half result = log2(b);
            result_accumulator += float(result);
            });

        suite.measure("half Log10", [&]() {
            half result = log10(b);
            result_accumulator += float(result);
            });

        suite.measure("half Pow", [&]() {
            half result = pow(a, b);
            result_accumulator += float(result);
            });

        suite.footer();
    }

    void benchmark_half_hlsl_functions()
    {
        BenchmarkSuite suite("half HLSL Functions", 1500000, true);
        suite.header();

        half a(0.3f);
        half b(0.7f);
        half c(0.5f);
        volatile float result_accumulator = 0.0f;

        // HLSL-style functions
        suite.measure("half Saturate", [&]() {
            half result = saturate(half(-0.5f));
            result_accumulator += float(result);
            });

        suite.measure("half Clamp", [&]() {
            half result = clamp(a, half(0.5f), half(1.0f));
            result_accumulator += float(result);
            });

        suite.measure("half Lerp", [&]() {
            half result = lerp(a, b, c);
            result_accumulator += float(result);
            });

        suite.measure("half Step", [&]() {
            half result = step(c, a);
            result_accumulator += float(result);
            });

        suite.measure("half Smoothstep", [&]() {
            half result = smoothstep(a, b, c);
            result_accumulator += float(result);
            });

        suite.measure("half Sign", [&]() {
            half result = sign(half(-2.0f));
            result_accumulator += float(result);
            });

        suite.measure("half Floor", [&]() {
            half result = floor(half(2.7f));
            result_accumulator += float(result);
            });

        suite.measure("half Ceil", [&]() {
            half result = ceil(half(2.3f));
            result_accumulator += float(result);
            });

        suite.measure("half Round", [&]() {
            half result = round(half(2.5f));
            result_accumulator += float(result);
            });

        suite.measure("half Frac", [&]() {
            half result = frac(half(3.75f));
            result_accumulator += float(result);
            });

        suite.footer();
    }

    void benchmark_half_special_operations()
    {
        BenchmarkSuite suite("half Special Operations", 1000000, true);
        suite.header();

        half normal(2.0f);
        half inf(std::numeric_limits<float>::infinity());
        half nan(std::numeric_limits<float>::quiet_NaN());
        half zero(0.0f);
        volatile float result_accumulator = 0.0f;

        // Special value checks
        suite.measure("half IsFinite", [&]() {
            bool result = is_finite(normal);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("half IsNan", [&]() {
            bool result = is_nan(nan);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("half IsInf", [&]() {
            bool result = is_inf(inf);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("half IsNormal", [&]() {
            bool result = is_normal(normal);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        // Bit operations
        suite.measure("half Bits", [&]() {
            auto result = normal.bits();
            result_accumulator += static_cast<float>(result);
            });

        suite.measure("half FromBits", [&]() {
            half result = half::from_bits(0x3C00); // 1.0
            result_accumulator += float(result);
            });

        // Utility functions
        suite.measure("half Abs", [&]() {
            half result = normal.abs();
            result_accumulator += float(result);
            });

        suite.measure("half Reciprocal", [&]() {
            half result = normal.reciprocal();
            result_accumulator += float(result);
            });

        suite.measure("half Approximately", [&]() {
            bool result = normal.approximately(half(2.0001f), 0.001f);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.footer();
    }

    void benchmark_half_memory_patterns()
    {
        BenchmarkSuite suite("half Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = HalfBenchmarkData::generate_half_data(ARRAY_SIZE);
        auto pairs = HalfBenchmarkData::generate_half_pairs(ARRAY_SIZE);
        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("half Sequential Sum", [&]() {
            half sum(0.0f);
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += float(sum);
            });

        suite.measure("half Sequential Product", [&]() {
            half product(1.0f);
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= data[i];
            }
            result_accumulator += float(product);
            });

        // Mathematical operations on arrays
        suite.measure("half Array Abs", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half result = abs(data[i]);
                result_accumulator += float(result);
            }
            });

        suite.measure("half Array Sqrt", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half result = sqrt(abs(data[i])); // Use abs to avoid NaN
                result_accumulator += float(result);
            }
            });

        // Pair operations
        suite.measure("half Pair Operations", [&]() {
            half sum(0.0f);
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += pairs[i].first * pairs[i].second;
            }
            result_accumulator += float(sum);
            });

        // Conversion patterns
        suite.measure("half Float Conversion Array", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += float(data[i]);
            }
            result_accumulator += sum;
            });

        // Random access pattern (simulated)
        suite.measure("half Strided Access", [&]() {
            half sum(0.0f);
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += float(sum);
            });

        suite.footer();
    }

    void benchmark_half_vs_float_memory()
    {
        BenchmarkSuite suite("half vs Float Memory Bandwidth", BenchmarkConfig::MEMORY_ITERATIONS / 10, true);
        suite.header();

        const int ARRAY_SIZE = 10000;

        // Generate test data
        std::vector<half> half_data;
        std::vector<float> float_data;
        half_data.reserve(ARRAY_SIZE);
        float_data.reserve(ARRAY_SIZE);

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            float val = static_cast<float>(std::sin(i * 0.1f) * 10.0f);
            half_data.emplace_back(val);
            float_data.push_back(val);
        }

        volatile float result_accumulator = 0.0f;

        // Memory bandwidth tests
        suite.measure("half Memory Sum", [&]() {
            half sum(0.0f);
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += half_data[i];
            }
            result_accumulator += float(sum);
            });

        suite.measure("float Memory Sum", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += float_data[i];
            }
            result_accumulator += sum;
            });

        suite.measure("half Memory Product", [&]() {
            half product(1.0f);
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= half_data[i];
            }
            result_accumulator += float(product);
            });

        suite.measure("float Memory Product", [&]() {
            float product = 1.0f;
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= float_data[i];
            }
            result_accumulator += product;
            });

        suite.footer();
    }

    // ============================================================================
    // Comprehensive Benchmark Runner for half
    // ============================================================================

    void run_half_benchmarks()
    {
        std::cout << "HALF PRECISION COMPREHENSIVE BENCHMARK SUITE" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "This benchmark compares half-precision performance" << std::endl;
        std::cout << "against native float implementations." << std::endl;
        std::cout << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        benchmark_half_operations();
        benchmark_half_mathematical_functions();
        benchmark_half_hlsl_functions();
        benchmark_half_special_operations();
        benchmark_half_memory_patterns();
        benchmark_half_vs_float_memory();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "HALF BENCHMARKS COMPLETED" << std::endl;
        std::cout << "Total benchmark time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ============================================================================
// Benchmark Data Generators for half2
// ============================================================================

    class Half2BenchmarkData
    {
    public:
        static std::vector<half2> generate_half2_data(int count)
        {
            std::vector<half2> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x = static_cast<float>(std::sin(i * 0.1f) * 2.0f);
                float y = static_cast<float>(std::cos(i * 0.1f) * 2.0f);
                data.emplace_back(x, y);
            }
            return data;
        }

        static std::vector<std::pair<half2, half2>> generate_half2_pairs(int count)
        {
            std::vector<std::pair<half2, half2>> pairs;
            pairs.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x1 = static_cast<float>(std::sin(i * 0.1f) * 2.0f);
                float y1 = static_cast<float>(std::cos(i * 0.1f) * 2.0f);
                float x2 = static_cast<float>(std::sin(i * 0.2f) * 2.0f);
                float y2 = static_cast<float>(std::cos(i * 0.2f) * 2.0f);
                pairs.emplace_back(half2(x1, y1), half2(x2, y2));
            }
            return pairs;
        }
    };

    // ============================================================================
    // half2 Benchmarks - Comprehensive
    // ============================================================================

    void benchmark_half2_operations()
    {
        BenchmarkSuite suite("half2 Operations", 1500000, true);
        suite.header();
        ComparisonBenchmark comparator("half2 Operations");

        // Test data
        half2 a(1.5f, 2.5f);
        half2 b(3.0f, 4.0f);
        half2 c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison (using float2)
        float2 na(1.5f, 2.5f);
        float2 nb(3.0f, 4.0f);
        float2 nc;

        // 1. Addition
        auto lib_time = suite.measure("half2 Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += float(c.x) + float(c.y);
            }).time_ms;

        auto native_time = suite.measure("half2 Addition (Native Float2)", [&]() {
            nc = na + nb;
            result_accumulator += nc.x + nc.y;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 1500000);

        // 2. Subtraction
        lib_time = suite.measure("half2 Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += float(c.x) + float(c.y);
            }).time_ms;

        native_time = suite.measure("half2 Subtraction (Native Float2)", [&]() {
            nc = na - nb;
            result_accumulator += nc.x + nc.y;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 1500000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("half2 Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += float(c.x) + float(c.y);
            }).time_ms;

        native_time = suite.measure("half2 Scalar Multiplication (Native Float2)", [&]() {
            nc = na * scalar;
            result_accumulator += nc.x + nc.y;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 1500000);

        // 4. Dot Product
        lib_time = suite.measure("half2 Dot Product (Library)", [&]() {
            half result = dot(a, b);
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half2 Dot Product (Native Float2)", [&]() {
            float result = dot(na, nb);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Dot Product", lib_time, native_time, 1500000);

        // 5. Length
        lib_time = suite.measure("half2 Length (Library)", [&]() {
            half result = a.length();
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half2 Length (Native Float2)", [&]() {
            float result = na.length();
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Length", lib_time, native_time, 1500000);

        // 6. Normalization
        lib_time = suite.measure("half2 Normalize (Library)", [&]() {
            c = a.normalize();
            result_accumulator += float(c.x) + float(c.y);
            }).time_ms;

        native_time = suite.measure("half2 Normalize (Native Float2)", [&]() {
            nc = na.normalize();
            result_accumulator += nc.x + nc.y;
            }).time_ms;

        comparator.compare("Normalize", lib_time, native_time, 1500000);

        // 7. Cross Product
        lib_time = suite.measure("half2 Cross Product (Library)", [&]() {
            half result = cross(a, b);
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half2 Cross Product (Native Float2)", [&]() {
            float result = cross(na, nb);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Cross Product", lib_time, native_time, 1500000);

        suite.footer();
    }

    void benchmark_half2_mathematical_functions()
    {
        BenchmarkSuite suite("half2 Mathematical Functions", 1000000, true);
        suite.header();

        half2 a(2.0f, 3.0f);
        half2 b(1.0f, 2.0f);
        volatile float result_accumulator = 0.0f;

        // Mathematical operations
        suite.measure("half2 Length Squared", [&]() {
            half result = a.length_sq();
            result_accumulator += float(result);
            });

        suite.measure("half2 Distance", [&]() {
            half result = distance(a, b);
            result_accumulator += float(result);
            });

        suite.measure("half2 Distance Squared", [&]() {
            half result = distance_sq(a, b);
            result_accumulator += float(result);
            });

        suite.measure("half2 Angle", [&]() {
            half result = angle(a);
            result_accumulator += float(result);
            });

        suite.measure("half2 Angle Between", [&]() {
            half result = angle_between(a, b);
            result_accumulator += float(result);
            });

        suite.measure("half2 Perpendicular", [&]() {
            half2 result = a.perpendicular();
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Lerp", [&]() {
            half2 result = lerp(a, b, 0.5f);
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.footer();
    }

    void benchmark_half2_hlsl_functions()
    {
        BenchmarkSuite suite("half2 HLSL Functions", 1000000, true);
        suite.header();

        half2 v(1.5f, -2.5f);
        half2 w(0.3f, 0.7f);
        volatile float result_accumulator = 0.0f;

        // HLSL-style functions
        suite.measure("half2 Abs", [&]() {
            half2 result = abs(v);
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Sign", [&]() {
            half2 result = sign(v);
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Floor", [&]() {
            half2 result = floor(half2(1.7f, -2.3f));
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Ceil", [&]() {
            half2 result = ceil(half2(1.2f, -2.7f));
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Round", [&]() {
            half2 result = round(half2(1.4f, 1.6f));
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Frac", [&]() {
            half2 result = frac(half2(1.7f, -2.3f));
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Saturate", [&]() {
            half2 result = saturate(half2(-0.5f, 1.5f));
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Step", [&]() {
            half2 result = step(half(1.0f), half2(0.5f, 1.5f));
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Min", [&]() {
            half2 result = min(v, w);
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Max", [&]() {
            half2 result = max(v, w);
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Clamp", [&]() {
            half2 result = clamp(v, 0.0f, 1.0f);
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 Smoothstep", [&]() {
            half2 result = smoothstep(half(0.0f), half(1.0f), w);
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.footer();
    }

    void benchmark_half2_conversion_operations()
    {
        BenchmarkSuite suite("half2 Conversion Operations", 1000000, true);
        suite.header();

        half2 h2(1.5f, 2.5f);
        float2 f2(3.0f, 4.0f);
        volatile float result_accumulator = 0.0f;

        // Conversion operations
        suite.measure("half2 to float2", [&]() {
            float2 result = static_cast<float2>(h2);
            result_accumulator += result.x + result.y;
            });

        suite.measure("float2 to half2", [&]() {
            half2 result = half2(f2);
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("to_float2", [&]() {
            float2 result = to_float2(h2);
            result_accumulator += result.x + result.y;
            });

        suite.measure("to_half2", [&]() {
            half2 result = to_half2(f2);
            result_accumulator += float(result.x) + float(result.y);
            });

        // Mixed type operations
        suite.measure("half2 + float2", [&]() {
            half2 result = h2 + f2;
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("float2 + half2", [&]() {
            half2 result = f2 + h2;
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.footer();
    }

    void benchmark_half2_swizzle_operations()
    {
        BenchmarkSuite suite("half2 Swizzle Operations", 2000000, true);
        suite.header();

        half2 v(1.0f, 2.0f);
        volatile float result_accumulator = 0.0f;

        // Swizzle operations
        suite.measure("half2 YX Swizzle", [&]() {
            half2 result = v.yx();
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 XX Swizzle", [&]() {
            half2 result = v.xx();
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half2 YY Swizzle", [&]() {
            half2 result = v.yy();
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.footer();
    }

    void benchmark_half2_memory_patterns()
    {
        BenchmarkSuite suite("half2 Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = Half2BenchmarkData::generate_half2_data(ARRAY_SIZE);
        auto pairs = Half2BenchmarkData::generate_half2_pairs(ARRAY_SIZE);
        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("half2 Sequential Sum", [&]() {
            half2 sum = half2::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += float(sum.x) + float(sum.y);
            });

        suite.measure("half2 Sequential Dot Products", [&]() {
            half sum = half(0.0f);
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(data[i], data[i + 1]);
            }
            result_accumulator += float(sum);
            });

        suite.measure("half2 Sequential Cross Products", [&]() {
            half sum = half(0.0f);
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += cross(data[i], data[i + 1]);
            }
            result_accumulator += float(sum);
            });

        suite.measure("half2 Sequential Normalization", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half2 normalized = data[i].normalize();
                result_accumulator += float(normalized.x) + float(normalized.y);
            }
            });

        // Pair operations
        suite.measure("half2 Pair Operations", [&]() {
            half2 sum = half2::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += pairs[i].first * pairs[i].second;
            }
            result_accumulator += float(sum.x) + float(sum.y);
            });

        // HLSL operations pattern
        suite.measure("half2 Sequential Abs", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half2 result = abs(data[i]);
                result_accumulator += float(result.x) + float(result.y);
            }
            });

        // Random access pattern (simulated)
        suite.measure("half2 Strided Access", [&]() {
            half2 sum = half2::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += float(sum.x) + float(sum.y);
            });

        suite.footer();
    }

    void benchmark_half2_vs_float2_memory()
    {
        BenchmarkSuite suite("half2 vs Float2 Memory Bandwidth", BenchmarkConfig::MEMORY_ITERATIONS / 10, true);
        suite.header();

        const int ARRAY_SIZE = 5000;

        // Generate test data
        std::vector<half2> half2_data;
        std::vector<float2> float2_data;
        half2_data.reserve(ARRAY_SIZE);
        float2_data.reserve(ARRAY_SIZE);

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            float x = static_cast<float>(std::sin(i * 0.1f) * 2.0f);
            float y = static_cast<float>(std::cos(i * 0.1f) * 2.0f);
            half2_data.emplace_back(x, y);
            float2_data.emplace_back(x, y);
        }

        volatile float result_accumulator = 0.0f;

        // Memory bandwidth tests
        suite.measure("half2 Memory Sum", [&]() {
            half2 sum = half2::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += half2_data[i];
            }
            result_accumulator += float(sum.x) + float(sum.y);
            });

        suite.measure("float2 Memory Sum", [&]() {
            float2 sum = float2::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += float2_data[i];
            }
            result_accumulator += sum.x + sum.y;
            });

        suite.measure("half2 Memory Dot Products", [&]() {
            half sum = half(0.0f);
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(half2_data[i], half2_data[i + 1]);
            }
            result_accumulator += float(sum);
            });

        suite.measure("float2 Memory Dot Products", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(float2_data[i], float2_data[i + 1]);
            }
            result_accumulator += sum;
            });

        suite.footer();
    }

    // ============================================================================
    // Comprehensive Benchmark Runner for half2
    // ============================================================================

    void run_half2_benchmarks()
    {
        std::cout << "HALF2 VECTOR COMPREHENSIVE BENCHMARK SUITE" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "This benchmark compares half2 performance" << std::endl;
        std::cout << "against native float2 implementations." << std::endl;
        std::cout << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        benchmark_half2_operations();
        benchmark_half2_mathematical_functions();
        benchmark_half2_hlsl_functions();
        benchmark_half2_conversion_operations();
        benchmark_half2_swizzle_operations();
        benchmark_half2_memory_patterns();
        benchmark_half2_vs_float2_memory();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "HALF2 BENCHMARKS COMPLETED" << std::endl;
        std::cout << "Total benchmark time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ============================================================================
    // Benchmark Data Generators for half3
    // ============================================================================

    class Half3BenchmarkData
    {
    public:
        static std::vector<half3> generate_half3_data(int count)
        {
            std::vector<half3> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x = static_cast<float>(std::sin(i * 0.1f) * 2.0f);
                float y = static_cast<float>(std::cos(i * 0.1f) * 2.0f);
                float z = static_cast<float>(std::sin(i * 0.2f) * 2.0f);
                data.emplace_back(x, y, z);
            }
            return data;
        }

        static std::vector<std::pair<half3, half3>> generate_half3_pairs(int count)
        {
            std::vector<std::pair<half3, half3>> pairs;
            pairs.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x1 = static_cast<float>(std::sin(i * 0.1f) * 2.0f);
                float y1 = static_cast<float>(std::cos(i * 0.1f) * 2.0f);
                float z1 = static_cast<float>(std::sin(i * 0.2f) * 2.0f);

                float x2 = static_cast<float>(std::sin(i * 0.3f) * 2.0f);
                float y2 = static_cast<float>(std::cos(i * 0.3f) * 2.0f);
                float z2 = static_cast<float>(std::sin(i * 0.4f) * 2.0f);

                pairs.emplace_back(half3(x1, y1, z1), half3(x2, y2, z2));
            }
            return pairs;
        }
    };

    // ============================================================================
    // half3 Benchmarks - Comprehensive
    // ============================================================================

    void benchmark_half3_operations()
    {
        BenchmarkSuite suite("half3 Operations", 1000000, true);
        suite.header();
        ComparisonBenchmark comparator("half3 Operations");

        // Test data
        half3 a(1.5f, 2.5f, 3.5f);
        half3 b(4.0f, 5.0f, 6.0f);
        half3 c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison (using float3)
        float3 na(1.5f, 2.5f, 3.5f);
        float3 nb(4.0f, 5.0f, 6.0f);
        float3 nc;

        // 1. Addition
        auto lib_time = suite.measure("half3 Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += float(c.x) + float(c.y) + float(c.z);
            }).time_ms;

        auto native_time = suite.measure("half3 Addition (Native Float3)", [&]() {
            nc = na + nb;
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 1000000);

        // 2. Subtraction
        lib_time = suite.measure("half3 Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += float(c.x) + float(c.y) + float(c.z);
            }).time_ms;

        native_time = suite.measure("half3 Subtraction (Native Float3)", [&]() {
            nc = na - nb;
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 1000000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("half3 Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += float(c.x) + float(c.y) + float(c.z);
            }).time_ms;

        native_time = suite.measure("half3 Scalar Multiplication (Native Float3)", [&]() {
            nc = na * scalar;
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 1000000);

        // 4. Dot Product
        lib_time = suite.measure("half3 Dot Product (Library)", [&]() {
            half result = dot(a, b);
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half3 Dot Product (Native Float3)", [&]() {
            float result = dot(na, nb);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Dot Product", lib_time, native_time, 1000000);

        // 5. Cross Product
        lib_time = suite.measure("half3 Cross Product (Library)", [&]() {
            c = cross(a, b);
            result_accumulator += float(c.x) + float(c.y) + float(c.z);
            }).time_ms;

        native_time = suite.measure("half3 Cross Product (Native Float3)", [&]() {
            nc = cross(na, nb);
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Cross Product", lib_time, native_time, 1000000);

        // 6. Length
        lib_time = suite.measure("half3 Length (Library)", [&]() {
            half result = a.length();
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half3 Length (Native Float3)", [&]() {
            float result = na.length();
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Length", lib_time, native_time, 1000000);

        // 7. Normalization
        lib_time = suite.measure("half3 Normalize (Library)", [&]() {
            c = a.normalize();
            result_accumulator += float(c.x) + float(c.y) + float(c.z);
            }).time_ms;

        native_time = suite.measure("half3 Normalize (Native Float3)", [&]() {
            nc = na.normalize();
            result_accumulator += nc.x + nc.y + nc.z;
            }).time_ms;

        comparator.compare("Normalize", lib_time, native_time, 1000000);

        // 8. Distance
        lib_time = suite.measure("half3 Distance (Library)", [&]() {
            half result = distance(a, b);
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half3 Distance (Native Float3)", [&]() {
            float result = distance(na, nb);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Distance", lib_time, native_time, 1000000);

        suite.footer();
    }

    void benchmark_half3_mathematical_functions()
    {
        BenchmarkSuite suite("half3 Mathematical Functions", 800000, true);
        suite.header();

        half3 a(2.0f, 3.0f, 4.0f);
        half3 b(1.0f, 2.0f, 3.0f);
        volatile float result_accumulator = 0.0f;

        // Mathematical operations
        suite.measure("half3 Length Squared", [&]() {
            half result = a.length_sq();
            result_accumulator += float(result);
            });

        suite.measure("half3 Distance Squared", [&]() {
            half result = distance_sq(a, b);
            result_accumulator += float(result);
            });

        suite.measure("half3 Angle Between", [&]() {
            half result = angle_between(a, b);
            result_accumulator += float(result);
            });

        suite.measure("half3 Project", [&]() {
            half3 result = project(a, b);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Reject", [&]() {
            half3 result = reject(a, b);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Reflect", [&]() {
            half3 result = reflect(a.normalize(), b.normalize());
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Lerp", [&]() {
            half3 result = lerp(a, b, 0.5f);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.footer();
    }

    void benchmark_half3_hlsl_functions()
    {
        BenchmarkSuite suite("half3 HLSL Functions", 800000, true);
        suite.header();

        half3 v(1.5f, -2.5f, 0.5f);
        half3 w(0.3f, 0.7f, -0.2f);
        volatile float result_accumulator = 0.0f;

        // HLSL-style functions
        suite.measure("half3 Abs", [&]() {
            half3 result = abs(v);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Sign", [&]() {
            half3 result = sign(v);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Floor", [&]() {
            half3 result = floor(half3(1.7f, -2.3f, 0.8f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Ceil", [&]() {
            half3 result = ceil(half3(1.2f, -2.7f, 0.3f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Round", [&]() {
            half3 result = round(half3(1.4f, 1.6f, -1.5f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Frac", [&]() {
            half3 result = frac(half3(1.7f, -2.3f, 0.8f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Saturate", [&]() {
            half3 result = saturate(half3(-0.5f, 1.5f, 0.5f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Step", [&]() {
            half3 result = step(half(1.0f), half3(0.5f, 1.5f, 1.0f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Min", [&]() {
            half3 result = min(v, w);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Max", [&]() {
            half3 result = max(v, w);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Clamp", [&]() {
            half3 result = clamp(v, 0.0f, 1.0f);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Smoothstep", [&]() {
            half3 result = smoothstep(half(0.0f), half(1.0f), w);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.footer();
    }

    void benchmark_half3_color_operations()
    {
        BenchmarkSuite suite("half3 Color Operations", 600000, true);
        suite.header();

        half3 color(0.8f, 0.6f, 0.4f);
        volatile float result_accumulator = 0.0f;

        // Color operations
        suite.measure("half3 Luminance", [&]() {
            half result = color.luminance();
            result_accumulator += float(result);
            });

        suite.measure("half3 RGB to Grayscale", [&]() {
            half3 result = color.rgb_to_grayscale();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Gamma Correct", [&]() {
            half3 result = color.gamma_correct(2.2f);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 sRGB to Linear", [&]() {
            half3 result = color.srgb_to_linear();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 Linear to sRGB", [&]() {
            half3 result = color.linear_to_srgb();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.footer();
    }

    void benchmark_half3_swizzle_operations()
    {
        BenchmarkSuite suite("half3 Swizzle Operations", 1000000, true);
        suite.header();

        half3 v(1.0f, 2.0f, 3.0f);
        volatile float result_accumulator = 0.0f;

        // 2D swizzles
        suite.measure("half3 XY Swizzle", [&]() {
            half2 result = v.xy();
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half3 XZ Swizzle", [&]() {
            half2 result = v.xz();
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half3 YZ Swizzle", [&]() {
            half2 result = v.yz();
            result_accumulator += float(result.x) + float(result.y);
            });

        // 3D swizzles
        suite.measure("half3 YXZ Swizzle", [&]() {
            half3 result = v.yxz();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 ZXY Swizzle", [&]() {
            half3 result = v.zxy();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 ZYX Swizzle", [&]() {
            half3 result = v.zyx();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        // Color swizzles
        suite.measure("half3 RGB Swizzle", [&]() {
            half3 result = v.rgb();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 BGR Swizzle", [&]() {
            half3 result = v.bgr();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half3 GBR Swizzle", [&]() {
            half3 result = v.gbr();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.footer();
    }

    void benchmark_half3_conversion_operations()
    {
        BenchmarkSuite suite("half3 Conversion Operations", 800000, true);
        suite.header();

        half3 h3(1.5f, 2.5f, 3.5f);
        float3 f3(3.0f, 4.0f, 5.0f);
        volatile float result_accumulator = 0.0f;

        // Conversion operations
        suite.measure("half3 to float3", [&]() {
            float3 result = static_cast<float3>(h3);
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float3 to half3", [&]() {
            half3 result = half3(f3);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("to_float3", [&]() {
            float3 result = to_float3(h3);
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("to_half3", [&]() {
            half3 result = to_half3(f3);
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        // Mixed type operations
        suite.measure("half3 + float3", [&]() {
            half3 result = h3 + f3;
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("float3 + half3", [&]() {
            half3 result = f3 + h3;
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.footer();
    }

    void benchmark_half3_memory_patterns()
    {
        BenchmarkSuite suite("half3 Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = Half3BenchmarkData::generate_half3_data(ARRAY_SIZE);
        auto pairs = Half3BenchmarkData::generate_half3_pairs(ARRAY_SIZE);
        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("half3 Sequential Sum", [&]() {
            half3 sum = half3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z);
            });

        suite.measure("half3 Sequential Dot Products", [&]() {
            half sum = half(0.0f);
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(data[i], data[i + 1]);
            }
            result_accumulator += float(sum);
            });

        suite.measure("half3 Sequential Cross Products", [&]() {
            half3 sum = half3::zero();
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += cross(data[i], data[i + 1]);
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z);
            });

        suite.measure("half3 Sequential Normalization", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half3 normalized = data[i].normalize();
                result_accumulator += float(normalized.x) + float(normalized.y) + float(normalized.z);
            }
            });

        // Pair operations
        suite.measure("half3 Pair Operations", [&]() {
            half3 sum = half3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += pairs[i].first * pairs[i].second;
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z);
            });

        // HLSL operations pattern
        suite.measure("half3 Sequential Abs", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half3 result = abs(data[i]);
                result_accumulator += float(result.x) + float(result.y) + float(result.z);
            }
            });

        // Color operations pattern
        suite.measure("half3 Sequential Grayscale", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half3 gray = data[i].rgb_to_grayscale();
                result_accumulator += float(gray.x) + float(gray.y) + float(gray.z);
            }
            });

        // Random access pattern (simulated)
        suite.measure("half3 Strided Access", [&]() {
            half3 sum = half3::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z);
            });

        suite.footer();
    }

    void benchmark_half3_vs_float3_memory()
    {
        BenchmarkSuite suite("half3 vs Float3 Memory Bandwidth", BenchmarkConfig::MEMORY_ITERATIONS / 10, true);
        suite.header();

        const int ARRAY_SIZE = 5000;

        // Generate test data
        std::vector<half3> half3_data;
        std::vector<float3> float3_data;
        half3_data.reserve(ARRAY_SIZE);
        float3_data.reserve(ARRAY_SIZE);

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            float x = static_cast<float>(std::sin(i * 0.1f) * 2.0f);
            float y = static_cast<float>(std::cos(i * 0.1f) * 2.0f);
            float z = static_cast<float>(std::sin(i * 0.2f) * 2.0f);
            half3_data.emplace_back(x, y, z);
            float3_data.emplace_back(x, y, z);
        }

        volatile float result_accumulator = 0.0f;

        // Memory bandwidth tests
        suite.measure("half3 Memory Sum", [&]() {
            half3 sum = half3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += half3_data[i];
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z);
            });

        suite.measure("float3 Memory Sum", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += float3_data[i];
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        suite.measure("half3 Memory Dot Products", [&]() {
            half sum = half(0.0f);
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(half3_data[i], half3_data[i + 1]);
            }
            result_accumulator += float(sum);
            });

        suite.measure("float3 Memory Dot Products", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(float3_data[i], float3_data[i + 1]);
            }
            result_accumulator += sum;
            });

        suite.measure("half3 Memory Cross Products", [&]() {
            half3 sum = half3::zero();
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += cross(half3_data[i], half3_data[i + 1]);
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z);
            });

        suite.measure("float3 Memory Cross Products", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += cross(float3_data[i], float3_data[i + 1]);
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        suite.footer();
    }

    // ============================================================================
    // Comprehensive Benchmark Runner for half3
    // ============================================================================

    void run_half3_benchmarks()
    {
        std::cout << "HALF3 VECTOR COMPREHENSIVE BENCHMARK SUITE" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "This benchmark compares half3 performance" << std::endl;
        std::cout << "against native float3 implementations." << std::endl;
        std::cout << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        benchmark_half3_operations();
        benchmark_half3_mathematical_functions();
        benchmark_half3_hlsl_functions();
        benchmark_half3_color_operations();
        benchmark_half3_swizzle_operations();
        benchmark_half3_conversion_operations();
        benchmark_half3_memory_patterns();
        benchmark_half3_vs_float3_memory();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "HALF3 BENCHMARKS COMPLETED" << std::endl;
        std::cout << "Total benchmark time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ============================================================================
    // Benchmark Data Generators for half4
    // ============================================================================

    class Half4BenchmarkData
    {
    public:
        static std::vector<half4> generate_half4_data(int count)
        {
            std::vector<half4> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x = static_cast<float>(std::sin(i * 0.1f) * 2.0f);
                float y = static_cast<float>(std::cos(i * 0.1f) * 2.0f);
                float z = static_cast<float>(std::sin(i * 0.2f) * 2.0f);
                float w = static_cast<float>(std::cos(i * 0.2f) * 2.0f);
                data.emplace_back(x, y, z, w);
            }
            return data;
        }

        static std::vector<std::pair<half4, half4>> generate_half4_pairs(int count)
        {
            std::vector<std::pair<half4, half4>> pairs;
            pairs.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float x1 = static_cast<float>(std::sin(i * 0.1f) * 2.0f);
                float y1 = static_cast<float>(std::cos(i * 0.1f) * 2.0f);
                float z1 = static_cast<float>(std::sin(i * 0.2f) * 2.0f);
                float w1 = static_cast<float>(std::cos(i * 0.2f) * 2.0f);

                float x2 = static_cast<float>(std::sin(i * 0.3f) * 2.0f);
                float y2 = static_cast<float>(std::cos(i * 0.3f) * 2.0f);
                float z2 = static_cast<float>(std::sin(i * 0.4f) * 2.0f);
                float w2 = static_cast<float>(std::cos(i * 0.4f) * 2.0f);

                pairs.emplace_back(half4(x1, y1, z1, w1), half4(x2, y2, z2, w2));
            }
            return pairs;
        }
    };

    // ============================================================================
    // half4 Benchmarks - Comprehensive
    // ============================================================================

    void benchmark_half4_operations()
    {
        BenchmarkSuite suite("half4 Operations", 800000, true);
        suite.header();
        ComparisonBenchmark comparator("half4 Operations");

        // Test data
        half4 a(1.5f, 2.5f, 3.5f, 4.5f);
        half4 b(5.0f, 6.0f, 7.0f, 8.0f);
        half4 c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison (using float4)
        float4 na(1.5f, 2.5f, 3.5f, 4.5f);
        float4 nb(5.0f, 6.0f, 7.0f, 8.0f);
        float4 nc;

        // 1. Addition
        auto lib_time = suite.measure("half4 Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += float(c.x) + float(c.y) + float(c.z) + float(c.w);
            }).time_ms;

        auto native_time = suite.measure("half4 Addition (Native Float4)", [&]() {
            nc = na + nb;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 800000);

        // 2. Subtraction
        lib_time = suite.measure("half4 Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += float(c.x) + float(c.y) + float(c.z) + float(c.w);
            }).time_ms;

        native_time = suite.measure("half4 Subtraction (Native Float4)", [&]() {
            nc = na - nb;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 800000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("half4 Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += float(c.x) + float(c.y) + float(c.z) + float(c.w);
            }).time_ms;

        native_time = suite.measure("half4 Scalar Multiplication (Native Float4)", [&]() {
            nc = na * scalar;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 800000);

        // 4. Dot Product
        lib_time = suite.measure("half4 Dot Product (Library)", [&]() {
            half result = dot(a, b);
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half4 Dot Product (Native Float4)", [&]() {
            float result = dot(na, nb);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Dot Product", lib_time, native_time, 800000);

        // 5. 3D Dot Product
        lib_time = suite.measure("half4 3D Dot Product (Library)", [&]() {
            half result = dot3(a, b);
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half4 3D Dot Product (Native Float4)", [&]() {
            float result = na.x * nb.x + na.y * nb.y + na.z * nb.z;
            result_accumulator += result;
            }).time_ms;

        comparator.compare("3D Dot Product", lib_time, native_time, 800000);

        // 6. Cross Product
        lib_time = suite.measure("half4 Cross Product (Library)", [&]() {
            c = cross(a, b);
            result_accumulator += float(c.x) + float(c.y) + float(c.z) + float(c.w);
            }).time_ms;

        native_time = suite.measure("half4 Cross Product (Native Float4)", [&]() {
            nc = cross(na, nb);
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Cross Product", lib_time, native_time, 800000);

        // 7. Length
        lib_time = suite.measure("half4 Length (Library)", [&]() {
            half result = a.length();
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half4 Length (Native Float4)", [&]() {
            float result = na.length();
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Length", lib_time, native_time, 800000);

        // 8. Normalization
        lib_time = suite.measure("half4 Normalize (Library)", [&]() {
            c = a.normalize();
            result_accumulator += float(c.x) + float(c.y) + float(c.z) + float(c.w);
            }).time_ms;

        native_time = suite.measure("half4 Normalize (Native Float4)", [&]() {
            nc = na.normalize();
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Normalize", lib_time, native_time, 800000);

        // 9. Distance
        lib_time = suite.measure("half4 Distance (Library)", [&]() {
            half result = distance(a, b);
            result_accumulator += float(result);
            }).time_ms;

        native_time = suite.measure("half4 Distance (Native Float4)", [&]() {
            float result = distance(na, nb);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Distance", lib_time, native_time, 800000);

        suite.footer();
    }

    void benchmark_half4_mathematical_functions()
    {
        BenchmarkSuite suite("half4 Mathematical Functions", 600000, true);
        suite.header();

        half4 a(2.0f, 3.0f, 4.0f, 5.0f);
        half4 b(1.0f, 2.0f, 3.0f, 4.0f);
        volatile float result_accumulator = 0.0f;

        // Mathematical operations
        suite.measure("half4 Length Squared", [&]() {
            half result = a.length_sq();
            result_accumulator += float(result);
            });

        suite.measure("half4 Distance Squared", [&]() {
            half result = distance_sq(a, b);
            result_accumulator += float(result);
            });

        suite.measure("half4 Lerp", [&]() {
            half4 result = lerp(a, b, 0.5f);
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.footer();
    }

    void benchmark_half4_hlsl_functions()
    {
        BenchmarkSuite suite("half4 HLSL Functions", 600000, true);
        suite.header();

        half4 v(1.5f, -2.5f, 0.5f, -0.8f);
        half4 w(0.3f, 0.7f, -0.2f, 1.2f);
        volatile float result_accumulator = 0.0f;

        // HLSL-style functions
        suite.measure("half4 Abs", [&]() {
            half4 result = abs(v);
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Sign", [&]() {
            half4 result = sign(v);
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Floor", [&]() {
            half4 result = floor(half4(1.7f, -2.3f, 0.8f, -0.5f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Ceil", [&]() {
            half4 result = ceil(half4(1.2f, -2.7f, 0.3f, -0.2f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Round", [&]() {
            half4 result = round(half4(1.4f, 1.6f, -1.5f, 2.5f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Frac", [&]() {
            half4 result = frac(half4(1.7f, -2.3f, 0.8f, -0.5f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Saturate", [&]() {
            half4 result = saturate(half4(-0.5f, 1.5f, 0.5f, 2.0f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Step", [&]() {
            half4 result = step(half(1.0f), half4(0.5f, 1.5f, 1.0f, 2.0f));
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Min", [&]() {
            half4 result = min(v, w);
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Max", [&]() {
            half4 result = max(v, w);
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Clamp", [&]() {
            half4 result = clamp(v, 0.0f, 1.0f);
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Smoothstep", [&]() {
            half4 result = smoothstep(half(0.0f), half(1.0f), w);
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.footer();
    }

    void benchmark_half4_color_operations()
    {
        BenchmarkSuite suite("half4 Color Operations", 500000, true);
        suite.header();

        half4 color(0.8f, 0.6f, 0.4f, 0.9f);
        volatile float result_accumulator = 0.0f;

        // Color operations
        suite.measure("half4 Luminance", [&]() {
            half result = color.luminance();
            result_accumulator += float(result);
            });

        suite.measure("half4 Brightness", [&]() {
            half result = color.brightness();
            result_accumulator += float(result);
            });

        suite.measure("half4 Grayscale", [&]() {
            half4 result = color.grayscale();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Premultiply Alpha", [&]() {
            half4 result = color.premultiply_alpha();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Unpremultiply Alpha", [&]() {
            half4 result = color.unpremultiply_alpha();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 sRGB to Linear", [&]() {
            half4 result = color.srgb_to_linear();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 Linear to sRGB", [&]() {
            half4 result = color.linear_to_srgb();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.footer();
    }

    void benchmark_half4_geometric_operations()
    {
        BenchmarkSuite suite("half4 Geometric Operations", 500000, true);
        suite.header();

        half4 homogeneous(2.0f, 4.0f, 6.0f, 2.0f);
        half4 point3d(1.0f, 2.0f, 3.0f, 0.0f);
        volatile float result_accumulator = 0.0f;

        // Geometric operations
        suite.measure("half4 Project", [&]() {
            half3 result = homogeneous.project();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half4 To Homogeneous", [&]() {
            half4 result = point3d.to_homogeneous();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.footer();
    }

    void benchmark_half4_swizzle_operations()
    {
        BenchmarkSuite suite("half4 Swizzle Operations", 800000, true);
        suite.header();

        half4 v(1.0f, 2.0f, 3.0f, 4.0f);
        volatile float result_accumulator = 0.0f;

        // 2D swizzles
        suite.measure("half4 XY Swizzle", [&]() {
            half2 result = v.xy();
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half4 XZ Swizzle", [&]() {
            half2 result = v.xz();
            result_accumulator += float(result.x) + float(result.y);
            });

        suite.measure("half4 XW Swizzle", [&]() {
            half2 result = v.xw();
            result_accumulator += float(result.x) + float(result.y);
            });

        // 3D swizzles
        suite.measure("half4 XYZ Swizzle", [&]() {
            half3 result = v.xyz();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half4 XYW Swizzle", [&]() {
            half3 result = v.xyw();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half4 YZW Swizzle", [&]() {
            half3 result = v.yzw();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        // 4D swizzles
        suite.measure("half4 YXZW Swizzle", [&]() {
            half4 result = v.yxzw();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 WZYX Swizzle", [&]() {
            half4 result = v.wzyx();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        // Color swizzles
        suite.measure("half4 RGB Swizzle", [&]() {
            half3 result = v.rgb();
            result_accumulator += float(result.x) + float(result.y) + float(result.z);
            });

        suite.measure("half4 BGRA Swizzle", [&]() {
            half4 result = v.bgra();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("half4 ABGR Swizzle", [&]() {
            half4 result = v.abgr();
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.footer();
    }

    void benchmark_half4_conversion_operations()
    {
        BenchmarkSuite suite("half4 Conversion Operations", 600000, true);
        suite.header();

        half4 h4(1.5f, 2.5f, 3.5f, 4.5f);
        float4 f4(3.0f, 4.0f, 5.0f, 6.0f);
        volatile float result_accumulator = 0.0f;

        // Conversion operations
        suite.measure("half4 to float4", [&]() {
            float4 result = static_cast<float4>(h4);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("float4 to half4", [&]() {
            half4 result = half4(f4);
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("to_float4", [&]() {
            float4 result = to_float4(h4);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("to_half4", [&]() {
            half4 result = to_half4(f4);
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        // Mixed type operations
        suite.measure("half4 + float4", [&]() {
            half4 result = h4 + f4;
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.measure("float4 + half4", [&]() {
            half4 result = f4 + h4;
            result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            });

        suite.footer();
    }

    void benchmark_half4_memory_patterns()
    {
        BenchmarkSuite suite("half4 Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = Half4BenchmarkData::generate_half4_data(ARRAY_SIZE);
        auto pairs = Half4BenchmarkData::generate_half4_pairs(ARRAY_SIZE);
        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("half4 Sequential Sum", [&]() {
            half4 sum = half4::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z) + float(sum.w);
            });

        suite.measure("half4 Sequential Dot Products", [&]() {
            half sum = half(0.0f);
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(data[i], data[i + 1]);
            }
            result_accumulator += float(sum);
            });

        suite.measure("half4 Sequential 3D Dot Products", [&]() {
            half sum = half(0.0f);
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot3(data[i], data[i + 1]);
            }
            result_accumulator += float(sum);
            });

        suite.measure("half4 Sequential Cross Products", [&]() {
            half4 sum = half4::zero();
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += cross(data[i], data[i + 1]);
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z) + float(sum.w);
            });

        suite.measure("half4 Sequential Normalization", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half4 normalized = data[i].normalize();
                result_accumulator += float(normalized.x) + float(normalized.y) + float(normalized.z) + float(normalized.w);
            }
            });

        // Pair operations
        suite.measure("half4 Pair Operations", [&]() {
            half4 sum = half4::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += pairs[i].first * pairs[i].second;
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z) + float(sum.w);
            });

        // HLSL operations pattern
        suite.measure("half4 Sequential Abs", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half4 result = abs(data[i]);
                result_accumulator += float(result.x) + float(result.y) + float(result.z) + float(result.w);
            }
            });

        // Color operations pattern
        suite.measure("half4 Sequential Grayscale", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                half4 gray = data[i].grayscale();
                result_accumulator += float(gray.x) + float(gray.y) + float(gray.z) + float(gray.w);
            }
            });

        // Random access pattern (simulated)
        suite.measure("half4 Strided Access", [&]() {
            half4 sum = half4::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z) + float(sum.w);
            });

        suite.footer();
    }

    void benchmark_half4_vs_float4_memory()
    {
        BenchmarkSuite suite("half4 vs Float4 Memory Bandwidth", BenchmarkConfig::MEMORY_ITERATIONS / 10, true);
        suite.header();

        const int ARRAY_SIZE = 5000;

        // Generate test data
        std::vector<half4> half4_data;
        std::vector<float4> float4_data;
        half4_data.reserve(ARRAY_SIZE);
        float4_data.reserve(ARRAY_SIZE);

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            float x = static_cast<float>(std::sin(i * 0.1f) * 2.0f);
            float y = static_cast<float>(std::cos(i * 0.1f) * 2.0f);
            float z = static_cast<float>(std::sin(i * 0.2f) * 2.0f);
            float w = static_cast<float>(std::cos(i * 0.2f) * 2.0f);
            half4_data.emplace_back(x, y, z, w);
            float4_data.emplace_back(x, y, z, w);
        }

        volatile float result_accumulator = 0.0f;

        // Memory bandwidth tests
        suite.measure("half4 Memory Sum", [&]() {
            half4 sum = half4::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += half4_data[i];
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z) + float(sum.w);
            });

        suite.measure("float4 Memory Sum", [&]() {
            float4 sum = float4::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += float4_data[i];
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        suite.measure("half4 Memory Dot Products", [&]() {
            half sum = half(0.0f);
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(half4_data[i], half4_data[i + 1]);
            }
            result_accumulator += float(sum);
            });

        suite.measure("float4 Memory Dot Products", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(float4_data[i], float4_data[i + 1]);
            }
            result_accumulator += sum;
            });

        suite.measure("half4 Memory Cross Products", [&]() {
            half4 sum = half4::zero();
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += cross(half4_data[i], half4_data[i + 1]);
            }
            result_accumulator += float(sum.x) + float(sum.y) + float(sum.z) + float(sum.w);
            });

        suite.measure("float4 Memory Cross Products", [&]() {
            float4 sum = float4::zero();
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += cross(float4_data[i], float4_data[i + 1]);
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        suite.footer();
    }

    // ============================================================================
    // Comprehensive Benchmark Runner for half4
    // ============================================================================

    void run_half4_benchmarks()
    {
        std::cout << "HALF4 VECTOR COMPREHENSIVE BENCHMARK SUITE" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "This benchmark compares half4 performance" << std::endl;
        std::cout << "against native float4 implementations." << std::endl;
        std::cout << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        benchmark_half4_operations();
        benchmark_half4_mathematical_functions();
        benchmark_half4_hlsl_functions();
        benchmark_half4_color_operations();
        benchmark_half4_geometric_operations();
        benchmark_half4_swizzle_operations();
        benchmark_half4_conversion_operations();
        benchmark_half4_memory_patterns();
        benchmark_half4_vs_float4_memory();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "HALF4 BENCHMARKS COMPLETED" << std::endl;
        std::cout << "Total benchmark time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ============================================================================
    // Benchmark Data Generators for float2x2
    // ============================================================================

    class Float2x2BenchmarkData
    {
    public:
        static std::vector<float2x2> generate_float2x2_data(int count)
        {
            std::vector<float2x2> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float angle = static_cast<float>(i) * 0.1f;
                float2x2 mat = float2x2::rotation(angle) * float2x2::scaling(1.0f + std::sin(angle) * 0.5f);
                data.push_back(mat);
            }
            return data;
        }

        static std::vector<std::pair<float2x2, float2x2>> generate_float2x2_pairs(int count)
        {
            std::vector<std::pair<float2x2, float2x2>> pairs;
            pairs.reserve(count);

            auto data = generate_float2x2_data(count * 2);
            for (int i = 0; i < count; ++i)
            {
                pairs.emplace_back(data[i * 2], data[i * 2 + 1]);
            }
            return pairs;
        }

        static std::vector<std::pair<float2x2, float2>> generate_float2x2_vector_pairs(int count)
        {
            std::vector<std::pair<float2x2, float2>> pairs;
            pairs.reserve(count);

            auto matrices = generate_float2x2_data(count);
            auto vectors = BenchmarkData::generate_float2_data(count);

            for (int i = 0; i < count; ++i)
            {
                pairs.emplace_back(matrices[i], vectors[i]);
            }
            return pairs;
        }
    };

    // ============================================================================
    // float2x2 Benchmarks - Comprehensive
    // ============================================================================

    void benchmark_float2x2_operations()
    {
        BenchmarkSuite suite("float2x2 Operations", 1000000, true);
        suite.header();
        ComparisonBenchmark comparator("float2x2 Operations");

        // Test data
        float2x2 a(1.0f, 2.0f, 3.0f, 4.0f);
        float2x2 b(5.0f, 6.0f, 7.0f, 8.0f);
        float2x2 c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison
        struct NativeFloat2x2 { float m00, m10, m01, m11; };
        NativeFloat2x2 na{ 1.0f, 2.0f, 3.0f, 4.0f };
        NativeFloat2x2 nb{ 5.0f, 6.0f, 7.0f, 8.0f };
        NativeFloat2x2 nc{ 0.0f, 0.0f, 0.0f, 0.0f };

        // 1. Addition
        auto lib_time = suite.measure("float2x2 Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += c(0, 0) + c(1, 1);
            }).time_ms;

        auto native_time = suite.measure("float2x2 Addition (Native)", [&]() {
            nc.m00 = na.m00 + nb.m00;
            nc.m10 = na.m10 + nb.m10;
            nc.m01 = na.m01 + nb.m01;
            nc.m11 = na.m11 + nb.m11;
            result_accumulator += nc.m00 + nc.m11;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 1000000);

        // 2. Subtraction
        lib_time = suite.measure("float2x2 Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += c(0, 0) + c(1, 1);
            }).time_ms;

        native_time = suite.measure("float2x2 Subtraction (Native)", [&]() {
            nc.m00 = na.m00 - nb.m00;
            nc.m10 = na.m10 - nb.m10;
            nc.m01 = na.m01 - nb.m01;
            nc.m11 = na.m11 - nb.m11;
            result_accumulator += nc.m00 + nc.m11;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 1000000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("float2x2 Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += c(0, 0) + c(1, 1);
            }).time_ms;

        native_time = suite.measure("float2x2 Scalar Multiplication (Native)", [&]() {
            nc.m00 = na.m00 * scalar;
            nc.m10 = na.m10 * scalar;
            nc.m01 = na.m01 * scalar;
            nc.m11 = na.m11 * scalar;
            result_accumulator += nc.m00 + nc.m11;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 1000000);

        // 4. Matrix Multiplication
        lib_time = suite.measure("float2x2 Matrix Multiplication (Library)", [&]() {
            c = a * b;
            result_accumulator += c(0, 0) + c(1, 1);
            }).time_ms;

        native_time = suite.measure("float2x2 Matrix Multiplication (Native)", [&]() {
            nc.m00 = na.m00 * nb.m00 + na.m01 * nb.m10;
            nc.m10 = na.m10 * nb.m00 + na.m11 * nb.m10;
            nc.m01 = na.m00 * nb.m01 + na.m01 * nb.m11;
            nc.m11 = na.m10 * nb.m01 + na.m11 * nb.m11;
            result_accumulator += nc.m00 + nc.m11;
            }).time_ms;

        comparator.compare("Matrix Multiplication", lib_time, native_time, 1000000);

        // 5. Determinant
        lib_time = suite.measure("float2x2 Determinant (Library)", [&]() {
            float result = a.determinant();
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float2x2 Determinant (Native)", [&]() {
            float result = na.m00 * na.m11 - na.m10 * na.m01;
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Determinant", lib_time, native_time, 1000000);

        // 6. Transpose
        lib_time = suite.measure("float2x2 Transpose (Library)", [&]() {
            c = a.transposed();
            result_accumulator += c(0, 0) + c(1, 1);
            }).time_ms;

        native_time = suite.measure("float2x2 Transpose (Native)", [&]() {
            nc.m00 = na.m00;
            nc.m10 = na.m01;
            nc.m01 = na.m10;
            nc.m11 = na.m11;
            result_accumulator += nc.m00 + nc.m11;
            }).time_ms;

        comparator.compare("Transpose", lib_time, native_time, 1000000);

        suite.footer();
    }

    void benchmark_float2x2_advanced_operations()
    {
        BenchmarkSuite suite("float2x2 Advanced Operations", 500000, true);
        suite.header();

        float2x2 a(1.0f, 2.0f, 3.0f, 4.0f);
        float2x2 b(5.0f, 6.0f, 7.0f, 8.0f);
        float2 vec(2.0f, 3.0f);
        volatile float result_accumulator = 0.0f;

        // Advanced matrix operations
        suite.measure("float2x2 Inverse", [&]() {
            float2x2 result = a.inverted();
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.measure("float2x2 Adjugate", [&]() {
            float2x2 result = a.adjugate();
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.measure("float2x2 Trace", [&]() {
            float result = a.trace();
            result_accumulator += result;
            });

        suite.measure("float2x2 Frobenius Norm", [&]() {
            float result = a.frobenius_norm();
            result_accumulator += result;
            });

        // Vector transformations
        suite.measure("float2x2 Transform Vector", [&]() {
            float2 result = a.transform_vector(vec);
            result_accumulator += result.x + result.y;
            });

        suite.measure("float2x2 Transform Point", [&]() {
            float2 result = a.transform_point(vec);
            result_accumulator += result.x + result.y;
            });

        suite.measure("float2 * float2x2", [&]() {
            float2 result = vec * a;
            result_accumulator += result.x + result.y;
            });

        // Utility functions
        suite.measure("float2x2 Is Identity", [&]() {
            bool result = a.is_identity();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("float2x2 Is Orthogonal", [&]() {
            bool result = a.is_orthogonal();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("float2x2 Is Rotation", [&]() {
            bool result = a.is_rotation();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.footer();
    }

    void benchmark_float2x2_static_constructors()
    {
        BenchmarkSuite suite("float2x2 Static Constructors", 1000000, true);
        suite.header();

        volatile float result_accumulator = 0.0f;
        float angle = Constants::PI / 4.0f;
        float2 scale(2.0f, 3.0f);
        float2 shear_factors(0.5f, 0.3f);

        // Static constructor benchmarks
        suite.measure("float2x2 Identity", [&]() {
            float2x2 result = float2x2::identity();
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.measure("float2x2 Zero", [&]() {
            float2x2 result = float2x2::zero();
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.measure("float2x2 Rotation", [&]() {
            float2x2 result = float2x2::rotation(angle);
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.measure("float2x2 Scaling (vector)", [&]() {
            float2x2 result = float2x2::scaling(scale);
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.measure("float2x2 Scaling (components)", [&]() {
            float2x2 result = float2x2::scaling(scale.x, scale.y);
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.measure("float2x2 Scaling (uniform)", [&]() {
            float2x2 result = float2x2::scaling(2.0f);
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.measure("float2x2 Shear (vector)", [&]() {
            float2x2 result = float2x2::shear(shear_factors);
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.measure("float2x2 Shear (components)", [&]() {
            float2x2 result = float2x2::shear(shear_factors.x, shear_factors.y);
            result_accumulator += result(0, 0) + result(1, 1);
            });

        suite.footer();
    }

    void benchmark_float2x2_transformation_components()
    {
        BenchmarkSuite suite("float2x2 Transformation Components", 500000, true);
        suite.header();

        float2x2 rotation_matrix = float2x2::rotation(Constants::PI / 6.0f);
        float2x2 scaling_matrix = float2x2::scaling(2.0f, 3.0f);
        float2x2 composite_matrix = rotation_matrix * scaling_matrix;

        volatile float result_accumulator = 0.0f;

        // Transformation component extraction
        suite.measure("float2x2 Get Rotation", [&]() {
            float result = composite_matrix.get_rotation();
            result_accumulator += result;
            });

        suite.measure("float2x2 Get Scale", [&]() {
            float2 result = composite_matrix.get_scale();
            result_accumulator += result.x + result.y;
            });

        suite.measure("float2x2 Set Rotation", [&]() {
            float2x2 mat = composite_matrix;
            mat.set_rotation(Constants::PI / 3.0f);
            result_accumulator += mat(0, 0) + mat(1, 1);
            });

        suite.measure("float2x2 Set Scale", [&]() {
            float2x2 mat = composite_matrix;
            mat.set_scale(float2(4.0f, 5.0f));
            result_accumulator += mat(0, 0) + mat(1, 1);
            });

        suite.footer();
    }

    void benchmark_float2x2_memory_patterns()
    {
        BenchmarkSuite suite("float2x2 Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = Float2x2BenchmarkData::generate_float2x2_data(ARRAY_SIZE);
        auto pairs = Float2x2BenchmarkData::generate_float2x2_pairs(ARRAY_SIZE);
        auto vector_pairs = Float2x2BenchmarkData::generate_float2x2_vector_pairs(ARRAY_SIZE);

        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("float2x2 Sequential Sum", [&]() {
            float2x2 sum = float2x2::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += sum(0, 0) + sum(1, 1);
            });

        suite.measure("float2x2 Sequential Multiplication", [&]() {
            float2x2 product = float2x2::identity();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= data[i];
            }
            result_accumulator += product(0, 0) + product(1, 1);
            });

        suite.measure("float2x2 Sequential Determinants", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i].determinant();
            }
            result_accumulator += sum;
            });

        // Pair operations
        suite.measure("float2x2 Pair Multiplication", [&]() {
            float2x2 sum = float2x2::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += pairs[i].first * pairs[i].second;
            }
            result_accumulator += sum(0, 0) + sum(1, 1);
            });

        // Vector transformation patterns
        suite.measure("float2x2 Sequential Vector Transform", [&]() {
            float2 sum = float2::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += vector_pairs[i].first.transform_vector(vector_pairs[i].second);
            }
            result_accumulator += sum.x + sum.y;
            });

        // Matrix operations pattern
        suite.measure("float2x2 Sequential Inverse", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float2x2 inv = data[i].inverted();
                result_accumulator += inv(0, 0) + inv(1, 1);
            }
            });

        // Random access pattern (simulated)
        suite.measure("float2x2 Strided Access", [&]() {
            float2x2 sum = float2x2::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += sum(0, 0) + sum(1, 1);
            });

        suite.footer();
    }

    void benchmark_float2x2_vs_native_memory()
    {
        BenchmarkSuite suite("float2x2 vs Native Memory Bandwidth", BenchmarkConfig::MEMORY_ITERATIONS / 10, true);
        suite.header();

        const int ARRAY_SIZE = 5000;

        // Generate test data
        std::vector<float2x2> float2x2_data;
        std::vector<std::array<float, 4>> native_data; // column-major storage

        float2x2_data.reserve(ARRAY_SIZE);
        native_data.reserve(ARRAY_SIZE);

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            float angle = static_cast<float>(i) * 0.1f;
            float2x2 mat = float2x2::rotation(angle) * float2x2::scaling(1.0f + std::sin(angle) * 0.5f);
            float2x2_data.push_back(mat);

            std::array<float, 4> native_mat = {
                mat(0, 0), mat(1, 0), mat(0, 1), mat(1, 1)
            };
            native_data.push_back(native_mat);
        }

        volatile float result_accumulator = 0.0f;

        // Memory bandwidth tests
        suite.measure("float2x2 Memory Sum", [&]() {
            float2x2 sum = float2x2::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += float2x2_data[i];
            }
            result_accumulator += sum(0, 0) + sum(1, 1);
            });

        suite.measure("Native Memory Sum", [&]() {
            std::array<float, 4> sum = { 0.0f, 0.0f, 0.0f, 0.0f };
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum[0] += native_data[i][0];
                sum[1] += native_data[i][1];
                sum[2] += native_data[i][2];
                sum[3] += native_data[i][3];
            }
            result_accumulator += sum[0] + sum[3];
            });

        suite.measure("float2x2 Memory Multiplication", [&]() {
            float2x2 product = float2x2::identity();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= float2x2_data[i];
            }
            result_accumulator += product(0, 0) + product(1, 1);
            });

        suite.measure("Native Memory Multiplication", [&]() {
            std::array<float, 4> product = { 1.0f, 0.0f, 0.0f, 1.0f }; // identity
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                std::array<float, 4> temp;
                temp[0] = product[0] * native_data[i][0] + product[2] * native_data[i][1];
                temp[1] = product[1] * native_data[i][0] + product[3] * native_data[i][1];
                temp[2] = product[0] * native_data[i][2] + product[2] * native_data[i][3];
                temp[3] = product[1] * native_data[i][2] + product[3] * native_data[i][3];
                product = temp;
            }
            result_accumulator += product[0] + product[3];
            });

        suite.footer();
    }

    // ============================================================================
    // Comprehensive Benchmark Runner for float2x2
    // ============================================================================

    void run_float2x2_benchmarks()
    {
        std::cout << "FLOAT2X2 MATRIX COMPREHENSIVE BENCHMARK SUITE" << std::endl;
        std::cout << "==============================================" << std::endl;
        std::cout << "This benchmark compares float2x2 performance" << std::endl;
        std::cout << "against native C++ implementations." << std::endl;
        std::cout << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        benchmark_float2x2_operations();
        benchmark_float2x2_advanced_operations();
        benchmark_float2x2_static_constructors();
        benchmark_float2x2_transformation_components();
        benchmark_float2x2_memory_patterns();
        benchmark_float2x2_vs_native_memory();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "FLOAT2X2 BENCHMARKS COMPLETED" << std::endl;
        std::cout << "Total benchmark time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ============================================================================
// Benchmark Data Generators for float3x3
// ============================================================================

    class Float3x3BenchmarkData
    {
    public:
        static std::vector<float3x3> generate_float3x3_data(int count)
        {
            std::vector<float3x3> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float angle = static_cast<float>(i) * 0.1f;
                float3x3 rot_x = float3x3::rotation_x(angle);
                float3x3 rot_y = float3x3::rotation_y(angle * 0.5f);
                float3x3 scale = float3x3::scaling(1.0f + std::sin(angle) * 0.5f);
                float3x3 mat = rot_x * rot_y * scale;
                data.push_back(mat);
            }
            return data;
        }

        static std::vector<std::pair<float3x3, float3x3>> generate_float3x3_pairs(int count)
        {
            std::vector<std::pair<float3x3, float3x3>> pairs;
            pairs.reserve(count);

            auto data = generate_float3x3_data(count * 2);
            for (int i = 0; i < count; ++i)
            {
                pairs.emplace_back(data[i * 2], data[i * 2 + 1]);
            }
            return pairs;
        }

        static std::vector<std::pair<float3x3, float3>> generate_float3x3_vector_pairs(int count)
        {
            std::vector<std::pair<float3x3, float3>> pairs;
            pairs.reserve(count);

            auto matrices = generate_float3x3_data(count);
            auto vectors = BenchmarkData::generate_float3_data(count);

            for (int i = 0; i < count; ++i)
            {
                pairs.emplace_back(matrices[i], vectors[i]);
            }
            return pairs;
        }
    };

    // ============================================================================
    // float3x3 Benchmarks - Comprehensive
    // ============================================================================

    void benchmark_float3x3_operations()
    {
        BenchmarkSuite suite("float3x3 Operations", 500000, true);
        suite.header();
        ComparisonBenchmark comparator("float3x3 Operations");

        // Test data
        float3x3 a(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
        float3x3 b(9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
        float3x3 c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison
        struct NativeFloat3x3 { float m00, m10, m20, m01, m11, m21, m02, m12, m22; };
        NativeFloat3x3 na{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
        NativeFloat3x3 nb{ 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
        NativeFloat3x3 nc{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        // 1. Addition
        auto lib_time = suite.measure("float3x3 Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += c(0, 0) + c(1, 1) + c(2, 2);
            }).time_ms;

        auto native_time = suite.measure("float3x3 Addition (Native)", [&]() {
            nc.m00 = na.m00 + nb.m00;
            nc.m10 = na.m10 + nb.m10;
            nc.m20 = na.m20 + nb.m20;
            nc.m01 = na.m01 + nb.m01;
            nc.m11 = na.m11 + nb.m11;
            nc.m21 = na.m21 + nb.m21;
            nc.m02 = na.m02 + nb.m02;
            nc.m12 = na.m12 + nb.m12;
            nc.m22 = na.m22 + nb.m22;
            result_accumulator += nc.m00 + nc.m11 + nc.m22;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 500000);

        // 2. Subtraction
        lib_time = suite.measure("float3x3 Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += c(0, 0) + c(1, 1) + c(2, 2);
            }).time_ms;

        native_time = suite.measure("float3x3 Subtraction (Native)", [&]() {
            nc.m00 = na.m00 - nb.m00;
            nc.m10 = na.m10 - nb.m10;
            nc.m20 = na.m20 - nb.m20;
            nc.m01 = na.m01 - nb.m01;
            nc.m11 = na.m11 - nb.m11;
            nc.m21 = na.m21 - nb.m21;
            nc.m02 = na.m02 - nb.m02;
            nc.m12 = na.m12 - nb.m12;
            nc.m22 = na.m22 - nb.m22;
            result_accumulator += nc.m00 + nc.m11 + nc.m22;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 500000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("float3x3 Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += c(0, 0) + c(1, 1) + c(2, 2);
            }).time_ms;

        native_time = suite.measure("float3x3 Scalar Multiplication (Native)", [&]() {
            nc.m00 = na.m00 * scalar;
            nc.m10 = na.m10 * scalar;
            nc.m20 = na.m20 * scalar;
            nc.m01 = na.m01 * scalar;
            nc.m11 = na.m11 * scalar;
            nc.m21 = na.m21 * scalar;
            nc.m02 = na.m02 * scalar;
            nc.m12 = na.m12 * scalar;
            nc.m22 = na.m22 * scalar;
            result_accumulator += nc.m00 + nc.m11 + nc.m22;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 500000);

        // 4. Matrix Multiplication
        lib_time = suite.measure("float3x3 Matrix Multiplication (Library)", [&]() {
            c = a * b;
            result_accumulator += c(0, 0) + c(1, 1) + c(2, 2);
            }).time_ms;

        native_time = suite.measure("float3x3 Matrix Multiplication (Native)", [&]() {
            nc.m00 = na.m00 * nb.m00 + na.m01 * nb.m10 + na.m02 * nb.m20;
            nc.m10 = na.m10 * nb.m00 + na.m11 * nb.m10 + na.m12 * nb.m20;
            nc.m20 = na.m20 * nb.m00 + na.m21 * nb.m10 + na.m22 * nb.m20;
            nc.m01 = na.m00 * nb.m01 + na.m01 * nb.m11 + na.m02 * nb.m21;
            nc.m11 = na.m10 * nb.m01 + na.m11 * nb.m11 + na.m12 * nb.m21;
            nc.m21 = na.m20 * nb.m01 + na.m21 * nb.m11 + na.m22 * nb.m21;
            nc.m02 = na.m00 * nb.m02 + na.m01 * nb.m12 + na.m02 * nb.m22;
            nc.m12 = na.m10 * nb.m02 + na.m11 * nb.m12 + na.m12 * nb.m22;
            nc.m22 = na.m20 * nb.m02 + na.m21 * nb.m12 + na.m22 * nb.m22;
            result_accumulator += nc.m00 + nc.m11 + nc.m22;
            }).time_ms;

        comparator.compare("Matrix Multiplication", lib_time, native_time, 500000);

        // 5. Determinant
        lib_time = suite.measure("float3x3 Determinant (Library)", [&]() {
            float result = a.determinant();
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("float3x3 Determinant (Native)", [&]() {
            float result = na.m00 * (na.m11 * na.m22 - na.m21 * na.m12)
                - na.m01 * (na.m10 * na.m22 - na.m20 * na.m12)
                + na.m02 * (na.m10 * na.m21 - na.m20 * na.m11);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Determinant", lib_time, native_time, 500000);

        // 6. Transpose
        lib_time = suite.measure("float3x3 Transpose (Library)", [&]() {
            c = a.transposed();
            result_accumulator += c(0, 0) + c(1, 1) + c(2, 2);
            }).time_ms;

        native_time = suite.measure("float3x3 Transpose (Native)", [&]() {
            nc.m00 = na.m00;
            nc.m10 = na.m01;
            nc.m20 = na.m02;
            nc.m01 = na.m10;
            nc.m11 = na.m11;
            nc.m21 = na.m12;
            nc.m02 = na.m20;
            nc.m12 = na.m21;
            nc.m22 = na.m22;
            result_accumulator += nc.m00 + nc.m11 + nc.m22;
            }).time_ms;

        comparator.compare("Transpose", lib_time, native_time, 500000);

        suite.footer();
    }

    void benchmark_float3x3_advanced_operations()
    {
        BenchmarkSuite suite("float3x3 Advanced Operations", 300000, true);
        suite.header();

        float3x3 a(2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f); // invertible
        float3x3 b(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
        float3 vec(2.0f, 3.0f, 4.0f);
        volatile float result_accumulator = 0.0f;

        // Advanced matrix operations
        suite.measure("float3x3 Inverse", [&]() {
            float3x3 result = a.inverted();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Trace", [&]() {
            float result = a.trace();
            result_accumulator += result;
            });

        suite.measure("float3x3 Frobenius Norm", [&]() {
            float result = a.frobenius_norm();
            result_accumulator += result;
            });

        suite.measure("float3x3 Diagonal", [&]() {
            float3 result = a.diagonal();
            result_accumulator += result.x + result.y + result.z;
            });

        // Vector transformations
        suite.measure("float3x3 Transform Vector", [&]() {
            float3 result = a.transform_vector(vec);
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float3x3 Transform Point", [&]() {
            float3 result = a.transform_point(vec);
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float3x3 Transform Normal", [&]() {
            float3 result = a.transform_normal(vec);
            result_accumulator += result.x + result.y + result.z;
            });

        // Decomposition methods
        suite.measure("float3x3 Extract Scale", [&]() {
            float3 result = a.extract_scale();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float3x3 Extract Rotation", [&]() {
            float3x3 result = a.extract_rotation();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        // Matrix properties
        suite.measure("float3x3 Is Identity", [&]() {
            bool result = a.is_identity();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("float3x3 Is Orthogonal", [&]() {
            bool result = a.is_orthogonal();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("float3x3 Approximately Equal", [&]() {
            bool result = a.approximately(b, 0.001f);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.footer();
    }

    void benchmark_float3x3_static_constructors()
    {
        BenchmarkSuite suite("float3x3 Static Constructors", 400000, true);
        suite.header();

        volatile float result_accumulator = 0.0f;
        float angle = Constants::PI / 4.0f;
        float3 scale(2.0f, 3.0f, 4.0f);
        float3 axis(1.0f, 1.0f, 1.0f);
        float3 euler_angles(Constants::PI / 6.0f, Constants::PI / 4.0f, Constants::PI / 3.0f);

        // Static constructor benchmarks
        suite.measure("float3x3 Identity", [&]() {
            float3x3 result = float3x3::identity();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Zero", [&]() {
            float3x3 result = float3x3::zero();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Rotation X", [&]() {
            float3x3 result = float3x3::rotation_x(angle);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Rotation Y", [&]() {
            float3x3 result = float3x3::rotation_y(angle);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Rotation Z", [&]() {
            float3x3 result = float3x3::rotation_z(angle);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Rotation Axis", [&]() {
            float3x3 result = float3x3::rotation_axis(axis.normalize(), angle);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Rotation Euler", [&]() {
            float3x3 result = float3x3::rotation_euler(euler_angles);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Scaling (vector)", [&]() {
            float3x3 result = float3x3::scaling(scale);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Scaling (uniform)", [&]() {
            float3x3 result = float3x3::scaling(2.0f);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Skew Symmetric", [&]() {
            float3x3 result = float3x3::skew_symmetric(axis);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("float3x3 Outer Product", [&]() {
            float3x3 result = float3x3::outer_product(axis, scale);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.footer();
    }

    void benchmark_float3x3_memory_patterns()
    {
        BenchmarkSuite suite("float3x3 Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = Float3x3BenchmarkData::generate_float3x3_data(ARRAY_SIZE);
        auto pairs = Float3x3BenchmarkData::generate_float3x3_pairs(ARRAY_SIZE);
        auto vector_pairs = Float3x3BenchmarkData::generate_float3x3_vector_pairs(ARRAY_SIZE);

        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("float3x3 Sequential Sum", [&]() {
            float3x3 sum = float3x3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += sum(0, 0) + sum(1, 1) + sum(2, 2);
            });

        suite.measure("float3x3 Sequential Multiplication", [&]() {
            float3x3 product = float3x3::identity();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= data[i];
            }
            result_accumulator += product(0, 0) + product(1, 1) + product(2, 2);
            });

        suite.measure("float3x3 Sequential Determinants", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i].determinant();
            }
            result_accumulator += sum;
            });

        suite.measure("float3x3 Sequential Transpose", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float3x3 transposed = data[i].transposed();
                result_accumulator += transposed(0, 0) + transposed(1, 1) + transposed(2, 2);
            }
            });

        // Pair operations
        suite.measure("float3x3 Pair Multiplication", [&]() {
            float3x3 sum = float3x3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += pairs[i].first * pairs[i].second;
            }
            result_accumulator += sum(0, 0) + sum(1, 1) + sum(2, 2);
            });

        // Vector transformation patterns
        suite.measure("float3x3 Sequential Vector Transform", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += vector_pairs[i].first.transform_vector(vector_pairs[i].second);
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        suite.measure("float3x3 Sequential Normal Transform", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += vector_pairs[i].first.transform_normal(vector_pairs[i].second);
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        // Matrix operations pattern
        suite.measure("float3x3 Sequential Inverse", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float3x3 inv = data[i].inverted();
                result_accumulator += inv(0, 0) + inv(1, 1) + inv(2, 2);
            }
            });

        // Random access pattern (simulated)
        suite.measure("float3x3 Strided Access", [&]() {
            float3x3 sum = float3x3::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += sum(0, 0) + sum(1, 1) + sum(2, 2);
            });

        suite.footer();
    }

    void benchmark_float3x3_vs_native_memory()
    {
        BenchmarkSuite suite("float3x3 vs Native Memory Bandwidth", BenchmarkConfig::MEMORY_ITERATIONS / 10, true);
        suite.header();

        const int ARRAY_SIZE = 5000;

        // Generate test data
        std::vector<float3x3> float3x3_data;
        std::vector<std::array<float, 9>> native_data; // column-major storage

        float3x3_data.reserve(ARRAY_SIZE);
        native_data.reserve(ARRAY_SIZE);

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            float angle = static_cast<float>(i) * 0.1f;
            float3x3 rot_x = float3x3::rotation_x(angle);
            float3x3 rot_y = float3x3::rotation_y(angle * 0.5f);
            float3x3 scale = float3x3::scaling(1.0f + std::sin(angle) * 0.5f);
            float3x3 mat = rot_x * rot_y * scale;
            float3x3_data.push_back(mat);

            std::array<float, 9> native_mat = {
                mat(0, 0), mat(1, 0), mat(2, 0),
                mat(0, 1), mat(1, 1), mat(2, 1),
                mat(0, 2), mat(1, 2), mat(2, 2)
            };
            native_data.push_back(native_mat);
        }

        volatile float result_accumulator = 0.0f;

        // Memory bandwidth tests
        suite.measure("float3x3 Memory Sum", [&]() {
            float3x3 sum = float3x3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += float3x3_data[i];
            }
            result_accumulator += sum(0, 0) + sum(1, 1) + sum(2, 2);
            });

        suite.measure("Native Memory Sum", [&]() {
            std::array<float, 9> sum = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                for (int j = 0; j < 9; ++j) {
                    sum[j] += native_data[i][j];
                }
            }
            result_accumulator += sum[0] + sum[4] + sum[8];
            });

        suite.measure("float3x3 Memory Multiplication", [&]() {
            float3x3 product = float3x3::identity();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= float3x3_data[i];
            }
            result_accumulator += product(0, 0) + product(1, 1) + product(2, 2);
            });

        suite.footer();
    }

    // ============================================================================
    // Comprehensive Benchmark Runner for float3x3
    // ============================================================================

    void run_float3x3_benchmarks()
    {
        std::cout << "FLOAT3X3 MATRIX COMPREHENSIVE BENCHMARK SUITE" << std::endl;
        std::cout << "==============================================" << std::endl;
        std::cout << "This benchmark compares float3x3 performance" << std::endl;
        std::cout << "against native C++ implementations." << std::endl;
        std::cout << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        benchmark_float3x3_operations();
        benchmark_float3x3_advanced_operations();
        benchmark_float3x3_static_constructors();
        benchmark_float3x3_memory_patterns();
        benchmark_float3x3_vs_native_memory();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "FLOAT3X3 BENCHMARKS COMPLETED" << std::endl;
        std::cout << "Total benchmark time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    class Float4x4BenchmarkData
    {
    public:
        static std::vector<float4x4> generate_float4x4_data(int count)
        {
            std::vector<float4x4> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float angle = static_cast<float>(i) * 0.1f;
                float3 translation(std::sin(angle), std::cos(angle), std::sin(angle * 0.5f));
                float3 scale(1.0f + std::sin(angle) * 0.2f, 1.0f + std::cos(angle) * 0.2f, 1.0f);

                float4x4 mat = float4x4::translation(translation) *
                    float4x4::rotation_y(angle) *
                    float4x4::scaling(scale);
                data.push_back(mat);
            }
            return data;
        }

        static std::vector<std::pair<float4x4, float4x4>> generate_float4x4_pairs(int count)
        {
            std::vector<std::pair<float4x4, float4x4>> pairs;
            pairs.reserve(count);

            auto data = generate_float4x4_data(count * 2);
            for (int i = 0; i < count; ++i)
            {
                pairs.emplace_back(data[i * 2], data[i * 2 + 1]);
            }
            return pairs;
        }

        static std::vector<std::pair<float4x4, float4>> generate_float4x4_vector_pairs(int count)
        {
            std::vector<std::pair<float4x4, float4>> pairs;
            pairs.reserve(count);

            auto matrices = generate_float4x4_data(count);
            auto vectors = BenchmarkData::generate_float4_data(count);

            for (int i = 0; i < count; ++i)
            {
                pairs.emplace_back(matrices[i], vectors[i]);
            }
            return pairs;
        }

        static std::vector<std::pair<float4x4, float3>> generate_float4x4_point_pairs(int count)
        {
            std::vector<std::pair<float4x4, float3>> pairs;
            pairs.reserve(count);

            auto matrices = generate_float4x4_data(count);
            auto points = BenchmarkData::generate_float3_data(count);

            for (int i = 0; i < count; ++i)
            {
                pairs.emplace_back(matrices[i], points[i]);
            }
            return pairs;
        }
    };

    // ============================================================================
    // float4x4 Basic Operations Benchmarks
    // ============================================================================

    void benchmark_float4x4_operations()
    {
        BenchmarkSuite suite("float4x4 Basic Operations", 200000, true);
        suite.header();
        ComparisonBenchmark comparator("float4x4 Operations");

        // Test data
        float4x4 a(1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f);

        float4x4 b(16.0f, 15.0f, 14.0f, 13.0f,
            12.0f, 11.0f, 10.0f, 9.0f,
            8.0f, 7.0f, 6.0f, 5.0f,
            4.0f, 3.0f, 2.0f, 1.0f);

        float4x4 c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison
        struct NativeFloat4x4 {
            float m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32, m03, m13, m23, m33;
        };

        NativeFloat4x4 na{ 1.0f, 5.0f, 9.0f, 13.0f, 2.0f, 6.0f, 10.0f, 14.0f, 3.0f, 7.0f, 11.0f, 15.0f, 4.0f, 8.0f, 12.0f, 16.0f };
        NativeFloat4x4 nb{ 16.0f, 12.0f, 8.0f, 4.0f, 15.0f, 11.0f, 7.0f, 3.0f, 14.0f, 10.0f, 6.0f, 2.0f, 13.0f, 9.0f, 5.0f, 1.0f };
        NativeFloat4x4 nc{ 0 };

        // 1. Addition
        auto lib_time = suite.measure("float4x4 Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += c(0, 0) + c(1, 1) + c(2, 2) + c(3, 3);
            }).time_ms;

        auto native_time = suite.measure("float4x4 Addition (Native)", [&]() {
            nc.m00 = na.m00 + nb.m00; nc.m10 = na.m10 + nb.m10; nc.m20 = na.m20 + nb.m20; nc.m30 = na.m30 + nb.m30;
            nc.m01 = na.m01 + nb.m01; nc.m11 = na.m11 + nb.m11; nc.m21 = na.m21 + nb.m21; nc.m31 = na.m31 + nb.m31;
            nc.m02 = na.m02 + nb.m02; nc.m12 = na.m12 + nb.m12; nc.m22 = na.m22 + nb.m22; nc.m32 = na.m32 + nb.m32;
            nc.m03 = na.m03 + nb.m03; nc.m13 = na.m13 + nb.m13; nc.m23 = na.m23 + nb.m23; nc.m33 = na.m33 + nb.m33;
            result_accumulator += nc.m00 + nc.m11 + nc.m22 + nc.m33;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 200000);

        // 2. Subtraction
        lib_time = suite.measure("float4x4 Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += c(0, 0) + c(1, 1) + c(2, 2) + c(3, 3);
            }).time_ms;

        native_time = suite.measure("float4x4 Subtraction (Native)", [&]() {
            nc.m00 = na.m00 - nb.m00; nc.m10 = na.m10 - nb.m10; nc.m20 = na.m20 - nb.m20; nc.m30 = na.m30 - nb.m30;
            nc.m01 = na.m01 - nb.m01; nc.m11 = na.m11 - nb.m11; nc.m21 = na.m21 - nb.m21; nc.m31 = na.m31 - nb.m31;
            nc.m02 = na.m02 - nb.m02; nc.m12 = na.m12 - nb.m12; nc.m22 = na.m22 - nb.m22; nc.m32 = na.m32 - nb.m32;
            nc.m03 = na.m03 - nb.m03; nc.m13 = na.m13 - nb.m13; nc.m23 = na.m23 - nb.m23; nc.m33 = na.m33 - nb.m33;
            result_accumulator += nc.m00 + nc.m11 + nc.m22 + nc.m33;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 200000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("float4x4 Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += c(0, 0) + c(1, 1) + c(2, 2) + c(3, 3);
            }).time_ms;

        native_time = suite.measure("float4x4 Scalar Multiplication (Native)", [&]() {
            nc.m00 = na.m00 * scalar; nc.m10 = na.m10 * scalar; nc.m20 = na.m20 * scalar; nc.m30 = na.m30 * scalar;
            nc.m01 = na.m01 * scalar; nc.m11 = na.m11 * scalar; nc.m21 = na.m21 * scalar; nc.m31 = na.m31 * scalar;
            nc.m02 = na.m02 * scalar; nc.m12 = na.m12 * scalar; nc.m22 = na.m22 * scalar; nc.m32 = na.m32 * scalar;
            nc.m03 = na.m03 * scalar; nc.m13 = na.m13 * scalar; nc.m23 = na.m23 * scalar; nc.m33 = na.m33 * scalar;
            result_accumulator += nc.m00 + nc.m11 + nc.m22 + nc.m33;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 200000);

        // 4. Matrix Multiplication
        lib_time = suite.measure("float4x4 Matrix Multiplication (Library)", [&]() {
            c = a * b;
            result_accumulator += c(0, 0) + c(1, 1) + c(2, 2) + c(3, 3);
            }).time_ms;

        native_time = suite.measure("float4x4 Matrix Multiplication (Native)", [&]() {
            // Manual 4x4 matrix multiplication
            nc.m00 = na.m00 * nb.m00 + na.m01 * nb.m10 + na.m02 * nb.m20 + na.m03 * nb.m30;
            nc.m10 = na.m10 * nb.m00 + na.m11 * nb.m10 + na.m12 * nb.m20 + na.m13 * nb.m30;
            nc.m20 = na.m20 * nb.m00 + na.m21 * nb.m10 + na.m22 * nb.m20 + na.m23 * nb.m30;
            nc.m30 = na.m30 * nb.m00 + na.m31 * nb.m10 + na.m32 * nb.m20 + na.m33 * nb.m30;

            nc.m01 = na.m00 * nb.m01 + na.m01 * nb.m11 + na.m02 * nb.m21 + na.m03 * nb.m31;
            nc.m11 = na.m10 * nb.m01 + na.m11 * nb.m11 + na.m12 * nb.m21 + na.m13 * nb.m31;
            nc.m21 = na.m20 * nb.m01 + na.m21 * nb.m11 + na.m22 * nb.m21 + na.m23 * nb.m31;
            nc.m31 = na.m30 * nb.m01 + na.m31 * nb.m11 + na.m32 * nb.m21 + na.m33 * nb.m31;

            nc.m02 = na.m00 * nb.m02 + na.m01 * nb.m12 + na.m02 * nb.m22 + na.m03 * nb.m32;
            nc.m12 = na.m10 * nb.m02 + na.m11 * nb.m12 + na.m12 * nb.m22 + na.m13 * nb.m32;
            nc.m22 = na.m20 * nb.m02 + na.m21 * nb.m12 + na.m22 * nb.m22 + na.m23 * nb.m32;
            nc.m32 = na.m30 * nb.m02 + na.m31 * nb.m12 + na.m32 * nb.m22 + na.m33 * nb.m32;

            nc.m03 = na.m00 * nb.m03 + na.m01 * nb.m13 + na.m02 * nb.m23 + na.m03 * nb.m33;
            nc.m13 = na.m10 * nb.m03 + na.m11 * nb.m13 + na.m12 * nb.m23 + na.m13 * nb.m33;
            nc.m23 = na.m20 * nb.m03 + na.m21 * nb.m13 + na.m22 * nb.m23 + na.m23 * nb.m33;
            nc.m33 = na.m30 * nb.m03 + na.m31 * nb.m13 + na.m32 * nb.m23 + na.m33 * nb.m33;

            result_accumulator += nc.m00 + nc.m11 + nc.m22 + nc.m33;
            }).time_ms;

        comparator.compare("Matrix Multiplication", lib_time, native_time, 200000);

        suite.footer();
    }

    // ============================================================================
    // float4x4 Advanced Operations Benchmarks
    // ============================================================================

    void benchmark_float4x4_advanced_operations()
    {
        BenchmarkSuite suite("float4x4 Advanced Operations", 100000, true);
        suite.header();

        // Use an invertible matrix for inverse operations
        float4x4 a = float4x4::translation(float3(1, 2, 3)) *
            float4x4::rotation_y(Constants::PI / 4.0f) *
            float4x4::scaling(2.0f, 3.0f, 4.0f);

        float4x4 b = float4x4::perspective(Constants::PI / 3.0f, 16.0f / 9.0f, 0.1f, 100.0f);
        float4 vec4(1.0f, 2.0f, 3.0f, 1.0f);
        float3 point(1.0f, 2.0f, 3.0f);
        float3 vector(1.0f, 0.0f, 0.0f);

        volatile float result_accumulator = 0.0f;

        // Matrix operations
        suite.measure("float4x4 Transpose", [&]() {
            float4x4 result = a.transposed();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Determinant", [&]() {
            float result = a.determinant();
            result_accumulator += result;
            });

        suite.measure("float4x4 Inverse", [&]() {
            float4x4 result = a.inverted();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Inverse Affine", [&]() {
            float4x4 result = a.inverted_affine();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Adjugate", [&]() {
            float4x4 result = a.adjugate();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Trace", [&]() {
            float result = a.trace();
            result_accumulator += result;
            });

        suite.measure("float4x4 Diagonal", [&]() {
            float4 result = a.diagonal();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("float4x4 Frobenius Norm", [&]() {
            float result = a.frobenius_norm();
            result_accumulator += result;
            });

        // Vector transformations
        suite.measure("float4x4 Transform Vector (float4)", [&]() {
            float4 result = a.transform_vector(vec4);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("float4x4 Transform Point", [&]() {
            float3 result = a.transform_point(point);
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float4x4 Transform Vector (float3)", [&]() {
            float3 result = a.transform_vector(vector);
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float4x4 Transform Direction", [&]() {
            float3 result = a.transform_direction(vector);
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float4 * float4x4", [&]() {
            float4 result = vec4 * a;
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("float3 * float4x4", [&]() {
            float3 result = point * a;
            result_accumulator += result.x + result.y + result.z;
            });

        suite.footer();
    }

    // ============================================================================
    // float4x4 Static Constructors Benchmarks
    // ============================================================================

    void benchmark_float4x4_static_constructors()
    {
        BenchmarkSuite suite("float4x4 Static Constructors", 200000, true);
        suite.header();

        volatile float result_accumulator = 0.0f;
        float angle = Constants::PI / 4.0f;
        float3 translation(1.0f, 2.0f, 3.0f);
        float3 scale(2.0f, 3.0f, 4.0f);
        float3 axis(1.0f, 1.0f, 1.0f);
        float3 euler_angles(Constants::PI / 6.0f, Constants::PI / 4.0f, Constants::PI / 3.0f);
        float3 eye(0.0f, 0.0f, 5.0f);
        float3 target(0.0f, 0.0f, 0.0f);
        float3 up(0.0f, 1.0f, 0.0f);
        quaternion q = quaternion::rotation_y(angle);

        // Static constructor benchmarks
        suite.measure("float4x4 Identity", [&]() {
            float4x4 result = float4x4::identity();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Zero", [&]() {
            float4x4 result = float4x4::zero();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Translation (vector)", [&]() {
            float4x4 result = float4x4::translation(translation);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Translation (components)", [&]() {
            float4x4 result = float4x4::translation(1.0f, 2.0f, 3.0f);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Scaling (vector)", [&]() {
            float4x4 result = float4x4::scaling(scale);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Scaling (components)", [&]() {
            float4x4 result = float4x4::scaling(2.0f, 3.0f, 4.0f);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Scaling (uniform)", [&]() {
            float4x4 result = float4x4::scaling(2.0f);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Rotation X", [&]() {
            float4x4 result = float4x4::rotation_x(angle);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Rotation Y", [&]() {
            float4x4 result = float4x4::rotation_y(angle);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Rotation Z", [&]() {
            float4x4 result = float4x4::rotation_z(angle);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Rotation Axis", [&]() {
            float4x4 result = float4x4::rotation_axis(axis.normalize(), angle);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Rotation Euler", [&]() {
            float4x4 result = float4x4::rotation_euler(euler_angles);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Orthographic", [&]() {
            float4x4 result = float4x4::orthographic(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Perspective", [&]() {
            float4x4 result = float4x4::perspective(Constants::PI / 3.0f, 16.0f / 9.0f, 0.1f, 100.0f);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 Look At", [&]() {
            float4x4 result = float4x4::look_at(eye, target, up);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 TRS", [&]() {
            float4x4 result = float4x4::TRS(translation, q, scale);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("float4x4 from quaternion", [&]() {
            float4x4 result = float4x4(q);
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.footer();
    }

    // ============================================================================
    // float4x4 Component Extraction Benchmarks
    // ============================================================================

    void benchmark_float4x4_component_extraction()
    {
        BenchmarkSuite suite("float4x4 Component Extraction", 200000, true);
        suite.header();

        float4x4 transform = float4x4::TRS(
            float3(1.0f, 2.0f, 3.0f),
            quaternion::rotation_y(Constants::PI / 4.0f),
            float3(2.0f, 3.0f, 4.0f)
        );

        volatile float result_accumulator = 0.0f;

        // Component extraction
        suite.measure("float4x4 Get Translation", [&]() {
            float3 result = transform.get_translation();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float4x4 Get Scale", [&]() {
            float3 result = transform.get_scale();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("float4x4 Get Rotation", [&]() {
            quaternion result = transform.get_rotation();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("float4x4 Normal Matrix", [&]() {
            float3x3 result = transform.normal_matrix();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        // Component setting
        suite.measure("float4x4 Set Translation", [&]() {
            float4x4 mat = transform;
            mat.set_translation(float3(4.0f, 5.0f, 6.0f));
            result_accumulator += mat(0, 0) + mat(1, 1) + mat(2, 2) + mat(3, 3);
            });

        suite.measure("float4x4 Set Scale", [&]() {
            float4x4 mat = transform;
            mat.set_scale(float3(5.0f, 6.0f, 7.0f));
            result_accumulator += mat(0, 0) + mat(1, 1) + mat(2, 2) + mat(3, 3);
            });

        // Property checks
        suite.measure("float4x4 Is Identity", [&]() {
            bool result = transform.is_identity();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("float4x4 Is Affine", [&]() {
            bool result = transform.is_affine();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("float4x4 Is Orthogonal", [&]() {
            bool result = transform.is_orthogonal();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("float4x4 Approximately Equal", [&]() {
            bool result = transform.approximately(float4x4::identity());
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.footer();
    }

    // ============================================================================
    // float4x4 Memory Patterns Benchmarks
    // ============================================================================

    void benchmark_float4x4_memory_patterns()
    {
        BenchmarkSuite suite("float4x4 Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = Float4x4BenchmarkData::generate_float4x4_data(ARRAY_SIZE);
        auto pairs = Float4x4BenchmarkData::generate_float4x4_pairs(ARRAY_SIZE);
        auto vector_pairs = Float4x4BenchmarkData::generate_float4x4_vector_pairs(ARRAY_SIZE);
        auto point_pairs = Float4x4BenchmarkData::generate_float4x4_point_pairs(ARRAY_SIZE);

        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("float4x4 Sequential Sum", [&]() {
            float4x4 sum = float4x4::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += sum(0, 0) + sum(1, 1) + sum(2, 2) + sum(3, 3);
            });

        suite.measure("float4x4 Sequential Multiplication", [&]() {
            float4x4 product = float4x4::identity();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= data[i];
            }
            result_accumulator += product(0, 0) + product(1, 1) + product(2, 2) + product(3, 3);
            });

        suite.measure("float4x4 Sequential Determinants", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i].determinant();
            }
            result_accumulator += sum;
            });

        suite.measure("float4x4 Sequential Transpose", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float4x4 transposed = data[i].transposed();
                result_accumulator += transposed(0, 0) + transposed(1, 1) + transposed(2, 2) + transposed(3, 3);
            }
            });

        // Pair operations
        suite.measure("float4x4 Pair Multiplication", [&]() {
            float4x4 sum = float4x4::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += pairs[i].first * pairs[i].second;
            }
            result_accumulator += sum(0, 0) + sum(1, 1) + sum(2, 2) + sum(3, 3);
            });

        // Vector transformation patterns
        suite.measure("float4x4 Sequential Vector Transform (float4)", [&]() {
            float4 sum = float4::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += vector_pairs[i].first.transform_vector(vector_pairs[i].second);
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        suite.measure("float4x4 Sequential Point Transform", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += point_pairs[i].first.transform_point(point_pairs[i].second);
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        suite.measure("float4x4 Sequential Direction Transform", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += point_pairs[i].first.transform_direction(point_pairs[i].second.normalize());
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        // Matrix operations pattern
        suite.measure("float4x4 Sequential Inverse", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float4x4 inv = data[i].inverted();
                result_accumulator += inv(0, 0) + inv(1, 1) + inv(2, 2) + inv(3, 3);
            }
            });

        // Component extraction pattern
        suite.measure("float4x4 Sequential Translation Extraction", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i].get_translation();
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        // Random access pattern (simulated)
        suite.measure("float4x4 Strided Access", [&]() {
            float4x4 sum = float4x4::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += sum(0, 0) + sum(1, 1) + sum(2, 2) + sum(3, 3);
            });

        suite.footer();
    }

    // ============================================================================
    // float4x4 vs Native Memory Benchmarks
    // ============================================================================

    void benchmark_float4x4_vs_native_memory()
    {
        BenchmarkSuite suite("float4x4 vs Native Memory Bandwidth", BenchmarkConfig::MEMORY_ITERATIONS / 10, true);
        suite.header();

        const int ARRAY_SIZE = 5000;

        // Generate test data
        std::vector<float4x4> float4x4_data;
        std::vector<std::array<float, 16>> native_data; // column-major storage

        float4x4_data.reserve(ARRAY_SIZE);
        native_data.reserve(ARRAY_SIZE);

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            float angle = static_cast<float>(i) * 0.1f;
            float3 translation(std::sin(angle), std::cos(angle), std::sin(angle * 0.5f));
            float3 scale(1.0f + std::sin(angle) * 0.2f, 1.0f + std::cos(angle) * 0.2f, 1.0f);

            float4x4 mat = float4x4::translation(translation) *
                float4x4::rotation_y(angle) *
                float4x4::scaling(scale);
            float4x4_data.push_back(mat);

            // Store in column-major order
            std::array<float, 16> native_mat;
            mat.to_column_major(native_mat.data());
            native_data.push_back(native_mat);
        }

        volatile float result_accumulator = 0.0f;

        // Memory bandwidth tests
        suite.measure("float4x4 Memory Sum", [&]() {
            float4x4 sum = float4x4::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += float4x4_data[i];
            }
            result_accumulator += sum(0, 0) + sum(1, 1) + sum(2, 2) + sum(3, 3);
            });

        suite.measure("Native Memory Sum", [&]() {
            std::array<float, 16> sum = { 0 };
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                for (int j = 0; j < 16; ++j) {
                    sum[j] += native_data[i][j];
                }
            }
            result_accumulator += sum[0] + sum[5] + sum[10] + sum[15]; // diagonal
            });

        suite.measure("float4x4 Memory Multiplication", [&]() {
            float4x4 product = float4x4::identity();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= float4x4_data[i];
            }
            result_accumulator += product(0, 0) + product(1, 1) + product(2, 2) + product(3, 3);
            });

        suite.measure("float4x4 Memory Transform Points", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += float4x4_data[i].transform_point(float3(1.0f, 2.0f, 3.0f));
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        suite.footer();
    }

    // ============================================================================
    // float4x4 Utility Functions Benchmarks
    // ============================================================================

    void benchmark_float4x4_utility_functions()
    {
        BenchmarkSuite suite("float4x4 Utility Functions", 100000, true);
        suite.header();

        float4x4 a = float4x4::TRS(
            float3(1.0f, 2.0f, 3.0f),
            quaternion::rotation_y(Constants::PI / 4.0f),
            float3(2.0f, 3.0f, 4.0f)
        );

        float4x4 b = float4x4::TRS(
            float3(1.1f, 2.1f, 3.1f),
            quaternion::rotation_y(Constants::PI / 4.1f),
            float3(2.1f, 3.1f, 4.1f)
        );

        volatile float result_accumulator = 0.0f;
        float column_major_data[16];
        float row_major_data[16];

        // Utility functions
        suite.measure("float4x4 To String", [&]() {
            std::string result = a.to_string();
            result_accumulator += result.length() * 0.001f; // Prevent optimization
            });

        suite.measure("float4x4 To Column Major", [&]() {
            a.to_column_major(column_major_data);
            result_accumulator += column_major_data[0] + column_major_data[15];
            });

        suite.measure("float4x4 To Row Major", [&]() {
            a.to_row_major(row_major_data);
            result_accumulator += row_major_data[0] + row_major_data[15];
            });

        suite.measure("float4x4 Column Accessors", [&]() {
            float4 col0 = a.col0();
            float4 col1 = a.col1();
            float4 col2 = a.col2();
            float4 col3 = a.col3();
            result_accumulator += col0.x + col1.y + col2.z + col3.w;
            });

        suite.measure("float4x4 Row Accessors", [&]() {
            float4 row0 = a.row0();
            float4 row1 = a.row1();
            float4 row2 = a.row2();
            float4 row3 = a.row3();
            result_accumulator += row0.x + row1.y + row2.z + row3.w;
            });

        suite.measure("float4x4 Element Access", [&]() {
            float sum = 0.0f;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    sum += a(row, col);
                }
            }
            result_accumulator += sum;
            });

        suite.measure("float4x4 Column Access", [&]() {
            float sum = 0.0f;
            for (int col = 0; col < 4; ++col) {
                float4 column = a[col];
                sum += column.x + column.y + column.z + column.w;
            }
            result_accumulator += sum;
            });

        // Comparison operations
        suite.measure("float4x4 Approximately", [&]() {
            bool result = a.approximately(b, 0.1f);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("float4x4 Operator ==", [&]() {
            bool result = (a == b);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("float4x4 Operator !=", [&]() {
            bool result = (a != b);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.footer();
    }

    // ============================================================================
    // Comprehensive Benchmark Runner for float4x4
    // ============================================================================

    void run_float4x4_benchmarks()
    {
        std::cout << "FLOAT4X4 MATRIX COMPREHENSIVE BENCHMARK SUITE" << std::endl;
        std::cout << "==============================================" << std::endl;
        std::cout << "This benchmark compares float4x4 performance" << std::endl;
        std::cout << "against native C++ implementations." << std::endl;
        std::cout << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        benchmark_float4x4_operations();
        benchmark_float4x4_advanced_operations();
        benchmark_float4x4_static_constructors();
        benchmark_float4x4_component_extraction();
        benchmark_float4x4_utility_functions();
        benchmark_float4x4_memory_patterns();
        benchmark_float4x4_vs_native_memory();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "FLOAT4X4 BENCHMARKS COMPLETED" << std::endl;
        std::cout << "Total benchmark time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ============================================================================
// Benchmark Data Generators for quaternion
// ============================================================================

    class QuaternionBenchmarkData
    {
    public:
        static std::vector<quaternion> generate_quaternion_data(int count)
        {
            std::vector<quaternion> data;
            data.reserve(count);

            for (int i = 0; i < count; ++i)
            {
                float angle = static_cast<float>(i) * 0.1f;
                float3 axis(std::sin(angle), std::cos(angle), std::sin(angle * 0.5f));
                axis = axis.normalize();
                data.emplace_back(axis, angle);
            }
            return data;
        }

        static std::vector<std::pair<quaternion, quaternion>> generate_quaternion_pairs(int count)
        {
            std::vector<std::pair<quaternion, quaternion>> pairs;
            pairs.reserve(count);

            auto data = generate_quaternion_data(count * 2);
            for (int i = 0; i < count; ++i)
            {
                pairs.emplace_back(data[i * 2], data[i * 2 + 1]);
            }
            return pairs;
        }

        static std::vector<std::pair<quaternion, float3>> generate_quaternion_vector_pairs(int count)
        {
            std::vector<std::pair<quaternion, float3>> pairs;
            pairs.reserve(count);

            auto quaternions = generate_quaternion_data(count);
            auto vectors = BenchmarkData::generate_float3_data(count);

            for (int i = 0; i < count; ++i)
            {
                pairs.emplace_back(quaternions[i], vectors[i]);
            }
            return pairs;
        }
    };

    // ============================================================================
    // quaternion Basic Operations Benchmarks
    // ============================================================================

    void benchmark_quaternion_operations()
    {
        BenchmarkSuite suite("quaternion Basic Operations", 1000000, true);
        suite.header();
        ComparisonBenchmark comparator("quaternion Operations");

        // Test data
        quaternion a(0.5f, 0.5f, 0.5f, 0.5f);
        quaternion b(0.3f, 0.3f, 0.3f, 0.3f);
        quaternion c;
        volatile float result_accumulator = 0.0f;
        const float scalar = 2.0f;

        // Native implementation for comparison
        struct NativeQuaternion { float x, y, z, w; };
        NativeQuaternion na{ 0.5f, 0.5f, 0.5f, 0.5f };
        NativeQuaternion nb{ 0.3f, 0.3f, 0.3f, 0.3f };
        NativeQuaternion nc{ 0.0f, 0.0f, 0.0f, 0.0f };

        // 1. Addition
        auto lib_time = suite.measure("quaternion Addition (Library)", [&]() {
            c = a + b;
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        auto native_time = suite.measure("quaternion Addition (Native)", [&]() {
            nc.x = na.x + nb.x;
            nc.y = na.y + nb.y;
            nc.z = na.z + nb.z;
            nc.w = na.w + nb.w;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Addition", lib_time, native_time, 1000000);

        // 2. Subtraction
        lib_time = suite.measure("quaternion Subtraction (Library)", [&]() {
            c = a - b;
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        native_time = suite.measure("quaternion Subtraction (Native)", [&]() {
            nc.x = na.x - nb.x;
            nc.y = na.y - nb.y;
            nc.z = na.z - nb.z;
            nc.w = na.w - nb.w;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Subtraction", lib_time, native_time, 1000000);

        // 3. Scalar Multiplication
        lib_time = suite.measure("quaternion Scalar Multiplication (Library)", [&]() {
            c = a * scalar;
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        native_time = suite.measure("quaternion Scalar Multiplication (Native)", [&]() {
            nc.x = na.x * scalar;
            nc.y = na.y * scalar;
            nc.z = na.z * scalar;
            nc.w = na.w * scalar;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Scalar Multiplication", lib_time, native_time, 1000000);

        // 4. Quaternion Multiplication
        lib_time = suite.measure("quaternion Multiplication (Library)", [&]() {
            c = a * b;
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        native_time = suite.measure("quaternion Multiplication (Native)", [&]() {
            nc.x = na.w * nb.x + na.x * nb.w + na.y * nb.z - na.z * nb.y;
            nc.y = na.w * nb.y - na.x * nb.z + na.y * nb.w + na.z * nb.x;
            nc.z = na.w * nb.z + na.x * nb.y - na.y * nb.x + na.z * nb.w;
            nc.w = na.w * nb.w - na.x * nb.x - na.y * nb.y - na.z * nb.z;
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Multiplication", lib_time, native_time, 1000000);

        // 5. Length
        lib_time = suite.measure("quaternion Length (Library)", [&]() {
            float result = a.length();
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("quaternion Length (Native)", [&]() {
            float result = std::sqrt(na.x * na.x + na.y * na.y + na.z * na.z + na.w * na.w);
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Length", lib_time, native_time, 1000000);

        // 6. Normalization
        lib_time = suite.measure("quaternion Normalize (Library)", [&]() {
            c = a.normalize();
            result_accumulator += c.x + c.y + c.z + c.w;
            }).time_ms;

        native_time = suite.measure("quaternion Normalize (Native)", [&]() {
            float len = std::sqrt(na.x * na.x + na.y * na.y + na.z * na.z + na.w * na.w);
            if (len > 0.0f) {
                nc.x = na.x / len;
                nc.y = na.y / len;
                nc.z = na.z / len;
                nc.w = na.w / len;
            }
            result_accumulator += nc.x + nc.y + nc.z + nc.w;
            }).time_ms;

        comparator.compare("Normalize", lib_time, native_time, 1000000);

        // 7. Dot Product
        lib_time = suite.measure("quaternion Dot Product (Library)", [&]() {
            float result = dot(a, b);
            result_accumulator += result;
            }).time_ms;

        native_time = suite.measure("quaternion Dot Product (Native)", [&]() {
            float result = na.x * nb.x + na.y * nb.y + na.z * nb.z + na.w * nb.w;
            result_accumulator += result;
            }).time_ms;

        comparator.compare("Dot Product", lib_time, native_time, 1000000);

        suite.footer();
    }

    void benchmark_quaternion_advanced_operations()
    {
        BenchmarkSuite suite("quaternion Advanced Operations", 500000, true);
        suite.header();

        quaternion a(0.5f, 0.5f, 0.5f, 0.5f);
        quaternion b(0.3f, 0.3f, 0.3f, 0.3f);
        float3 vec(1.0f, 2.0f, 3.0f);
        volatile float result_accumulator = 0.0f;

        // Advanced quaternion operations
        suite.measure("quaternion Conjugate", [&]() {
            quaternion result = a.conjugate();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Inverse", [&]() {
            quaternion result = a.inverse();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Length Squared", [&]() {
            float result = a.length_sq();
            result_accumulator += result;
            });

        // Vector transformations
        suite.measure("quaternion Transform Vector", [&]() {
            float3 result = a.transform_vector(vec);
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("quaternion Transform Direction", [&]() {
            float3 result = a.transform_direction(vec);
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("quaternion * float3", [&]() {
            float3 result = a * vec;
            result_accumulator += result.x + result.y + result.z;
            });

        suite.footer();
    }

    void benchmark_quaternion_conversion_operations()
    {
        BenchmarkSuite suite("quaternion Conversion Operations", 300000, true);
        suite.header();

        quaternion q(0.5f, 0.5f, 0.5f, 0.5f);
        float3 axis(1.0f, 0.0f, 0.0f);
        float angle = Constants::PI / 4.0f;
        float3 euler_angles(Constants::PI / 6.0f, Constants::PI / 4.0f, Constants::PI / 3.0f);
        volatile float result_accumulator = 0.0f;

        // Conversion operations
        suite.measure("quaternion to Matrix3x3", [&]() {
            float3x3 result = q.to_matrix3x3();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2);
            });

        suite.measure("quaternion to Matrix4x4", [&]() {
            float4x4 result = q.to_matrix4x4();
            result_accumulator += result(0, 0) + result(1, 1) + result(2, 2) + result(3, 3);
            });

        suite.measure("quaternion to Euler Angles", [&]() {
            float3 result = q.to_euler();
            result_accumulator += result.x + result.y + result.z;
            });

        suite.measure("quaternion to Axis-Angle", [&]() {
            float3 axis_result;
            float angle_result;
            q.to_axis_angle(axis_result, angle_result);
            result_accumulator += axis_result.x + axis_result.y + axis_result.z + angle_result;
            });

        // Static constructors
        suite.measure("quaternion from Axis-Angle", [&]() {
            quaternion result = quaternion::from_axis_angle(axis, angle);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion from Euler Angles", [&]() {
            quaternion result = quaternion::from_euler(euler_angles);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion from Matrix3x3", [&]() {
            float3x3 mat = q.to_matrix3x3();
            quaternion result = quaternion::from_matrix(mat);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion from Matrix4x4", [&]() {
            float4x4 mat = q.to_matrix4x4();
            quaternion result = quaternion::from_matrix(mat);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.footer();
    }

    void benchmark_quaternion_interpolation_operations()
    {
        BenchmarkSuite suite("quaternion Interpolation Operations", 400000, true);
        suite.header();

        quaternion a = quaternion::rotation_x(Constants::PI / 6.0f);
        quaternion b = quaternion::rotation_y(Constants::PI / 3.0f);
        volatile float result_accumulator = 0.0f;

        // Interpolation operations
        suite.measure("quaternion Slerp", [&]() {
            quaternion result = slerp(a, b, 0.5f);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Nlerp", [&]() {
            quaternion result = nlerp(a, b, 0.5f);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Lerp", [&]() {
            quaternion result = lerp(a, b, 0.5f);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Slerp (Static)", [&]() {
            quaternion result = quaternion::slerp(a, b, 0.5f);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Lerp (Static)", [&]() {
            quaternion result = quaternion::lerp(a, b, 0.5f);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.footer();
    }

    void benchmark_quaternion_static_constructors()
    {
        BenchmarkSuite suite("quaternion Static Constructors", 500000, true);
        suite.header();

        float3 axis(1.0f, 1.0f, 1.0f);
        float angle = Constants::PI / 4.0f;
        float3 from_dir(1.0f, 0.0f, 0.0f);
        float3 to_dir(0.0f, 1.0f, 0.0f);
        volatile float result_accumulator = 0.0f;

        // Static constructor benchmarks
        suite.measure("quaternion Identity", [&]() {
            quaternion result = quaternion::identity();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Zero", [&]() {
            quaternion result = quaternion::zero();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion One", [&]() {
            quaternion result = quaternion::one();
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Rotation X", [&]() {
            quaternion result = quaternion::rotation_x(angle);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Rotation Y", [&]() {
            quaternion result = quaternion::rotation_y(angle);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion Rotation Z", [&]() {
            quaternion result = quaternion::rotation_z(angle);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.measure("quaternion From-To Rotation", [&]() {
            quaternion result = quaternion::from_to_rotation(from_dir, to_dir);
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.footer();
    }

    void benchmark_quaternion_utility_operations()
    {
        BenchmarkSuite suite("quaternion Utility Operations", 600000, true);
        suite.header();

        quaternion a(0.5f, 0.5f, 0.5f, 0.5f);
        quaternion b(0.5f, 0.5f, 0.5f, 0.5f);
        quaternion c(0.3f, 0.3f, 0.3f, 0.3f);
        volatile float result_accumulator = 0.0f;

        // Utility operations
        suite.measure("quaternion Is Identity", [&]() {
            bool result = a.is_identity();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("quaternion Is Normalized", [&]() {
            bool result = a.normalize().is_normalized();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("quaternion Is Valid", [&]() {
            bool result = a.is_valid();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("quaternion Approximately Equal", [&]() {
            bool result = a.approximately(b);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("quaternion Approximately Zero", [&]() {
            bool result = quaternion::zero().approximately_zero();
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("quaternion Operator ==", [&]() {
            bool result = (a == b);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("quaternion Operator !=", [&]() {
            bool result = (a != c);
            result_accumulator += result ? 1.0f : 0.0f;
            });

        suite.measure("quaternion Unary Minus", [&]() {
            quaternion result = -a;
            result_accumulator += result.x + result.y + result.z + result.w;
            });

        suite.footer();
    }

    void benchmark_quaternion_memory_patterns()
    {
        BenchmarkSuite suite("quaternion Memory Patterns", BenchmarkConfig::MEMORY_ITERATIONS, true);
        suite.header();

        const int ARRAY_SIZE = 1000;
        auto data = QuaternionBenchmarkData::generate_quaternion_data(ARRAY_SIZE);
        auto pairs = QuaternionBenchmarkData::generate_quaternion_pairs(ARRAY_SIZE);
        auto vector_pairs = QuaternionBenchmarkData::generate_quaternion_vector_pairs(ARRAY_SIZE);

        volatile float result_accumulator = 0.0f;

        // Sequential access patterns
        suite.measure("quaternion Sequential Sum", [&]() {
            quaternion sum = quaternion::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += data[i];
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        suite.measure("quaternion Sequential Multiplication", [&]() {
            quaternion product = quaternion::identity();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= data[i];
            }
            result_accumulator += product.x + product.y + product.z + product.w;
            });

        suite.measure("quaternion Sequential Dot Products", [&]() {
            float sum = 0.0f;
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                sum += dot(data[i], data[i + 1]);
            }
            result_accumulator += sum;
            });

        suite.measure("quaternion Sequential Normalization", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                quaternion normalized = data[i].normalize();
                result_accumulator += normalized.x + normalized.y + normalized.z + normalized.w;
            }
            });

        // Pair operations
        suite.measure("quaternion Pair Multiplication", [&]() {
            quaternion sum = quaternion::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += pairs[i].first * pairs[i].second;
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        // Vector transformation patterns
        suite.measure("quaternion Sequential Vector Transform", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += vector_pairs[i].first.transform_vector(vector_pairs[i].second);
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        // Conversion patterns
        suite.measure("quaternion Sequential to Matrix3x3", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float3x3 mat = data[i].to_matrix3x3();
                result_accumulator += mat(0, 0) + mat(1, 1) + mat(2, 2);
            }
            });

        suite.measure("quaternion Sequential to Euler", [&]() {
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                float3 euler = data[i].to_euler();
                result_accumulator += euler.x + euler.y + euler.z;
            }
            });

        // Interpolation patterns
        suite.measure("quaternion Sequential Slerp", [&]() {
            for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
                quaternion result = slerp(data[i], data[i + 1], 0.5f);
                result_accumulator += result.x + result.y + result.z + result.w;
            }
            });

        // Random access pattern (simulated)
        suite.measure("quaternion Strided Access", [&]() {
            quaternion sum = quaternion::zero();
            for (int i = 0; i < ARRAY_SIZE; i += 4) {
                sum += data[i];
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        suite.footer();
    }

    void benchmark_quaternion_vs_native_memory()
    {
        BenchmarkSuite suite("quaternion vs Native Memory Bandwidth", BenchmarkConfig::MEMORY_ITERATIONS / 10, true);
        suite.header();

        const int ARRAY_SIZE = 5000;

        // Generate test data
        std::vector<quaternion> quaternion_data;
        std::vector<std::array<float, 4>> native_data;

        quaternion_data.reserve(ARRAY_SIZE);
        native_data.reserve(ARRAY_SIZE);

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            float angle = static_cast<float>(i) * 0.1f;
            float3 axis(std::sin(angle), std::cos(angle), std::sin(angle * 0.5f));
            axis = axis.normalize();
            quaternion q(axis, angle);
            quaternion_data.push_back(q);

            std::array<float, 4> native_q = { q.x, q.y, q.z, q.w };
            native_data.push_back(native_q);
        }

        volatile float result_accumulator = 0.0f;

        // Memory bandwidth tests
        suite.measure("quaternion Memory Sum", [&]() {
            quaternion sum = quaternion::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += quaternion_data[i];
            }
            result_accumulator += sum.x + sum.y + sum.z + sum.w;
            });

        suite.measure("Native Memory Sum", [&]() {
            std::array<float, 4> sum = { 0.0f, 0.0f, 0.0f, 0.0f };
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum[0] += native_data[i][0];
                sum[1] += native_data[i][1];
                sum[2] += native_data[i][2];
                sum[3] += native_data[i][3];
            }
            result_accumulator += sum[0] + sum[1] + sum[2] + sum[3];
            });

        suite.measure("quaternion Memory Multiplication", [&]() {
            quaternion product = quaternion::identity();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                product *= quaternion_data[i];
            }
            result_accumulator += product.x + product.y + product.z + product.w;
            });

        suite.measure("quaternion Memory Vector Transform", [&]() {
            float3 sum = float3::zero();
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += quaternion_data[i].transform_vector(float3(1.0f, 2.0f, 3.0f));
            }
            result_accumulator += sum.x + sum.y + sum.z;
            });

        suite.footer();
    }

    // ============================================================================
    // Comprehensive Benchmark Runner for quaternion
    // ============================================================================

    void run_quaternion_benchmarks()
    {
        std::cout << "QUATERNION COMPREHENSIVE BENCHMARK SUITE" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "This benchmark compares quaternion performance" << std::endl;
        std::cout << "against native C++ implementations." << std::endl;
        std::cout << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        benchmark_quaternion_operations();
        benchmark_quaternion_advanced_operations();
        benchmark_quaternion_conversion_operations();
        benchmark_quaternion_interpolation_operations();
        benchmark_quaternion_static_constructors();
        benchmark_quaternion_utility_operations();
        benchmark_quaternion_memory_patterns();
        benchmark_quaternion_vs_native_memory();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "QUATERNION BENCHMARKS COMPLETED" << std::endl;
        std::cout << "Total benchmark time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // ============================================================================
    // Comprehensive Benchmark Runner
    // ============================================================================

    void run_all_benchmarks()
    {
        std::cout << "MATH LIBRARY COMPREHENSIVE BENCHMARK SUITE" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "This benchmark compares SSE-optimized math library" << std::endl;
        std::cout << "performance against native C++ implementations." << std::endl;
        std::cout << std::endl;

        auto total_start = std::chrono::steady_clock::now();

        benchmark_float2_operations();
        benchmark_float2_advanced_operations();
        benchmark_float2_memory_patterns();

        benchmark_float3_operations();
        benchmark_float3_advanced_operations();
        benchmark_float3_swizzle_operations();
        benchmark_float3_memory_patterns();

        benchmark_float4_operations();
        benchmark_float4_advanced_operations();
        benchmark_float4_swizzle_operations();
        benchmark_float4_memory_patterns();

        run_half_benchmarks();
        run_half2_benchmarks();
        run_half3_benchmarks();
        run_half4_benchmarks();

        run_float2x2_benchmarks();
        run_float3x3_benchmarks();
        run_float4x4_benchmarks();

        run_quaternion_benchmarks();

        auto total_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "ALL BENCHMARKS COMPLETED" << std::endl;
        std::cout << "Total benchmark time: " << total_duration << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

} // namespace MathBenchmarks

// Main function for standalone benchmark executable
#ifdef MATH_BENCHMARKS_STANDALONE
int main()
{
    MathBenchmarks::run_all_benchmarks();
    return 0;
}
#endif
