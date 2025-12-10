#include <iostream>
#include <windows.h>

#include "Demo/benchmark.h"
#include "Demo/autotests.h"

int main()
{
    //MathTests::run_all_tests();
    MathBenchmarks::run_all_benchmarks();
    Sleep(10000000);
    return 0;
}
