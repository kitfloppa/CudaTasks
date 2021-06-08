#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

struct monte_carlo : public thrust::unary_function<unsigned int, double> {
    __host__ __device__
    double operator()(unsigned int tthread) {
        unsigned int seed = 21^tthread;
        double x, y, sum = 0;

        thrust::default_random_engine ran(seed);
        thrust::uniform_real_distribution<double> gen(0, 1);
        for (size_t i = 0; i < 1000000; ++i) {
                x = gen(ran); y = gen(ran);
                if (((x * x) + (y * y)) <= 1) sum += 1;
        }
        
        return sum;
    }
};

int main() {
    double pi = 0, result = 0, n = 10000;
    auto begin = thrust::counting_iterator<double>(0);
    auto end = thrust::counting_iterator<double>(n);

    result = thrust::transform_reduce(begin, end, monte_carlo(), 0.0, thrust::plus<double>());
    pi = 4 * result / n;

    std::cout << "PI = " << pi;

    return 0;
} // nvcc monte_carlo_thrust.cu -o monte_carlo_thrust
