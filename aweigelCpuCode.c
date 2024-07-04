#include <stdio.h>
#include <stdlib.h>
#include <omp.h> //OpenMP for parallel processing???
#include <stddef.h>

#define min(x, y) (((x) < (y)) ? (x) : (y))

void matmul(const double *__restrict a, const double *__restrict b,
            double *__restrict c, size_t m, size_t k, size_t n) {

    //windowing and unrolling stuff
    const size_t BLOCK_SIZE = 16;
    const size_t unroll_factor = 4;
   
    #pragma omp parallel for collapse(2) //pareallel processing pipin
      for (size_t i = 0; i < m; i += BLOCK_SIZE) { //windowing
        for (size_t j = 0; j < n; j += BLOCK_SIZE) {
            for (size_t ii = i; ii < min(i + BLOCK_SIZE, m); ii++) {
                for (size_t jj = j; jj < min(j + BLOCK_SIZE, n); jj++) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum) //loop unrollin

                    //multiplication block
                    for (size_t l = 0; l < k; l += unroll_factor) {
                        for (size_t u = 0; u < unroll_factor; u++) { //loop unrolled
                            sum += a[ii * k + l + u] * b[(l + u) * n + jj];
                        }
                    }
                    c[ii * n + jj] += sum;
                }
            }
        }
    }
}

int main() { //dummy main
    size_t m = 500;
    size_t k = 400;
    size_t n = 500;

    double *a = (double *)malloc(m * k * sizeof(double));
    double *b = (double *)malloc(k * n * sizeof(double));
    double *c = (double *)calloc(m * n, sizeof(double)); //zeros

    //1 an 2s
    for (size_t i = 0; i < m * k; i++) {
        a[i] = 1.0;
    }
    for (size_t i = 0; i < k * n; i++) {
        b[i] = 2.0;
    }

    matmul(a, b, c, m, k, n);
    
    free(a);
    free(b);
    free(c);
    
    return 0;
}
