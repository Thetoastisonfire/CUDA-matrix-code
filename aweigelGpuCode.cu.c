#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h> //OpenMP for parallel processing???

extern "C" {
    #include <stddef.h>
    #include <cuda_runtime.h>
}


__global__ void GrimbloTheGrumbler(const double *__restrict a,
 const double *__restrict b, double *__restrict c,
 size_t m, size_t k, size_t n) {

    constexpr size_t unroll_factor = 4;

    //global thread thingies
    size_t ii = blockIdx.x * blockDim.x + threadIdx.x;
    size_t jj = blockIdx.y * blockDim.y + threadIdx.y;

    //threads within matrix bounds check
    if (ii < m && jj < n) {
        double sum[unroll_factor] = {0.0};

        //multiplication block
        for (size_t l = 0; l < k; l += unroll_factor) {
            #pragma unroll
            for (size_t u = 0; u < unroll_factor; u++) {
                sum[u] += a[ii * k + l + u] * b[(l + u) * n + jj];
            }
        }

        double total_sum = 0.0;
        #pragma unroll //unroll adding to c
        for (size_t u = 0; u < unroll_factor; u++) {
            total_sum += sum[u];
        }
        c[ii * n + jj] = total_sum;
    }


}

extern "C" void matmul(const double *__restrict a, const double *__restrict b,
                      double *__restrict c, size_t m, size_t k, size_t n) {
    double *cuda_a, *cuda_b, *cuda_c;
    cudaMalloc(&cuda_a, m * k * sizeof(double));
    cudaMalloc(&cuda_b, k * n * sizeof(double));
    cudaMalloc(&cuda_c, m * n * sizeof(double));
    cudaMemcpy(cuda_a, a, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, k * n * sizeof(double), cudaMemcpyHostToDevice);

  
    constexpr size_t BLOCK_SIZE = 16;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //method call
    GrimbloTheGrumbler<<<grid_size, block_size>>>(cuda_a, cuda_b, cuda_c, m, k, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c, cuda_c, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

}

int main() { //dummy main
    size_t m = 500;
    size_t k = 400;
    size_t n = 500;

    std::vector<double> a(m * k, 1.0); //init matrix 'a' with all 1's
    std::vector<double> b(k * n, 2.0); //intittitititit matrix 'b' with all 2's
    std::vector<double> c(m * n);

    matmul(a.data(), b.data(), c.data(), m, k, n);

     std::cout << c[n] << "\t" << std::endl;
     
    return 0;
}

