#include <iostream> 
#include "cublas_v2.h"
#define BLOCK_SIZE 2
using namespace std;

__global__ void matmul_naive_1(double *a, double *b, double *c, int matrix_dim) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < matrix_dim && y < matrix_dim) {
        float tmp = 0.0;
        for (int i = 0; i < matrix_dim; ++i) {
            tmp += a[x * matrix_dim + i] * b[i * matrix_dim + y];
        }
        c[x * matrix_dim + y] = tmp;
    }
}

__global__ void matmul_naive_2(double *a, double *b, double *c, int matrix_dim) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < matrix_dim && y < matrix_dim) {
        float tmp = 0.0;
        for (int i = 0; i < matrix_dim; ++i) {
            tmp += a[y * matrix_dim + i] * b[i * matrix_dim + x];
        }
        c[y * matrix_dim + x] = tmp;
    }
}

__global__ void matmul_block(double *a, double *b, double *c, int matrix_dim) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //row
    int y = blockIdx.y * blockDim.y + threadIdx.y; //column

    const int N_BLOCKS = matrix_dim / BLOCK_SIZE;

    __shared__ double a_temp[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ double b_temp[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ double c_temp[BLOCK_SIZE*BLOCK_SIZE];
    c_temp[threadIdx.y + threadIdx.x] = 0.0;

    for (int k = 0; k != N_BLOCKS; k++){
        a_temp[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 
        a[y * matrix_dim + (threadIdx.x + BLOCK_SIZE * k)];

        b_temp[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 
        b[(threadIdx.y + BLOCK_SIZE * k) * matrix_dim + x];

        __syncthreads();

        float tmp = 0.0;
        for (int i = 0; i != BLOCK_SIZE; ++i) {
            tmp += a_temp[threadIdx.y * BLOCK_SIZE + i] * b_temp[i * BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();
        c_temp[threadIdx.y * BLOCK_SIZE + threadIdx.x] += tmp;


    }
    c[y * matrix_dim + x] = c_temp[threadIdx.y * BLOCK_SIZE + threadIdx.x];

}

int check_if_equal(double *a, double *b, int N){

    for (int i = 0; i != N; i++){
        if (abs(a[i] - b[i]) > 1E-1){
            return 1;
        }
    }
    
    return 0;
}

int check_if_equal_transpose(double *a, double *b, int matrix_dim){

    for (size_t i = 0; i != matrix_dim; i++){
        for (size_t j = 0; j != matrix_dim; j++){

            if (abs(a[j * matrix_dim + i] - b[i * matrix_dim + j]) > 1E-1){
                cout << a[j * matrix_dim + i] <<" " <<  b[i * matrix_dim + j] <<
                 " " << abs(a[j * matrix_dim + i] - b[i * matrix_dim + j]) << endl;
                return 1;
            }

        }
    }
    
    return 0;
}

void random_fill(double *array, int N, double random_lowest, double random_highest){

    const long max_rand = 1000000L;
    static double timep = 0.0;
    timep += 1.0;
    srandom(time(NULL) + timep);
    for (int i = 0; i != N; ++i){
    array[i] = random_lowest+(random_highest - random_lowest)*(random() % max_rand)/max_rand;
    }
}

int main() {

    double *a, *b, *c_host, *c_device; 
    double *d_a, *d_b, *d_c;
    const int N = 2; 
    const int NN = pow(N, 2);
    double random_lowest = 1.0; 
    double random_highest = 5.0; 
    double alpha = 1.0;
    double beta = 0.0;
    cublasHandle_t handle;
    int size = NN * sizeof(double);

    dim3 gridDim(N / BLOCK_SIZE, N / BLOCK_SIZE, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Alloc space for host copies of a, b, c and setup input values
    a        = (double *)malloc(size);
    b        = (double *)malloc(size);
    c_host   = (double *)malloc(size);
    c_device = (double *)malloc(size);

    random_fill(a, NN, random_lowest, random_highest);
    random_fill(b, NN, random_lowest, random_highest);

    for (int x = 0; x != N; x++){
        for (int y = 0; y != N; y++){
            float tmp = 0.0;
            for (int i = 0; i != N; ++i) {
                tmp += a[x * N + i] * b[i * N + y];
            }
            c_host[x * N + y] = tmp;
        }
    }


    // Alloc space for device copies of a, b
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    matmul_naive_1<<<gridDim,blockDim>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c_device, d_c, size, cudaMemcpyDeviceToHost);

    //check if two results are equal
    if (check_if_equal(c_device, c_host, NN)){
        cout << "Results from host and device are not equal!" << endl;
    }
    else {
        cout << "Results from host and device are equal!" << endl;
    }


    matmul_naive_2<<<gridDim,blockDim>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c_device, d_c, size, cudaMemcpyDeviceToHost);

    //check if two results are equal
    if (check_if_equal(c_device, c_host, NN)){
        cout << "Results from host and device are not equal!" << endl;
    }
    else {
        cout << "Results from host and device are equal!" << endl;
    }

    matmul_block<<<gridDim,blockDim>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c_device, d_c, size, cudaMemcpyDeviceToHost);

    //check if two results are equal
    if (check_if_equal(c_device, c_host, NN)){
        cout << "Results from host and device are not equal!" << endl;
    }
    else {
        cout << "Results from host and device are equal!" << endl;
    }


    cublasCreate(&handle);
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, d_a, N, d_b, N, &beta, d_c, N);

    // Copy result back to host
    cudaMemcpy(c_device, d_c, size, cudaMemcpyDeviceToHost);

    if (check_if_equal_transpose(c_device, c_host, N)){
        cout << "Results from host and device are not equal!" << endl;
    }
    else {
        cout << "Results from host and device are equal!" << endl;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); 
    free(a); free(b); free(c_host); free(c_device);
    return 0;
}
