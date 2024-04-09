#include <iostream> 
#include "cublas_v2.h"
using namespace std;


int check_if_equal(double *a, double *b, int N){

    for (size_t i = 0; i < N; i++){
        if (abs(a[i] - b[i]) > 1E-6){
            return 1;
        }
    }
    
    return 0;
}

void random_fill(double *array, int N, double random_lowest, double random_highest){

    const long max_rand = 1000000L;
    static double timep = 0.0;
    timep += 1.0;
    srandom(time(NULL) + timep);
    for (size_t i = 0; i < N; ++i){
    array[i] = random_lowest+(random_highest - random_lowest)*(random() % max_rand)/max_rand;
    }
}

int main() {

    double *a, *b, *c, *c_host;
    double *dev_a, *dev_b, *dev_c;
    double random_lowest = 1.0; 
    double random_highest = 5.0; 
    double alpha = 1.0;
    double beta = 0.0;
    const int N = 1024; //размер матрицы
    const int NN = pow(N, 2); //число элементов в матрице
    const int NA = 3; //сколько раз матрица A в матрице B
    cublasStatus_t status;
    cudaError_t cudaerr;
    cublasHandle_t handle;
    cudaStream_t stream[NA];
    int size = NN * sizeof(double);

    a = (double *)malloc(size);
    b = (double *)malloc(size*NA);
    c = (double *)malloc(size*NA);
    c_host = (double *)malloc(size*NA);

    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size*NA);
    cudaMalloc((void **)&dev_c, size*NA);

    random_fill(a, NN, random_lowest, random_highest);
    random_fill(b, NN*NA, random_lowest, random_highest);
    

    //CPU
    for (int x = 0; x < N; x++){
        for (int y = 0; y < N*NA; y++){
            double tmp = 0.0;
            for (int i = 0; i < N; ++i) {
                tmp += a[i * N + x] * b[y * N + i];
            }
            c_host[y * N + x] = tmp;
        }
    }


    //GPU cublas
    cublasCreate(&handle);
    status = cublasSetVector(NN, sizeof(double), a, 1, dev_a, 1);
    status = cublasSetVector(NN*NA, sizeof(double), b, 1, dev_b, 1);
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N*NA, N,
    &alpha, dev_a, N, dev_b, N, &beta, dev_c, N);
    status = cublasGetVector(NN*NA, sizeof(double), dev_c, 1, c, 1);


    
    if (check_if_equal(c, c_host, NN*NA)){
        cout << "Results from host and serial cublas are not equal!" << endl;
    }
    else {
        cout << "Results from host and serial cublas are equal!" << endl;
    }

    cudaFree(dev_c); cudaMalloc((void **)&dev_c, size*NA);
    free(c); c = (double *)malloc(size*NA);



    //GPU async cublas
    status = cublasSetVector(NN, sizeof(double), a, 1, dev_a, 1);

    for (int i = 0; i < NA; i++){
        cudaerr = cudaStreamCreate(&stream[i]);
    }
    for (int istream = 0; istream < NA; istream++){
        status = cublasSetVectorAsync(NN, sizeof(double), b + NN * istream,
        1, dev_b + NN * istream, 1, stream[istream]);
    }

    for (int istream = 0; istream < NA; istream++){
        status = cublasSetStream(handle, stream[istream]);
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
        &alpha, dev_a, N, dev_b + NN * istream, N, &beta, dev_c + NN * istream, N);

    }

    for (int istream = 0; istream < NA; istream++){
        status = cublasGetVectorAsync(NN, sizeof(double), dev_c + NN * istream,
        1, c + NN * istream, 1, stream[istream]);
    }

    for (int i = 0; i < NA; i++){
        cudaerr = cudaStreamDestroy(stream[i]);
    }


    if (check_if_equal(c, c_host, NN*NA)){
        cout << "Results from host and async cublas are not equal!" << endl;
    }
    else {
        cout << "Results from host and async cublas are equal!" << endl;
    }

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    free(a); free(b); free(c); free(c_host);
    return 0;
}
