#include <iostream> 
#include "cusolverDn.h"
using namespace std;
#include <cstddef>

void print_matrix(string array_name, double *a, int N, int M){
    for (size_t i = 0; i != N; i++){
        for (size_t j = 0; j != M; j++){

            cout << a[j * N + i] << " ";

        }

        cout << endl;
    }
}

void print_array(string array_name, double *a, int N){
    cout << array_name + " = (";
    for (size_t i = 0; i != N; i++){
        cout << a[i];
        if (i != N-1) cout << ",";
        if (i > 20) {
            cout << "...";
            break;
        }
    }
    cout << ")" << endl;
}

int check_if_equal(double *a, double *b, int N){

    for (size_t i = 0; i != N; i++){
        if (abs(a[i] - b[i]) > 1E-6){
            cout << a[i] <<" " <<  b[i] << " " << abs(a[i] - b[i]) << endl;
            return 1;
        }
    }
    
    return 0;
}

void random_fill_sym(double *a, int N, double random_lowest, double random_highest){

    const long max_rand = 1000000L;
    static double timep = 0.0;
    timep += 1.0;
    srandom(time(NULL) + timep);

    for (size_t i = 0; i != N; i++){
        for (size_t j = 0; j <= i; j++){
            a[j * N + i] = random_lowest+(random_highest - random_lowest)*(random() % max_rand)/max_rand;
            if (i != j) a[i * N + j] = a[j * N + i];
        }
    }
}

int main() {

    double *a, *eig;
    double *dev_a, *dev_eig;
    const int N = 4; //размер матрицы
    const int NN = pow(N, 2); //число элементов в матрице
    double random_lowest = 1.0; 
    double random_highest = 5.0;
    cusolverDnHandle_t cusolverH;
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
    cusolverDnParams_t dn_params;

    int size = NN * sizeof(double);
    int size2 = N * sizeof(double);

    a = (double *)malloc(size);
    eig = (double *)malloc(size2);


    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_eig, size2);

    random_fill_sym(a, N, random_lowest, random_highest);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

    cout << "Матрица A:" << endl;
    print_matrix("A", a, N, N);
    cout << endl;

    
    
    cusolverDnCreateParams(&dn_params);
    cusolver_status = cusolverDnCreate(&cusolverH);

    size_t workspaceDevice = 0;
    size_t workspaceHost = 0;
    cusolver_status = cusolverDnXsyevd_bufferSize(
        cusolverH, dn_params, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
        N, CUDA_R_64F, dev_a, N, CUDA_R_64F, dev_eig, CUDA_R_64F,
        &workspaceDevice, &workspaceHost);


    double *d_work, *h_work = 0;
    int *d_dev_info = 0;
    cudaMalloc((void**)&d_dev_info, sizeof(int));
    cudaMalloc((void**)&d_work, workspaceDevice);
    cudaMemset((void*)d_dev_info, 0, sizeof(int));
    cudaMemset((void*)d_work, 0, workspaceDevice);
    h_work = (double*)malloc(workspaceHost);
    
    
    cusolver_status = cusolverDnXsyevd(
        cusolverH, dn_params, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
        N, CUDA_R_64F, dev_a, N, CUDA_R_64F,
        dev_eig, CUDA_R_64F, d_work, workspaceDevice, h_work, workspaceHost,
        d_dev_info);

    cudaMemcpy(eig, dev_eig, size2, cudaMemcpyDeviceToHost);

    print_array("Собств. числа", eig, N);

    free(h_work);
    cudaFree(d_work);
    cudaFree(d_dev_info);
    cudaFree(dev_eig);    
    cudaFree(dev_a); 
    free(a); 
    return 0;
}
