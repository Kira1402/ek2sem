#include <iostream> 
using namespace std;

__global__ void add(double *a, double *b, double *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_fill(double *array, int N, double random_lowest, double random_highest){

    const long max_rand = 1000000L;
    static double timep = 0.0;
    timep += 1.0;
    srandom(time(NULL) + timep);
    for (size_t i = 0; i != N; ++i){
    array[i] = random_lowest+(random_highest - random_lowest)*(random() % max_rand)/max_rand;
    
    }

}

int main() {

    double *a, *b, *c, *d; // host copies of a, b, c
    double *d_a, *d_b, *d_c; // device copies of a, b, c
    const int N = 512; // size of the arrays
    double random_lowest = 1.0; //lowest possible random double
    double random_highest = 10.0; //highest possible random double

    int size = N * sizeof(double);

    a = (double *)malloc(size);
    b = (double *)malloc(size);
    c = (double *)malloc(size);
    d = (double *)malloc(size);

    random_fill(a, N, random_lowest, random_highest);
    random_fill(b, N, random_lowest, random_highest);
    
    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    add<<<N,1>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    for (int i = 0; i != N; i++){
    
        d[i] = a[i] + b[i];
        
    }


    for (int i = 0; i != N; i++){
    
        if (c[i] != d[i]){
            cout << "c != d" << endl;
            return -1;
            
        }
        
    }
    
    cout << "c = d" << endl;


    free(a); free(b); free(c); free(d);
    return 0;

}

