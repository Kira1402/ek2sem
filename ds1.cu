п#include <iostream> 
#define BLOCK_SIZE 8
#define RADIUS 4

using namespace std;


__global__ void stencil_1d(double *in, double *out, int N) {
    __shared__ double temp[BLOCK_SIZE + 2 * RADIUS]; 
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    int lindex = threadIdx.x + RADIUS; 

    temp[lindex] = in[gindex]; 
    if (threadIdx.x < RADIUS) {
        if (gindex - RADIUS >= 0)
            temp[lindex - RADIUS] = in[gindex - RADIUS];
        if (gindex + BLOCK_SIZE < N)
            temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE]; 
    }

    __syncthreads(); // sync cache memory
    
    double result = 0;
    if (gindex >= RADIUS && gindex <= N - 1 - RADIUS){
        for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
            result += temp[lindex + offset];
        out[gindex - RADIUS] = result;
    }


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

    double *a, *b, *c; // host copies of a, b, c
    double *d_a, *d_b; // device copies of a, b, c
    const int N = 20; // size of the arrays
    double random_lowest = 1.0; //lowest possible random double
    double random_highest = 10.0; //highest possible random double

    int size = N * sizeof(double);

    a = (double *)malloc(size);
    b = (double *)malloc(size);
    c = (double *)malloc(size);

    random_fill(a, N, random_lowest, random_highest);
    
    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    stencil_1d<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(d_a, d_b, N);

    // Copy result back to host
    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b);

    double result;
    for (int i = RADIUS; i != N - RADIUS; i++){
        result = 0;
        for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
            result += a[i + offset];
        // Store the result
        c[i - RADIUS] = result;
    }


    for (int i = 0; i != N; i++){
        if (b[i] != c[i]){
            cout << "b != c" << endl;
            return 1;
        }
    }
    cout << "b = c" << endl;


    free(a); free(b); free(c);
    return 0;

}
