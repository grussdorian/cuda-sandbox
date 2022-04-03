#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define LOG_INPUT if(0)
#define LOG_OUTPUT if(1)
#define LOG if(0)


__global__ void hadamard(float *A, float *B, float *C, int M, int N)
{
    // Complete the kernel code snippet
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if((i<M) && (j<N))
    {
        C[i*N + j] = A[i*N + j]*B[i*N + j]; 
    }
}

/**
 * Host main routine
 */
void print_matrix(float *A,int m,int n)
{
    for(int i =0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            printf("%.2f ",A[i*n+j]);
        printf("\n");
    }

}
int main(void)
{
    float *d_A = NULL , *d_B = NULL , *d_C = NULL;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;


    int t; //number of test cases
    scanf("%d",&t);
    while(t--)
    {
        int m,n;
        scanf("%d %d",&m,&n);
        if(m<1 || n<1){
            LOG printf("Please enter correct dimensions\n");
            continue;
        }
        // Print the vector length to be used, and compute its size
        size_t size = m*n * sizeof(float);
        
        LOG printf("[Hadamard product of two matrices ]\n");
        LOG printf("n = %d m = %d\n", n,m);

        // Allocate the host input vector A
        float *h_A = (float*)malloc(sizeof(float)*size);
        // Allocate the host input vector B
        float *h_B = (float*)malloc(sizeof(float)*size);
        // Allocate the host output vector C
        float *h_C = (float*)malloc(sizeof(float)*size);

        // Initialize the host input vectors
        for (int i = 0; i < n*m; ++i)
            scanf("%f",&h_A[i]);

        for (int i = 0; i < n*m; ++i)
            scanf("%f",&h_B[i]);
        
        // Verify that allocations succeeded
        if (h_A == NULL || h_B == NULL || h_C == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }
        // Allocate the device input vector A
        err = cudaMalloc((void**)&d_A, size);
        if(err != cudaSuccess){
            fprintf ( stderr , " Failed to allocate device vector A ( error code %s) !\n",cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        // Allocate the device input vector B
        err = cudaMalloc((void**)&d_B, size);
        if(err != cudaSuccess){
            fprintf ( stderr , " Failed to allocate device vector B ( error code %s) !\n",cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        // Allocate the device output vector C
        err = cudaMalloc((void**)&d_C, size);
        if(err != cudaSuccess){
            fprintf ( stderr , " Failed to allocate device vector C ( error code %s) !\n",cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        // Copy the host input vectors A and B in host memory to the device input vectors in
        // device memory
        err = cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
        if ( err != cudaSuccess )
        {
            fprintf ( stderr, "Failed to copy vector A from host to device ( error code %s)!\n", cudaGetErrorString (err));
            exit ( EXIT_FAILURE );
        }
        err = cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
        if ( err != cudaSuccess )
        {
            fprintf ( stderr, "Failed to copy vector B from host to device ( error code %s)!\n", cudaGetErrorString (err));
            exit ( EXIT_FAILURE );
        }
        // initialize blocksPerGrid and threads Per Block
        // m=m<32?m:32;
        // n=n<32?n:32;
        int ty=m>32?32:m;
        int tx=n>32?32:n;
        int bx=32,by=32;
        if(m<32){
            bx = 1;
        }else{
            bx=m/32 +1;
        }
        if(n<32){
            by = 1;
        }else{
            bx= n/32 +1;
        }
        dim3 blocksPerGrid(bx,by,1);
        dim3 threadsPerBlock(tx,ty,1);
        //printf("m = %d n = %d bx = %d by = %d tx=%d ty=%d \n",m,n,bx,by,tx,ty);
        hadamard<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the device result vector in device memory to the host result vector
        // in host memory.
        err = cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
        if ( err != cudaSuccess )
        {
            fprintf ( stderr, "Failed to copy vector C from device to host ( error code %s)! \n", cudaGetErrorString(err));
            exit ( EXIT_FAILURE ) ;
        }
        // Verify that the result vector is correct
        for (int i = 0; i < n*m; ++i)
        {
            if (fabs(h_A[i] * h_B[i] - h_C[i]) > 1e-5)
            {
                printf("h_A = %0.2f   h_B = %0.2f   h_C = %0.2f\n",h_A[i],h_B[i],h_C[i]);
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }

        LOG printf("Test PASSED\n");
        // Free device global memory
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        
        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        print_matrix(h_C,m,n);
        // Free host memory
        free(h_A); free(h_B); free(h_C);
        LOG printf("Done\n");
    }
    return 0;
}

