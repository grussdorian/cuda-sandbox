#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>



void print_array(float *A, int N)
{
    for(int i=0;i<N;i++)
        printf("%.2f ",A[i]);
    printf("\n");
}


__global__ void
compute_kernel1(float *input1, float *input2, float *output, int datasize)
{
    int numElements = datasize / sizeof(float);

    //Write code for i

    if (i < numElements)
    {
        //Write code for compute
    }
}


__global__ void
compute_kernel2(float *input, float *output, int datasize)
{
    int numElements = datasize / sizeof(float);

     //Write code for i

    if (i < numElements)
    {
        //Write code for compute
    }
}


__global__ void
compute_kernel3(float *input, float *output, int datasize)
{
    int numElements = datasize / sizeof(float);

    //Write code for i
    
    if (i < numElements)
    {
 	   //Write code for compute
    }
}



int main(void)
{
    cudaError_t err = cudaSuccess;

    int numElements = 16384;
    size_t size = numElements * sizeof(float);

    float *h_input1 = (float *)malloc(size);

    float *h_input2 = (float *)malloc(size);

    float *h_output1 = (float *)malloc(size);

    float *h_output2 = (float *)malloc(size);

    float *h_output3 = (float *)malloc(size);

    if (h_input1 == NULL || h_input2 == NULL || h_output1 == NULL || h_output2 == NULL || h_output3 == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }


   
    for (int i = 0; i < numElements; ++i)
    {
        scanf("%f",&h_input1[i]);
        
    }
    for (int i = 0; i < numElements; ++i)
    {
        scanf("%f",&h_input2[i]);
        
    }
    


    float *d_input1 = NULL;
    err = cudaMalloc((void **)&d_input1, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_input2 = NULL;
    err = cudaMalloc((void **)&d_input2, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_output1 = NULL;
    err = cudaMalloc((void **)&d_output1, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector h_output1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_output2 = NULL;
    err = cudaMalloc((void **)&d_output2, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector h_output2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_output3 = NULL;
    err = cudaMalloc((void **)&d_output3, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector h_output3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_input1, h_input1, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector h_input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_input2, h_input2, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector h_input2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  //Complete Code for launching compute_kernel1
  
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Complete Code for launching compute_kernel2


    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel2 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Complete Code for launching compute_kernel3 


    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel3 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
1
    
    // printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_output1, d_output1, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_output1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_output2, d_output2, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_output2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_output3, d_output3, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_output3 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


   

    print_array(h_output3,numElements);
    

    err = cudaFree(d_input1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_input1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_input2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_input2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_output1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_output2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output3);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_output3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(h_input1);
    free(h_input2);
    free(h_output1);
    free(h_output2);
    free(h_output3);

    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   
    return 0;
}

