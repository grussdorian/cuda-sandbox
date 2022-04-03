#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void convolution(float *mat_input1, float *mat_conv_input1, float *mat_output1, int mat_datasize, int mat_dim);

__global__ void
convolution(float *mat_input1, float *mat_conv_input1, float *mat_output1, int mat_datasize, int mat_dim)
{
    int blockNum = blockIdx.x;
    int threadNum = threadIdx.x;
    int globalThreadId = blockNum*(blockDim.x) + threadNum;
    int i, j;

    int row_val = globalThreadId/mat_dim;
    int col_val = globalThreadId%mat_dim;

    if(row_val<mat_dim && col_val<mat_dim)
    {

	//Write code for convolution

    }
}
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
    cudaError_t err = cudaSuccess;

    int i, j, k;
    int t;
    scanf("%d",&t);
    while(t--)
    {
        int mat_dim;
        scanf("%d",&mat_dim);
        int mat_num_eles = mat_dim*mat_dim;
        size_t mat_size = mat_num_eles*sizeof(float);


	/* populate code for allocating host memory
        float *h_mat_input1 = ______;
        float *h_mat_output1 = ________;
        float *h_mat_output2 = ____________;

        int mat_conv_dim = 3;
        int mat_conv_num_eles = mat_conv_dim*mat_conv_dim;
        size_t mat_conv_size = mat_conv_num_eles*sizeof(float);

        float *h_mat_conv_input = ______________;

        if (h_mat_input1 == NULL || h_mat_output1 == NULL || h_mat_output2 == NULL || h_mat_conv_input == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        for(i=0;i<mat_num_eles;i++)
        {
            j = i/mat_dim;
            k = i%mat_dim;

            scanf("%f",&h_mat_input1[mat_dim*j + k]);         
	}
        
        for(i=0;i<mat_conv_num_eles;i++)
        {
            j = i/mat_conv_dim;
            k = i%mat_conv_dim;
            h_mat_conv_input[mat_conv_dim*j + k] = 1.0/9.0;
                     
        }
       

        float *d_mat_input1 = NULL;
        err = cudaMalloc((void **)&d_mat_input1, mat_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector d_mat_input1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        float *d_mat_conv_input = NULL;
        err = cudaMalloc((void **)&d_mat_conv_input, mat_conv_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector d_mat_conv_input (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        float *d_mat_output1 = NULL;
        err = cudaMalloc((void **)&d_mat_output1, mat_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector d_mat_output1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

       
        err = cudaMemcpy(d_mat_input1, h_mat_input1, mat_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector h_mat_input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(d_mat_conv_input, h_mat_conv_input, mat_conv_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector h_mat_conv_input from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        int mat_block_dim = 16;
        int mat_grid_dim = __________________;
        convolution<<<mat_grid_dim, mat_block_dim>>>(d_mat_input1, d_mat_conv_input, d_mat_output1, mat_size, mat_dim);
       
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch process_kernel2 kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        err = cudaMemcpy(h_mat_output1, d_mat_output1, mat_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_mat_output1 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
  
        err = cudaFree(d_mat_input1);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector d_mat_input1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaFree(d_mat_conv_input);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector d_mat_conv_input (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaFree(d_mat_output1);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector d_mat_output1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
       print_matrix(h_mat_output1,mat_dim,mat_dim);
        
        free(h_mat_input1);
        free(h_mat_output1);
        free(h_mat_output2);
        free(h_mat_conv_input);

        err = cudaDeviceReset();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

    }
    return 0;
}
