#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void convolution(float *mat_input1, float *mat_conv_input1, float *mat_output1, int mat_datasize, int mat_dim);
__global__ void convolution2(float *mat_input1, float *mat_conv_input1, float *mat_output1, int mat_datasize, int mat_dim);

__device__
int getGlobalIdx_3D_3D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) 
    + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

// VERTICAL MASK
__global__ void
convolution2(float *mat_input1, float *mat_conv_input1, float *mat_output1, int mat_datasize, int mat_dim)
{

    mat_conv_input1[4] =0;

    int globalThreadId = getGlobalIdx_3D_3D();

    int row_val = globalThreadId/mat_dim;
    int col_val = globalThreadId%mat_dim;

    if(row_val<mat_dim && col_val<mat_dim)
    {
    int index = mat_dim*row_val + col_val;
	//Write code for convolution

    // corner points

        //top left corner
        if(globalThreadId == 0 ){
            mat_output1[globalThreadId] += mat_input1[globalThreadId] * mat_conv_input1[4]+  (mat_input1[mat_dim*(row_val+1) + col_val]*mat_conv_input1[7]);
        }
        //bottom left corner
        else if(globalThreadId == mat_dim*(mat_dim -1 ) ) {
            mat_output1[globalThreadId] = (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[mat_dim*(row_val-1) + col_val]*mat_conv_input1[1]);
        }
        //bottom right corner
        else if(globalThreadId == mat_dim*mat_dim - 1 ) {
            mat_output1[globalThreadId] += (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[mat_dim*(row_val-1) + col_val]*mat_conv_input1[1]);
        }
        //edge points

        //leftmost column
        else if( col_val == 0){
            mat_output1[globalThreadId] += (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[mat_dim*(row_val-1) + col_val]*mat_conv_input1[1])
                +  (mat_input1[mat_dim*(row_val+1) + col_val]*mat_conv_input1[7]);
        }
        //top row
        else if(globalThreadId < mat_dim-1 && globalThreadId!=0 ){
            mat_output1[globalThreadId] += (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[mat_dim*(row_val+1) + col_val]*mat_conv_input1[7]);
        }
        //rightmost column
        else if(col_val == mat_dim-1 && globalThreadId != mat_dim-1 ){
            mat_output1[globalThreadId] += (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[mat_dim*(row_val-1) + col_val]*mat_conv_input1[1])
                +  (mat_input1[mat_dim*(row_val+1) + col_val]*mat_conv_input1[7]);
        }
        //top right corner
        else if(globalThreadId == mat_dim-1){
            mat_output1[globalThreadId] += (mat_input1[index] * mat_conv_input1[4]) + (mat_input1[mat_dim*(row_val+1) + col_val]*mat_conv_input1[7]);
        }
        //bottom row
        else if(globalThreadId > mat_dim*(mat_dim-1) && globalThreadId < ((mat_dim*mat_dim)-1) ){
             mat_output1[globalThreadId] +=  (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[mat_dim*(row_val-1) + col_val]*mat_conv_input1[1]);
        } else{
                mat_output1[index] += (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[mat_dim*(row_val-1) + col_val]*mat_conv_input1[1])
                +  (mat_input1[mat_dim*(row_val+1) + col_val]*mat_conv_input1[7]);
        }

    }

}

// HORIZONTAL MASK 
__global__ void
convolution(float *mat_input1, float *mat_conv_input1, float *mat_output1, int mat_datasize, int mat_dim)
{
    int globalThreadId = getGlobalIdx_3D_3D();

    int row_val = globalThreadId/mat_dim;
    int col_val = globalThreadId%mat_dim;

    if(row_val<mat_dim && col_val<mat_dim)
    {
    int index = mat_dim*row_val + col_val;
	//Write code for convolution

    // corner points

        //top left corner
        if(globalThreadId == 0 ){
            mat_output1[globalThreadId] = mat_input1[globalThreadId] * mat_conv_input1[4] + (mat_input1[index +1 ]*mat_conv_input1[5]);
        }
        //bottom left corner
        else if(globalThreadId == mat_dim*(mat_dim -1 ) ) {
            mat_output1[globalThreadId] = (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[index + 1 ]*mat_conv_input1[5]);
        }
        //bottom right corner
        else if(globalThreadId == mat_dim*mat_dim - 1 ) {
            mat_output1[globalThreadId] = (mat_input1[index] * mat_conv_input1[4])
                 +(mat_input1[index-1]*mat_conv_input1[3]);
        }
        //edge points

        //leftmost column
        else if( col_val == 0){
            mat_output1[globalThreadId] = (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[index +1 ]*mat_conv_input1[5]);
        }
        //top row
        else if(globalThreadId < mat_dim-1 && globalThreadId!=0 ){
            mat_output1[globalThreadId] = (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[index -1 ]*mat_conv_input1[3])
                +  (mat_input1[index +1 ]*mat_conv_input1[5]);
        }
        //rightmost column
        else if(col_val == mat_dim-1 && globalThreadId != mat_dim-1 ){
            mat_output1[globalThreadId] =(mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[index -1 ]*mat_conv_input1[3]);
        }
        //top right corner
        else if(globalThreadId == mat_dim-1){
            mat_output1[globalThreadId] = (mat_input1[index] * mat_conv_input1[4]) +  (mat_input1[index -1 ]*mat_conv_input1[3]);
        }
        //bottom row
        else if(globalThreadId > mat_dim*(mat_dim-1) && globalThreadId < ((mat_dim*mat_dim)-1) ){
             mat_output1[globalThreadId] =  (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[index -1 ]*mat_conv_input1[3])
                +  (mat_input1[index +1 ]*mat_conv_input1[5]);;
        } else{
                mat_output1[index] = (mat_input1[index] * mat_conv_input1[4])
                +  (mat_input1[index -1 ]*mat_conv_input1[3])
                +  (mat_input1[index +1 ]*mat_conv_input1[5]);
        }

    }

}

void print_matrix(float *A,int m,int n)
{
    for(int i =0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            printf("%.2f ",A[i*m+j]);
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


	//populate code for allocating host memory

        float *h_mat_input1 = (float*)malloc(mat_size);
        float *h_mat_output1 = (float*)malloc(mat_size);
        int mat_conv_dim = 3;
        int mat_conv_num_eles = mat_conv_dim*mat_conv_dim;
        size_t mat_conv_size = mat_conv_num_eles*sizeof(float);

        float h_mat_conv_input[] = {0,-1,0,-1,5,-1,0,-1,0};

        if (h_mat_input1 == NULL || h_mat_output1 == NULL || h_mat_conv_input == NULL)
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
        // total matrix size upper bounded by 2^26 so size allocated for kernel is adjusted accordingly
        int gx = ceil(mat_dim/32);
        int gy = ceil(mat_dim/32);
        dim3 mat_grid_dim(gx,gy,4);
        // max threads per block = 1024 for backwards compatibility
        dim3 mat_block_dim(32,32,1);
        //successive use of 1D masks
        convolution<<<mat_grid_dim, mat_block_dim>>>(d_mat_input1, d_mat_conv_input, d_mat_output1, mat_size, mat_dim);
        convolution2<<<mat_grid_dim, mat_block_dim>>>(d_mat_input1, d_mat_conv_input, d_mat_output1, mat_size, mat_dim);

       
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

        err = cudaDeviceReset();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    return 0;
}
