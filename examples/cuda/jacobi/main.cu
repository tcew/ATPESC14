#include <iostream>
#include <cuda.h>
#include <math.h>

using namespace std;

#define BDIM 256

/* 

   Poisson problem:  diff(u, x, 2) + diff(u, y, 2) = f

   Coordinates:      x -> -1 + delta*i, 
                     y -> -1 + delta*j

   2nd order finite difference:   4*u(j,i) - u(j-1,i) - u(j+1,i) - u(j,i-1) - u(j,i+1) = -delta*delta*f(j,i) 

*/

__global__ void jacobi(const int N,
                       const float *rhs,
                       const float *u,
                       float *newu){
  // Get thread ID
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if((i < N) && (j < N)){

    // Get padded grid ID
    const int pid = (j + 1)*(N + 2) + (i + 1);

    float invD = 0.25;

    newu[pid] = invD*(rhs[pid]
		      + u[pid - (N+2)]
		      + u[pid + (N+2)]
		      + u[pid - 1]
		      + u[pid + 1]);
  }
}

__global__ void partialReduceResidual(const int entries,
                               float *u,
                               float *newu,
                               float *red){
  __shared__ double s_red[BDIM];

  const int id = blockIdx.x*blockDim.x + threadIdx.x;

  s_red[threadIdx.x] = 0;

  if(id < entries){
    const float diff = u[id] - newu[id];
    s_red[threadIdx.x] = diff*diff;
  }

  __syncthreads();  // barrier (make sure s_red is ready)

  // manually unrolled reduction (assumes BDIM=256)
  if(BDIM>128) {
    if(threadIdx.x<128)
      s_red[threadIdx.x] += s_red[threadIdx.x+128];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>64){
    if(threadIdx.x<64)
      s_red[threadIdx.x] += s_red[threadIdx.x+64];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>32){
    if(threadIdx.x<32)
      s_red[threadIdx.x] += s_red[threadIdx.x+32];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>16){
    if(threadIdx.x<16)
      s_red[threadIdx.x] += s_red[threadIdx.x+16];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>8){
    if(threadIdx.x<8)
      s_red[threadIdx.x] += s_red[threadIdx.x+8];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>4){
    if(threadIdx.x<4)
      s_red[threadIdx.x] += s_red[threadIdx.x+4];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>2){
    if(threadIdx.x<2)
      s_red[threadIdx.x] += s_red[threadIdx.x+2];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>1){
    if(threadIdx.x<1)
      s_red[threadIdx.x] += s_red[threadIdx.x+1];
  }

  // store result of this block reduction
  if(threadIdx.x==0)
    red[blockIdx.x] = s_red[threadIdx.x];
}

int main(int argc, char** argv){

  if(argc != 3){
    printf("Usage: ./main N tol \n");
    return 0;
  }

  const int N     = atoi(argv[1]);
  const float tol = atof(argv[2]);
  float res = 0;

  const int iterationChunk = 100; // needs to be multiple of 2
  int iterationsTaken = 0;

  // Setup jacobi kernel block-grid sizes
  dim3 jBlock(16,16);
  dim3 jGrid((N + jBlock.x - 1)/jBlock.x, (N + jBlock.y - 1)/jBlock.y);

  // Setup reduceResidual kernel block-grid sizes
  dim3 rBlock(BDIM);
  dim3 rGrid((N*N + rBlock.x - 1)/rBlock.x);

  // Host Arrays
  float *h_u   = (float*) calloc((N+2)*(N+2), sizeof(float));
  float *h_u2  = (float*) calloc((N+2)*(N+2), sizeof(float));
  float *h_rhs = (float*) calloc((N+2)*(N+2), sizeof(float));

  float *h_res = (float*) calloc(rGrid.x, sizeof(float));

  float delta = 2./(N+1);
  float normRHS = 0;
  for(int j = 0; j < N+2; ++j){
    for(int i = 0; i < N+2; ++i){
      float x = -1 + delta*i;
      float y = -1 + delta*j;

      h_rhs[i + (N+2)*j] = delta*delta*(2.*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y));
      normRHS += pow(h_rhs[i+(N+2)*j],2);
    }
  }
  normRHS = sqrt(normRHS);
  
  // Device Arrays
  float *c_u, *c_u2, *c_rhs, *c_res;

  cudaMalloc((void**) &c_u  , (N+2)*(N+2)*sizeof(float));
  cudaMalloc((void**) &c_u2 , (N+2)*(N+2)*sizeof(float));
  cudaMalloc((void**) &c_rhs ,(N+2)*(N+2)*sizeof(float));

  cudaMalloc((void**) &c_res, rGrid.x*sizeof(float));

  // Setting device vectors to 0
  cudaMemcpy(c_u ,  h_u ,  (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_u2,  h_u2,  (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_rhs, h_rhs, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);

  // Create CUDA events
  cudaEvent_t startEvent, endEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&endEvent);

  cudaEventRecord(startEvent, 0);
  do {
    // Call jacobi [iterationChunk] times before calculating residual
    for(int i = 0; i < iterationChunk/2; ++i){

      // first iteration 
      jacobi<<<jGrid, jBlock>>>(N, c_rhs, c_u, c_u2);
      
      // flip flop arguments
      jacobi<<<jGrid, jBlock>>>(N, c_rhs, c_u2, c_u);
    }

    // Calculate residual
    partialReduceResidual<<<rGrid, rBlock>>>(N*N, c_u, c_u2, c_res);

    // Finish reduce in host
    cudaMemcpy(h_res, c_res, rGrid.x*sizeof(float), cudaMemcpyDeviceToHost);

    res = 0;
    for(int i = 0; i < rGrid.x; ++i)
      res += h_res[i];

    res = sqrt(res);

    iterationsTaken += iterationChunk;

    printf("residual = %g after %d steps \n", res, iterationsTaken);

  } while(res > tol);

  cudaEventRecord(endEvent, 0);
  cudaEventSynchronize(endEvent);

  // Get time taken
  float timeTaken;
  cudaEventElapsedTime(&timeTaken, startEvent, endEvent);

  // Copy final solution from device array to host
  cudaMemcpy(h_u, c_u, (N+2)*(N+2)*sizeof(float), cudaMemcpyDeviceToHost);

  // Compute maximum error in finite difference solution
  FILE *fp = fopen("result.dat", "w");
  float maxError = 0;
  for(int j = 0; j < N+2; ++j){
    for(int i = 0; i < N+2; ++i){
      float x = -1 + delta*i;
      float y = -1 + delta*j;
      float error = fabs( sin(M_PI*x)*sin(M_PI*y) - h_u[i + (N+2)*j]);
      maxError = (error > maxError) ? error:maxError;
      fprintf(fp, "%g %g %g %g\n", x, y, h_u[i+(N+2)*j],error);
    }
  }
  fclose(fp);

  const float avgTimePerIteration = timeTaken/((float) iterationsTaken);

  printf("Top right current          : %7.9e\n"     , h_u[N*(N+2) + N]);
  printf("Residual                   : %7.9e\n"     , res);
  printf("Iterations                 : %d\n"        , iterationsTaken);
  printf("Average time per iteration : %3.5e ms\n"  , avgTimePerIteration);
  printf("Bandwidth                  : %3.5e GB/s\n", (1.0e-6)*(6*N*N*sizeof(float))/avgTimePerIteration);
  printf("Maximum absolute error     : %7.9e\n"     ,  maxError);

  // Free all the mess
  cudaFree(c_u);
  cudaFree(c_u2);
  cudaFree(c_res);

  free(h_u);
  free(h_u2);
  free(h_res);
}
