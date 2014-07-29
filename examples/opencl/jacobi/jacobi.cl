
/* 

   Poisson problem:               diff(u, x, 2) + diff(u, y, 2) = f

   Coordinate transform:          x -> -1 + delta*i, 
                                  y -> -1 + delta*j

   2nd order finite difference:   4*u(j,i) - u(j-1,i) - u(j+1,i) - u(j,i-1) - u(j,i+1) = -delta*delta*f(j,i) 

   define: rhs(j,i) = -delta*delta*f(j,i)

   Jacobi iteration: newu(j,i) = 0.25*(u(j-1,i) + u(j+1,i) + u(j,i-1) + u(j,i+1) + rhs(j,i))

   To run with a 402x402 grid until the solution changes less than 1e-7 per iteration (in l2): ./main 400 1e-7  

*/

__kernel void jacobi(const int N,
                     __global const datafloat *rhs,
                     __global const datafloat *u,
                     __global  datafloat *newu){

  // Get thread indices
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  if((i < N) && (j < N)){

    // Get padded grid ID
    const int pid = (j + 1)*(N + 2) + (i + 1);

    datafloat invD = 0.25;

    newu[pid] = invD*(rhs[pid]
		      + u[pid - (N+2)]
		      + u[pid + (N+2)]
		      + u[pid - 1]
		      + u[pid + 1]);
  }
}
