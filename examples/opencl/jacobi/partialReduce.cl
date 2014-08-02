
__kernel void partialReduce(const int entries,
			    __global const datafloat *u,
			    __global const datafloat *newu,
			    __global datafloat *blocksum){

  __local datafloat s_blocksum[BDIM];

  const int id = get_global_id(0);

  int alive = get_local_size(0);

  int t = get_local_id(0);

  // load block of vector into shared memory
  s_blocksum[t] = 0;

  if(id < entries){
    const datafloat diff = u[id] - newu[id];
    s_blocksum[t] = diff*diff;
  }

  while(alive>1){

    barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_blocksum is ready)                                                                                                        
    alive /= 2;
    if(t < alive)
      s_blocksum[t] += s_blocksum[t+alive];
  }

  if(t==0)
    blocksum[get_group_id(0)] = s_blocksum[0];
}

