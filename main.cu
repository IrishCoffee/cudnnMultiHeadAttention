#include <cuda_runtime.h>
#include <cstdio>
#define FINAL_MASK 0xffffffff

    __inline__ __device__
float warpReduceSum(float val)
{
    for(int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

/* Calculate the sum of all elements in a block */
    __inline__ __device__
float blockReduceSum(float val)
{
    static __shared__ float shared[32]; 
    int lane = threadIdx.x & 0x1f; 
    int wid = threadIdx.x >> 5;  

    val = warpReduceSum(val);

    if(lane == 0)
        shared[wid] = val;

    __syncthreads();


    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
    val = warpReduceSum(val);

    return val;
}

    __inline__ __device__
float warpReduceMax(float val)
{
    for(int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

/* Calculate the maximum of all elements in a block */
    __inline__ __device__
float blockReduceMax(float val)
{
    static __shared__ float shared[32]; 
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid = threadIdx.x >> 5;  // warp idx

    val = warpReduceMax(val); // get maxx in each warp

    if(lane == 0) // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();


    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
    val = warpReduceMax(val);

    return val;
}


    __global__
void KeMultiHeadAttention(
        const float *q, 
        const float *k, 
        const float *v, 
        const int beam_size, 
        const int n_steps, 
        const int qk_col, 
        const int v_col, 
        const int nhead, 
        const float scale,
        const int THRESHOLD,
        float *dst)
{
    /* 
       Each block processes one head from one candidate.

       _dim_per_head is the size of partition processed by each head.
       
       candidate_id is the index of candidate processed by this block. 
       We have beam_size candidates in total.

       head_id is the index of head processed by this block.

    */
    int _dim_per_head = qk_col / nhead;
    int candidate_id = blockIdx.x / nhead;
    int head_id = blockIdx.x % nhead;

    /*
       sq is the query vector shared by all threads inside the same block.

       The size of sq should be _dim_per_head.
       
       Each block only load the a part of the query vector belongs to the corresponding candidate.
    */
    extern __shared__ float buffer[];
    float *sq = buffer;
    float *logits = (float*)&(buffer[_dim_per_head]);


    /*
       pos is the start position of the corresponding query matrix prococessed by this block.
    */
    int pos = candidate_id * qk_col + head_id * _dim_per_head + threadIdx.x;
    if(threadIdx.x < _dim_per_head)
        sq[threadIdx.x] = q[pos];
    __syncthreads();

    
    /* 
       Step 1 calculate the correlation between the query and key QK^T/sqrt(d_k)
     */

    float summ = 0;
    const float* m2; 
    if(threadIdx.x < n_steps)
    {   
        m2 = k + candidate_id * qk_col * n_steps + threadIdx.x * qk_col + head_id * _dim_per_head;
        for (int i = 0; i < _dim_per_head; i++)
            summ += sq[i] * m2[i];
        summ *= scale;
    }   

    
    /*
       Step 2 Calculate the softmax value of the first step softmax(QK^T/sqrt(d_k)) using warp shuffle.

    */
    __shared__ float s_max_val;
    __shared__ float s_sum;


    float local_i = threadIdx.x < n_steps ? summ : -1e-20;
    float local_o;

    float max_val = blockReduceMax(local_i);

    if(threadIdx.x == 0)
        s_max_val = max_val;
    __syncthreads();

    local_i -= s_max_val;

    if(local_i < -THRESHOLD)
        local_i = -THRESHOLD;

    local_o = exp(local_i);

    float val = (threadIdx.x < n_steps) ? local_o : 0.0f;
    val = blockReduceSum(val);
    if(threadIdx.x == 0)
        s_sum = val;
    __syncthreads();

    if(threadIdx.x < n_steps)
        logits[threadIdx.x] = local_o / s_sum;
    __syncthreads();


    /* 
       Step 3 Calculate the weighted sum on value matrix V softmax(QK^T/sqrt(d_k))V 
    */
    summ = 0;
    if(threadIdx.x < _dim_per_head)
    {
        int tid = candidate_id * v_col * n_steps + head_id * _dim_per_head + threadIdx.x;
        for(int i = 0; i < n_steps; ++i)
            summ += logits[i] * v[tid + (i << 10)];
        dst[candidate_id * v_col + head_id * _dim_per_head + threadIdx.x] = summ;
    }
}
/*
    Q: Query matrix of dimension beam_size * qk_col
    K: Key matrix of dimension beam_size * (n_steps * qk_col)
    V: Value matrix of dimension beam_size * (n_steps * v_col)
    beamsize: Dimension used in beam size, also called the number of candidates
    n_steps: The number of words that have already been decoded
    qk_col: Dimension of the query/value feature
    v_col: Dimension of the value feature
    nhead: The number of heads
    scaler: Pre-computed scaler 
    THRESHOLD: Customer-defined value for soft-max maximum value calculation
    dst: Output. The attention value of this query over all decoded word keys
*/
cudaError_t multiHeadAttention(
        const float *Q, 
        const float *K, 
        const float *V, 
        const int beamsize,
        const int n_steps, 
        const int qk_col, 
        const int v_col, 
        const int nhead, 
        const float scaler,
        const int THRESHOLD,
        float *dst)
{
    /* 
       We have beamsize candidates.
       
       We use one block to process one head of a candidate.

       The block dimension equals to the dimension of the subhead of each candidate.

    */

    dim3 grid(nhead * beamsize, 1); 
    dim3 block(qk_col / nhead, 1); 

    int shared_size = sizeof(float) * ((qk_col / nhead) + n_steps);
    
    KeMultiHeadAttention<<<grid, block, shared_size, 0 >>> (Q, K, V,
            beamsize, n_steps, qk_col, v_col, nhead, scaler, THRESHOLD, dst);

    return cudaGetLastError();

}
int main()
{
    /*
       Each step, we maintain beamsize candidates for the beam search.

       We have nhead heads, which contains dim_feature/nhead values.

       Currently, we have already decoded n_steps words.

    */
    const int beamsize = 4;
    const int nhead = 16;
    const int dim_feature = 1024;
    const int n_steps = 9;
    
    /* 
       Calculate sqrt(d_k) 
    */
    float scaler = sqrt(nhead * 1.0 / dim_feature);

    //qk_col can be different with v_col.
    int qk_col = dim_feature;
    int v_col = dim_feature;
    int THRESHOLD = 64;

    float *dq, *dk, *dv, *dst;
    cudaMalloc((void**)&dq, sizeof(float) * beamsize * dim_feature);
    cudaMalloc((void**)&dk, sizeof(float) * beamsize * dim_feature * n_steps);
    cudaMalloc((void**)&dv, sizeof(float) * beamsize * dim_feature * n_steps);
    cudaMalloc((void**)&dst, sizeof(float) * beamsize * dim_feature);

    float *hq, *hk, *hv;
    hq = (float*)malloc(sizeof(float) * beamsize * dim_feature);
    hk = (float*)malloc(sizeof(float) * beamsize * n_steps * dim_feature);
    hv = (float*)malloc(sizeof(float) * beamsize * n_steps * dim_feature);

    /* 
       Load the query, key and value matrices used in this step from text.
    */
    FILE *fd = fopen("data/query.txt", "r");
    for(int i = 0; i < beamsize * dim_feature; ++i)
        fscanf(fd, "%f", &hq[i]);

    fd = fopen("data/key.txt", "r");
    for(int i = 0; i < beamsize * n_steps * dim_feature; ++i)
        fscanf(fd, "%f", &hk[i]);

    fd = fopen("data/value.txt", "r");
    for(int i = 0; i < beamsize * n_steps * dim_feature; ++i)
        fscanf(fd, "%f", &hv[i]);

    cudaMemcpy(dq, hq, sizeof(float) * beamsize * dim_feature, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk, sizeof(float) * beamsize * n_steps * dim_feature, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv, sizeof(float) * beamsize * n_steps * dim_feature, cudaMemcpyHostToDevice);

    cudaError_t error;
    error = multiHeadAttention(dq, dk, dv, beamsize, n_steps, qk_col, v_col, nhead, scaler, THRESHOLD, dst);

    float *h_dst = (float*)malloc(sizeof(float) * beamsize * dim_feature);
    cudaMemcpy(h_dst, dst, sizeof(float) * beamsize * dim_feature, cudaMemcpyDeviceToHost);
    for(int i = 0; i < beamsize * dim_feature; ++i)
        printf("%f\n", h_dst[i]);

    free(hq);
    free(hk);
    free(hv);
    free(h_dst);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dst);
}


