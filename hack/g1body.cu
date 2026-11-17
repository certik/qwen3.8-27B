{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    // src-cuda's gdn_kernel with 256 threads per value head: thread (part, w, lane) = (tid / 64, tid / 32 % 2,
    // tid % 32) holds state[k = 64w .. 64w+63][v = 32 part + lane].
    const int GD = 128;
    const int nv = __scalar_args.__dim_st_2, nk = __scalar_args.nk, rep = nv / nk;
    const int hv = blockIdx.x, hk = hv / rep;
    const int tid = threadIdx.x, part = tid >> 6, w = (tid >> 5) & 1, lane = tid & 31, dv = part * 32 + lane;
    const int C = __scalar_args.__dim_cw_1, p = __scalar_args.pos;
    const float eps = __scalar_args.eps;
    float* Sp = st + (size_t)hv * GD * GD + (size_t)(w * 64) * GD + dv;
    float sreg[64];
#pragma unroll
    for (int j = 0; j < 64; j++) sreg[j] = Sp[j * GD];
    const float* st_in = cs + (size_t)(p & 1) * 3 * C;
    float* st_out = cs + (size_t)((p + 1) & 1) * 3 * C;
    __shared__ float qs[128], ks[128], vs[128], red[4][2][32], sums[8];
    const float4* cw4 = reinterpret_cast<const float4*>(cw);
    const float alog = a_log[hv], dtb = dt_bias[hv], nwv = nw[dv];
    const float* ab = y + C + nv * GD;
    const float zz = y[C + hv * GD + dv], bb = ab[hv], aa = ab[nv + hv];
    {   // q (tid < 128) and k (tid >= 128) channels of key head hk
        const int c = tid < GD ? hk * GD + tid : nk * GD + hk * GD + (tid - GD);
        const float4 wc = cw4[c];
        const float s0 = st_in[c], s1 = st_in[C + c], s2 = st_in[2 * C + c], xin = y[c];
        float v = wc.x * s0 + wc.y * s1 + wc.z * s2 + wc.w * xin;
        v = v * (1.f / (1.f + __expf(-v)));
        if (hv % rep == 0) { st_out[c] = s1; st_out[C + c] = s2; st_out[2 * C + c] = xin; }
        float sq = v * v;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) sums[tid >> 5] = sq;
        if (tid < GD) qs[tid] = v; else ks[tid - GD] = v;
    }
    if (tid < GD) {   // v channel tid of value head hv
        const int c = 2 * nk * GD + hv * GD + tid;
        const float4 wc = cw4[c];
        const float s0 = st_in[c], s1 = st_in[C + c], s2 = st_in[2 * C + c], xin = y[c];
        float v = wc.x * s0 + wc.y * s1 + wc.z * s2 + wc.w * xin;
        v = v * (1.f / (1.f + __expf(-v)));
        st_out[c] = s1; st_out[C + c] = s2; st_out[2 * C + c] = xin;
        vs[tid] = v;
    }
    const float x = aa + dtb;
    const float softplus = x > 20.f ? x : log1pf(expf(x));
    const float g = expf(-expf(alog) * softplus);
    const float beta = 1.f / (1.f + expf(-bb));
    __syncthreads();
    const float qn = rsqrtf((sums[0] + sums[1]) + (sums[2] + sums[3]) + 1e-6f) * rsqrtf((float)GD);
    const float kn = rsqrtf((sums[4] + sums[5]) + (sums[6] + sums[7]) + 1e-6f);
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 64; j++) a += sreg[j] * ks[w * 64 + j];
    red[part][w][lane] = a;
    __syncthreads();
    const float kvm = (red[part][0][lane] + red[part][1][lane]) * kn;
    const float delta = (vs[dv] - g * kvm) * beta;
    const float dk = delta * kn;
    float o = 0.f;
#pragma unroll
    for (int j = 0; j < 64; j++) {
        const float sn = sreg[j] * g + ks[w * 64 + j] * dk;
        Sp[j * GD] = sn;
        o += sn * qs[w * 64 + j];
    }
    __syncthreads();
    red[part][w][lane] = o;
    __syncthreads();
    if (w == 0) {
        o = (red[part][0][lane] + red[part][1][lane]) * qn;
        float s2 = o * o;
        for (int off = 16; off > 0; off >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, off);
        if (lane == 0) sums[part] = s2;
    }
    __syncthreads();
    if (w == 0) {
        const float tot = (sums[0] + sums[1]) + (sums[2] + sums[3]);
        out[hv * GD + dv] = o * rsqrtf(tot / GD + eps) * nwv * (zz * (1.f / (1.f + __expf(-zz))));
    }
}
