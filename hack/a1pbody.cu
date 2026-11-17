{
    namespace cg = cooperative_groups;
    constexpr int HD = 256, G = 6, ACL = 8, NW = ACL * 8;
    cg::cluster_group cluster = cg::this_cluster();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __shared__ float qs[G * HD], kb[HD], wacc[4 * G * HD], bacc[G * HD], bml[2 * G];
    __shared__ float wm[8][G], wl[8][G];
    const int rank = blockIdx.x % ACL, kh = blockIdx.x / ACL;
    const int d = threadIdx.x, warp = d >> 5, lane = d & 31;
    const int nkv = __scalar_args.__dim_kc_1, nq = nkv * G;
    const int p = __scalar_args.p, L = p + 1;
    const size_t stride = (size_t)nkv * HD;
    const float eps = __scalar_args.eps;
    const int rcd = __scalar_args.__dim_rc_0;
    const int tfirst = rank * 8 + warp;
    float4 k0, k1, v0, v1;
    if (tfirst < p) {
        const float* kp = kc + tfirst * stride + kh * HD + lane * 8;
        const float* vp = vc + tfirst * stride + kh * HD + lane * 8;
        k0 = *(const float4*)kp; k1 = *(const float4*)(kp + 4);
        v0 = *(const float4*)vp; v1 = *(const float4*)(vp + 4);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    const float cs = rc[lane + rcd * p], sn = rs[lane + rcd * p];
    if (warp < G) cu_warp_norm_rope(y + (size_t)(kh * G + warp) * 2 * HD, qs + warp * HD, qw, cs, sn, eps);
    const float* vnew = y + (size_t)nq * 2 * HD + nkv * HD + kh * HD;
    if (warp == G) {
        cu_warp_norm_rope(y + (size_t)nq * 2 * HD + kh * HD, kb, kw, cs, sn, eps);
        if (rank == 0)
            for (int j = lane; j < HD; j += 32) {
                kc[p * stride + kh * HD + j] = kb[j];
                vc[p * stride + kh * HD + j] = vnew[j];
            }
    }
    __syncthreads();
    float qr[G][8], acc[G][8], m[G], l[G];
#pragma unroll
    for (int i = 0; i < G; i++) {
        m[i] = -INFINITY; l[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) { qr[i][j] = qs[i * HD + lane * 8 + j] * 0.0625f; acc[i][j] = 0.f; }
    }
    auto load = [&](int t, float4& k0, float4& k1, float4& v0, float4& v1) {
        const float* kp = t == p ? kb + lane * 8 : kc + t * stride + kh * HD + lane * 8;
        const float* vp = t == p ? vnew + lane * 8 : vc + t * stride + kh * HD + lane * 8;
        k0 = *(const float4*)kp; k1 = *(const float4*)(kp + 4);
        v0 = *(const float4*)vp; v1 = *(const float4*)(vp + 4);
    };
    int t = tfirst;
    if (t == p) load(t, k0, k1, v0, v1);
    for (; t < L; t += NW) {
        const float kk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
        const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        if (t + NW < L) load(t + NW, k0, k1, v0, v1);
        float sc[G];
#pragma unroll
        for (int i = 0; i < G; i++) {
            float dot = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) dot += qr[i][j] * kk[j];
            sc[i] = dot;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int i = 0; i < G; i++) sc[i] += __shfl_xor_sync(0xffffffffu, sc[i], o);
#pragma unroll
        for (int i = 0; i < G; i++) {
            const float mn = fmaxf(m[i], sc[i]);
            const float corr = __expf(m[i] - mn), pr = __expf(sc[i] - mn);
            m[i] = mn;
            l[i] = l[i] * corr + pr;
#pragma unroll
            for (int j = 0; j < 8; j++) acc[i][j] = acc[i][j] * corr + pr * vv[j];
        }
    }
    // Merge the block's 8 warps in two halves (static shared memory): warps 4..7 into warps 0..3.
#pragma unroll
    for (int i = 0; i < G; i++) {
        if (warp >= 4) {
#pragma unroll
            for (int j = 0; j < 8; j++) wacc[((warp - 4) * G + i) * HD + lane * 8 + j] = acc[i][j];
        }
        if (lane == 0) { wm[warp][i] = m[i]; wl[warp][i] = l[i]; }
    }
    __syncthreads();
    if (warp < 4) {
#pragma unroll
        for (int i = 0; i < G; i++) {
            const float m2 = wm[warp + 4][i], l2 = wl[warp + 4][i], M = fmaxf(m[i], m2);
            if (M != -INFINITY) {
                const float e1 = __expf(m[i] - M), e2 = __expf(m2 - M);
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = acc[i][j] * e1 + wacc[(warp * G + i) * HD + lane * 8 + j] * e2;
                l[i] = l[i] * e1 + l2 * e2;
                m[i] = M;
            }
        }
    }
    __syncthreads();
    if (warp < 4) {
#pragma unroll
        for (int i = 0; i < G; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) wacc[(warp * G + i) * HD + lane * 8 + j] = acc[i][j];
            if (lane == 0) { wm[warp][i] = m[i]; wl[warp][i] = l[i]; }
        }
    }
    __syncthreads();
    for (int i = 0; i < G; i++) {
        float M = -INFINITY;
        for (int w = 0; w < 4; w++) M = fmaxf(M, wm[w][i]);
        float num = 0.f, den = 0.f;
        if (M != -INFINITY)
            for (int w = 0; w < 4; w++) {
                const float e = __expf(wm[w][i] - M);
                num += e * wacc[(w * G + i) * HD + d];
                den += e * wl[w][i];
            }
        bacc[i * HD + d] = num;
        if (d == 0) { bml[2 * i] = M; bml[2 * i + 1] = den; }
    }
    cluster.sync();
    if (d < G * 32) {
        const int i = d / 32, dim = rank * 32 + (d % 32);
        float mr[ACL], M = -INFINITY;
#pragma unroll
        for (int r = 0; r < ACL; r++) { mr[r] = cluster.map_shared_rank(bml, r)[2 * i]; M = fmaxf(M, mr[r]); }
        float num = 0.f, den = 0.f;
#pragma unroll
        for (int r = 0; r < ACL; r++) {
            if (mr[r] == -INFINITY) continue;
            const float e = __expf(mr[r] - M);
            num += e * cluster.map_shared_rank(bacc, r)[i * HD + dim];
            den += e * cluster.map_shared_rank(bml, r)[2 * i + 1];
        }
        const int h = kh * G + i;
        const float gz = y[(size_t)h * 2 * HD + HD + dim];
        out[h * HD + dim] = num / den * (1.f / (1.f + __expf(-gz)));
    }
    cluster.sync();
}
