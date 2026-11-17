# Hack: g1 + attn_all's body replaced by src-cuda's attention_kernel (cluster of 8 blocks per kv head)
import sys, re, subprocess, os
D = os.path.dirname(os.path.abspath(__file__))
p = sys.argv[1]
import os
subprocess.check_call(['python3', os.path.join(D, ('g1p_cg.py' if os.environ.get('P1G','1')=='1' else 'g1.py')), p])
s = open(p).read(); body = open(os.path.join(D, ('a1pbody.cu' if os.environ.get('P1A','1')=='1' else 'a1body.cu'))).read()
helper = r'''#include <cooperative_groups.h>
__device__ void cu_warp_norm_rope(const float* src, float* dst, const float* w, float cs, float sn, float eps) {
    const int lane = threadIdx.x & 31;
    float v[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) { v[j] = src[lane + 32 * j]; ss += v[j] * v[j]; }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float r = rsqrtf(ss / 256 + eps);
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] *= r * w[lane + 32 * j];
    const float a = v[0], b = v[1];
    v[0] = a * cs - b * sn;
    v[1] = b * cs + a * sn;
#pragma unroll
    for (int j = 0; j < 8; j++) dst[lane + 32 * j] = v[j];
}
'''
out = []; first = True
for k in re.split(r'(?=extern "C" __global__)', s):
    if 'attn_all(' in k.split('\n')[0]:
        k = k.replace('__launch_bounds__(256) __lfortran_gpu_kernel', '__cluster_dims__(8, 1, 1) __launch_bounds__(256) __lfortran_gpu_kernel', 1)
        a = k.index('\n{\n'); e = k.index('\n}\n', a)
        k = helper + k[:a] + '\n' + body + k[e + 3:]
    out.append(k)
s2 = ''.join(out)
W = 'asm volatile("griddepcontrol.wait;" ::: "memory");'
T = 'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");'
out2 = []
for k in re.split(r'(?=extern "C" __global__)', s2):
    h = k.split('\n')[0]
    if os.environ.get('P1S','1')=='1' and ('embed(' in h or 'argmax_part(' in h) and T not in k:
        k = k.replace(W, T + '\n    ' + W, 1)
    out2.append(k)
open(p, 'w').write(''.join(out2))
