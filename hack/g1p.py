# Hack: t1 (atomic norm sums + CUDA gemv bodies) + gdn_all's body replaced by src-cuda's gdn_kernel (256 threads)
import sys, re, subprocess, os
D = os.path.dirname(os.path.abspath(__file__))
p = sys.argv[1]
subprocess.check_call(['python3', os.path.join(D, 't1.py'), p])
s = open(p).read(); body = open(os.path.join(D, 'g1pbody.cu')).read()
out = []
for k in re.split(r'(?=extern "C" __global__)', s):
    if 'gdn_all(' in k.split('\n')[0]:
        a = k.index('\n{\n'); e = k.index('\n}\n', a)
        k = k[:a] + '\n' + body + k[e + 3:]
    out.append(k)
open(p, 'w').write(''.join(out))
