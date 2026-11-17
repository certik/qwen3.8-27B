# Hack: the stores of the norm partial ssp(i+k-1) become atomic adds into slot k; block 0 zeroes the other slot.
import sys, re, os
D = os.path.dirname(os.path.abspath(__file__))
p = sys.argv[1]; s = open(p).read()
pat = re.compile(r'ssp\[\(\(int\)\(\(\(i__dc \+ k\) - 1\)\) - \(__lb_ssp_0\)\)\] = (.*?);\n', re.S)
s, n = pat.subn(lambda m: '{ unsigned long long* sl = (unsigned long long*)ssp; if (blockIdx.x == 0 && threadIdx.x == 0) sl[2 - k] = 0; '
                'atomicAdd(&sl[k - 1], (unsigned long long)((' + m.group(1) + ') * 16777216.f + 0.5f)); }\n', s)
sys.stderr.write('h1: %d stores\n' % n)
open(p, 'w').write(s)
