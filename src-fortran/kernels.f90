! GPU kernels for Qwen3.8-27B (MLX 8-bit) single-token decode: the kernels of src-cuda/kernels.cu as
! do concurrent loops, which LFortran offloads to the GPU (--gpu=cuda). fp32 activations; the weights
! stay in the checkpoint's q8 + bf16 scale/bias format.
!
! A layer is 4 GEMVs plus a Gated DeltaNet or an attention: RMSNorm, the residual adds and SwiGLU are
! fused into the GEMVs. The residual stream and its RMSNorm (as `resid` in model.h):
!   x    residual (H)
!   xw   x * (weight of the RMSNorm that comes next), the input of the next GEMV
!   ssp  sum of x**2 in parts: ssp(i) = x(4i-3)**2 + ... + x(4i)**2
! The GEMV that updates x (gemv_resid, embed) writes all three. The GEMV that consumes the norm
! (gemv_norm) adds up ssp alongside its dot products and multiplies its rows by rsqrt(sum/H + eps),
! which gives W (rmsnorm(x) * weight). (CUDA accumulates the sum with atomics instead of parts.)
!
! In a do concurrent over 2 or 3 indices the last one varies fastest between GPU threads. An inner
! sequential loop that only accumulates into scalars is shared out over the 32 lanes of a warp.
module kernels
    use, intrinsic :: iso_fortran_env, only: int16, int32
    implicit none
    private
    public :: qmat, bf, embed, gemv_norm, gemv_resid, gdn_conv, gdn_step, gdn_norm, gdn_all, attn_all, argmax_final
    public :: attn_qkv, attn_scores, attn_out, argmax_part

    ! One MLX affine 8-bit matrix (group size 64) in the checkpoint's own layout: w(:, n) holds the K
    ! bytes of row n (the U32 tensor [N, K/4]), s(:, n) and b(:, n) its K/64 bf16 scales and biases.
    ! Element (k, n) = byte * scale + bias.
    type :: qmat
        integer(int32), allocatable :: w(:,:)
        integer(int16), allocatable :: s(:,:), b(:,:)
    end type

    integer, parameter :: GD = 128   ! linear attention key/value head dim
    integer, parameter :: HD = 256   ! full attention head dim

contains

    ! bf16 bits -> real
    pure elemental real function bf(v)
        integer(int16), intent(in) :: v
        bf = transfer(shiftl(int(v, int32), 16), 0.0)
    end function

    pure real function sigmoid(x)
        real, intent(in) :: x
        sigmoid = 1 / (1 + exp(-x))
    end function

    ! acc + the 4 bytes of w dotted with x0..x3, as a chain of fused multiply-adds. A byte as a real:
    ! 2**23 + byte has it as its low mantissa bits.
    pure real function dot4(acc, w, x0, x1, x2, x3)
        real, intent(in) :: acc, x0, x1, x2, x3
        integer(int32), intent(in) :: w
        dot4 = acc + (transfer(ior(iand(w, 255), 1258291200), 0.0) - 8388608.0) * x0 &
             + (transfer(ior(iand(shiftr(w, 8), 255), 1258291200), 0.0) - 8388608.0) * x1 &
             + (transfer(ior(iand(shiftr(w, 16), 255), 1258291200), 0.0) - 8388608.0) * x2 &
             + (transfer(ior(shiftr(w, 24), 1258291200), 0.0) - 8388608.0) * x3
    end function

    ! ------------------------------------------------------------ embedding

    ! Element (k, n) of the q8 matrix (w, s, b)
    pure real function q8(w, s, b, k, n)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        integer, intent(in) :: k, n
        q8 = ibits(w((k-1)/4+1, n), 8*modulo(k-1, 4), 8) * bf(s((k-1)/64+1, n)) + bf(b((k-1)/64+1, n))
    end function

    ! x = E(:, tok); xw = x * wn; ssp
    subroutine embed(w, s, b, tokbuf, wn, k, x, xw, ssp)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        integer, intent(in) :: tokbuf(:), k
        real, intent(in) :: wn(:)
        real, intent(inout) :: x(:), xw(:), ssp(:)
        do concurrent (integer :: i = 1:size(x)/4)
            x(4*i-3) = q8(w, s, b, 4*i-3, tokbuf(1)+1)
            x(4*i-2) = q8(w, s, b, 4*i-2, tokbuf(1)+1)
            x(4*i-1) = q8(w, s, b, 4*i-1, tokbuf(1)+1)
            x(4*i) = q8(w, s, b, 4*i, tokbuf(1)+1)
            xw(4*i-3) = x(4*i-3) * wn(4*i-3)
            xw(4*i-2) = x(4*i-2) * wn(4*i-2)
            xw(4*i-1) = x(4*i-1) * wn(4*i-1)
            xw(4*i) = x(4*i) * wn(4*i)
            ssp(i+k-1) = (x(4*i-3)*x(4*i-3) + x(4*i-2)*x(4*i-2)) + (x(4*i-1)*x(4*i-1) + x(4*i)*x(4*i))
        end do
    end subroutine

    ! ------------------------------------------------------------ q8 GEMV
    ! An iteration computes 4 consecutive rows; its lanes (a warp, or two for a matrix with few rows)
    ! take its chunks of 16 bytes, c = lane, lane + 32, ..., and combine their sums at the end.

    ! Chunk c (16 bytes) of row n dotted with x: scale * sum(byte * x) + bias * sum(x), with one scale
    ! and bias per 4 chunks.
    pure real function qdot(w, s, b, x, n, c)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        real, intent(in) :: x(:)
        integer, intent(in) :: n, c
        qdot = bf(s((c+3)/4,n)) * dot4(dot4(dot4(dot4(0.0, w(4*c-3,n), x(16*c-15), x(16*c-14), x(16*c-13), x(16*c-12)), &
            w(4*c-2,n), x(16*c-11), x(16*c-10), x(16*c-9), x(16*c-8)), w(4*c-1,n), x(16*c-7), x(16*c-6), x(16*c-5), x(16*c-4)), &
            w(4*c,n), x(16*c-3), x(16*c-2), x(16*c-1), x(16*c)) &
            + bf(b((c+3)/4,n)) * ((((x(16*c-15) + x(16*c-14)) + (x(16*c-13) + x(16*c-12))) &
            + ((x(16*c-11) + x(16*c-10)) + (x(16*c-9) + x(16*c-8)))) + (((x(16*c-7) + x(16*c-6)) &
            + (x(16*c-5) + x(16*c-4))) + ((x(16*c-3) + x(16*c-2)) + (x(16*c-1) + x(16*c)))))
    end function

    ! y = rsqrt(sum(ssp)/H + eps) * (W xw). swiglu: the rows are interleaved gate/up pairs
    ! (g1 g2 u1 u2 g3 g4 u3 u4 ...) and y(i) = silu(gate_i) * up_i, for N/2 outputs.
    subroutine gemv_norm(w, s, b, xw, ssp, k, eps, swiglu, y)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        real, intent(in) :: xw(:), ssp(:), eps
        logical, intent(in) :: swiglu
        integer, intent(in) :: k
        real, intent(inout) :: y(:)
        do concurrent (integer :: i = 1:size(w, 2)/4)
            block
                real :: a1, a2, a3, a4, ss, r
                integer :: c, n
                n = 4*i - 3
                a1 = 0; a2 = 0; a3 = 0; a4 = 0
                ss = ssp(k)
                do c = 1, size(w, 1)/4
                    a1 = a1 + qdot(w, s, b, xw, n, c)
                    a2 = a2 + qdot(w, s, b, xw, n+1, c)
                    a3 = a3 + qdot(w, s, b, xw, n+2, c)
                    a4 = a4 + qdot(w, s, b, xw, n+3, c)
                end do
                r = 1 / sqrt(ss / size(xw) + eps)
                if (swiglu) then
                    y(2*i-1) = a1 * r * sigmoid(a1 * r) * (a3 * r)
                    y(2*i) = a2 * r * sigmoid(a2 * r) * (a4 * r)
                else
                    y(n) = a1 * r
                    y(n+1) = a2 * r
                    y(n+2) = a3 * r
                    y(n+3) = a4 * r
                end if
            end block
        end do
    end subroutine

    ! x += W v; then xw = x * wnext and ssp, for the next RMSNorm (weight wnext)
    subroutine gemv_resid(w, s, b, v, wnext, k, x, xw, ssp)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        real, intent(in) :: v(:), wnext(:)
        integer, intent(in) :: k
        real, intent(inout) :: x(:), xw(:), ssp(:)
        do concurrent (integer :: i = 1:size(w, 2)/4)
            block
                real :: a1, a2, a3, a4
                integer :: c, n
                n = 4*i - 3
                a1 = 0; a2 = 0; a3 = 0; a4 = 0
                do c = 1, size(w, 1)/4
                    a1 = a1 + qdot(w, s, b, v, n, c)
                    a2 = a2 + qdot(w, s, b, v, n+1, c)
                    a3 = a3 + qdot(w, s, b, v, n+2, c)
                    a4 = a4 + qdot(w, s, b, v, n+3, c)
                end do
                x(n) = x(n) + a1
                x(n+1) = x(n+1) + a2
                x(n+2) = x(n+2) + a3
                x(n+3) = x(n+3) + a4
                xw(n) = x(n) * wnext(n)
                xw(n+1) = x(n+1) * wnext(n+1)
                xw(n+2) = x(n+2) * wnext(n+2)
                xw(n+3) = x(n+3) * wnext(n+3)
                ssp(i+k-1) = (x(n)*x(n) + x(n+1)*x(n+1)) + (x(n+2)*x(n+2) + x(n+3)*x(n+3))
            end block
        end do
    end subroutine

    ! ------------------------------------------------------------ linear attention (Gated DeltaNet)
    ! y = in_proj output [qkv (C) | z (nv*128) | b (nv) | a (nv)], C = 2*nk*128 + nv*128. The state
    ! S(k, v, h) of value head h is updated as S = g*S + khat * delta^T, delta = (v - g*S^T khat)*beta,
    ! and the output o = S^T qhat comes from the old state: o = g*(S^T qhat) + delta*(khat . qhat).

    ! Causal conv (4 taps; cs: the last 3 inputs) + SiLU of the qkv channels.
    subroutine gdn_conv(y, cw, cs, qkv)
        real, intent(in) :: y(:), cw(:,:)
        real, intent(inout) :: cs(:,:), qkv(:)
        do concurrent (integer :: c = 1:size(qkv))
            block
                real :: v
                v = cw(1,c)*cs(c,1) + cw(2,c)*cs(c,2) + cw(3,c)*cs(c,3) + cw(4,c)*y(c)
                cs(c,1) = cs(c,2)
                cs(c,2) = cs(c,3)
                cs(c,3) = y(c)
                qkv(c) = v * sigmoid(v)
            end block
        end do
    end subroutine

    ! Column (v, h): the sums over the old state, the gates, delta and the output; then the column
    ! of the new state, S = g*S + khat * delta^T.
    subroutine gdn_step(qkv, y, a_log, dt_bias, nk, st, o)
        real, intent(in) :: qkv(:), y(:), a_log(:), dt_bias(:)
        integer, intent(in) :: nk
        real, intent(inout) :: st(:,:,:), o(:,:)
        integer :: nv, c
        nv = size(st, 3)
        c = size(qkv)
        do concurrent (integer :: h = 1:nv, v = 1:GD)
            block
                real :: sk, sq, kk, qq, kq, kn, qn, g, beta, delta, sp, dk
                integer :: j, kq0, kk0
                kq0 = ((h-1) / (nv/nk)) * GD
                kk0 = nk*GD + kq0
                sk = 0; sq = 0; kk = 0; qq = 0; kq = 0
                do j = 1, GD
                    sk = sk + st(j,v,h) * qkv(kk0+j)
                    sq = sq + st(j,v,h) * qkv(kq0+j)
                    kk = kk + qkv(kk0+j) * qkv(kk0+j)
                    qq = qq + qkv(kq0+j) * qkv(kq0+j)
                    kq = kq + qkv(kk0+j) * qkv(kq0+j)
                end do
                kn = 1 / sqrt(kk + 1e-6)
                qn = 1 / sqrt(qq + 1e-6) / sqrt(real(GD))
                sp = y(c + nv*GD + nv + h) + dt_bias(h)
                if (sp <= 20) sp = log(1 + exp(sp))
                g = exp(-exp(a_log(h)) * sp)
                beta = sigmoid(y(c + nv*GD + h))
                delta = (qkv(2*nk*GD + (h-1)*GD + v) - g * sk * kn) * beta
                o(v,h) = (g * sq + delta * kn * kq) * qn
                dk = delta * kn
                do j = 1, GD
                    st(j,v,h) = g * st(j,v,h) + qkv(kk0+j) * dk
                end do
            end block
        end do
    end subroutine

    ! Gated RMSNorm: out = rmsnorm(o) * w * silu(z), per value head; z = y(z0+1:).
    subroutine gdn_norm(o, y, z0, nw, eps, out)
        real, intent(in) :: o(:,:), y(:), nw(:), eps
        integer, intent(in) :: z0
        real, intent(inout) :: out(:)
        do concurrent (integer :: h = 1:size(o, 2))
            block
                real :: ss, r, z
                integer :: j, v
                ss = 0
                do j = 1, GD
                    ss = ss + o(j,h) * o(j,h)
                end do
                r = 1 / sqrt(ss / GD + eps)
                do v = 1, GD
                    z = y(z0 + (h-1)*GD + v)
                    out((h-1)*GD + v) = o(v,h) * r * nw(v) * z * sigmoid(z)
                end do
            end block
        end do
    end subroutine

    ! HACK placeholder: one block of 256 threads per value head; the generated body is replaced by
    ! src-cuda's gdn_kernel (adapted to 256 threads). cs: conv inputs, double buffered by position parity.
    subroutine gdn_all(y, cw, a_log, dt_bias, nw, nk, pos, eps, cs, st, out)
        real, intent(in) :: y(:), cw(:,:), a_log(:), dt_bias(:), nw(:), eps
        integer, intent(in) :: nk, pos
        real, intent(inout) :: cs(:,:), st(:,:,:), out(:)
        do concurrent (integer :: i = 1:size(st, 3)*256)
            if (i < 0) then
                out(1) = y(1) + cw(1,1) + a_log(1) + dt_bias(1) + nw(1) + eps + nk + pos + cs(1,1) + st(1,1,1)
                cs(1,1) = 0
                st(1,1,1) = 0
            end if
        end do
    end subroutine

    ! HACK placeholder: 8 blocks of 256 threads per kv head; the generated body is replaced by src-cuda's
    ! attention_kernel. kc and vc: kc(d, kh, t), vc(d, kh, t).
    subroutine attn_all(y, qw, kw, rc, rs, eps, p, kc, vc, out)
        real, intent(in) :: y(:), qw(:), kw(:), rc(:,:), rs(:,:), eps
        integer, intent(in) :: p
        real, intent(inout) :: kc(:,:,:), vc(:,:,:), out(:)
        do concurrent (integer :: i = 1:size(kc, 2)*8*256)
            if (i < 0) then
                out(1) = y(1) + qw(1) + kw(1) + rc(1,1) + rs(1,1) + eps + p + kc(1,1,1) + vc(1,1,1)
                kc(1,1,1) = 0
                vc(1,1,1) = 0
            end if
        end do
    end subroutine

    ! ------------------------------------------------------------ full attention
    ! y = [q|gate per head (nq*512) | k (nkv*256) | v (nkv*256)]. q and k get an RMSNorm and a
    ! rotate-half RoPE on their first 64 dims (cos/sin of the 32 angles of position t: rc(:, t+1), rs(:, t+1)).

    ! The q heads and the k head (RMSNorm, RoPE) of position p: q into qr (scaled by 1/sqrt(256)), k and
    ! v appended to the caches. Head hh = 1..nq is a q head, nq+1..nq+nkv a k head.
    subroutine attn_qkv(y, qw, kw, rc, rs, eps, p, qr, kc, vc)
        real, intent(in) :: y(:), qw(:), kw(:), rc(:,:), rs(:,:), eps
        integer, intent(in) :: p
        real, intent(inout) :: qr(:,:), kc(:,:,:), vc(:,:,:)
        integer :: nq, nkv
        nq = size(qr, 2)
        nkv = size(kc, 2)
        do concurrent (integer :: hh = 1:nq+nkv, d = 1:HD)
            block
                real :: ss, cs, sn, w, wp, r
                integer :: j, b, dp
                if (hh <= nq) then
                    b = (hh-1)*2*HD
                else
                    b = nq*2*HD + (hh-nq-1)*HD
                end if
                ss = 0
                do j = 1, HD
                    ss = ss + y(b+j) * y(b+j)
                end do
                if (d <= 32) then   ! rotate-half RoPE on dims 1..64: pairs (d, d+32)
                    dp = d + 32; cs = rc(d,p+1); sn = -rs(d,p+1)
                else if (d <= 64) then
                    dp = d - 32; cs = rc(d-32,p+1); sn = rs(d-32,p+1)
                else
                    dp = d; cs = 1; sn = 0
                end if
                if (hh <= nq) then
                    w = qw(d); wp = qw(dp)
                else
                    w = kw(d); wp = kw(dp)
                end if
                r = (y(b+d) * w * cs + y(b+dp) * wp * sn) / sqrt(ss / HD + eps)
                if (hh <= nq) then
                    qr(d,hh) = r / 16
                else
                    kc(d,hh-nq,p+1) = r
                    vc(p+1,d,hh-nq) = y(nq*2*HD + nkv*HD + (hh-nq-1)*HD + d)
                end if
            end block
        end do
    end subroutine

    ! sc(t, h) = q_h . k_t for the positions t = 1..n
    subroutine attn_scores(qr, kc, n, sc)
        real, intent(in) :: qr(:,:), kc(:,:,:)
        integer, intent(in) :: n
        real, intent(inout) :: sc(:,:)
        integer :: nq, nkv
        nq = size(qr, 2)
        nkv = size(kc, 2)
        do concurrent (integer :: h = 1:nq, t = 1:n)
            block
                real :: dot
                integer :: d, kh
                kh = (h-1) / (nq/nkv) + 1
                dot = 0
                do d = 1, HD
                    dot = dot + qr(d,h) * kc(d,kh,t)
                end do
                sc(t,h) = dot
            end block
        end do
    end subroutine

    ! out = softmax(sc) V * sigmoid(gate). The V cache is vc(t, d, kh), so that the sum over the
    ! positions t reads along its first index and the lanes share it.
    subroutine attn_out(y, sc, vc, n, out)
        real, intent(in) :: y(:), sc(:,:), vc(:,:,:)
        integer, intent(in) :: n
        real, intent(inout) :: out(:)
        integer :: nq, nkv
        nq = size(sc, 2)
        nkv = size(vc, 3)
        do concurrent (integer :: h = 1:nq, d = 1:HD)
            block
                real :: m, l, acc, e
                integer :: t, kh
                kh = (h-1) / (nq/nkv) + 1
                m = -huge(m)
                do t = 1, n
                    m = max(m, sc(t,h))
                end do
                l = 0
                acc = 0
                do t = 1, n
                    e = exp(sc(t,h) - m)
                    l = l + e
                    acc = acc + e * vc(t,d,kh)
                end do
                out((h-1)*HD + d) = acc / l * sigmoid(y((h-1)*2*HD + HD + d))
            end block
        end do
    end subroutine

    ! ------------------------------------------------------------ argmax
    ! Part i of np: the largest of its x(j) and the first j it is at.
    subroutine argmax_part(x, np, pv, pi)
        real, intent(in) :: x(:)
        integer, intent(in) :: np
        real, intent(inout) :: pv(:)
        integer, intent(inout) :: pi(:)
        integer :: len
        len = (size(x) + np - 1) / np
        do concurrent (integer :: i = 1:np)
            block
                real :: m
                integer :: j, bi
                m = -huge(m)
                do j = (i-1)*len + 1, min(i*len, size(x))
                    m = max(m, x(j))
                end do
                bi = huge(bi)
                do j = (i-1)*len + 1, min(i*len, size(x))
                    if (x(j) == m) bi = min(bi, j)
                end do
                pv(i) = m
                pi(i) = bi
            end block
        end do
    end subroutine

    ! HACK: the token of the largest of the parts (pv, pi) into tokbuf(1) (for the next embed) and hist(pos+1)
    subroutine argmax_final(pv, pi, np, pos, tokbuf, hist)
        real, intent(in) :: pv(:)
        integer, intent(in) :: pi(:), np, pos
        integer, intent(inout) :: tokbuf(:), hist(:)
        do concurrent (integer :: i = 1:1)
            block
                real :: m
                integer :: j, bi
                m = -huge(m)
                do j = 1, np
                    m = max(m, pv(j))
                end do
                bi = huge(bi)
                do j = 1, np
                    if (pv(j) == m) bi = min(bi, pi(j))
                end do
                tokbuf(i) = bi - 1
                hist(pos+i) = bi - 1
            end block
        end do
    end subroutine

end module kernels
