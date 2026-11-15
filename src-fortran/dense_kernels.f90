module dense_kernels
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32
    use, intrinsic :: iso_c_binding, only: c_int
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_negative_inf, ieee_is_finite
    use precision_formats, only: bf16_bits_to_real32, real32_to_bf16_bits, packed_q8_byte
    implicit none
    private

    interface
        ! The CUDA integer dot product of four signed bytes with four signed bytes,
        ! plus an integer.
        pure function dp4a(a, b, c) bind(c, name="__dp4a") result(r)
            import :: c_int
            integer(c_int), value, intent(in) :: a, b, c
            integer(c_int) :: r
        end function dp4a
    end interface
    public :: embed_dequant_gather_q8, linear_q8_bf16
    public :: rmsnorm_bf16, rmsnorm_scale_bf16, argmax_bf16
    public :: silu_mul_bf16, silu_mul_f32_gate_bf16, elemwise_sigmoid_mul_bf16
    public :: residual_add_bf16, silu_inplace_bf16, sigmoid_inplace_bf16
    public :: compute_g_f32, conv1d_depthwise_causal_bf16, gated_delta_step_bf16, keep_finite_rows_f32
    public :: conv1d_silu_split_bf16, conv1d_silu_split_norm_bf16, rmsnorm_scale_halves_bf16, gated_delta_fused_bf16, sdpa_decode_bf16
    public :: gated_delta_lazy_bf16, gated_delta_flush_bf16, gated_rmsnorm_q_delta_bf16
    public :: gated_rmsnorm_bf16, residual_rmsnorm_bf16, silu_mul2_bf16
    public :: residual_rmsnorm_q_bf16, linear_q8_q_bf16
    public :: silu_mul2_q_bf16, sigmoid_mul_q_bf16, gated_rmsnorm_q_bf16
    public :: qknorm_rope_bf16, rope_partial_inplace_bf16, sdpa_gqa_bf16, knorm_rope_append_bf16, qnorm_rope_gate_bf16, rope_tables_f32
    public :: native_qmv_eligible, native_matrix_parts, native_attention_blocks
    public :: sample_native_bf16

contains

    pure subroutine require_size(count)
        integer(int64), intent(in) :: count

        if (count < 1_int64 .or. count > int(huge(0), int64)) &
            error stop "dense_kernels requires nonempty arrays with default-integer-sized element counts"
    end subroutine require_size

    pure subroutine require_q8_shape(w, scales, offsets, k)
        integer(int32), intent(in) :: w(:, :)
        integer(int16), intent(in) :: scales(:, :), offsets(:, :)
        integer, intent(in) :: k

        call require_size(size(w, kind=int64))
        call require_size(size(scales, kind=int64))
        call require_size(size(offsets, kind=int64))
        if (k < 64 .or. modulo(k, 64) /= 0) error stop "q8 requires a positive multiple of 64 input features"
        if (size(w, 1) /= k / 4 .or. size(scales, 1) /= k / 64 .or. &
                size(scales, 2) /= size(w, 2) .or. any(shape(offsets) /= shape(scales))) &
            error stop "q8 packed weight, scale or offset shape mismatch"
    end subroutine require_q8_shape

    pure elemental function round_bf16(value) result(rounded)
        real(real32), intent(in) :: value
        real(real32) :: rounded

        rounded = bf16_bits_to_real32(real32_to_bf16_bits(value))
    end function round_bf16

    pure elemental function sigmoid_bf16_value(x) result(y)
        real(real32), intent(in) :: x
        real(real32) :: y, e

        e = round_bf16(exp(abs(x)))
        y = round_bf16(1.0_real32 / round_bf16(1.0_real32 + e))
        if (x >= 0.0_real32) y = round_bf16(1.0_real32 - y)
    end function sigmoid_bf16_value

    pure function sum_squares_f32(x) result(total)
        integer(int16), intent(in) :: x(:)
        real(real32) :: total

        total = sum(bf16_bits_to_real32(x)**2)
    end function sum_squares_f32

    pure function sum_strided_bf16(partials) result(total)
        integer(int16), intent(in) :: partials(:)
        real(real32) :: total
        integer(int16) :: work(size(partials)), next(size(partials))
        integer :: active, pairs

        ! The native strided reducer has BF16 typed accumulators. Its device
        ! association is not a portable semantic requirement. Use a typed,
        ! balanced reduction rather than encoding SIMD lanes or a serial fold.
        work=partials
        active=size(partials)
        do while (active > 1)
            pairs=active/2
            do concurrent (integer :: p=1:pairs)
                next(p)=real32_to_bf16_bits(bf16_bits_to_real32(work(2*p-1))+bf16_bits_to_real32(work(2*p)))
            end do
            if (modulo(active,2) /= 0) next(pairs+1)=work(active)
            active=(active+1)/2
            work(:active)=next(:active)
        end do
        total=bf16_bits_to_real32(work(1))
    end function sum_strided_bf16

    pure logical function native_qmv_eligible(rows, width, outputs) result(eligible)
        integer, intent(in) :: rows, width, outputs
        integer :: limit

        limit = 10
        if (width <= 4096 .and. outputs <= 4096) limit = 12
        if (width <= 2048 .and. outputs <= 2048) limit = 18
        eligible = rows > 0 .and. rows < limit .and. modulo(width,512) == 0 .and. modulo(outputs,8) == 0
    end function native_qmv_eligible

    pure integer function native_matrix_parts(rows, width, outputs) result(parts)
        integer, intent(in) :: rows, width, outputs
        integer(int64) :: tiles

        if (rows < 1 .or. outputs < 1 .or. width < 64 .or. modulo(width,64) /= 0) &
            error stop "invalid native matrix shape"
        tiles = ((int(rows,int64)+31)/32)*((int(outputs,int64)+31)/32)
        parts = 1
        if (tiles < 512) parts = int(512/tiles)
        parts = min(parts,width/64)
        do while (modulo(width/64,parts) /= 0)
            parts = parts-1
        end do
    end function native_matrix_parts

    pure subroutine embed_dequant_gather_q8(ids, w, scales, offsets, y)
        integer(int32), contiguous, intent(in) :: ids(:), w(:, :)
        integer(int16), contiguous, intent(in) :: scales(:, :), offsets(:, :)
        integer(int16), contiguous, intent(out) :: y(:, :)

        call require_size(size(ids, kind=int64))
        call require_size(size(y, kind=int64))
        call require_q8_shape(w, scales, offsets, size(y, 1))
        if (size(y, 2) /= size(ids)) error stop "embedding output shape mismatch"
        if (any(ids < 0_int32) .or. any(int(ids, int64) >= size(w, 2, kind=int64))) &
            error stop "embedding token id outside 0..vocabulary-1"
        do concurrent (integer :: m = 1:size(ids), k = 1:size(y, 1))
            block
                integer :: row, group
                integer(int32) :: q
                real(real32) :: value

                row = int(ids(m)) + 1
                group = (k - 1) / 64 + 1
                q = packed_q8_byte(w((k - 1) / 4 + 1, row), modulo(k - 1, 4))
                value = real(q, real32) * bf16_bits_to_real32(scales(group, row)) + &
                    bf16_bits_to_real32(offsets(group, row))
                y(k, m) = real32_to_bf16_bits(value)
            end block
        end do
    end subroutine embed_dequant_gather_q8

    pure subroutine linear_q8_bf16(x, w, scales, offsets, y, bias)
        integer(int16), contiguous, intent(in) :: x(:, :), scales(:, :), offsets(:, :)
        integer(int32), contiguous, intent(in) :: w(:, :)
        integer(int16), contiguous, intent(out) :: y(:, :)
        integer(int16), contiguous, optional, intent(in) :: bias(:)
        integer :: k, groups, parts, partition
        logical :: vector
        real(real32) :: xf(size(x,1),size(x,2)), sums(size(x,1)/64,size(x,2)), sums16(size(x,1)/16,size(x,2)), sx16(size(x,1)/16,size(x,2))
        integer(int32) :: xw(size(x,1)/4,size(x,2))

        call require_size(size(x, kind=int64))
        call require_size(size(y, kind=int64))
        k = size(x, 1)
        call require_q8_shape(w, scales, offsets, k)
        if (size(y, 1) /= size(w, 2) .or. size(y, 2) /= size(x, 2)) &
            error stop "linear output shape mismatch"
        if (present(bias)) then
            if (size(bias, kind=int64) /= size(w, 2, kind=int64)) error stop "linear bias shape mismatch"
        end if
        groups = k / 64
        vector = native_qmv_eligible(size(x,2),k,size(y,1))
        parts = 1
        if (.not. vector) parts = native_matrix_parts(size(x,2),k,size(y,1))
        partition = k/parts
        ! The activations stay where the previous kernel wrote them: chunks of 16
        ! are quantized (vector path) or groups of 64 widened to FP32 and summed.
        if (vector) then
            do concurrent (integer :: input_token = 1:size(x,2), input_chunk = 1:k/16)
                call quantize_chunk(x,xw,sx16,sums16,input_token,input_chunk)
            end do
        else
            do concurrent (integer :: input_token = 1:size(x,2), input_group = 1:groups)
                xf((input_group-1)*64+1:input_group*64,input_token) = &
                    bf16_bits_to_real32(x((input_group-1)*64+1:input_group*64,input_token))
                sums(input_group,input_token) = sum(xf((input_group-1)*64+1:input_group*64,input_token))
            end do
        end if
        if (vector) then
            ! The activations are quantized to bytes with one scale per chunk of 16
            ! and the codes dotted with them by the integer dot product of the GPU:
            ! a code c of 0..255 is the signed byte c-128 plus 128, the 128 going
            ! with the offset and the sum of the activations.
            do concurrent (integer :: n = 1:size(y,1), m = 1:size(x,2))
                block
                    real(real32) :: total, sc
                    integer :: c

                    total = 0.0_real32
                    do c = 1, k/16
                        sc = bf16_bits_to_real32(scales((c-1)/4+1,n))
                        total = total + (sc*sx16(c,m)*real(chunk_idot(w,xw,n,m,c),real32) &
                            + (bf16_bits_to_real32(offsets((c-1)/4+1,n)) + 128.0_real32*sc)*sums16(c,m))
                    end do
                    total = round_bf16(total)
                    if (present(bias)) total = total + bf16_bits_to_real32(bias(n))
                    y(n,m) = real32_to_bf16_bits(total)
                end block
            end do
        else
            do concurrent (integer :: n = 1:size(y,1))
                block
                    real(real32) :: weights(k)

                    do concurrent (integer :: d = 1:k)
                        weights(d) = round_bf16( &
                            real(packed_q8_byte(w((d-1)/4+1,n),modulo(d-1,4)),real32)* &
                            bf16_bits_to_real32(scales((d-1)/64+1,n)) + &
                            bf16_bits_to_real32(offsets((d-1)/64+1,n)))
                    end do
                    do concurrent (integer :: m = 1:size(x,2))
                        block
                            integer(int16) :: partial_values(parts)
                            real(real32) :: total
                            integer :: c

                            if (parts == 1) then
                                total = sum(weights*xf(:,m))
                            else
                                do concurrent (integer :: p = 1:parts)
                                    partial_values(p) = real32_to_bf16_bits(sum( &
                                        weights((p-1)*partition+1:p*partition)*xf((p-1)*partition+1:p*partition,m)))
                                end do
                                total = sum_strided_bf16(partial_values)
                            end if
                            total = round_bf16(total)
                            if (present(bias)) total = total + bf16_bits_to_real32(bias(n))
                            y(n,m) = real32_to_bf16_bits(total)
                        end block
                    end do
                end block
            end do
        end if
    end subroutine linear_q8_bf16

    ! The activations of chunk c as bytes: the largest of the 16 maps to 127.
    pure subroutine quantize_chunk(x, xw, sx, sums, m, c)
        integer(int16), intent(in) :: x(:,:)
        integer(int32), intent(inout) :: xw(:,:)
        real(real32), intent(inout) :: sx(:,:), sums(:,:)
        integer, intent(in) :: m, c
        real(real32) :: a0
        real(real32) :: a1
        real(real32) :: a2
        real(real32) :: a3
        real(real32) :: a4
        real(real32) :: a5
        real(real32) :: a6
        real(real32) :: a7
        real(real32) :: a8
        real(real32) :: a9
        real(real32) :: a10
        real(real32) :: a11
        real(real32) :: a12
        real(real32) :: a13
        real(real32) :: a14
        real(real32) :: a15
        real(real32) :: amax, inv
        integer(int32) :: q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15

        a0 = bf16_bits_to_real32(x(16*c-15,m))
        a1 = bf16_bits_to_real32(x(16*c-14,m))
        a2 = bf16_bits_to_real32(x(16*c-13,m))
        a3 = bf16_bits_to_real32(x(16*c-12,m))
        a4 = bf16_bits_to_real32(x(16*c-11,m))
        a5 = bf16_bits_to_real32(x(16*c-10,m))
        a6 = bf16_bits_to_real32(x(16*c-9,m))
        a7 = bf16_bits_to_real32(x(16*c-8,m))
        a8 = bf16_bits_to_real32(x(16*c-7,m))
        a9 = bf16_bits_to_real32(x(16*c-6,m))
        a10 = bf16_bits_to_real32(x(16*c-5,m))
        a11 = bf16_bits_to_real32(x(16*c-4,m))
        a12 = bf16_bits_to_real32(x(16*c-3,m))
        a13 = bf16_bits_to_real32(x(16*c-2,m))
        a14 = bf16_bits_to_real32(x(16*c-1,m))
        a15 = bf16_bits_to_real32(x(16*c,m))
        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
        sums(c,m) = (a0 + a2 + a4 + a6 + a8 + a10 + a12 + a14) + (a1 + a3 + a5 + a7 + a9 + a11 + a13 + a15)
        sx(c,m) = amax/127.0_real32
        inv = 0.0_real32
        if (amax > 0.0_real32) inv = 127.0_real32/amax
        q0 = nint(a0*inv)
        q1 = nint(a1*inv)
        q2 = nint(a2*inv)
        q3 = nint(a3*inv)
        q4 = nint(a4*inv)
        q5 = nint(a5*inv)
        q6 = nint(a6*inv)
        q7 = nint(a7*inv)
        q8 = nint(a8*inv)
        q9 = nint(a9*inv)
        q10 = nint(a10*inv)
        q11 = nint(a11*inv)
        q12 = nint(a12*inv)
        q13 = nint(a13*inv)
        q14 = nint(a14*inv)
        q15 = nint(a15*inv)
        xw(4*c-3,m) = ior(ior(iand(q0,255),ishft(iand(q1,255),8)), &
            ior(ishft(iand(q2,255),16),ishft(iand(q3,255),24)))
        xw(4*c-2,m) = ior(ior(iand(q4,255),ishft(iand(q5,255),8)), &
            ior(ishft(iand(q6,255),16),ishft(iand(q7,255),24)))
        xw(4*c-1,m) = ior(ior(iand(q8,255),ishft(iand(q9,255),8)), &
            ior(ishft(iand(q10,255),16),ishft(iand(q11,255),24)))
        xw(4*c,m) = ior(ior(iand(q12,255),ishft(iand(q13,255),8)), &
            ior(ishft(iand(q14,255),16),ishft(iand(q15,255),24)))
    end subroutine quantize_chunk

    ! The integer dot product of the 16 codes of chunk c (four words of a column
    ! of w, as signed bytes) with the 16 quantized activations.
    pure function chunk_idot(w, xw, n, m, c) result(r)
        integer(int32), intent(in) :: w(:,:), xw(:,:)
        integer, intent(in) :: n, m, c
        integer(int32) :: r

        r = dp4a(ieor(w(4*c-3,n),-2139062144_int32),xw(4*c-3,m), &
            dp4a(ieor(w(4*c-2,n),-2139062144_int32),xw(4*c-2,m), &
            dp4a(ieor(w(4*c-1,n),-2139062144_int32),xw(4*c-1,m), &
            dp4a(ieor(w(4*c,n),-2139062144_int32),xw(4*c,m),0_int32))))
    end function chunk_idot

    pure subroutine require_norm_shape(x, y, eps)
        integer(int16), intent(in) :: x(:, :)
        integer(int16), intent(in) :: y(:, :)
        real(real32), intent(in) :: eps

        call require_size(size(x, kind=int64))
        call require_size(size(y, kind=int64))
        if (any(shape(x) /= shape(y))) error stop "rmsnorm output shape mismatch"
        if (.not. (eps > 0.0_real32)) error stop "rmsnorm requires positive epsilon"
    end subroutine require_norm_shape

    pure subroutine rmsnorm_bf16(x, y, eps, weight)
        integer(int16), contiguous, intent(in) :: x(:, :)
        integer(int16), contiguous, intent(out) :: y(:, :)
        real(real32), intent(in) :: eps
        integer(int16), contiguous, optional, intent(in) :: weight(:)

        call require_norm_shape(x, y, eps)
        if (present(weight)) then
            if (size(weight, kind=int64) /= size(x, 1, kind=int64)) error stop "rmsnorm weight shape mismatch"
        end if
        do concurrent (integer :: m = 1:size(x, 2))
            block
                real(real32) :: rrms

                rrms = 1.0_real32 / sqrt(sum_squares_f32(x(:, m)) / real(size(x, 1), real32) + eps)
                do concurrent (integer :: d = 1:size(x, 1))
                    block
                        real(real32) :: normalized

                        normalized = bf16_bits_to_real32(x(d, m)) * rrms
                        if (present(weight)) normalized = round_bf16(normalized) * bf16_bits_to_real32(weight(d))
                        y(d, m) = real32_to_bf16_bits(normalized)
                    end block
                end do
            end block
        end do
    end subroutine rmsnorm_bf16

    pure subroutine rmsnorm_scale_bf16(x, y, eps, scalar)
        integer(int16), contiguous, intent(in) :: x(:, :)
        integer(int16), contiguous, intent(out) :: y(:, :)
        real(real32), intent(in) :: eps, scalar

        call require_norm_shape(x, y, eps)
        do concurrent (integer :: m = 1:size(x, 2))
            block
                real(real32) :: rrms, scale

                scale = round_bf16(scalar)
                rrms = 1.0_real32 / sqrt(sum_squares_f32(x(:, m)) / real(size(x, 1), real32) + eps)
                do concurrent (integer :: d = 1:size(x, 1))
                    block
                        real(real32) :: normalized

                        normalized = round_bf16(bf16_bits_to_real32(x(d, m)) * rrms)
                        y(d, m) = real32_to_bf16_bits(normalized * scale)
                    end block
                end do
            end block
        end do
    end subroutine rmsnorm_scale_bf16

    pure subroutine require_element_shape(x, y, out)
        integer(int16), intent(in) :: x(:), y(:)
        integer(int16), intent(in) :: out(:)

        call require_size(size(x, kind=int64))
        if (size(x, kind=int64) /= size(y, kind=int64) .or. &
                size(x, kind=int64) /= size(out, kind=int64)) error stop "elementwise shape mismatch"
    end subroutine require_element_shape

    pure subroutine silu_mul_bf16(gate, up, out)
        integer(int16), contiguous, intent(in) :: gate(:), up(:)
        integer(int16), contiguous, intent(out) :: out(:)

        call require_element_shape(gate, up, out)
        do concurrent (integer :: i = 1:size(gate))
            block
                real(real32) :: g, s

                g = bf16_bits_to_real32(gate(i))
                s = round_bf16(g * sigmoid_bf16_value(g))
                out(i) = real32_to_bf16_bits(s * bf16_bits_to_real32(up(i)))
            end block
        end do
    end subroutine silu_mul_bf16

    pure subroutine silu_mul2_bf16(gate, up, out)
        integer(int16), contiguous, intent(in) :: gate(:, :), up(:, :)
        integer(int16), contiguous, intent(out) :: out(:, :)

        call require_size(size(gate, kind=int64))
        if (any(shape(gate) /= shape(up)) .or. any(shape(gate) /= shape(out))) &
            error stop "elementwise shape mismatch"
        do concurrent (integer :: t = 1:size(gate, 2), i = 1:size(gate, 1))
            block
                real(real32) :: g, sv

                g = bf16_bits_to_real32(gate(i, t))
                sv = round_bf16(g * sigmoid_bf16_value(g))
                out(i, t) = real32_to_bf16_bits(sv * bf16_bits_to_real32(up(i, t)))
            end block
        end do
    end subroutine silu_mul2_bf16

    pure subroutine silu_mul_f32_gate_bf16(gate, up, out)
        integer(int16), contiguous, intent(in) :: gate(:), up(:)
        integer(int16), contiguous, intent(out) :: out(:)

        call require_element_shape(gate, up, out)
        do concurrent (integer :: i = 1:size(gate))
            block
                real(real32) :: g, s

                g = bf16_bits_to_real32(gate(i))
                s = 1.0_real32/(1.0_real32 + exp(abs(g)))
                if (g >= 0.0_real32) s = 1.0_real32-s
                s = g*s
                out(i) = real32_to_bf16_bits(s * bf16_bits_to_real32(up(i)))
            end block
        end do
    end subroutine silu_mul_f32_gate_bf16

    pure subroutine elemwise_sigmoid_mul_bf16(gate, x, out)
        integer(int16), contiguous, intent(in) :: gate(:), x(:)
        integer(int16), contiguous, intent(out) :: out(:)

        call require_element_shape(gate, x, out)
        do concurrent (integer :: i = 1:size(gate))
            out(i) = real32_to_bf16_bits(sigmoid_bf16_value(bf16_bits_to_real32(gate(i))) * &
                bf16_bits_to_real32(x(i)))
        end do
    end subroutine elemwise_sigmoid_mul_bf16

    pure subroutine residual_add_bf16(x, delta)
        integer(int16), contiguous, intent(inout) :: x(:)
        integer(int16), contiguous, intent(in) :: delta(:)

        call require_size(size(x, kind=int64))
        if (size(x, kind=int64) /= size(delta, kind=int64)) error stop "residual shape mismatch"
        do concurrent (integer :: i = 1:size(x))
            x(i) = real32_to_bf16_bits(bf16_bits_to_real32(x(i)) + bf16_bits_to_real32(delta(i)))
        end do
    end subroutine residual_add_bf16

    pure subroutine silu_inplace_bf16(x)
        integer(int16), contiguous, intent(inout) :: x(:)

        call require_size(size(x, kind=int64))
        do concurrent (integer :: i = 1:size(x))
            block
                real(real32) :: value

                value = bf16_bits_to_real32(x(i))
                x(i) = real32_to_bf16_bits(value * sigmoid_bf16_value(value))
            end block
        end do
    end subroutine silu_inplace_bf16

    pure subroutine sigmoid_inplace_bf16(x)
        integer(int16), contiguous, intent(inout) :: x(:)

        call require_size(size(x, kind=int64))
        do concurrent (integer :: i = 1:size(x))
            x(i) = real32_to_bf16_bits(sigmoid_bf16_value(bf16_bits_to_real32(x(i))))
        end do
    end subroutine sigmoid_inplace_bf16

    pure elemental function g_value(a, a_log, dt_bias) result(g)
        integer(int16), intent(in) :: a, a_log, dt_bias
        real(real32) :: g
        real(real32) :: summed, e, u, logarithm, sp

        summed = round_bf16(bf16_bits_to_real32(a) + bf16_bits_to_real32(dt_bias))
        e = round_bf16(exp(-abs(summed)))
        ! Standard Fortran has no LOG1P. Correct the rounded 1+e before
        ! the BF16 log1p store, including e too small to change 1.
        u = 1.0_real32 + e
        logarithm = log(u) - ((u - 1.0_real32) - e) / u
        sp = round_bf16(max(summed, 0.0_real32) + round_bf16(logarithm))
        g = exp(-exp(bf16_bits_to_real32(a_log)) * sp)
    end function g_value

    pure subroutine compute_g_f32(a, a_log, dt_bias, g)
        integer(int16), contiguous, intent(in) :: a(:, :), a_log(:), dt_bias(:)
        real(real32), contiguous, intent(out) :: g(:, :)

        call require_size(size(a, kind=int64))
        call require_size(size(g, kind=int64))
        if (size(a_log) /= size(a, 1) .or. size(dt_bias) /= size(a, 1) .or. &
                any(shape(g) /= shape(a))) error stop "compute_g shape mismatch"
        do concurrent (integer :: l = 1:size(a, 2), h = 1:size(a, 1))
            g(h, l) = g_value(a(h, l), a_log(h), dt_bias(h))
        end do
    end subroutine compute_g_f32

    pure subroutine conv1d_depthwise_causal_bf16(x, state, w, y)
        integer(int16), contiguous, intent(in) :: x(:, :), w(:, :)
        integer(int16), contiguous, intent(inout) :: state(:, :)
        integer(int16), contiguous, intent(out) :: y(:, :)
        integer :: channels, length, kernel, carry

        call require_size(size(x, kind=int64))
        call require_size(size(w, kind=int64))
        call require_size(size(y, kind=int64))
        channels = size(x, 1)
        length = size(x, 2)
        kernel = size(w, 1)
        carry = kernel - 1
        if (size(w, 2) /= channels .or. any(shape(y) /= shape(x)) .or. &
                any(shape(state) /= [channels, carry])) error stop "causal convolution shape mismatch"
        if (carry > 0) call require_size(size(state, kind=int64))
        do concurrent (integer :: c = 1:channels)
            block
                integer :: l, j, t, p
                real(real32) :: acc, value

                do l = 1, length
                    acc = 0.0_real32
                    do j = 1, kernel
                        t = l - (kernel - j)
                        if (t <= 0) then
                            value = bf16_bits_to_real32(state(c, t + carry))
                        else
                            value = bf16_bits_to_real32(x(c, t))
                        end if
                        acc = acc + bf16_bits_to_real32(w(j, c)) * value
                    end do
                    y(c, l) = real32_to_bf16_bits(acc)
                end do
                ! Each channel reads its old carry until all outputs are done.
                ! Ascending copies shift left safely when length < carry.
                do p = 1, carry
                    if (length >= carry .or. p > carry - length) then
                        state(c, p) = x(c, length - carry + p)
                    else
                        state(c, p) = state(c, p + length)
                    end if
                end do
            end block
        end do
    end subroutine conv1d_depthwise_causal_bf16

    pure subroutine gated_delta_step_bf16(q, k, v, g, beta, state, y)
        integer(int16), contiguous, intent(in) :: q(:, :), k(:, :), v(:, :), beta(:)
        real(real32), contiguous, intent(in) :: g(:)
        real(real32), contiguous, intent(inout) :: state(:, :, :)
        integer(int16), contiguous, intent(out) :: y(:, :)
        integer :: hk, hv, dk, dv, rep

        call require_size(size(q, kind=int64))
        call require_size(size(v, kind=int64))
        call require_size(size(state, kind=int64))
        call require_size(size(y, kind=int64))
        hk = size(q, 2)
        hv = size(v, 2)
        dk = size(q, 1)
        dv = size(v, 1)
        if (modulo(hv, hk) /= 0) error stop "gated delta requires Hv divisible by Hk"
        if (any(shape(k) /= shape(q)) .or. size(g) /= hv .or. size(beta) /= hv .or. &
                any(shape(state) /= [dv, dk, hv]) .or. any(shape(y) /= shape(v))) &
            error stop "gated delta shape mismatch"
        rep = hv / hk
        do concurrent (integer :: h = 1:hv, row = 1:dv)
            block
                integer :: kh
                real(real32) :: kv_mem, delta

                kh = (h - 1) / rep + 1
                state(:,row,h) = state(:,row,h)*g(h)
                kv_mem = sum(state(:,row,h)*bf16_bits_to_real32(k(:,kh)))
                ! Complete the old, decayed row reduction before any rank-one update.
                delta = (bf16_bits_to_real32(v(row, h)) - kv_mem) * bf16_bits_to_real32(beta(h))
                state(:,row,h) = state(:,row,h) + bf16_bits_to_real32(k(:,kh))*delta
                y(row,h) = real32_to_bf16_bits(sum(state(:,row,h)*bf16_bits_to_real32(q(:,kh))))
            end block
        end do
    end subroutine gated_delta_step_bf16

    ! conv1d_depthwise_causal_bf16 followed by silu_inplace_bf16 and the split of
    ! each token's channels: the first 2*key_dim channels go to qk, as one column
    ! of kdh values per head, token by token (the Hk query heads, then the Hk key
    ! heads); the remaining channels go to v. One launch instead of four.
    ! The same launch finishes the small per-head inputs of the recurrent step:
    ! g = compute_g_f32(a, a_log, dt_bias) and beta = sigmoid(b) in place
    ! (sigmoid_inplace_bf16), done by the threads of the first channels.
    pure subroutine conv1d_silu_split_bf16(x, state, w, qk, v, a, b, a_log, dt_bias, g)
        integer(int16), contiguous, intent(in) :: x(:, :), w(:, :), a(:, :), a_log(:), dt_bias(:)
        integer(int16), contiguous, intent(inout) :: state(:, :), b(:, :)
        integer(int16), contiguous, intent(out) :: qk(:, :), v(:, :)
        real(real32), contiguous, intent(out) :: g(:, :)
        integer :: channels, length, kernel, carry, kdh, hk, key_dim

        call require_size(size(x, kind=int64))
        call require_size(size(w, kind=int64))
        channels = size(x, 1)
        length = size(x, 2)
        kernel = size(w, 1)
        carry = kernel - 1
        kdh = size(qk, 1)
        hk = size(qk, 2) / (2 * length)
        key_dim = kdh * hk
        if (size(w, 2) /= channels .or. any(shape(state) /= [channels, carry]) .or. &
                size(qk, 2) /= 2 * hk * length .or. size(v, 2) /= length .or. &
                size(v, 1) /= channels - 2 * key_dim .or. size(a, 2) /= length .or. &
                any(shape(b) /= shape(a)) .or. any(shape(g) /= shape(a)) .or. &
                size(a_log) /= size(a, 1) .or. size(dt_bias) /= size(a, 1) .or. &
                size(a, 1) > channels) error stop "causal convolution shape mismatch"
        if (carry > 0) call require_size(size(state, kind=int64))
        do concurrent (integer :: c = 1:channels)
            block
                integer :: l, j, t, p, r
                real(real32) :: acc, value
                integer(int16) :: y

                do l = 1, length
                    acc = 0.0_real32
                    do j = 1, kernel
                        t = l - (kernel - j)
                        if (t <= 0) then
                            value = bf16_bits_to_real32(state(c, t + carry))
                        else
                            value = bf16_bits_to_real32(x(c, t))
                        end if
                        acc = acc + bf16_bits_to_real32(w(j, c)) * value
                    end do
                    y = real32_to_bf16_bits(acc)
                    value = bf16_bits_to_real32(y)
                    y = real32_to_bf16_bits(value * sigmoid_bf16_value(value))
                    if (c <= size(a, 1)) then
                        g(c, l) = g_value(a(c, l), a_log(c), dt_bias(c))
                        b(c, l) = real32_to_bf16_bits(sigmoid_bf16_value(bf16_bits_to_real32(b(c, l))))
                    end if
                    if (c <= 2 * key_dim) then
                        r = modulo(c - 1, key_dim)
                        qk(modulo(r, kdh) + 1, (l - 1) * 2 * hk + ((c - 1) / key_dim) * hk + r / kdh + 1) = y
                    else
                        v(c - 2 * key_dim, l) = y
                    end if
                end do
                do p = 1, carry
                    if (length >= carry .or. p > carry - length) then
                        state(c, p) = x(c, length - carry + p)
                    else
                        state(c, p) = state(c, p + length)
                    end if
                end do
            end block
        end do
    end subroutine conv1d_silu_split_bf16

    ! conv1d_silu_split_bf16 of one token followed by rmsnorm_scale_halves_bf16 of the
    ! query and key heads it produces, in one launch: a block takes the kdh channels of
    ! one head (or of kdh values), convolves them on its threads, and where they are a query
    ! or key head, normalizes them with the same threads after the block's barrier.
    pure subroutine conv1d_silu_split_norm_bf16(x, state, w, qk, v, a, b, a_log, dt_bias, g, qn, eps, scalar_a, scalar_b, half, period)
        integer(int16), contiguous, intent(in) :: x(:, :), w(:, :), a(:, :), a_log(:), dt_bias(:)
        integer(int16), contiguous, intent(inout) :: state(:, :), b(:, :)
        integer(int16), contiguous, intent(out) :: qk(:, :), v(:, :), qn(:, :)
        real(real32), contiguous, intent(out) :: g(:, :)
        real(real32), intent(in) :: eps, scalar_a, scalar_b
        integer, intent(in) :: half, period
        integer :: channels, kernel, carry, kdh, hk, key_dim, blocks

        call require_size(size(x, kind=int64))
        call require_size(size(w, kind=int64))
        channels = size(x, 1)
        kernel = size(w, 1)
        carry = kernel - 1
        kdh = size(qk, 1)
        hk = size(qk, 2) / 2
        key_dim = kdh * hk
        if (size(x, 2) /= 1 .or. modulo(channels, kdh) /= 0 .or. size(w, 2) /= channels .or. &
                any(shape(state) /= [channels, carry]) .or. size(qk, 2) /= 2 * hk .or. size(v, 2) /= 1 .or. &
                size(v, 1) /= channels - 2 * key_dim .or. size(a, 2) /= 1 .or. any(shape(b) /= shape(a)) .or. &
                any(shape(g) /= shape(a)) .or. size(a_log) /= size(a, 1) .or. size(dt_bias) /= size(a, 1) .or. &
                size(a, 1) > channels .or. any(shape(qn) /= shape(qk)) .or. carry < 1) &
            error stop "causal convolution shape mismatch"
        call require_size(size(state, kind=int64))
        blocks = channels / kdh
        do concurrent (integer :: blk = 1:blocks)
            block
                integer :: col
                real(real32) :: ss, rrms, scale

                do concurrent (integer :: ii = 1:kdh)
                    block
                        integer :: cc, jj, tt, pp, r
                        real(real32) :: accum, val
                        integer(int16) :: yy

                        cc = (blk - 1) * kdh + ii
                        accum = 0.0_real32
                        do jj = 1, kernel
                            tt = 1 - (kernel - jj)
                            if (tt <= 0) then
                                val = bf16_bits_to_real32(state(cc, tt + carry))
                            else
                                val = bf16_bits_to_real32(x(cc, tt))
                            end if
                            accum = accum + bf16_bits_to_real32(w(jj, cc)) * val
                        end do
                        yy = real32_to_bf16_bits(accum)
                        val = bf16_bits_to_real32(yy)
                        yy = real32_to_bf16_bits(val * sigmoid_bf16_value(val))
                        if (cc <= size(a, 1)) then
                            g(cc, 1) = g_value(a(cc, 1), a_log(cc), dt_bias(cc))
                            b(cc, 1) = real32_to_bf16_bits(sigmoid_bf16_value(bf16_bits_to_real32(b(cc, 1))))
                        end if
                        if (cc <= 2 * key_dim) then
                            r = modulo(cc - 1, key_dim)
                            qk(modulo(r, kdh) + 1, ((cc - 1) / key_dim) * hk + r / kdh + 1) = yy
                        else
                            v(cc - 2 * key_dim, 1) = yy
                        end if
                        do pp = 1, carry
                            if (pp > carry - 1) then
                                state(cc, pp) = x(cc, 1)
                            else
                                state(cc, pp) = state(cc, pp + 1)
                            end if
                        end do
                    end block
                end do
                if (blk <= 2 * hk) then
                    col = blk
                    scale = round_bf16(scalar_b)
                    if (modulo(col - 1, period) < half) scale = round_bf16(scalar_a)
                    ss = 0.0_real32
                    do concurrent (integer :: d = 1:kdh) reduce(+:ss)
                        ss = ss + bf16_bits_to_real32(qk(d, col))**2
                    end do
                    rrms = 1.0_real32 / sqrt(ss / real(kdh, real32) + eps)
                    do concurrent (integer :: d = 1:kdh)
                        block
                            real(real32) :: normalized

                            normalized = round_bf16(bf16_bits_to_real32(qk(d, col)) * rrms)
                            qn(d, col) = real32_to_bf16_bits(normalized * scale)
                        end block
                    end do
                end if
            end block
        end do
    end subroutine conv1d_silu_split_norm_bf16

    ! rmsnorm_scale_bf16 of the columns of x, where the columns of each period
    ! start with `half` columns that take scalar_a and end with columns that
    ! take scalar_b: the query and key heads of a token, in one launch.
    pure subroutine rmsnorm_scale_halves_bf16(x, y, eps, scalar_a, scalar_b, half, period)
        integer(int16), contiguous, intent(in) :: x(:, :)
        integer(int16), contiguous, intent(out) :: y(:, :)
        real(real32), intent(in) :: eps, scalar_a, scalar_b
        integer, intent(in) :: half, period

        call require_norm_shape(x, y, eps)
        do concurrent (integer :: m = 1:size(x, 2))
            block
                real(real32) :: rrms, scale

                scale = round_bf16(scalar_b)
                if (modulo(m - 1, period) < half) scale = round_bf16(scalar_a)
                rrms = 1.0_real32 / sqrt(sum_squares_f32(x(:, m)) / real(size(x, 1), real32) + eps)
                do concurrent (integer :: d = 1:size(x, 1))
                    block
                        real(real32) :: normalized

                        normalized = round_bf16(bf16_bits_to_real32(x(d, m)) * rrms)
                        y(d, m) = real32_to_bf16_bits(normalized * scale)
                    end block
                end do
            end block
        end do
    end subroutine rmsnorm_scale_halves_bf16

    ! gated_delta_step_bf16 plus the finiteness flags of the new state rows
    ! (keep_finite_rows_f32), kept up to date by the same launch. A non-finite
    ! state value never becomes finite again (inf and nan stay non-finite under
    ! the update), so checking the state after every step decides the same as
    ! checking it after the last.
    pure subroutine gated_delta_fused_bf16(qk, v, g, beta, state, y, finite, token, hk, dl)
        ! qk holds, token by token, hk query heads then hk key heads as columns;
        ! v, g, beta and y hold every token, and this step is the token-th.
        integer(int16), contiguous, intent(in) :: qk(:, :), v(:, :), beta(:, :)
        real(real32), contiguous, intent(in) :: g(:, :)
        real(real32), contiguous, intent(inout) :: state(:, :, :)
        integer(int16), contiguous, intent(inout) :: y(:, :, :)
        logical, contiguous, intent(inout) :: finite(:, :)
        real(real32), contiguous, intent(inout) :: dl(:, :, :)
        integer, intent(in) :: token, hk
        integer, parameter :: np = 4
        integer :: hv, dk, dv, rep, kp

        call require_size(size(qk, kind=int64))
        call require_size(size(v, kind=int64))
        call require_size(size(state, kind=int64))
        call require_size(size(y, kind=int64))
        dk = size(qk, 1)
        dv = size(y, 1)
        hv = size(y, 2)
        if (modulo(hv, hk) /= 0) error stop "gated delta requires Hv divisible by Hk"
        if (size(g, 1) /= hv .or. size(beta, 1) /= hv .or. size(v, 1) /= dv * hv .or. &
                token < 1 .or. token > size(y, 3) .or. size(qk, 2) < 2 * hk * token .or. &
                any(shape(state) /= [dv, dk, hv]) .or. any(shape(finite) /= [dv, hv]) .or. any(shape(dl) /= [dv, hv, 3*np]) .or. modulo(dk, np) /= 0 .or. modulo(dk, 4) /= 0) &
            error stop "gated delta shape mismatch"
        rep = hv / hk
        kp = dk / np
        ! The sums over the keys of a state row, in np parts that one thread each takes (the threads of a
        ! warp take consecutive rows, which are consecutive in memory): 4 times the threads of one per row.
        do concurrent (integer :: part = 1:np, h = 1:hv, row = 1:dv)
            block
                integer :: kh, qc, kc, k
                real(real32) :: a, b, kq, kk, qq, sv

                kh = (h - 1) / rep + 1
                qc = (token - 1) * 2 * hk + kh
                kc = qc + hk
                a = 0.0_real32
                b = 0.0_real32
                kq = 0.0_real32
                do k = (part-1)*kp + 1, part*kp
                    sv = state(row,k,h)
                    kk = bf16_bits_to_real32(qk(k,kc))
                    qq = bf16_bits_to_real32(qk(k,qc))
                    a = a + sv*kk
                    b = b + sv*qq
                    kq = kq + kk*qq
                end do
                dl(row,h,part) = a
                dl(row,h,np+part) = b
                dl(row,h,2*np+part) = kq
            end block
        end do
        ! The state update, 4 keys per thread (dk/4 apart, so that the delta of the row, which is made out
        ! of the np parts, is worked out once for them); the thread of the first keys also gives the output
        ! and the finiteness flag.
        do concurrent (integer :: h = 1:hv, k = 1:dk/4, row = 1:dv)
            block
                integer :: kc, p
                real(real32) :: a, b, kq, gg, delta

                kc = (token - 1) * 2 * hk + (h - 1) / rep + 1 + hk
                gg = g(h,token)
                a = 0.0_real32
                b = 0.0_real32
                kq = 0.0_real32
                do p = 1, np
                    a = a + dl(row,h,p)
                    b = b + dl(row,h,np+p)
                    kq = kq + dl(row,h,2*np+p)
                end do
                delta = (bf16_bits_to_real32(v((h-1)*dv+row, token)) - gg*a) * &
                    bf16_bits_to_real32(beta(h,token))
                state(row,k,h) = state(row,k,h)*gg + bf16_bits_to_real32(qk(k,kc))*delta
                state(row,k+dk/4,h) = state(row,k+dk/4,h)*gg + bf16_bits_to_real32(qk(k+dk/4,kc))*delta
                state(row,k+dk/2,h) = state(row,k+dk/2,h)*gg + bf16_bits_to_real32(qk(k+dk/2,kc))*delta
                state(row,k+3*dk/4,h) = state(row,k+3*dk/4,h)*gg + bf16_bits_to_real32(qk(k+3*dk/4,kc))*delta
                if (k == 1) then
                    y(row,h,token) = real32_to_bf16_bits(gg*b + delta*kq)
                    ! A non-finite new state value comes from a non-finite old one, which makes a and b
                    ! non-finite, or from a non-finite delta.
                    finite(row,h) = finite(row,h) .and. ieee_is_finite(a) .and. &
                        ieee_is_finite(b) .and. ieee_is_finite(delta)
                end if
            end block
        end do
    end subroutine gated_delta_fused_bf16

    ! The decode form of the recurrent step: the state update of a token is not made by a launch of its own;
    ! the pending update of the previous token (delta and decay in the columns of pd, the decay as its last row, and key column pk, written by
    ! gated_rmsnorm_q_delta_bf16) is applied to each state value as this launch reads it for the sums of the
    ! token, and the updated value is written to the other state array (reads and writes of one array in a
    ! loop would be ordered one after the other). The result is that of gated_delta_fused_bf16.
    pure subroutine gated_delta_lazy_bf16(qk, state, state2, pd, pk, token, hk, dl)
        integer(int16), contiguous, intent(in) :: qk(:, :), pk(:, :)
        real(real32), contiguous, intent(in) :: state(:, :, :)
        real(real32), contiguous, intent(out) :: state2(:, :, :)
        real(real32), contiguous, intent(in) :: pd(:, :)
        real(real32), contiguous, intent(inout) :: dl(:, :, :)
        integer, intent(in) :: token, hk
        integer, parameter :: np = 4
        integer :: hv, dk, dv, rep, kp

        call require_size(size(qk, kind=int64))
        call require_size(size(state, kind=int64))
        dk = size(qk, 1)
        dv = size(state, 1)
        hv = size(state, 3)
        if (modulo(hv, hk) /= 0) error stop "gated delta requires Hv divisible by Hk"
        if (size(pd, 1) /= dv + 1 .or. size(pd, 2) /= hv .or. size(pk, 1) /= dk .or. &
                size(pk, 2) /= hk .or. token < 1 .or. size(qk, 2) < 2 * hk * token .or. &
                any(shape(state) /= [dv, dk, hv]) .or. any(shape(state2) /= [dv, dk, hv]) .or. &
                any(shape(dl) /= [dv, hv, 3*np]) .or. modulo(dk, np) /= 0) error stop "gated delta shape mismatch"
        rep = hv / hk
        kp = dk / np
        do concurrent (integer :: part = 1:np, h = 1:hv, row = 1:dv)
            block
                integer :: kh, qc, kc, k
                real(real32) :: a, b, kq, kk, qq, sv, pgg, pdl

                kh = (h - 1) / rep + 1
                qc = (token - 1) * 2 * hk + kh
                kc = qc + hk
                pgg = pd(dv+1,h)
                pdl = pd(row,h)
                a = 0.0_real32
                b = 0.0_real32
                kq = 0.0_real32
                do k = (part-1)*kp + 1, part*kp
                    sv = state(row,k,h)*pgg + bf16_bits_to_real32(pk(k,kh))*pdl
                    state2(row,k,h) = sv
                    kk = bf16_bits_to_real32(qk(k,kc))
                    qq = bf16_bits_to_real32(qk(k,qc))
                    a = a + sv*kk
                    b = b + sv*qq
                    kq = kq + kk*qq
                end do
                dl(row,h,part) = a
                dl(row,h,np+part) = b
                dl(row,h,2*np+part) = kq
            end block
        end do
    end subroutine gated_delta_lazy_bf16

    ! Applies the pending update of the lazy form to the state and clears it.
    pure subroutine gated_delta_flush_bf16(state, pd, pk, hk)
        real(real32), contiguous, intent(inout) :: state(:, :, :), pd(:, :)
        integer(int16), contiguous, intent(in) :: pk(:, :)
        integer, intent(in) :: hk
        integer :: rep

        call require_size(size(state, kind=int64))
        rep = size(state, 3) / hk
        do concurrent (integer :: h = 1:size(state, 3), k = 1:size(state, 2), row = 1:size(state, 1))
            block
                state(row,k,h) = state(row,k,h)*pd(size(state,1)+1,h) + bf16_bits_to_real32(pk(k,(h-1)/rep+1))*pd(row,h)
            end block
        end do
        do concurrent (integer :: h = 1:size(state, 3), row = 1:size(state, 1))
            block
                pd(row,h) = 0.0_real32
                if (row == 1) pd(size(state,1)+1,h) = 1.0_real32
            end block
        end do
    end subroutine gated_delta_flush_bf16

    ! quantize_chunk of chunk c of a vector x, writing the results at chunk cg.
    pure subroutine quantize_chunk1(x, xw, sx, sums, c, cg)
        integer(int16), intent(in) :: x(:)
        integer(int32), intent(inout) :: xw(:)
        real(real32), intent(inout) :: sx(:), sums(:)
        integer, intent(in) :: c, cg
        real(real32) :: a0
        real(real32) :: a1
        real(real32) :: a2
        real(real32) :: a3
        real(real32) :: a4
        real(real32) :: a5
        real(real32) :: a6
        real(real32) :: a7
        real(real32) :: a8
        real(real32) :: a9
        real(real32) :: a10
        real(real32) :: a11
        real(real32) :: a12
        real(real32) :: a13
        real(real32) :: a14
        real(real32) :: a15
        real(real32) :: amax, inv
        integer(int32) :: q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15

        a0 = bf16_bits_to_real32(x(16*c-15))
        a1 = bf16_bits_to_real32(x(16*c-14))
        a2 = bf16_bits_to_real32(x(16*c-13))
        a3 = bf16_bits_to_real32(x(16*c-12))
        a4 = bf16_bits_to_real32(x(16*c-11))
        a5 = bf16_bits_to_real32(x(16*c-10))
        a6 = bf16_bits_to_real32(x(16*c-9))
        a7 = bf16_bits_to_real32(x(16*c-8))
        a8 = bf16_bits_to_real32(x(16*c-7))
        a9 = bf16_bits_to_real32(x(16*c-6))
        a10 = bf16_bits_to_real32(x(16*c-5))
        a11 = bf16_bits_to_real32(x(16*c-4))
        a12 = bf16_bits_to_real32(x(16*c-3))
        a13 = bf16_bits_to_real32(x(16*c-2))
        a14 = bf16_bits_to_real32(x(16*c-1))
        a15 = bf16_bits_to_real32(x(16*c))
        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
        sums(cg) = (a0 + a2 + a4 + a6 + a8 + a10 + a12 + a14) + (a1 + a3 + a5 + a7 + a9 + a11 + a13 + a15)
        sx(cg) = amax/127.0_real32
        inv = 0.0_real32
        if (amax > 0.0_real32) inv = 127.0_real32/amax
        q0 = nint(a0*inv)
        q1 = nint(a1*inv)
        q2 = nint(a2*inv)
        q3 = nint(a3*inv)
        q4 = nint(a4*inv)
        q5 = nint(a5*inv)
        q6 = nint(a6*inv)
        q7 = nint(a7*inv)
        q8 = nint(a8*inv)
        q9 = nint(a9*inv)
        q10 = nint(a10*inv)
        q11 = nint(a11*inv)
        q12 = nint(a12*inv)
        q13 = nint(a13*inv)
        q14 = nint(a14*inv)
        q15 = nint(a15*inv)
        xw(4*cg-3) = ior(ior(iand(q0,255),ishft(iand(q1,255),8)), &
            ior(ishft(iand(q2,255),16),ishft(iand(q3,255),24)))
        xw(4*cg-2) = ior(ior(iand(q4,255),ishft(iand(q5,255),8)), &
            ior(ishft(iand(q6,255),16),ishft(iand(q7,255),24)))
        xw(4*cg-1) = ior(ior(iand(q8,255),ishft(iand(q9,255),8)), &
            ior(ishft(iand(q10,255),16),ishft(iand(q11,255),24)))
        xw(4*cg) = ior(ior(iand(q12,255),ishft(iand(q13,255),8)), &
            ior(ishft(iand(q14,255),16),ishft(iand(q15,255),24)))
    end subroutine quantize_chunk1

    ! residual_rmsnorm_bf16 of one column whose result is not stored but handed
    ! to linear_q8_q_bf16 as quantized activations (see quantize_vals), in one
    ! launch of one iteration per group of 64: each recomputes the norm of the
    ! column from x and delta, so no iteration waits for another. The new
    ! residual goes to xo, because the other iterations still read x and delta.
    pure subroutine residual_rmsnorm_q_bf16(x, delta, xo, y, xw, sx, sums, eps, weight)
        integer(int16), contiguous, intent(in) :: x(:), delta(:), weight(:)
        integer(int16), contiguous, intent(out) :: xo(:), y(:)
        integer(int32), contiguous, intent(out) :: xw(:)
        real(real32), contiguous, intent(out) :: sx(:), sums(:)
        real(real32), intent(in) :: eps
        integer :: n

        n = size(x)
        call require_size(size(x, kind=int64))
        if (size(delta) /= n .or. size(xo) /= n .or. size(y) /= n .or. size(weight) /= n .or. &
            modulo(n, 64) /= 0 .or. size(sums) < n / 16 .or. size(sx) < n / 16 .or. size(xw) < n / 4) &
            error stop "residual rmsnorm shape mismatch"
        do concurrent (integer :: g = 1:n/64)
            block
                real(real32) :: rrms, ssq
                integer :: c

                ! The sum of squares in chunks of 8 elements: a lane reads the 16 bytes of 8 values of x and of
                ! delta at once (gpu_wide_loads).
                ssq = 0.0_real32
                do c = 1, n/8
                    ssq = ssq + (((sq_sum(x(8*c-7), delta(8*c-7)) + sq_sum(x(8*c-6), delta(8*c-6))) + &
                        (sq_sum(x(8*c-5), delta(8*c-5)) + sq_sum(x(8*c-4), delta(8*c-4)))) + &
                        ((sq_sum(x(8*c-3), delta(8*c-3)) + sq_sum(x(8*c-2), delta(8*c-2))) + &
                        (sq_sum(x(8*c-1), delta(8*c-1)) + sq_sum(x(8*c), delta(8*c)))))
                end do
                rrms = 1.0_real32 / sqrt(ssq / real(n, real32) + eps)
                do concurrent (integer :: d = (g-1)*64+1:g*64)
                    block
                        real(real32) :: normalized
                        integer(int16) :: sum_bits

                        sum_bits = real32_to_bf16_bits(bf16_bits_to_real32(x(d)) + bf16_bits_to_real32(delta(d)))
                        xo(d) = sum_bits
                        normalized = round_bf16(bf16_bits_to_real32(sum_bits) * rrms) * bf16_bits_to_real32(weight(d))
                        y(d) = real32_to_bf16_bits(normalized)
                    end block
                end do
                ! The words of the quantized vector (4 values each) are shared out over the lanes; the scale and
                ! the sum of a chunk of 16 are computed again by each word of it (quantize_chunk1 inlined: a call
                ! inside the lanes of a launch makes it serial). Each iteration assigns one element of xw.
                do concurrent (integer :: wd = 1:16)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax, inv, b0, b1, b2, b3
                        integer(int32) :: q0, q1, q2, q3
                        integer :: c, e
                        c = (wd-1)/4
                        e = 4*modulo(wd-1,4)
                        a0 = bf16_bits_to_real32(y((g-1)*64+16*c+1))
                        a1 = bf16_bits_to_real32(y((g-1)*64+16*c+2))
                        a2 = bf16_bits_to_real32(y((g-1)*64+16*c+3))
                        a3 = bf16_bits_to_real32(y((g-1)*64+16*c+4))
                        a4 = bf16_bits_to_real32(y((g-1)*64+16*c+5))
                        a5 = bf16_bits_to_real32(y((g-1)*64+16*c+6))
                        a6 = bf16_bits_to_real32(y((g-1)*64+16*c+7))
                        a7 = bf16_bits_to_real32(y((g-1)*64+16*c+8))
                        a8 = bf16_bits_to_real32(y((g-1)*64+16*c+9))
                        a9 = bf16_bits_to_real32(y((g-1)*64+16*c+10))
                        a10 = bf16_bits_to_real32(y((g-1)*64+16*c+11))
                        a11 = bf16_bits_to_real32(y((g-1)*64+16*c+12))
                        a12 = bf16_bits_to_real32(y((g-1)*64+16*c+13))
                        a13 = bf16_bits_to_real32(y((g-1)*64+16*c+14))
                        a14 = bf16_bits_to_real32(y((g-1)*64+16*c+15))
                        a15 = bf16_bits_to_real32(y((g-1)*64+16*c+16))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        inv = 0.0_real32
                        if (amax > 0.0_real32) inv = 127.0_real32/amax
                        b0 = bf16_bits_to_real32(y((g-1)*64+16*c+e+1))
                        b1 = bf16_bits_to_real32(y((g-1)*64+16*c+e+2))
                        b2 = bf16_bits_to_real32(y((g-1)*64+16*c+e+3))
                        b3 = bf16_bits_to_real32(y((g-1)*64+16*c+e+4))
                        q0 = nint(b0*inv)
                        q1 = nint(b1*inv)
                        q2 = nint(b2*inv)
                        q3 = nint(b3*inv)
                        xw((g-1)*16+wd) = ior(ior(iand(q0,255),ishft(iand(q1,255),8)), &
                            ior(ishft(iand(q2,255),16),ishft(iand(q3,255),24)))
                    end block
                end do
                do concurrent (integer :: j = 1:4)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax
                        integer :: c
                        c = j-1
                        a0 = bf16_bits_to_real32(y((g-1)*64+16*c+1))
                        a1 = bf16_bits_to_real32(y((g-1)*64+16*c+2))
                        a2 = bf16_bits_to_real32(y((g-1)*64+16*c+3))
                        a3 = bf16_bits_to_real32(y((g-1)*64+16*c+4))
                        a4 = bf16_bits_to_real32(y((g-1)*64+16*c+5))
                        a5 = bf16_bits_to_real32(y((g-1)*64+16*c+6))
                        a6 = bf16_bits_to_real32(y((g-1)*64+16*c+7))
                        a7 = bf16_bits_to_real32(y((g-1)*64+16*c+8))
                        a8 = bf16_bits_to_real32(y((g-1)*64+16*c+9))
                        a9 = bf16_bits_to_real32(y((g-1)*64+16*c+10))
                        a10 = bf16_bits_to_real32(y((g-1)*64+16*c+11))
                        a11 = bf16_bits_to_real32(y((g-1)*64+16*c+12))
                        a12 = bf16_bits_to_real32(y((g-1)*64+16*c+13))
                        a13 = bf16_bits_to_real32(y((g-1)*64+16*c+14))
                        a14 = bf16_bits_to_real32(y((g-1)*64+16*c+15))
                        a15 = bf16_bits_to_real32(y((g-1)*64+16*c+16))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        sums((g-1)*4+j) = (a0 + a2 + a4 + a6 + a8 + a10 + a12 + a14) + (a1 + a3 + a5 + a7 + a9 + a11 + a13 + a15)
                        sx((g-1)*4+j) = amax/127.0_real32
                    end block
                end do
            end block
        end do
    end subroutine residual_rmsnorm_q_bf16

    ! linear_q8_bf16 of one column on the native vector path, with the activations
    ! given quantized (xw, sx, sums as quantize_vals writes them).
    pure subroutine linear_q8_q_bf16(xw, sx, sums, w, scales, offsets, y)
        integer(int32), contiguous, intent(in) :: xw(:), w(:, :)
        real(real32), contiguous, intent(in) :: sx(:), sums(:)
        integer(int16), contiguous, intent(in) :: scales(:, :), offsets(:, :)
        integer(int16), contiguous, intent(out) :: y(:, :)
        integer :: k

        k = 4*size(w, 1)
        call require_q8_shape(w, scales, offsets, k)
        if (size(y, 1) /= size(w, 2) .or. size(y, 2) /= 1 .or. size(xw) < k/4 .or. size(sx) < k/16 .or. &
            size(sums) < k/16 .or. modulo(k, 1024) /= 0 .or. .not. native_qmv_eligible(1, k, size(y, 1))) error stop "linear q shape mismatch"
        ! A lane takes two chunks of 16 at a time, which share a group of 64: the
        ! scale and offset are read once, and the activations of both together.
        do concurrent (integer :: n = 1:size(y,1))
            block
                real(real32) :: total, sc, off
                integer :: c, q

                total = 0.0_real32
                do q = 1, k/32
                    c = 2*q - 1
                    sc = bf16_bits_to_real32(scales((c-1)/4+1,n))
                    off = bf16_bits_to_real32(offsets((c-1)/4+1,n)) + 128.0_real32*sc
                    ! The dot products are written out here, not in chunk_idot1: the words of
                    ! the two chunks are then read as vectors in the loop itself, which the
                    ! compiler writes twice, for an aligned array (no test per vector) and not.
                    ! The two chunks are one statement, total + chunk + chunk (the order of two
                    ! statements): the compiler needs lfortran PR 17815 to share it out between lanes.
                    total = total + (sc*sx(c)*real(dp4a(ieor(w(4*c-3,n),-2139062144_int32),xw(4*c-3), &
                        dp4a(ieor(w(4*c-2,n),-2139062144_int32),xw(4*c-2), &
                        dp4a(ieor(w(4*c-1,n),-2139062144_int32),xw(4*c-1), &
                        dp4a(ieor(w(4*c,n),-2139062144_int32),xw(4*c),0_int32)))),real32) + off*sums(c)) &
                        + (sc*sx(c+1)*real(dp4a(ieor(w(4*(c+1)-3,n),-2139062144_int32),xw(4*(c+1)-3), &
                        dp4a(ieor(w(4*(c+1)-2,n),-2139062144_int32),xw(4*(c+1)-2), &
                        dp4a(ieor(w(4*(c+1)-1,n),-2139062144_int32),xw(4*(c+1)-1), &
                        dp4a(ieor(w(4*(c+1),n),-2139062144_int32),xw(4*(c+1)),0_int32)))),real32) + off*sums(c+1))
                end do
                y(n,1) = real32_to_bf16_bits(round_bf16(total))
            end block
        end do
    end subroutine linear_q8_q_bf16

    ! The square of the bf16 sum of x and delta.
    pure elemental function sq_sum(x, delta) result(v)
        integer(int16), intent(in) :: x, delta
        real(real32) :: v

        v = round_bf16(bf16_bits_to_real32(x) + bf16_bits_to_real32(delta))
        v = v * v
    end function sq_sum

    ! rmsnorm_bf16 (with weight) of the columns of x, followed by
    ! silu_mul_f32_gate_bf16 with the gate z: the gated norm of the delta layers.
    pure subroutine gated_rmsnorm_bf16(x, z, y, eps, weight)
        integer(int16), contiguous, intent(in) :: x(:, :), z(:, :), weight(:)
        integer(int16), contiguous, intent(out) :: y(:, :)
        real(real32), intent(in) :: eps

        call require_norm_shape(x, y, eps)
        if (any(shape(z) /= shape(x)) .or. size(weight, kind=int64) /= size(x, 1, kind=int64)) &
            error stop "gated rmsnorm shape mismatch"
        do concurrent (integer :: m = 1:size(x, 2))
            block
                real(real32) :: rrms

                rrms = 1.0_real32 / sqrt(sum_squares_f32(x(:, m)) / real(size(x, 1), real32) + eps)
                do concurrent (integer :: d = 1:size(x, 1))
                    block
                        real(real32) :: normalized, gt, s

                        normalized = round_bf16(round_bf16(bf16_bits_to_real32(x(d, m)) * rrms) * &
                            bf16_bits_to_real32(weight(d)))
                        gt = bf16_bits_to_real32(z(d, m))
                        s = 1.0_real32/(1.0_real32 + exp(abs(gt)))
                        if (gt >= 0.0_real32) s = 1.0_real32-s
                        s = gt*s
                        y(d, m) = real32_to_bf16_bits(s * normalized)
                    end block
                end do
            end block
        end do
    end subroutine gated_rmsnorm_bf16

    ! silu_mul2_bf16 of one column, quantized for linear_q8_q_bf16, in one launch.
    pure subroutine silu_mul2_q_bf16(gate, up, out, xw, sx, sums)
        integer(int16), contiguous, intent(in) :: gate(:), up(:)
        integer(int16), contiguous, intent(out) :: out(:)
        integer(int32), contiguous, intent(out) :: xw(:)
        real(real32), contiguous, intent(out) :: sx(:), sums(:)

        call require_size(size(gate, kind=int64))
        if (size(up) /= size(gate) .or. size(out) /= size(gate) .or. modulo(size(gate), 64) /= 0 .or. &
            size(sums) < size(gate) / 16 .or. size(sx) < size(gate) / 16 .or. size(xw) < size(gate) / 4) &
            error stop "elementwise q shape mismatch"
        do concurrent (integer :: g = 1:size(gate)/64)
            block
                do concurrent (integer :: i = (g-1)*64+1:g*64)
                    block
                        real(real32) :: gv, sv

                        gv = bf16_bits_to_real32(gate(i))
                        sv = round_bf16(gv * sigmoid_bf16_value(gv))
                        out(i) = real32_to_bf16_bits(sv * bf16_bits_to_real32(up(i)))
                    end block
                end do
                ! The words of the quantized vector (4 values each) are shared out over the lanes; the scale and
                ! the sum of a chunk of 16 are computed again by each word of it (quantize_chunk1 inlined: a call
                ! inside the lanes of a launch makes it serial). Each iteration assigns one element of xw.
                do concurrent (integer :: wd = 1:16)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax, inv, b0, b1, b2, b3
                        integer(int32) :: q0, q1, q2, q3
                        integer :: c, e
                        c = (wd-1)/4
                        e = 4*modulo(wd-1,4)
                        a0 = bf16_bits_to_real32(out((g-1)*64+16*c+1))
                        a1 = bf16_bits_to_real32(out((g-1)*64+16*c+2))
                        a2 = bf16_bits_to_real32(out((g-1)*64+16*c+3))
                        a3 = bf16_bits_to_real32(out((g-1)*64+16*c+4))
                        a4 = bf16_bits_to_real32(out((g-1)*64+16*c+5))
                        a5 = bf16_bits_to_real32(out((g-1)*64+16*c+6))
                        a6 = bf16_bits_to_real32(out((g-1)*64+16*c+7))
                        a7 = bf16_bits_to_real32(out((g-1)*64+16*c+8))
                        a8 = bf16_bits_to_real32(out((g-1)*64+16*c+9))
                        a9 = bf16_bits_to_real32(out((g-1)*64+16*c+10))
                        a10 = bf16_bits_to_real32(out((g-1)*64+16*c+11))
                        a11 = bf16_bits_to_real32(out((g-1)*64+16*c+12))
                        a12 = bf16_bits_to_real32(out((g-1)*64+16*c+13))
                        a13 = bf16_bits_to_real32(out((g-1)*64+16*c+14))
                        a14 = bf16_bits_to_real32(out((g-1)*64+16*c+15))
                        a15 = bf16_bits_to_real32(out((g-1)*64+16*c+16))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        inv = 0.0_real32
                        if (amax > 0.0_real32) inv = 127.0_real32/amax
                        b0 = bf16_bits_to_real32(out((g-1)*64+16*c+e+1))
                        b1 = bf16_bits_to_real32(out((g-1)*64+16*c+e+2))
                        b2 = bf16_bits_to_real32(out((g-1)*64+16*c+e+3))
                        b3 = bf16_bits_to_real32(out((g-1)*64+16*c+e+4))
                        q0 = nint(b0*inv)
                        q1 = nint(b1*inv)
                        q2 = nint(b2*inv)
                        q3 = nint(b3*inv)
                        xw((g-1)*16+wd) = ior(ior(iand(q0,255),ishft(iand(q1,255),8)), &
                            ior(ishft(iand(q2,255),16),ishft(iand(q3,255),24)))
                    end block
                end do
                do concurrent (integer :: j = 1:4)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax
                        integer :: c
                        c = j-1
                        a0 = bf16_bits_to_real32(out((g-1)*64+16*c+1))
                        a1 = bf16_bits_to_real32(out((g-1)*64+16*c+2))
                        a2 = bf16_bits_to_real32(out((g-1)*64+16*c+3))
                        a3 = bf16_bits_to_real32(out((g-1)*64+16*c+4))
                        a4 = bf16_bits_to_real32(out((g-1)*64+16*c+5))
                        a5 = bf16_bits_to_real32(out((g-1)*64+16*c+6))
                        a6 = bf16_bits_to_real32(out((g-1)*64+16*c+7))
                        a7 = bf16_bits_to_real32(out((g-1)*64+16*c+8))
                        a8 = bf16_bits_to_real32(out((g-1)*64+16*c+9))
                        a9 = bf16_bits_to_real32(out((g-1)*64+16*c+10))
                        a10 = bf16_bits_to_real32(out((g-1)*64+16*c+11))
                        a11 = bf16_bits_to_real32(out((g-1)*64+16*c+12))
                        a12 = bf16_bits_to_real32(out((g-1)*64+16*c+13))
                        a13 = bf16_bits_to_real32(out((g-1)*64+16*c+14))
                        a14 = bf16_bits_to_real32(out((g-1)*64+16*c+15))
                        a15 = bf16_bits_to_real32(out((g-1)*64+16*c+16))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        sums((g-1)*4+j) = (a0 + a2 + a4 + a6 + a8 + a10 + a12 + a14) + (a1 + a3 + a5 + a7 + a9 + a11 + a13 + a15)
                        sx((g-1)*4+j) = amax/127.0_real32
                    end block
                end do
            end block
        end do
    end subroutine silu_mul2_q_bf16

    ! elemwise_sigmoid_mul_bf16, quantized for linear_q8_q_bf16.
    pure subroutine sigmoid_mul_q_bf16(gate, x, out, xw, sx, sums)
        integer(int16), contiguous, intent(in) :: gate(:), x(:)
        integer(int16), contiguous, intent(out) :: out(:)
        integer(int32), contiguous, intent(out) :: xw(:)
        real(real32), contiguous, intent(out) :: sx(:), sums(:)

        call require_size(size(gate, kind=int64))
        if (size(x) /= size(gate) .or. size(out) /= size(gate) .or. modulo(size(gate), 64) /= 0 .or. &
            size(sums) < size(gate) / 16 .or. size(sx) < size(gate) / 16 .or. size(xw) < size(gate) / 4) &
            error stop "elementwise q shape mismatch"
        do concurrent (integer :: g = 1:size(gate)/64)
            block
                do concurrent (integer :: i = (g-1)*64+1:g*64)
                    out(i) = real32_to_bf16_bits(sigmoid_bf16_value(bf16_bits_to_real32(gate(i))) * &
                        bf16_bits_to_real32(x(i)))
                end do
                ! The words of the quantized vector (4 values each) are shared out over the lanes; the scale and
                ! the sum of a chunk of 16 are computed again by each word of it (quantize_chunk1 inlined: a call
                ! inside the lanes of a launch makes it serial). Each iteration assigns one element of xw.
                do concurrent (integer :: wd = 1:16)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax, inv, b0, b1, b2, b3
                        integer(int32) :: q0, q1, q2, q3
                        integer :: c, e
                        c = (wd-1)/4
                        e = 4*modulo(wd-1,4)
                        a0 = bf16_bits_to_real32(out((g-1)*64+16*c+1))
                        a1 = bf16_bits_to_real32(out((g-1)*64+16*c+2))
                        a2 = bf16_bits_to_real32(out((g-1)*64+16*c+3))
                        a3 = bf16_bits_to_real32(out((g-1)*64+16*c+4))
                        a4 = bf16_bits_to_real32(out((g-1)*64+16*c+5))
                        a5 = bf16_bits_to_real32(out((g-1)*64+16*c+6))
                        a6 = bf16_bits_to_real32(out((g-1)*64+16*c+7))
                        a7 = bf16_bits_to_real32(out((g-1)*64+16*c+8))
                        a8 = bf16_bits_to_real32(out((g-1)*64+16*c+9))
                        a9 = bf16_bits_to_real32(out((g-1)*64+16*c+10))
                        a10 = bf16_bits_to_real32(out((g-1)*64+16*c+11))
                        a11 = bf16_bits_to_real32(out((g-1)*64+16*c+12))
                        a12 = bf16_bits_to_real32(out((g-1)*64+16*c+13))
                        a13 = bf16_bits_to_real32(out((g-1)*64+16*c+14))
                        a14 = bf16_bits_to_real32(out((g-1)*64+16*c+15))
                        a15 = bf16_bits_to_real32(out((g-1)*64+16*c+16))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        inv = 0.0_real32
                        if (amax > 0.0_real32) inv = 127.0_real32/amax
                        b0 = bf16_bits_to_real32(out((g-1)*64+16*c+e+1))
                        b1 = bf16_bits_to_real32(out((g-1)*64+16*c+e+2))
                        b2 = bf16_bits_to_real32(out((g-1)*64+16*c+e+3))
                        b3 = bf16_bits_to_real32(out((g-1)*64+16*c+e+4))
                        q0 = nint(b0*inv)
                        q1 = nint(b1*inv)
                        q2 = nint(b2*inv)
                        q3 = nint(b3*inv)
                        xw((g-1)*16+wd) = ior(ior(iand(q0,255),ishft(iand(q1,255),8)), &
                            ior(ishft(iand(q2,255),16),ishft(iand(q3,255),24)))
                    end block
                end do
                do concurrent (integer :: j = 1:4)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax
                        integer :: c
                        c = j-1
                        a0 = bf16_bits_to_real32(out((g-1)*64+16*c+1))
                        a1 = bf16_bits_to_real32(out((g-1)*64+16*c+2))
                        a2 = bf16_bits_to_real32(out((g-1)*64+16*c+3))
                        a3 = bf16_bits_to_real32(out((g-1)*64+16*c+4))
                        a4 = bf16_bits_to_real32(out((g-1)*64+16*c+5))
                        a5 = bf16_bits_to_real32(out((g-1)*64+16*c+6))
                        a6 = bf16_bits_to_real32(out((g-1)*64+16*c+7))
                        a7 = bf16_bits_to_real32(out((g-1)*64+16*c+8))
                        a8 = bf16_bits_to_real32(out((g-1)*64+16*c+9))
                        a9 = bf16_bits_to_real32(out((g-1)*64+16*c+10))
                        a10 = bf16_bits_to_real32(out((g-1)*64+16*c+11))
                        a11 = bf16_bits_to_real32(out((g-1)*64+16*c+12))
                        a12 = bf16_bits_to_real32(out((g-1)*64+16*c+13))
                        a13 = bf16_bits_to_real32(out((g-1)*64+16*c+14))
                        a14 = bf16_bits_to_real32(out((g-1)*64+16*c+15))
                        a15 = bf16_bits_to_real32(out((g-1)*64+16*c+16))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        sums((g-1)*4+j) = (a0 + a2 + a4 + a6 + a8 + a10 + a12 + a14) + (a1 + a3 + a5 + a7 + a9 + a11 + a13 + a15)
                        sx((g-1)*4+j) = amax/127.0_real32
                    end block
                end do
            end block
        end do
    end subroutine sigmoid_mul_q_bf16

    ! gated_rmsnorm_bf16, quantized for linear_q8_q_bf16 over the columns of x
    ! one after another (the width is a multiple of 64).
    pure subroutine gated_rmsnorm_q_bf16(x, z, y, eps, weight, xw, sx, sums)
        integer(int16), contiguous, intent(in) :: x(:, :), z(:, :), weight(:)
        integer(int16), contiguous, intent(out) :: y(:, :)
        integer(int32), contiguous, intent(out) :: xw(:)
        real(real32), contiguous, intent(out) :: sx(:), sums(:)
        real(real32), intent(in) :: eps
        integer :: w

        call require_norm_shape(x, y, eps)
        w = size(x, 1)
        if (any(shape(z) /= shape(x)) .or. size(weight) /= w .or. modulo(w, 64) /= 0 .or. &
            size(sums) < size(x) / 16 .or. size(sx) < size(x) / 16 .or. size(xw) < size(x) / 4) &
            error stop "gated rmsnorm q shape mismatch"
        do concurrent (integer :: m = 1:size(x, 2))
            block
                real(real32) :: rrms

                rrms = 1.0_real32 / sqrt(sum_squares_f32(x(:, m)) / real(w, real32) + eps)
                do concurrent (integer :: d = 1:w)
                    block
                        real(real32) :: normalized, gt, s

                        normalized = round_bf16(round_bf16(bf16_bits_to_real32(x(d, m)) * rrms) * &
                            bf16_bits_to_real32(weight(d)))
                        gt = bf16_bits_to_real32(z(d, m))
                        s = 1.0_real32/(1.0_real32 + exp(abs(gt)))
                        if (gt >= 0.0_real32) s = 1.0_real32-s
                        s = gt*s
                        y(d, m) = real32_to_bf16_bits(s * normalized)
                    end block
                end do
                ! The words of the quantized vector (4 values each) are shared out over the lanes; the scale and
                ! the sum of a chunk of 16 are computed again by each word of it (quantize_chunk1 inlined: a call
                ! inside the lanes of a launch makes it serial). Each iteration assigns one element of xw.
                do concurrent (integer :: wd = 1:w/4)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax, inv, b0, b1, b2, b3
                        integer(int32) :: q0, q1, q2, q3
                        integer :: c, e
                        c = (wd-1)/4
                        e = 4*modulo(wd-1,4)
                        a0 = bf16_bits_to_real32(y(16*c+1,m))
                        a1 = bf16_bits_to_real32(y(16*c+2,m))
                        a2 = bf16_bits_to_real32(y(16*c+3,m))
                        a3 = bf16_bits_to_real32(y(16*c+4,m))
                        a4 = bf16_bits_to_real32(y(16*c+5,m))
                        a5 = bf16_bits_to_real32(y(16*c+6,m))
                        a6 = bf16_bits_to_real32(y(16*c+7,m))
                        a7 = bf16_bits_to_real32(y(16*c+8,m))
                        a8 = bf16_bits_to_real32(y(16*c+9,m))
                        a9 = bf16_bits_to_real32(y(16*c+10,m))
                        a10 = bf16_bits_to_real32(y(16*c+11,m))
                        a11 = bf16_bits_to_real32(y(16*c+12,m))
                        a12 = bf16_bits_to_real32(y(16*c+13,m))
                        a13 = bf16_bits_to_real32(y(16*c+14,m))
                        a14 = bf16_bits_to_real32(y(16*c+15,m))
                        a15 = bf16_bits_to_real32(y(16*c+16,m))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        inv = 0.0_real32
                        if (amax > 0.0_real32) inv = 127.0_real32/amax
                        b0 = bf16_bits_to_real32(y(16*c+e+1,m))
                        b1 = bf16_bits_to_real32(y(16*c+e+2,m))
                        b2 = bf16_bits_to_real32(y(16*c+e+3,m))
                        b3 = bf16_bits_to_real32(y(16*c+e+4,m))
                        q0 = nint(b0*inv)
                        q1 = nint(b1*inv)
                        q2 = nint(b2*inv)
                        q3 = nint(b3*inv)
                        xw((m-1)*(w/4)+wd) = ior(ior(iand(q0,255),ishft(iand(q1,255),8)), &
                            ior(ishft(iand(q2,255),16),ishft(iand(q3,255),24)))
                    end block
                end do
                do concurrent (integer :: j = 1:w/16)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax
                        integer :: c
                        c = j-1
                        a0 = bf16_bits_to_real32(y(16*c+1,m))
                        a1 = bf16_bits_to_real32(y(16*c+2,m))
                        a2 = bf16_bits_to_real32(y(16*c+3,m))
                        a3 = bf16_bits_to_real32(y(16*c+4,m))
                        a4 = bf16_bits_to_real32(y(16*c+5,m))
                        a5 = bf16_bits_to_real32(y(16*c+6,m))
                        a6 = bf16_bits_to_real32(y(16*c+7,m))
                        a7 = bf16_bits_to_real32(y(16*c+8,m))
                        a8 = bf16_bits_to_real32(y(16*c+9,m))
                        a9 = bf16_bits_to_real32(y(16*c+10,m))
                        a10 = bf16_bits_to_real32(y(16*c+11,m))
                        a11 = bf16_bits_to_real32(y(16*c+12,m))
                        a12 = bf16_bits_to_real32(y(16*c+13,m))
                        a13 = bf16_bits_to_real32(y(16*c+14,m))
                        a14 = bf16_bits_to_real32(y(16*c+15,m))
                        a15 = bf16_bits_to_real32(y(16*c+16,m))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        sums((m-1)*(w/16)+j) = (a0 + a2 + a4 + a6 + a8 + a10 + a12 + a14) + (a1 + a3 + a5 + a7 + a9 + a11 + a13 + a15)
                        sx((m-1)*(w/16)+j) = amax/127.0_real32
                    end block
                end do
            end block
        end do
    end subroutine gated_rmsnorm_q_bf16

    pure subroutine gated_rmsnorm_q_delta_bf16(dl, v, g, beta, finite, pd, pk, qk, token, hk, x, z, y, eps, weight, xw, sx, sums)
        ! The head of a column: the output of the recurrent step (from the partial sums dl of
        ! gated_delta_lazy_bf16) is made here, and the pending update of the state for the next token is left.
        real(real32), contiguous, intent(in) :: dl(:, :, :), g(:, :)
        integer(int16), contiguous, intent(in) :: v(:, :), beta(:, :), qk(:, :)
        logical, contiguous, intent(inout) :: finite(:, :)
        real(real32), contiguous, intent(inout) :: pd(:, :)
        integer(int16), contiguous, intent(inout) :: pk(:, :)
        integer, intent(in) :: token, hk
        integer, parameter :: np = 4
        integer(int16), contiguous, intent(inout) :: x(:, :)
        integer(int16), contiguous, intent(in) :: z(:, :), weight(:)
        integer(int16), contiguous, intent(out) :: y(:, :)
        integer(int32), contiguous, intent(out) :: xw(:)
        real(real32), contiguous, intent(out) :: sx(:), sums(:)
        real(real32), intent(in) :: eps
        integer :: w, rep

        call require_norm_shape(x, y, eps)
        call require_size(size(dl, kind=int64))
        w = size(x, 1)
        if (any(shape(z) /= shape(x)) .or. size(weight) /= w .or. modulo(w, 64) /= 0 .or. &
            size(sums) < size(x) / 16 .or. size(sx) < size(x) / 16 .or. size(xw) < size(x) / 4) &
            error stop "gated rmsnorm q shape mismatch"
        if (size(x, 2) /= size(g, 1) .or. size(pd, 1) /= size(x, 1) + 1 .or. size(pd, 2) /= size(x, 2) .or. modulo(size(x, 2), hk) /= 0 .or. &
            size(x, 1) /= size(pk, 1)) error stop "gated rmsnorm q delta shape mismatch"
        rep = size(x, 2) / hk
        do concurrent (integer :: m = 1:size(x, 2))
            block
                real(real32) :: rrms

                do concurrent (integer :: d = 1:w)
                    block
                        real(real32) :: at, bt, kqt, gg, delta
                        integer :: p

                        gg = g(m,token)
                        at = 0.0_real32
                        bt = 0.0_real32
                        kqt = 0.0_real32
                        do p = 1, np
                            at = at + dl(d,m,p)
                            bt = bt + dl(d,m,np+p)
                            kqt = kqt + dl(d,m,2*np+p)
                        end do
                        delta = (bf16_bits_to_real32(v((m-1)*w+d, token)) - gg*at) * &
                            bf16_bits_to_real32(beta(m,token))
                        x(d,m) = real32_to_bf16_bits(gg*bt + delta*kqt)
                        pd(d,m) = delta
                        if (d == 1) pd(w+1,m) = gg
                        if (modulo(m-1, rep) == 0) pk(d,(m-1)/rep+1) = qk(d,(token-1)*2*hk+(m-1)/rep+1+hk)
                        finite(d,m) = finite(d,m) .and. ieee_is_finite(at) .and. &
                            ieee_is_finite(bt) .and. ieee_is_finite(delta)
                    end block
                end do
                rrms = 1.0_real32 / sqrt(sum_squares_f32(x(:, m)) / real(w, real32) + eps)
                do concurrent (integer :: d = 1:w)
                    block
                        real(real32) :: normalized, gt, s

                        normalized = round_bf16(round_bf16(bf16_bits_to_real32(x(d, m)) * rrms) * &
                            bf16_bits_to_real32(weight(d)))
                        gt = bf16_bits_to_real32(z(d, m))
                        s = 1.0_real32/(1.0_real32 + exp(abs(gt)))
                        if (gt >= 0.0_real32) s = 1.0_real32-s
                        s = gt*s
                        y(d, m) = real32_to_bf16_bits(s * normalized)
                    end block
                end do
                ! The words of the quantized vector (4 values each) are shared out over the lanes; the scale and
                ! the sum of a chunk of 16 are computed again by each word of it (quantize_chunk1 inlined: a call
                ! inside the lanes of a launch makes it serial). Each iteration assigns one element of xw.
                do concurrent (integer :: wd = 1:w/4)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax, inv, b0, b1, b2, b3
                        integer(int32) :: q0, q1, q2, q3
                        integer :: c, e
                        c = (wd-1)/4
                        e = 4*modulo(wd-1,4)
                        a0 = bf16_bits_to_real32(y(16*c+1,m))
                        a1 = bf16_bits_to_real32(y(16*c+2,m))
                        a2 = bf16_bits_to_real32(y(16*c+3,m))
                        a3 = bf16_bits_to_real32(y(16*c+4,m))
                        a4 = bf16_bits_to_real32(y(16*c+5,m))
                        a5 = bf16_bits_to_real32(y(16*c+6,m))
                        a6 = bf16_bits_to_real32(y(16*c+7,m))
                        a7 = bf16_bits_to_real32(y(16*c+8,m))
                        a8 = bf16_bits_to_real32(y(16*c+9,m))
                        a9 = bf16_bits_to_real32(y(16*c+10,m))
                        a10 = bf16_bits_to_real32(y(16*c+11,m))
                        a11 = bf16_bits_to_real32(y(16*c+12,m))
                        a12 = bf16_bits_to_real32(y(16*c+13,m))
                        a13 = bf16_bits_to_real32(y(16*c+14,m))
                        a14 = bf16_bits_to_real32(y(16*c+15,m))
                        a15 = bf16_bits_to_real32(y(16*c+16,m))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        inv = 0.0_real32
                        if (amax > 0.0_real32) inv = 127.0_real32/amax
                        b0 = bf16_bits_to_real32(y(16*c+e+1,m))
                        b1 = bf16_bits_to_real32(y(16*c+e+2,m))
                        b2 = bf16_bits_to_real32(y(16*c+e+3,m))
                        b3 = bf16_bits_to_real32(y(16*c+e+4,m))
                        q0 = nint(b0*inv)
                        q1 = nint(b1*inv)
                        q2 = nint(b2*inv)
                        q3 = nint(b3*inv)
                        xw((m-1)*(w/4)+wd) = ior(ior(iand(q0,255),ishft(iand(q1,255),8)), &
                            ior(ishft(iand(q2,255),16),ishft(iand(q3,255),24)))
                    end block
                end do
                do concurrent (integer :: j = 1:w/16)
                    block
                        real(real32) :: a0
                        real(real32) :: a1
                        real(real32) :: a2
                        real(real32) :: a3
                        real(real32) :: a4
                        real(real32) :: a5
                        real(real32) :: a6
                        real(real32) :: a7
                        real(real32) :: a8
                        real(real32) :: a9
                        real(real32) :: a10
                        real(real32) :: a11
                        real(real32) :: a12
                        real(real32) :: a13
                        real(real32) :: a14
                        real(real32) :: a15
                        real(real32) :: amax
                        integer :: c
                        c = j-1
                        a0 = bf16_bits_to_real32(y(16*c+1,m))
                        a1 = bf16_bits_to_real32(y(16*c+2,m))
                        a2 = bf16_bits_to_real32(y(16*c+3,m))
                        a3 = bf16_bits_to_real32(y(16*c+4,m))
                        a4 = bf16_bits_to_real32(y(16*c+5,m))
                        a5 = bf16_bits_to_real32(y(16*c+6,m))
                        a6 = bf16_bits_to_real32(y(16*c+7,m))
                        a7 = bf16_bits_to_real32(y(16*c+8,m))
                        a8 = bf16_bits_to_real32(y(16*c+9,m))
                        a9 = bf16_bits_to_real32(y(16*c+10,m))
                        a10 = bf16_bits_to_real32(y(16*c+11,m))
                        a11 = bf16_bits_to_real32(y(16*c+12,m))
                        a12 = bf16_bits_to_real32(y(16*c+13,m))
                        a13 = bf16_bits_to_real32(y(16*c+14,m))
                        a14 = bf16_bits_to_real32(y(16*c+15,m))
                        a15 = bf16_bits_to_real32(y(16*c+16,m))
                        amax = max(max(max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3))), max(max(abs(a4), abs(a5)), max(abs(a6), abs(a7)))), max(max(max(abs(a8), abs(a9)), max(abs(a10), abs(a11))), max(max(abs(a12), abs(a13)), max(abs(a14), abs(a15)))))
                        sums((m-1)*(w/16)+j) = (a0 + a2 + a4 + a6 + a8 + a10 + a12 + a14) + (a1 + a3 + a5 + a7 + a9 + a11 + a13 + a15)
                        sx((m-1)*(w/16)+j) = amax/127.0_real32
                    end block
                end do
            end block
        end do
    end subroutine gated_rmsnorm_q_delta_bf16

    ! residual_add_bf16 of delta into the columns of x, followed by rmsnorm_bf16
    ! of the new columns into y.
    pure subroutine residual_rmsnorm_bf16(x, delta, y, eps, weight)
        integer(int16), contiguous, intent(inout) :: x(:, :)
        integer(int16), contiguous, intent(in) :: delta(:, :), weight(:)
        integer(int16), contiguous, intent(out) :: y(:, :)
        real(real32), intent(in) :: eps

        call require_norm_shape(x, y, eps)
        if (any(shape(delta) /= shape(x)) .or. size(weight, kind=int64) /= size(x, 1, kind=int64)) &
            error stop "residual rmsnorm shape mismatch"
        do concurrent (integer :: m = 1:size(x, 2))
            block
                real(real32) :: rrms

                rrms = 1.0_real32 / sqrt(sum(bf16_bits_to_real32(real32_to_bf16_bits( &
                    bf16_bits_to_real32(x(:, m)) + bf16_bits_to_real32(delta(:, m))))**2) / &
                    real(size(x, 1), real32) + eps)
                do concurrent (integer :: d = 1:size(x, 1))
                    block
                        real(real32) :: normalized
                        integer(int16) :: sum_bits

                        sum_bits = real32_to_bf16_bits(bf16_bits_to_real32(x(d, m)) + bf16_bits_to_real32(delta(d, m)))
                        x(d, m) = sum_bits
                        normalized = round_bf16(bf16_bits_to_real32(sum_bits) * rrms) * bf16_bits_to_real32(weight(d))
                        y(d, m) = real32_to_bf16_bits(normalized)
                    end block
                end do
            end block
        end do
    end subroutine residual_rmsnorm_bf16

    pure subroutine keep_finite_rows_f32(x, finite)
        real(real32), contiguous, intent(in) :: x(:, :, :)
        logical, contiguous, intent(inout) :: finite(:, :)

        ! finite(j,k) stays true only while every x(:,j,k) is finite. Each row
        ! keeps its own flag, so the device checks a state where it is and the
        ! host reads the flags once, when it needs the answer.
        if (size(finite, 1) /= size(x, 2) .or. size(finite, 2) /= size(x, 3)) &
            error stop "finite rows shape mismatch"
        do concurrent (integer :: k = 1:size(x, 3), j = 1:size(x, 2))
            finite(j, k) = finite(j, k) .and. all(ieee_is_finite(x(:, j, k)))
        end do
    end subroutine keep_finite_rows_f32

    pure subroutine require_positions(q_offset, length)
        integer(int64), intent(in) :: q_offset
        integer, intent(in) :: length
        integer(int64), parameter :: max_position = 4294967295_int64

        if (q_offset < 0_int64 .or. q_offset > max_position) error stop "position offset outside uint32 range"
        if (int(length - 1, int64) > max_position - q_offset) error stop "position range exceeds uint32"
    end subroutine require_positions

    ! The cosines and sines of the rotations of one position.
    pure subroutine rope_tables_f32(inv_freqs, q_offset, cosv, sinv)
        real(real32), contiguous, intent(in) :: inv_freqs(:)
        integer(int64), intent(in) :: q_offset
        real(real32), contiguous, intent(out) :: cosv(:), sinv(:)

        if (size(cosv) /= size(inv_freqs) .or. size(sinv) /= size(inv_freqs)) error stop "rope table shape mismatch"
        call require_positions(q_offset, 1)
        do concurrent (integer :: d = 1:size(inv_freqs))
            cosv(d) = cos(real(q_offset, real32) * inv_freqs(d))
            sinv(d) = sin(real(q_offset, real32) * inv_freqs(d))
        end do
    end subroutine rope_tables_f32

    ! rmsnorm_bf16 (with weight) of the key heads of one token, rope_partial_inplace_bf16
    ! of the result and the append of it and of the value heads to the caches
    ! at position q_offset + 1, in one launch of one iteration per head.
    pure subroutine knorm_rope_append_bf16(k, v, weight, cosv, sinv, d_rot, q_offset, eps, k_cache, v_cache)
        integer(int16), contiguous, intent(in) :: k(:), v(:), weight(:)
        real(real32), contiguous, intent(in) :: cosv(:), sinv(:)
        integer, intent(in) :: d_rot
        integer(int64), intent(in) :: q_offset
        real(real32), intent(in) :: eps
        integer(int16), contiguous, intent(inout) :: k_cache(:, :, :), v_cache(:, :, :)
        integer :: hd, half, nh, at

        call require_size(size(k, kind=int64))
        hd = size(weight)
        nh = size(k) / hd
        half = d_rot / 2
        at = int(q_offset) + 1
        if (.not. (eps > 0.0_real32) .or. nh * hd /= size(k) .or. size(v) /= size(k) .or. &
            size(k_cache, 1) /= hd .or. size(k_cache, 2) /= nh .or. any(shape(v_cache) /= shape(k_cache)) .or. &
            at < 1 .or. at > size(k_cache, 3) .or. size(cosv) /= half .or. size(sinv) /= half .or. d_rot < 0 .or. d_rot > hd .or. &
            modulo(d_rot, 2) /= 0) error stop "key norm rope append shape mismatch"
        do concurrent (integer :: h = 1:nh)
            block
                real(real32) :: rrms

                rrms = 1.0_real32 / sqrt(sum_squares_f32(k((h-1)*hd+1:h*hd)) / real(hd, real32) + eps)
                do concurrent (integer :: d = 1:half)
                    block
                        real(real32) :: a, b, c, s

                        c = cosv(d)
                        s = sinv(d)
                        a = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(k((h-1)*hd+d)) * rrms) * bf16_bits_to_real32(weight(d))))
                        b = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(k((h-1)*hd+d+half)) * rrms) * bf16_bits_to_real32(weight(d+half))))
                        k_cache(d, h, at) = real32_to_bf16_bits(a * c - b * s)
                    end block
                end do
                do concurrent (integer :: d = 1:half)
                    block
                        real(real32) :: a, b, c, s

                        c = cosv(d)
                        s = sinv(d)
                        a = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(k((h-1)*hd+d)) * rrms) * bf16_bits_to_real32(weight(d))))
                        b = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(k((h-1)*hd+d+half)) * rrms) * bf16_bits_to_real32(weight(d+half))))
                        k_cache(d+half, h, at) = real32_to_bf16_bits(a * s + b * c)
                    end block
                end do
                do concurrent (integer :: d = d_rot+1:hd)
                    k_cache(d, h, at) = real32_to_bf16_bits(round_bf16( &
                        bf16_bits_to_real32(k((h-1)*hd+d)) * rrms) * bf16_bits_to_real32(weight(d)))
                end do
                do concurrent (integer :: d = 1:hd)
                    v_cache(d, h, at) = v((h-1)*hd+d)
                end do
            end block
        end do
    end subroutine knorm_rope_append_bf16

    ! The query heads of one token, each followed by its gate, in q: the gate
    ! of every head to gate, and rmsnorm_bf16 (with weight) and
    ! rope_partial_inplace_bf16 of the query to qr, in one launch of one iteration per head.
    pure subroutine qnorm_rope_gate_bf16(q, weight, cosv, sinv, d_rot, q_offset, eps, qr, gate)
        integer(int16), contiguous, intent(in) :: q(:), weight(:)
        real(real32), contiguous, intent(in) :: cosv(:), sinv(:)
        integer, intent(in) :: d_rot
        integer(int64), intent(in) :: q_offset
        real(real32), intent(in) :: eps
        integer(int16), contiguous, intent(out) :: qr(:, :), gate(:)
        integer :: hd, half, nh

        call require_size(size(q, kind=int64))
        hd = size(weight)
        nh = size(q) / (2 * hd)
        half = d_rot / 2
        if (.not. (eps > 0.0_real32) .or. 2 * nh * hd /= size(q) .or. size(qr, 1) /= hd .or. size(qr, 2) /= nh .or. &
            size(gate) /= nh * hd .or. size(cosv) /= half .or. size(sinv) /= half .or. d_rot < 0 .or. d_rot > hd .or. &
            modulo(d_rot, 2) /= 0) error stop "query norm rope gate shape mismatch"
        do concurrent (integer :: h = 1:nh)
            block
                real(real32) :: rrms

                rrms = 1.0_real32 / sqrt(sum_squares_f32(q((h-1)*2*hd+1:(h-1)*2*hd+hd)) / real(hd, real32) + eps)
                do concurrent (integer :: d = 1:half)
                    block
                        real(real32) :: a, b, c, s

                        c = cosv(d)
                        s = sinv(d)
                        a = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(q((h-1)*2*hd+d)) * rrms) * bf16_bits_to_real32(weight(d))))
                        b = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(q((h-1)*2*hd+d+half)) * rrms) * bf16_bits_to_real32(weight(d+half))))
                        qr(d,h) = real32_to_bf16_bits(a * c - b * s)
                    end block
                end do
                do concurrent (integer :: d = 1:half)
                    block
                        real(real32) :: a, b, c, s

                        c = cosv(d)
                        s = sinv(d)
                        a = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(q((h-1)*2*hd+d)) * rrms) * bf16_bits_to_real32(weight(d))))
                        b = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(q((h-1)*2*hd+d+half)) * rrms) * bf16_bits_to_real32(weight(d+half))))
                        qr(d+half,h) = real32_to_bf16_bits(a * s + b * c)
                    end block
                end do
                do concurrent (integer :: d = d_rot+1:hd)
                    qr(d,h) = real32_to_bf16_bits(round_bf16( &
                        bf16_bits_to_real32(q((h-1)*2*hd+d)) * rrms) * bf16_bits_to_real32(weight(d)))
                end do
                do concurrent (integer :: d = 1:hd)
                    gate((h-1)*hd+d) = q((h-1)*2*hd+hd+d)
                end do
            end block
        end do
    end subroutine qnorm_rope_gate_bf16

    ! knorm_rope_append_bf16 and qnorm_rope_gate_bf16 in one launch: the first nq iterations are the
    ! query heads, the next nkv the key (and value) heads.
    pure subroutine qknorm_rope_bf16(q, k, v, qweight, kweight, cosv, sinv, d_rot, q_offset, eps, qr, gate, k_cache, v_cache)
        integer(int16), contiguous, intent(in) :: q(:), k(:), v(:), qweight(:), kweight(:)
        real(real32), contiguous, intent(in) :: cosv(:), sinv(:)
        integer, intent(in) :: d_rot
        integer(int64), intent(in) :: q_offset
        real(real32), intent(in) :: eps
        integer(int16), contiguous, intent(out) :: qr(:, :), gate(:)
        integer(int16), contiguous, intent(inout) :: k_cache(:, :, :), v_cache(:, :, :)
        integer :: hd, half, nq, nh, at

        call require_size(size(q, kind=int64))
        call require_size(size(k, kind=int64))
        hd = size(qweight)
        nq = size(q) / (2 * hd)
        nh = size(k) / hd
        half = d_rot / 2
        at = int(q_offset) + 1
        if (.not. (eps > 0.0_real32) .or. size(kweight) /= hd .or. 2 * nq * hd /= size(q) .or. size(qr, 1) /= hd .or. &
            size(qr, 2) /= nq .or. size(gate) /= nq * hd .or. nh * hd /= size(k) .or. size(v) /= size(k) .or. &
            size(k_cache, 1) /= hd .or. size(k_cache, 2) /= nh .or. any(shape(v_cache) /= shape(k_cache)) .or. &
            at < 1 .or. at > size(k_cache, 3) .or. size(cosv) /= half .or. size(sinv) /= half .or. d_rot < 0 .or. &
            d_rot > hd .or. modulo(d_rot, 2) /= 0) error stop "query key norm rope shape mismatch"
        do concurrent (integer :: hh = 1:nq+nh)
            block
                integer :: h

                if (hh <= nq) then
                    h = hh
            block
                real(real32) :: rrms

                rrms = 1.0_real32 / sqrt(sum_squares_f32(q((h-1)*2*hd+1:(h-1)*2*hd+hd)) / real(hd, real32) + eps)
                do concurrent (integer :: d = 1:half)
                    block
                        real(real32) :: a, b, c, s

                        c = cosv(d)
                        s = sinv(d)
                        a = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(q((h-1)*2*hd+d)) * rrms) * bf16_bits_to_real32(qweight(d))))
                        b = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(q((h-1)*2*hd+d+half)) * rrms) * bf16_bits_to_real32(qweight(d+half))))
                        qr(d,h) = real32_to_bf16_bits(a * c - b * s)
                    end block
                end do
                do concurrent (integer :: d = 1:half)
                    block
                        real(real32) :: a, b, c, s

                        c = cosv(d)
                        s = sinv(d)
                        a = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(q((h-1)*2*hd+d)) * rrms) * bf16_bits_to_real32(qweight(d))))
                        b = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(q((h-1)*2*hd+d+half)) * rrms) * bf16_bits_to_real32(qweight(d+half))))
                        qr(d+half,h) = real32_to_bf16_bits(a * s + b * c)
                    end block
                end do
                do concurrent (integer :: d = d_rot+1:hd)
                    qr(d,h) = real32_to_bf16_bits(round_bf16( &
                        bf16_bits_to_real32(q((h-1)*2*hd+d)) * rrms) * bf16_bits_to_real32(qweight(d)))
                end do
                do concurrent (integer :: d = 1:hd)
                    gate((h-1)*hd+d) = q((h-1)*2*hd+hd+d)
                end do
            end block
                else
                    h = hh - nq
            block
                real(real32) :: rrms

                rrms = 1.0_real32 / sqrt(sum_squares_f32(k((h-1)*hd+1:h*hd)) / real(hd, real32) + eps)
                do concurrent (integer :: d = 1:half)
                    block
                        real(real32) :: a, b, c, s

                        c = cosv(d)
                        s = sinv(d)
                        a = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(k((h-1)*hd+d)) * rrms) * bf16_bits_to_real32(kweight(d))))
                        b = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(k((h-1)*hd+d+half)) * rrms) * bf16_bits_to_real32(kweight(d+half))))
                        k_cache(d, h, at) = real32_to_bf16_bits(a * c - b * s)
                    end block
                end do
                do concurrent (integer :: d = 1:half)
                    block
                        real(real32) :: a, b, c, s

                        c = cosv(d)
                        s = sinv(d)
                        a = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(k((h-1)*hd+d)) * rrms) * bf16_bits_to_real32(kweight(d))))
                        b = bf16_bits_to_real32(real32_to_bf16_bits(round_bf16( &
                            bf16_bits_to_real32(k((h-1)*hd+d+half)) * rrms) * bf16_bits_to_real32(kweight(d+half))))
                        k_cache(d+half, h, at) = real32_to_bf16_bits(a * s + b * c)
                    end block
                end do
                do concurrent (integer :: d = d_rot+1:hd)
                    k_cache(d, h, at) = real32_to_bf16_bits(round_bf16( &
                        bf16_bits_to_real32(k((h-1)*hd+d)) * rrms) * bf16_bits_to_real32(kweight(d)))
                end do
                do concurrent (integer :: d = 1:hd)
                    v_cache(d, h, at) = v((h-1)*hd+d)
                end do
            end block
                end if
            end block
        end do
    end subroutine qknorm_rope_bf16

    pure subroutine rope_partial_inplace_bf16(x, inv_freqs, d_rot, q_offset)
        integer(int16), contiguous, intent(inout) :: x(:, :, :)
        real(real32), contiguous, intent(in) :: inv_freqs(:)
        integer, intent(in) :: d_rot
        integer(int64), intent(in) :: q_offset
        integer :: half

        call require_size(size(x, kind=int64))
        if (d_rot < 0 .or. d_rot > size(x, 1) .or. modulo(d_rot, 2) /= 0) &
            error stop "rope requires even rotation width in 0..D"
        half = d_rot / 2
        if (size(inv_freqs) /= half) error stop "rope frequency shape mismatch"
        call require_positions(q_offset, size(x, 3))
        do concurrent (integer :: l = 1:size(x, 3), h = 1:size(x, 2))
            block
                integer :: d
                real(real32) :: pos, c, s, a, b

                pos = real(q_offset + int(l - 1, int64), real32)
                do d = 1, half
                    c = cos(pos * inv_freqs(d))
                    s = sin(pos * inv_freqs(d))
                    a = bf16_bits_to_real32(x(d, h, l))
                    b = bf16_bits_to_real32(x(d + half, h, l))
                    x(d, h, l) = real32_to_bf16_bits(a * c - b * s)
                    x(d + half, h, l) = real32_to_bf16_bits(a * s + b * c)
                end do
            end block
        end do
    end subroutine rope_partial_inplace_bf16

    pure integer function native_attention_blocks(rows, length, gqa, width) result(blocks)
        integer, intent(in) :: rows, length, gqa, width

        if (min(rows,length,gqa,width) < 1) error stop "invalid native attention shape"
        blocks = 0
        if (rows > 8 .or. int(rows,int64)*gqa > 32 .or. &
                .not. any(width == [64,96,128,256])) return
        blocks = 1
        if (length < 1024) return
        blocks = 64
        if (length > 1024) blocks = 128
        if (length > 8192) blocks = 256
        if (length > 32768) blocks = 512
        if (length > 65536) blocks = 1024
    end function native_attention_blocks

    ! sdpa_gqa_bf16 of a single query token that sees the first `length`
    ! positions of the cache (no rounded scores, as in the native attention
    ! path): one launch for the scores of all heads and positions, one for the
    ! softmax and the weighted values. scores(length, nq) is workspace.
    pure subroutine sdpa_decode_bf16(q, k, v, out, scores, scale, length)
        integer(int16), contiguous, intent(in) :: q(:, :, :), k(:, :, :), v(:, :, :)
        integer(int16), contiguous, intent(out) :: out(:, :, :)
        real(real32), contiguous, intent(inout) :: scores(:, :)
        real(real32), intent(in) :: scale
        integer, intent(in) :: length
        integer :: d_head, nq, nkv, group

        call require_size(size(k, kind=int64))
        d_head = size(q, 1)
        nq = size(q, 2)
        nkv = size(k, 2)
        if (modulo(nq, nkv) /= 0) error stop "sdpa requires Nq divisible by Nkv"
        if (size(k, 1) /= d_head .or. any(shape(v) /= shape(k)) .or. any(shape(out) /= shape(q)) .or. size(q, 3) /= 1 .or. &
                length < 1 .or. length > size(k, 3) .or. size(scores, 1) < length + 1 .or. &
                size(scores, 2) /= nq) error stop "sdpa shape mismatch"
        group = nq / nkv
        do concurrent (integer :: h = 1:nq, t = 1:length)
            block
                integer :: kh, d
                real(real32) :: s

                kh = (h - 1) / group + 1
                s = 0.0_real32
                do d = 1, d_head
                    s = s + bf16_bits_to_real32(q(d,h,1))*scale*bf16_bits_to_real32(k(d,kh,t))
                end do
                scores(t,h) = s
            end block
        end do
        do concurrent (integer :: h = 1:nq)
            block
                integer :: u
                real(real32) :: max_s, total

                max_s = maxval(scores(1:length,h))
                do concurrent (integer :: t = 1:length)
                    scores(t,h) = exp(scores(t,h) - max_s)
                end do
                total = sum(scores(1:length,h))
                scores(length+1,h) = total
            end block
        end do
        do concurrent (integer :: h = 1:nq, i = 1:d_head)
            block
                integer :: kh, u
                real(real32) :: acc

                kh = (h - 1) / group + 1
                acc = 0.0_real32
                do u = 1, length
                    acc = acc + scores(u,h)*bf16_bits_to_real32(v(i,kh,u))
                end do
                out(i,h,1) = real32_to_bf16_bits(acc/scores(length+1,h))
            end block
        end do
    end subroutine sdpa_decode_bf16

    pure subroutine sdpa_gqa_bf16(q, k, v, out, q_offset, scale, causal, length)
        integer(int16), contiguous, intent(in) :: q(:, :, :), k(:, :, :), v(:, :, :)
        integer(int16), contiguous, intent(out) :: out(:, :, :)
        integer(int64), intent(in) :: q_offset
        real(real32), intent(in) :: scale
        logical, intent(in) :: causal
        integer, intent(in), optional :: length
        integer :: d_head, nq, nkv, lq, lk, group, blocks
        logical :: rounded_scores
        real(real32) :: input_scale

        call require_size(size(q, kind=int64))
        call require_size(size(k, kind=int64))
        call require_size(size(v, kind=int64))
        call require_size(size(out, kind=int64))
        d_head = size(q, 1)
        nq = size(q, 2)
        nkv = size(k, 2)
        lq = size(q, 3)
        lk = size(k, 3)
        ! k and v may be a longer cache of which only the first `length` positions are valid.
        if (present(length)) then
            if (length < 1 .or. length > lk) error stop "sdpa length out of range"
            lk = length
        end if
        if (modulo(nq, nkv) /= 0) error stop "sdpa requires Nq divisible by Nkv"
        if (size(k, 1) /= d_head .or. any(shape(v) /= shape(k)) .or. any(shape(out) /= shape(q))) &
            error stop "sdpa shape mismatch"
        call require_positions(q_offset, lq)
        group = nq / nkv
        blocks = native_attention_blocks(lq,lk,group,d_head)
        rounded_scores = blocks == 0
        input_scale = round_bf16(scale)
        do concurrent (integer :: token = 1:lq, h = 1:nq)
            block
                integer :: kh, visible, d, t, c
                integer(int64) :: q_abs
                real(real32) :: scores(lk), max_s, total, qv(d_head), s

                kh = (h - 1) / group + 1
                q_abs = q_offset + int(token - 1, int64)
                visible = lk
                if (causal) visible = int(min(int(lk,int64),q_abs+1))
                do d = 1, d_head
                    qv(d) = bf16_bits_to_real32(q(d,h,token))*scale
                    if (rounded_scores) qv(d) = round_bf16(bf16_bits_to_real32(q(d,h,token))*input_scale)
                end do
                do t = 1, visible
                    s = 0.0_real32
                    do d = 1, d_head
                        s = s + qv(d)*bf16_bits_to_real32(k(d,kh,t))
                    end do
                    scores(t) = s
                    if (rounded_scores) scores(t) = round_bf16(s)
                end do
                if (blocks > 1) then
                    block
                        real(real32) :: sums(blocks), maxima(blocks), factors(blocks)
                        real(real32) :: pacc(d_head,blocks)
                        integer(int16) :: partials(d_head,blocks)

                        do concurrent (integer :: b = 1:blocks)
                            block
                                integer :: i, cc
                                real(real32) :: old_max, new_max, factor, e

                                old_max = -huge(0.0_real32)
                                sums(b) = 0.0_real32
                                do i = 1, d_head
                                    pacc(i,b) = 0.0_real32
                                end do
                                do i = b, visible, blocks
                                    new_max = max(old_max,scores(i))
                                    factor = exp(old_max-new_max)
                                    e = exp(scores(i)-new_max)
                                    sums(b) = sums(b)*factor+e
                                    do cc = 1, d_head
                                        pacc(cc,b) = pacc(cc,b)*factor+e*bf16_bits_to_real32(v(cc,kh,i))
                                    end do
                                    old_max = new_max
                                end do
                                maxima(b) = old_max
                                do i = 1, d_head
                                    partials(i,b) = real32_to_bf16_bits(pacc(i,b))
                                end do
                            end block
                        end do
                        max_s = maxima(1)
                        do c = 2, blocks
                            max_s = max(max_s,maxima(c))
                        end do
                        total = 0.0_real32
                        do c = 1, blocks
                            factors(c) = exp(maxima(c)-max_s)
                            total = total + sums(c)*factors(c)
                        end do
                        do concurrent (integer :: i = 1:d_head)
                            block
                                integer :: b
                                real(real32) :: acc
                                acc = 0.0_real32
                                do b = 1, blocks
                                    acc = acc + bf16_bits_to_real32(partials(i,b))*factors(b)
                                end do
                                out(i,h,token) = real32_to_bf16_bits(acc/total)
                            end block
                        end do
                    end block
                else
                    max_s = scores(1)
                    do t = 2, visible
                        max_s = max(max_s,scores(t))
                    end do
                    total = 0.0_real32
                    do t = 1, visible
                        scores(t) = exp(scores(t)-max_s)
                        total = total + scores(t)
                    end do
                    if (rounded_scores) then
                        do t = 1, visible
                            scores(t) = round_bf16(scores(t)/total)
                        end do
                    end if
                    do concurrent (integer :: i = 1:d_head)
                        block
                            integer :: u
                            real(real32) :: acc
                            acc = 0.0_real32
                            do u = 1, visible
                                acc = acc + scores(u)*bf16_bits_to_real32(v(i,kh,u))
                            end do
                            if (.not. rounded_scores) acc = acc/total
                            out(i,h,token) = real32_to_bf16_bits(acc)
                        end block
                    end do
                end if
            end block
        end do
    end subroutine sdpa_gqa_bf16

    ! The logits are normalised (the log-sum-exp subtracted and rounded to
    ! BF16) and the index of the first largest one is taken, as passes over the
    ! logits in launches that follow one another on the device with no host
    ! access in between: the maximum, the sum of the exponentials, and the
    ! largest normalised logit, each over 2048 chunks and then the 64 groups of
    ! 32 chunks that those make up. The launch after a pass reads the results
    ! of the pass before directly, so the host waits only for the index.
    pure subroutine sample_native_bf16(logits, index, logprobs)
        integer(int16), contiguous, intent(in) :: logits(:)
        integer(int16), contiguous, optional, intent(out) :: logprobs(:)
        integer(int32), intent(out) :: index
        integer, parameter :: nc = 2048, cg = 32, ng = 64
        real(real32), allocatable :: cmax(:), csum(:), cbest(:)
        integer(int32), allocatable :: cidx(:), cbad(:)
        ! The results of the groups (maximum, sum, best, its index, count of non-finite) and the
        ! index are in one array, so that the host copies one array back, not seven (indices are
        ! exact in a real(real32) up to 2**24).
        real(real32), allocatable :: gw(:)
        integer :: n, chunk, used, groups

        call require_size(size(logits,kind=int64))
        n = size(logits)
        if (present(logprobs)) then
            if (size(logprobs) /= n) error stop "sampler output shape mismatch"
        end if
        chunk = (n + nc - 1)/nc
        used = (n + chunk - 1)/chunk
        groups = (used + cg - 1)/cg
        allocate(cmax(nc), csum(nc), cbest(nc), cidx(nc), cbad(nc), gw(5*ng+1))
        do concurrent (integer :: k = 1:used)
            block
                integer :: i, lo, hi
                real(real32) :: m
                integer :: bad
                lo = (k-1)*chunk + 1
                hi = min(k*chunk, n)
                m = bf16_bits_to_real32(logits(lo))
                bad = 0
                do i = lo, hi
                    m = max(m, bf16_bits_to_real32(logits(i)))
                    if (.not. ieee_is_finite(bf16_bits_to_real32(logits(i)))) bad = bad + 1
                end do
                cmax(k) = m
                cbad(k) = bad
            end block
        end do
        do concurrent (integer :: g = 1:groups)
            block
                integer :: c, lo, hi, bad
                real(real32) :: m
                lo = (g-1)*cg + 1
                hi = min(g*cg, used)
                m = cmax(lo)
                bad = 0
                do c = lo, hi
                    m = max(m, cmax(c))
                    bad = bad + cbad(c)
                end do
                gw(g) = m
                gw(4*ng+g) = real(bad, real32)
            end block
        end do
        do concurrent (integer :: k = 1:used)
            block
                integer :: i, lo, hi, g
                real(real32) :: maximum, t
                maximum = gw(1)
                do g = 2, groups
                    maximum = max(maximum, gw(g))
                end do
                lo = (k-1)*chunk + 1
                hi = min(k*chunk, n)
                t = 0.0_real32
                do i = lo, hi
                    t = t + exp(bf16_bits_to_real32(logits(i)) - maximum)
                end do
                csum(k) = t
            end block
        end do
        do concurrent (integer :: g = 1:groups)
            block
                integer :: c, lo, hi
                real(real32) :: t
                lo = (g-1)*cg + 1
                hi = min(g*cg, used)
                t = 0.0_real32
                do c = lo, hi
                    t = t + csum(c)
                end do
                gw(ng+g) = t
            end block
        end do
        do concurrent (integer :: k = 1:used)
            block
                integer :: i, lo, hi, at, g
                real(real32) :: maximum, total, lse, value, top
                integer(int16) :: r
                maximum = gw(1)
                total = gw(ng+1)
                do g = 2, groups
                    maximum = max(maximum, gw(g))
                    total = total + gw(ng+g)
                end do
                lse = round_bf16(maximum+log(total))
                lo = (k-1)*chunk + 1
                hi = min(k*chunk, n)
                at = lo
                top = bf16_bits_to_real32(real32_to_bf16_bits(bf16_bits_to_real32(logits(lo)) - lse))
                do i = lo, hi
                    r = real32_to_bf16_bits(bf16_bits_to_real32(logits(i)) - lse)
                    if (present(logprobs)) logprobs(i) = r
                    value = bf16_bits_to_real32(r)
                    if (value > top) then
                        top = value
                        at = i
                    end if
                end do
                cbest(k) = top
                cidx(k) = int(at - 1, int32)
            end block
        end do
        do concurrent (integer :: g = 1:groups)
            block
                integer :: c, lo, hi
                real(real32) :: top
                integer(int32) :: at
                lo = (g-1)*cg + 1
                hi = min(g*cg, used)
                top = cbest(lo)
                at = cidx(lo)
                do c = lo, hi
                    if (cbest(c) > top) then
                        top = cbest(c)
                        at = cidx(c)
                    end if
                end do
                gw(2*ng+g) = top
                gw(3*ng+g) = real(at, real32)
            end block
        end do
        do concurrent (integer :: k = 1:1)
            block
                integer :: g, bad
                real(real32) :: top
                integer(int32) :: at
                top = gw(2*ng+1)
                at = int(gw(3*ng+1), int32)
                bad = int(gw(4*ng+1))
                do g = 2, groups
                    bad = bad + int(gw(4*ng+g))
                    if (gw(2*ng+g) > top) then
                        top = gw(2*ng+g)
                        at = int(gw(3*ng+g), int32)
                    end if
                end do
                if (bad /= 0) error stop 'forward: non-finite logits'
                gw(5*ng+1) = real(at, real32)
            end block
        end do
        index = int(gw(5*ng+1), int32)
    end subroutine sample_native_bf16

    pure function argmax_bf16(logits) result(index)
        integer(int16), contiguous, intent(in) :: logits(:)
        integer(int32) :: index
        real(real32) :: best, value
        integer :: i

        call require_size(size(logits, kind=int64))
        if (size(logits, kind=int64) > int(huge(index), int64)) error stop "argmax index exceeds int32"
        index = 0_int32
        best = bf16_bits_to_real32(logits(1))
        do i = 2, size(logits)
            value = bf16_bits_to_real32(logits(i))
            if (value > best) then
                best = value
                index = int(i - 1, int32)
            end if
        end do
    end function argmax_bf16

end module dense_kernels
