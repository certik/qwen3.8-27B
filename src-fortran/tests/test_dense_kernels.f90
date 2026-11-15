program test_dense_kernels
    use, intrinsic :: iso_fortran_env, only: int8, int16, int32, int64, real32, error_unit, iostat_end
    use precision_formats, only: require_precision_environment, real32_to_bf16_bits
    use dense_kernels
    implicit none
    character(len=4096) :: path
    integer :: unit, status, records, seen(11)
    integer(int32) :: header(3), op, dims(4), footer(2)
    integer(int16) :: marker
    integer(int8) :: tail
    real(real32) :: real_marker
    integer(int64) :: comparisons, fp32_comparisons

    call require_precision_environment()
    if (command_argument_count() /= 1) error stop "usage: test_dense_kernels oracle.bin"
    call get_command_argument(1, path, status=status)
    if (status /= 0) error stop "cannot read complete dense C oracle path"
    if (index(path, "--") == 1) then
        call reject_case(trim(path))
        error stop "negative test unexpectedly returned"
    end if
    open(newunit=unit, file=trim(path), access="stream", form="unformatted", status="old", &
        action="read", iostat=status)
    if (status /= 0) error stop "cannot open dense C oracle"
    read(unit, iostat=status) header, marker, real_marker
    call require_read(status)
    if (any(header /= [int(z'464b3031', int32), int(z'01020304', int32), int(z'3f800000', int32)]) .or. &
            marker /= ibset(int(z'0123', int16), 15) .or. &
            transfer(real_marker, 0_int32) /= int(z'3fa00000', int32)) &
        error stop "dense C oracle native representation mismatch"
    records = 0
    comparisons = 0_int64
    fp32_comparisons = 0_int64
    seen = 0
    do
        read(unit, iostat=status) op
        call require_read(status)
        if (op == 0_int32) exit
        if (op < 1 .or. op > 11) error stop "unknown dense C oracle operation"
        read(unit, iostat=status) dims
        call require_read(status)
        if (any(dims < 0_int32) .or. dims(1) < 1 .or. dims(2) < 1 .or. dims(4) > 1) &
            error stop "invalid dense C oracle dimensions"
        if (int(dims(1), int64) * int(dims(2), int64) > 2000000_int64 .or. &
                int(dims(1), int64) * int(dims(3), int64) > 2000000_int64 .or. &
                int(dims(2), int64) * int(dims(3), int64) > 2000000_int64) &
            error stop "dense C oracle fixture too large"
        records = records + 1
        seen(op) = seen(op) + 1
        select case (op)
        case (1)
            call embedding_case(int(dims(1)), int(dims(2)), int(dims(3)))
        case (2)
            call linear_case(int(dims(1)), int(dims(2)), int(dims(3)), dims(4) == 1)
        case (3, 4)
            call norm_case(int(dims(1)), int(dims(2)), dims(4) == 1, op == 4)
        case (5)
            call element_case(int(dims(1)))
        case (6)
            call argmax_case(int(dims(1)))
        case (7)
            call decay_case(int(dims(1)), int(dims(2)), dims(4) == 1)
        case (8)
            call conv_case(int(dims(1)), int(dims(2)), int(dims(3)))
        case (9)
            call rope_case(int(dims(1)), int(dims(2)), int(dims(3)))
        case (10)
            call attention_case(int(dims(1)), int(dims(2)), int(dims(3)), dims(4) == 1)
        case (11)
            call recurrent_case(int(dims(1)), int(dims(2)), int(dims(3)))
        end select
    end do
    read(unit, iostat=status) footer
    call require_read(status)
    if (records /= footer(1) .or. comparisons /= int(footer(2), int64) .or. &
            any(seen /= [2, 22, 14, 7, 1, 5, 4, 4, 5, 18, 4])) error stop "incomplete dense C oracle corpus"
    read(unit, iostat=status) tail
    if (status /= iostat_end) error stop "unexpected data after dense C oracle footer"
    close(unit, iostat=status)
    if (status /= 0) error stop "cannot close dense C oracle"
    print '(a,i0,a,i0,a)', "PASS: ", records, " C-oracle records, ", comparisons, " bit-exact output comparisons"
    print '(a)', "PASS: q8 vector/matrix profiles, split-K folds, RMS rounding, 6 elementwise ops, first-index argmax"
    print '(a,i0,a)', "PASS: ", fp32_comparisons, " exact FP32 g/state comparisons; max absolute error=0, ULP error=0"
    print '(a)', "PASS: causal/GQA SDPA profiles/cache prefixes, partial RoPE chunks, " // &
        "convolution carry/reset, recurrent row ownership"

contains

    subroutine require_read(io_status)
        integer, intent(in) :: io_status

        if (io_status /= 0) error stop "missing/truncated dense C oracle"
    end subroutine require_read

    subroutine check(got, expected, label)
        integer(int16), intent(in) :: got(:, :), expected(:, :)
        character(len=*), intent(in) :: label
        integer :: i, j

        do j = 1, size(got, 2)
            do i = 1, size(got, 1)
                if (got(i, j) /= expected(i, j)) then
                    write(error_unit, '(a,a,a,i0,a,i0,a,i0,a,z4.4,a,z4.4)') &
                        "C oracle mismatch: ", label, " record=", records, " row=", j, " column=", i, &
                        " got=0x", got(i, j), " expected=0x", expected(i, j)
                    error stop 1
                end if
            end do
        end do
        comparisons = comparisons + size(got, kind=int64)
    end subroutine check

    subroutine check_f32(got, expected, label)
        real(real32), intent(in) :: got(:, :), expected(:, :)
        character(len=*), intent(in) :: label
        integer :: i, j
        integer(int32) :: got_bits, expected_bits

        do j = 1, size(got, 2)
            do i = 1, size(got, 1)
                got_bits = transfer(got(i, j), 0_int32)
                expected_bits = transfer(expected(i, j), 0_int32)
                if (got_bits /= expected_bits) then
                    write(error_unit, '(a,a,a,i0,a,i0,a,i0,a,z8.8,a,z8.8)') &
                        "C oracle mismatch: ", label, " record=", records, " row=", j, " column=", i, &
                        " got=0x", got_bits, " expected=0x", expected_bits
                    error stop 1
                end if
            end do
        end do
        fp32_comparisons = fp32_comparisons + size(got, kind=int64)
        comparisons = comparisons + size(got, kind=int64)
    end subroutine check_f32

    subroutine embedding_case(d, m, n)
        integer, intent(in) :: d, m, n
        integer(int32), allocatable :: w(:, :), ids(:)
        integer(int16), allocatable :: s(:, :), b(:, :), y(:, :), expected(:, :)

        allocate(w(d / 4, n), s(d / 64, n), b(d / 64, n), ids(m), y(d, m), expected(d, m))
        read(unit, iostat=status) w, s, b, ids, expected
        call require_read(status)
        call embed_dequant_gather_q8(ids, w, s, b, y)
        call check(y, expected, "embedding")
        print '(a,3(i0,1x))', "PASS: embedding D/M/V=", d, m, n
    end subroutine embedding_case

    subroutine linear_case(k, m, n, has_bias)
        integer, intent(in) :: k, m, n
        logical, intent(in) :: has_bias
        integer(int32), allocatable :: w(:, :)
        integer(int16), allocatable :: x(:, :), s(:, :), b(:, :), bias(:), y(:, :), expected(:, :)

        allocate(x(k, m), w(k / 4, n), s(k / 64, n), b(k / 64, n), y(n, m), expected(n, m))
        read(unit, iostat=status) x, w, s, b
        call require_read(status)
        if (has_bias) then
            allocate(bias(n))
            read(unit, iostat=status) bias
            call require_read(status)
            call linear_q8_bf16(x, w, s, b, y, bias)
        else
            call linear_q8_bf16(x, w, s, b, y)
        end if
        read(unit, iostat=status) expected
        call require_read(status)
        call check(y, expected, "linear")
        print '(a,3(i0,1x),a,l1)', "PASS: linear K/M/N=", k, m, n, "bias=", has_bias
    end subroutine linear_case

    subroutine norm_case(d, m, weighted, scaled)
        integer, intent(in) :: d, m
        logical, intent(in) :: weighted, scaled
        integer(int16), allocatable :: x(:, :), w(:), y(:, :), expected(:, :)
        real(real32) :: eps, scalar

        allocate(x(d, m), y(d, m), expected(d, m))
        read(unit, iostat=status) x, eps
        call require_read(status)
        if (scaled) then
            read(unit, iostat=status) scalar
            call require_read(status)
            call rmsnorm_scale_bf16(x, y, eps, scalar)
        else if (weighted) then
            allocate(w(d))
            read(unit, iostat=status) w
            call require_read(status)
            call rmsnorm_bf16(x, y, eps, w)
        else
            call rmsnorm_bf16(x, y, eps)
        end if
        read(unit, iostat=status) expected
        call require_read(status)
        call check(y, expected, "rmsnorm")
        print '(a,2(i0,1x),a,l1,a,l1)', "PASS: norm D/M=", d, m, "weighted=", weighted, " scaled=", scaled
    end subroutine norm_case

    subroutine element_case(n)
        integer, intent(in) :: n
        integer(int16), allocatable :: gate(:), up(:), y(:, :), expected(:, :)

        allocate(gate(n), up(n), y(n, 6), expected(n, 6))
        read(unit, iostat=status) gate, up, expected
        call require_read(status)
        call silu_mul_bf16(gate, up, y(:, 1))
        call silu_mul_f32_gate_bf16(gate, up, y(:, 2))
        call elemwise_sigmoid_mul_bf16(gate, up, y(:, 3))
        y(:, 4) = gate
        call residual_add_bf16(y(:, 4), up)
        y(:, 5) = gate
        call silu_inplace_bf16(y(:, 5))
        y(:, 6) = gate
        call sigmoid_inplace_bf16(y(:, 6))
        call check(y, expected, "elementwise")
        if (all(y(:, 1) == y(:, 2))) error stop "SiLU variants must differ in this corpus"
        print '(a,i0)', "PASS: six elementwise operations N=", n
    end subroutine element_case

    subroutine argmax_case(n)
        integer, intent(in) :: n
        integer(int16), allocatable :: logits(:)
        integer(int32) :: got, expected

        allocate(logits(n))
        read(unit, iostat=status) logits, expected
        call require_read(status)
        got = argmax_bf16(logits)
        if (got /= expected) error stop "C oracle argmax mismatch"
        comparisons = comparisons + 1_int64
        print '(a,i0,a,i0)', "PASS: argmax N=", n, " zero-based index=", got
    end subroutine argmax_case

    subroutine decay_case(hv, length, shared_parameters)
        integer, intent(in) :: hv, length
        logical, intent(in) :: shared_parameters
        integer(int16), allocatable :: a(:, :), alg(:), dt(:)
        real(real32), allocatable :: g(:, :), expected(:, :)

        allocate(a(hv, length), alg(hv), dt(hv), g(hv, length), expected(hv, length))
        read(unit, iostat=status) a, alg, dt, expected
        call require_read(status)
        if (shared_parameters) then
            if (any(alg /= dt)) error stop "invalid shared compute_g oracle inputs"
            call compute_g_f32(a, alg, alg, g)
        else
            call compute_g_f32(a, alg, dt, g)
        end if
        call check_f32(g, expected, "compute_g")
        print '(a,2(i0,1x),a,l1)', "PASS: compute_g Hv/L=", hv, length, "read-only alias=", shared_parameters
    end subroutine decay_case

    subroutine conv_case(channels, calls, kernel)
        integer, intent(in) :: channels, calls, kernel
        integer(int16), allocatable :: w(:, :), state(:, :), expected_state(:, :), x(:, :), y(:, :), expected(:, :)
        integer(int32) :: meta(2)
        integer :: step, length

        if (kernel < 1 .or. kernel > 100 .or. calls > 100) error stop "invalid convolution oracle dimensions"
        allocate(w(kernel, channels), state(channels, kernel - 1), expected_state(channels, kernel - 1))
        read(unit, iostat=status) w, state
        call require_read(status)
        do step = 1, calls
            read(unit, iostat=status) meta
            call require_read(status)
            if (meta(1) < 1 .or. meta(1) > 100 .or. meta(2) < 0 .or. meta(2) > 1) &
                error stop "invalid convolution oracle call"
            length = int(meta(1))
            allocate(x(channels, length), y(channels, length), expected(channels, length))
            read(unit, iostat=status) x, expected, expected_state
            call require_read(status)
            if (meta(2) == 1) state = 0_int16
            call conv1d_depthwise_causal_bf16(x, state, w, y)
            call check(y, expected, "causal convolution")
            call check(state, expected_state, "convolution carry")
            deallocate(x, y, expected)
        end do
        print '(a,3(i0,1x))', "PASS: causal convolution C/K/calls (including reset)=", channels, kernel, calls
    end subroutine conv_case

    subroutine rope_case(d, length, heads)
        integer, intent(in) :: d, length, heads
        integer(int16), allocatable :: x(:, :, :), chunked(:, :, :), expected(:, :, :)
        real(real32), allocatable :: freqs(:)
        integer(int32) :: rot
        integer(int64) :: offset
        integer :: first, last

        read(unit, iostat=status) rot, offset
        call require_read(status)
        if (rot < 0 .or. rot > d .or. modulo(rot, 2) /= 0 .or. int(d, int64) * length * heads > 2000000) &
            error stop "invalid rope oracle dimensions"
        allocate(freqs(rot / 2), x(d, heads, length), chunked(d, heads, length), expected(d, heads, length))
        read(unit, iostat=status) freqs, x, expected
        call require_read(status)
        chunked = x
        call rope_partial_inplace_bf16(x, freqs, int(rot), offset)
        call check(reshape(x, [d, heads * length]), reshape(expected, [d, heads * length]), "rope in-place")
        first = 1
        do while (first <= length)
            last = min(first + 1, length)
            call rope_partial_inplace_bf16(chunked(:, :, first:last), freqs, int(rot), offset + int(first - 1, int64))
            first = last + 1
        end do
        call check(reshape(chunked, [d, heads * length]), reshape(expected, [d, heads * length]), "rope chunked")
        print '(a,4(i0,1x),a,i0)', "PASS: rope D/L/H/rot=", d, length, heads, rot, "offset=", offset
    end subroutine rope_case

    subroutine attention_case(d, lq, lk, causal)
        integer, intent(in) :: d, lq, lk
        logical, intent(in) :: causal
        integer(int32) :: heads(2)
        integer(int64) :: offset
        real(real32) :: scale
        integer(int16), allocatable :: q(:, :, :), k(:, :, :), v(:, :, :), y(:, :, :), expected(:, :, :)
        integer :: nq, nkv

        read(unit, iostat=status) heads, offset, scale
        call require_read(status)
        if (any(heads < 1) .or. any(heads > 100) .or. &
                int(d, int64) * lq * heads(1) > 2000000 .or. &
                int(d, int64) * (lk + 2) * heads(2) > 2000000) error stop "invalid sdpa oracle dimensions"
        nq = int(heads(1))
        nkv = int(heads(2))
        allocate(q(d, nq, lq), k(d, nkv, lk + 2), v(d, nkv, lk + 2), y(d, nq, lq), expected(d, nq, lq))
        ! Extra cache capacity must not enter the visible cache prefix.
        k = real32_to_bf16_bits(100.0_real32)
        v = real32_to_bf16_bits(-100.0_real32)
        read(unit, iostat=status) q, k(:, :, :lk), v(:, :, :lk), expected
        call require_read(status)
        call sdpa_gqa_bf16(q, k(:, :, :lk), v(:, :, :lk), y, offset, scale, causal)
        call check(reshape(y, [d, nq * lq]), reshape(expected, [d, nq * lq]), "sdpa")
        print '(a,5(i0,1x),a,i0,a,l1)', "PASS: sdpa D/Lq/Lk/Nq/Nkv=", d, lq, lk, nq, nkv, &
            "offset=", offset, " causal=", causal
    end subroutine attention_case

    subroutine recurrent_case(dk, dv, hk)
        integer, intent(in) :: dk, dv, hk
        integer(int32) :: meta(2), reset
        integer :: hv, calls, step
        integer(int16), allocatable :: q(:, :), k(:, :), v(:, :), beta(:), y(:, :), expected(:, :)
        real(real32), allocatable :: g(:), state(:, :, :), expected_state(:, :, :)

        read(unit, iostat=status) meta
        call require_read(status)
        if (any(meta < 1) .or. any(meta > 100) .or. int(dk, int64) * dv * meta(1) > 2000000) &
            error stop "invalid recurrent oracle dimensions"
        hv = int(meta(1))
        calls = int(meta(2))
        allocate(q(dk, hk), k(dk, hk), v(dv, hv), beta(hv), g(hv), y(dv, hv), expected(dv, hv))
        allocate(state(dk, dv, hv), expected_state(dk, dv, hv))
        read(unit, iostat=status) state
        call require_read(status)
        do step = 1, calls
            read(unit, iostat=status) reset, q, k, v, g, beta, expected, expected_state
            call require_read(status)
            if (reset < 0 .or. reset > 1) error stop "invalid recurrent oracle reset"
            if (reset == 1) state = 0.0_real32
            if (step == 2) then
                if (any(q /= k)) error stop "invalid shared recurrent oracle inputs"
                call gated_delta_step_bf16(q, q, v, g, beta, state, y)
            else
                call gated_delta_step_bf16(q, k, v, g, beta, state, y)
            end if
            call check(y, expected, "gated delta output")
            call check_f32(reshape(state, [dk, dv * hv]), reshape(expected_state, [dk, dv * hv]), "gated delta state")
        end do
        print '(a,5(i0,1x))', "PASS: gated delta Dk/Dv/Hk/Hv/calls (reset, q=k)=", dk, dv, hk, hv, calls
    end subroutine recurrent_case

    subroutine reject_case(mode)
        character(len=*), intent(in) :: mode
        integer(int32) :: ids(1), w(16, 2), bad_w(15, 2), result
        integer(int16) :: s(1, 2), x(64, 1), y(2, 1), wrong_y(3, 1), bad_x(65, 1)
        integer(int16) :: v(2), short(1), long(3), empty(0)

        ids = 0_int32
        w = 0_int32
        bad_w = 0_int32
        s = 0_int16
        x = 0_int16
        bad_x = 0_int16
        v = 0_int16
        short = 0_int16
        long = 0_int16
        select case (mode)
        case ("--bad-id-low", "--bad-id-high")
            ids = -1_int32
            if (mode == "--bad-id-high") ids = 2_int32
            call embed_dequant_gather_q8(ids, w, s, s, x)
        case ("--bad-q8-group")
            call linear_q8_bf16(bad_x, w, s, s, y)
        case ("--bad-q8-shape")
            call linear_q8_bf16(x, bad_w, s, s, y)
        case ("--bad-linear-output")
            call linear_q8_bf16(x, w, s, s, wrong_y)
        case ("--bad-bias")
            call linear_q8_bf16(x, w, s, s, y, short)
        case ("--bad-norm-output")
            call rmsnorm_bf16(x, y, 1.0e-6_real32)
        case ("--bad-norm-weight")
            call rmsnorm_bf16(x, bad_x(:64, :), 1.0e-6_real32, short)
        case ("--bad-epsilon")
            call rmsnorm_scale_bf16(x, bad_x(:64, :), 0.0_real32, 1.0_real32)
        case ("--bad-element")
            call silu_mul_bf16(v, long, short)
        case ("--bad-residual")
            call residual_add_bf16(v, short)
        case ("--empty-argmax")
            result = argmax_bf16(empty)
            print *, result
        case ("--empty-silu")
            call silu_inplace_bf16(empty)
        case default
            call reject_attention_state_case(mode)
        end select
    end subroutine reject_case

    subroutine reject_attention_state_case(mode)
        character(len=*), intent(in) :: mode
        integer(int16) :: tokens(3, 2), weights(4, 3), carry(3, 3), convout(3, 2)
        integer(int16) :: q(4, 2), v(5, 6), y(5, 6), beta(6)
        integer(int16) :: x3(4, 3, 2), y3(4, 3, 2), kv3(4, 1, 3)
        real(real32) :: g(3, 2), decay(6), state(4, 5, 6), freqs(2)

        tokens = 0_int16
        weights = 0_int16
        carry = 0_int16
        q = 0_int16
        v = 0_int16
        beta = 0_int16
        x3 = 0_int16
        kv3 = 0_int16
        decay = 0.0_real32
        state = 0.0_real32
        freqs = 0.0_real32
        select case (mode)
        case ("--bad-g-parameter")
            call compute_g_f32(tokens, beta(:2), beta(:3), g)
        case ("--bad-g-output")
            call compute_g_f32(tokens, beta(:3), beta(:3), g(:2, :))
        case ("--empty-g")
            call compute_g_f32(tokens(:, :0), beta(:3), beta(:3), g(:, :0))
        case ("--bad-conv-carry")
            call conv1d_depthwise_causal_bf16(tokens, carry(:, :2), weights, convout)
        case ("--bad-conv-weights")
            call conv1d_depthwise_causal_bf16(tokens, carry, weights(:, :2), convout)
        case ("--bad-conv-output")
            call conv1d_depthwise_causal_bf16(tokens, carry, weights, convout(:2, :))
        case ("--empty-conv-kernel")
            call conv1d_depthwise_causal_bf16(tokens, carry, weights(:0, :), convout)
        case ("--bad-delta-ratio")
            call gated_delta_step_bf16(q, q, v(:, :5), decay(:5), beta(:5), state(:, :, :5), y(:, :5))
        case ("--bad-delta-k")
            call gated_delta_step_bf16(q, q(:3, :), v, decay, beta, state, y)
        case ("--bad-delta-state")
            call gated_delta_step_bf16(q, q, v, decay, beta, state(:3, :, :), y)
        case ("--bad-delta-output")
            call gated_delta_step_bf16(q, q, v, decay, beta, state, y(:4, :))
        case ("--bad-delta-g")
            call gated_delta_step_bf16(q, q, v, decay(:5), beta, state, y)
        case ("--bad-delta-beta")
            call gated_delta_step_bf16(q, q, v, decay, beta(:5), state, y)
        case ("--empty-delta-q")
            call gated_delta_step_bf16(q(:, :0), q(:, :0), v, decay, beta, state, y)
        case ("--odd-rope")
            call rope_partial_inplace_bf16(x3, freqs, 3, 0_int64)
        case ("--wide-rope")
            call rope_partial_inplace_bf16(x3, freqs, 6, 0_int64)
        case ("--negative-rope")
            call rope_partial_inplace_bf16(x3, freqs, -2, 0_int64)
        case ("--bad-rope-frequencies")
            call rope_partial_inplace_bf16(x3, freqs(:1), 4, 0_int64)
        case ("--negative-position")
            call rope_partial_inplace_bf16(x3, freqs, 4, -1_int64)
        case ("--wide-position")
            call rope_partial_inplace_bf16(x3, freqs, 4, 4294967296_int64)
        case ("--overflow-position")
            call rope_partial_inplace_bf16(x3, freqs, 4, 4294967295_int64)
        case ("--bad-sdpa-ratio")
            call sdpa_gqa_bf16(x3(:, :2, :), x3, x3, y3(:, :2, :), 0_int64, 1.0_real32, .true.)
        case ("--bad-sdpa-head-width")
            call sdpa_gqa_bf16(x3, kv3(:3, :, :), kv3(:3, :, :), y3, 0_int64, 1.0_real32, .true.)
        case ("--bad-sdpa-v")
            call sdpa_gqa_bf16(x3, kv3, kv3(:, :, :2), y3, 0_int64, 1.0_real32, .true.)
        case ("--bad-sdpa-output")
            call sdpa_gqa_bf16(x3, kv3, kv3, y3(:3, :, :), 0_int64, 1.0_real32, .true.)
        case ("--empty-sdpa-cache")
            call sdpa_gqa_bf16(x3, kv3(:, :, :0), kv3(:, :, :0), y3, 0_int64, 1.0_real32, .true.)
        case ("--bad-sdpa-position")
            call sdpa_gqa_bf16(x3, kv3, kv3, y3, -1_int64, 1.0_real32, .true.)
        case ("--overflow-sdpa-position")
            call sdpa_gqa_bf16(x3, kv3, kv3, y3, 4294967295_int64, 1.0_real32, .true.)
        case default
            error stop "unknown negative dense kernel test"
        end select
    end subroutine reject_attention_state_case

end program test_dense_kernels
