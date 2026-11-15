module precision_formats
    use, intrinsic :: iso_fortran_env, only: int16, int32, real32
    use, intrinsic :: ieee_arithmetic, only: ieee_support_datatype, &
        ieee_support_denormal, ieee_support_inf, ieee_support_nan
    implicit none
    private
    public :: require_precision_environment
    public :: bf16_bits_to_fp32_bits, fp32_bits_to_bf16_bits
    public :: bf16_bits_to_real32, real32_to_bf16_bits
    public :: packed_q8_byte, dp4a

contains

    subroutine require_precision_environment()
        if (storage_size(0_int16) /= 16 .or. bit_size(0_int16) /= 16) &
            error stop "precision_formats requires 16-bit integer(int16)"
        if (storage_size(0_int32) /= 32 .or. bit_size(0_int32) /= 32) &
            error stop "precision_formats requires 32-bit integer(int32)"
        if (storage_size(0.0_real32) /= 32) &
            error stop "precision_formats requires 32-bit real(real32)"
        if (radix(0.0_real32) /= 2 .or. digits(0.0_real32) /= 24 .or. &
                minexponent(0.0_real32) /= -125 .or. maxexponent(0.0_real32) /= 128) &
            error stop "precision_formats requires IEEE binary32 real(real32)"
        if (.not. ieee_support_datatype(0.0_real32) .or. &
                .not. ieee_support_denormal(0.0_real32) .or. &
                .not. ieee_support_inf(0.0_real32) .or. &
                .not. ieee_support_nan(0.0_real32)) &
            error stop "precision_formats requires IEEE subnormal, infinity and NaN support"
        if (transfer(1.0_real32, 0_int32) /= int(z'3f800000', int32) .or. &
                transfer(-0.0_real32, 0_int32) /= ibset(0_int32, 31) .or. &
                transfer(tiny(0.0_real32), 0_int32) /= int(z'00800000', int32) .or. &
                transfer(huge(0.0_real32), 0_int32) /= int(z'7f7fffff', int32)) &
            error stop "precision_formats requires matching IEEE binary32 integer bit layout"
        if (not(0_int16) /= -1_int16 .or. not(0_int32) /= -1_int32) &
            error stop "precision_formats requires two's-complement integer storage"
    end subroutine require_precision_environment

    pure elemental function bf16_bits_to_fp32_bits(bits) result(word)
        integer(int16), intent(in) :: bits
        integer(int32) :: word

        word = shiftl(int(bits, int32), 16)
    end function bf16_bits_to_fp32_bits

    pure elemental function fp32_bits_to_bf16_bits(word) result(bits)
        integer(int32), intent(in) :: word
        integer(int16) :: bits
        integer(int32) :: upper, lower, rounded

        upper = ibits(word, 16, 16)
        lower = ibits(word, 0, 16)
        ! Round to nearest even: the carry of the lower half plus 0x7fff plus the lowest bit kept. The
        ! largest non-NaN upper half is 0xff80, so the addition cannot overflow. A NaN keeps sign and top
        ! payload and gets the quiet bit, including low-only payloads.
        rounded = upper + shiftr(lower + 32767_int32 + iand(upper, 1_int32), 16)
        if (ibits(word, 23, 8) == 255_int32 .and. ibits(word, 0, 23) /= 0_int32) rounded = ibset(upper, 6)
        bits = int(ibits(rounded, 0, 15), int16)
        if (btest(rounded, 15)) bits = ibset(bits, 15)
    end function fp32_bits_to_bf16_bits

    pure elemental function bf16_bits_to_real32(bits) result(value)
        integer(int16), intent(in) :: bits
        real(real32) :: value

        value = transfer(bf16_bits_to_fp32_bits(bits), 0.0_real32)
    end function bf16_bits_to_real32

    pure elemental function real32_to_bf16_bits(value) result(bits)
        real(real32), intent(in) :: value
        integer(int16) :: bits

        bits = fp32_bits_to_bf16_bits(transfer(value, 0_int32))
    end function real32_to_bf16_bits

    pure elemental function packed_q8_byte(word, byte_index) result(value)
        integer(int32), intent(in) :: word
        integer, intent(in) :: byte_index
        integer(int32) :: value

        if (byte_index < 0 .or. byte_index > 3) &
            error stop "packed_q8_byte requires byte_index in 0..3"
        value = ibits(word, 8 * byte_index, 8)
    end function packed_q8_byte

    ! Integer dot product of four signed bytes with four signed bytes, plus c: the
    ! bytes of a and of b are read as int8 (sign-extended by an arithmetic shift),
    ! multiplied pairwise and summed into c. The four products add up to at most
    ! 2**16 in magnitude, so the result does not overflow as long as |c| < 2**30
    ! (callers pass 0 or a partial sum of the same kind). Written out as plain
    ! integer arithmetic so that any compiler can run it; a compiler may recognize
    ! this form and use a hardware dot-product instruction (such as dp4a on NVIDIA
    ! GPUs) for it. The byte order does not matter as long as a and b use the same.
    pure elemental function dp4a(a, b, c) result(r)
        !$acc routine seq
        integer(int32), intent(in) :: a, b, c
        integer(int32) :: r

        r = c + shifta(shiftl(a, 24), 24) * shifta(shiftl(b, 24), 24) &
              + shifta(shiftl(a, 16), 24) * shifta(shiftl(b, 16), 24) &
              + shifta(shiftl(a,  8), 24) * shifta(shiftl(b,  8), 24) &
              + shifta(a, 24) * shifta(b, 24)
    end function dp4a

end module precision_formats
