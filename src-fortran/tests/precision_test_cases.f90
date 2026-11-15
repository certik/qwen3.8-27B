module precision_test_cases
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32, &
        error_unit, iostat_end, iostat_eor
    use precision_formats
    implicit none
    private
    public :: prepare_precision_tests, run_precision_tests, report_precision_tests
    integer, parameter :: oracle_records = 2 * 256 * 7 * 5 + 4096 + 16

contains

    subroutine prepare_precision_tests(path)
        character(len=*), intent(out) :: path
        integer :: argument_status
        integer(int32) :: rejected_byte

        call require_precision_environment()
        if (command_argument_count() /= 1) error stop "usage: test_precision oracle.txt"
        call get_command_argument(1, path, status=argument_status)
        if (argument_status /= 0) error stop "cannot read complete C oracle path"
        if (trim(path) == "--invalid-low") then
            rejected_byte = packed_q8_byte(0_int32, -1)
            print *, rejected_byte
            stop
        else if (trim(path) == "--invalid-high") then
            rejected_byte = packed_q8_byte(0_int32, 4)
            print *, rejected_byte
            stop
        end if
    end subroutine prepare_precision_tests

    subroutine run_precision_tests(path)
        character(len=*), intent(in) :: path

        call test_exhaustive_bf16()
        call test_q8()
        call test_oracle(trim(path))
    end subroutine run_precision_tests

    subroutine report_precision_tests()
        print '(a)', "PASS: 65536 BF16 roundtrips and independent DO CONCURRENT outputs"
        print '(a,i0,a)', "PASS: ", oracle_records, " C-oracle binary32 conversions"
        print '(a)', "PASS: all 256 q8 boundary-byte combinations"
    end subroutine report_precision_tests

    pure function signed16(code) result(bits)
        integer(int32), intent(in) :: code
        integer(int16) :: bits

        bits = int(iand(code, 32767_int32), int16)
        if (btest(code, 15)) bits = ibset(bits, 15)
    end function signed16

    pure function unsigned16(bits) result(code)
        integer(int16), intent(in) :: bits
        integer(int32) :: code

        code = int(ibits(bits, 0, 15), int32)
        if (btest(bits, 15)) code = code + 32768_int32
    end function unsigned16

    pure function widened_word(code) result(word)
        integer(int32), intent(in) :: code
        integer(int32) :: word

        word = iand(code, 32767_int32) * 65536_int32
        if (btest(code, 15)) word = ibset(word, 31)
    end function widened_word

    subroutine test_exhaustive_bf16()
        integer(int16), allocatable :: source(:), narrowed(:)
        real(real32), allocatable :: widened(:)
        integer(int32) :: code, expected, raw
        integer :: i

        allocate(source(65536), narrowed(65536), widened(65536))
        do i = 1, size(source)
            source(i) = signed16(int(i - 1, int32))
        end do
        ! Each iteration owns its array elements; only the index is an iteration-local scalar.
        do concurrent (i = 1:65536)
            widened(i) = bf16_bits_to_real32(source(i))
            narrowed(i) = real32_to_bf16_bits(widened(i))
        end do
        do i = 1, size(source)
            code = int(i - 1, int32)
            expected = code
            if (iand(code, int(z'7f80', int32)) == int(z'7f80', int32) .and. &
                    iand(code, 127_int32) /= 0) expected = ibset(expected, 6)
            raw = widened_word(code)
            if (transfer(widened(i), 0_int32) /= raw .or. &
                    bf16_bits_to_fp32_bits(source(i)) /= raw .or. &
                    unsigned16(narrowed(i)) /= expected .or. &
                    unsigned16(fp32_bits_to_bf16_bits(raw)) /= expected) then
                write(error_unit, '(a,z4.4)') "exhaustive BF16 mismatch at 0x", code
                error stop 1
            end if
        end do
    end subroutine test_exhaustive_bf16

    subroutine test_q8()
        integer(int32), parameter :: boundaries(4) = [0_int32, 127_int32, 128_int32, 255_int32]
        integer(int32) :: word, expected, actual
        integer :: combination, position, bit, digit

        do combination = 0, 255
            word = 0_int32
            do position = 0, 3
                digit = mod(combination / 4**position, 4) + 1
                do bit = 0, 7
                    if (btest(boundaries(digit), bit)) word = ibset(word, 8 * position + bit)
                end do
            end do
            do position = 0, 3
                expected = boundaries(mod(combination / 4**position, 4) + 1)
                actual = packed_q8_byte(word, position)
                if (actual < 0_int32 .or. actual > 255_int32 .or. actual /= expected) &
                    error stop "unsigned q8 extraction mismatch"
            end do
        end do
        if (packed_q8_byte(ibset(0_int32, 31), 3) /= 128_int32 .or. &
                packed_q8_byte(not(0_int32), 0) /= 255_int32) &
            error stop "signed q8 word handling mismatch"
    end subroutine test_q8

    subroutine test_oracle(filename)
        character(len=*), intent(in) :: filename
        integer :: unit, status, row, j, characters_read
        integer(int64) :: unsigned_word
        integer(int32) :: word, expected, actual, exponent, fraction
        integer(int16) :: result_bits
        character(len=128) :: line
        character(len=*), parameter :: hex_digits = "0123456789abcdefABCDEF"
        character(len=64) :: expected_header, expected_footer
        logical :: input_nan, output_nan, input_inf
        real(real32) :: value

        write(expected_header, '(a,i0)') "BF16_ORACLE_V1 ", oracle_records
        write(expected_footer, '(a,i0)') "END ", oracle_records
        open(newunit=unit, file=filename, status="old", action="read", iostat=status)
        if (status /= 0) error stop "cannot open C oracle output"
        call read_oracle_line(unit, line)
        if (trim(line) /= trim(expected_header)) error stop "invalid C oracle header"
        do row = 1, oracle_records
            call read_oracle_line(unit, line)
            if (len_trim(line) /= 13 .or. line(9:9) /= " ") error stop "invalid C oracle record"
            do j = 1, 13
                if (j == 9) cycle
                if (index(hex_digits, line(j:j)) == 0) error stop "non-hex C oracle record"
            end do
            read(line(1:8), '(z8)', iostat=status) unsigned_word
            if (status /= 0) error stop "invalid C oracle input bits"
            read(line(10:13), '(z4)', iostat=status) expected
            if (status /= 0) error stop "invalid C oracle expected bits"
            word = int(ibits(unsigned_word, 0, 31), int32)
            if (btest(unsigned_word, 31)) word = ibset(word, 31)
            exponent = ibits(word, 23, 8)
            fraction = ibits(word, 0, 23)
            input_nan = exponent == 255_int32 .and. fraction /= 0_int32
            input_inf = exponent == 255_int32 .and. fraction == 0_int32
            output_nan = iand(expected, int(z'7f80', int32)) == int(z'7f80', int32) .and. &
                iand(expected, 127_int32) /= 0_int32
            if (input_nan .neqv. output_nan) error stop "unexpected NaN classification in C oracle"
            if (input_nan) then
                if (.not. btest(expected, 6)) error stop "C oracle did not quiet a NaN"
                if (expected /= ior(ibits(word, 16, 16), 64_int32)) &
                    error stop "C oracle NaN sign/payload mismatch"
            else if (iand(expected, int(z'7fff', int32)) == int(z'7f80', int32)) then
                if (.not. input_inf .and. ibits(word, 0, 31) < int(z'7f7f8000', int32)) &
                    error stop "unexpected finite-to-infinity conversion in C oracle"
            end if
            value = transfer(word, 0.0_real32)
            result_bits = real32_to_bf16_bits(value)
            actual = unsigned16(result_bits)
            if (actual /= expected .or. &
                    unsigned16(fp32_bits_to_bf16_bits(word)) /= expected) then
                write(error_unit, '(a,i0,a,z8.8,a,z4.4,a,z4.4)') &
                    "C oracle mismatch at record ", row, ": 0x", word, &
                    " expected 0x", expected, " actual 0x", actual
                error stop 1
            end if
            if (transfer(bf16_bits_to_real32(result_bits), 0_int32) /= widened_word(expected)) &
                error stop "C oracle result widening mismatch"
        end do
        call read_oracle_line(unit, line)
        if (trim(line) /= trim(expected_footer)) error stop "missing C oracle footer"
        read(unit, '(a)', advance="no", size=characters_read, iostat=status) line
        if (status /= iostat_end) then
            write(error_unit, '(a,i0,a,i0,a,a)') "post-footer status ", status, &
                ", expected ", iostat_end, ", record: ", trim(line)
            error stop "unexpected extra data or I/O error after C oracle footer"
        end if
        close(unit, iostat=status)
        if (status /= 0) error stop "cannot close C oracle output"
    end subroutine test_oracle

    subroutine read_oracle_line(unit, line)
        integer, intent(in) :: unit
        character(len=*), intent(out) :: line
        integer :: status, characters_read

        read(unit, '(a)', advance="no", size=characters_read, iostat=status) line
        if (status == iostat_end) error stop "missing/truncated C oracle output"
        if (status /= iostat_eor) error stop "overlong C oracle record or read error"
        if (characters_read == 0) error stop "empty C oracle record"
    end subroutine read_oracle_line

end module precision_test_cases
