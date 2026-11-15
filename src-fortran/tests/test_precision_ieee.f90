program test_precision_ieee
    use, intrinsic :: iso_fortran_env, only: real32
    use, intrinsic :: ieee_exceptions, only: ieee_invalid, ieee_overflow, ieee_underflow, &
        ieee_support_flag, ieee_set_flag, ieee_get_flag
    use precision_test_cases, only: prepare_precision_tests, run_precision_tests, report_precision_tests
    implicit none
    character(len=4096) :: path
    logical :: raised

    call prepare_precision_tests(path)
    if (.not. ieee_support_flag(ieee_invalid, 0.0_real32) .or. &
            .not. ieee_support_flag(ieee_overflow, 0.0_real32) .or. &
            .not. ieee_support_flag(ieee_underflow, 0.0_real32)) &
        error stop "tests require IEEE exception flag support"
    call ieee_set_flag(ieee_invalid, .false.)
    call ieee_set_flag(ieee_overflow, .false.)
    call ieee_set_flag(ieee_underflow, .false.)
    call run_precision_tests(path)
    call ieee_get_flag(ieee_invalid, raised)
    if (raised) error stop "bit conversions raised IEEE invalid"
    call ieee_get_flag(ieee_overflow, raised)
    if (raised) error stop "bit conversions raised IEEE overflow"
    call ieee_get_flag(ieee_underflow, raised)
    if (raised) error stop "bit conversions raised IEEE underflow"
    call report_precision_tests()
    print '(a)', "PASS: no IEEE invalid, overflow or underflow flags"
end program test_precision_ieee
