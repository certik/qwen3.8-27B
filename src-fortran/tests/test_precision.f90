program test_precision
    use precision_test_cases, only: prepare_precision_tests, run_precision_tests, report_precision_tests
    implicit none
    character(len=4096) :: path

    call prepare_precision_tests(path)
    call run_precision_tests(path)
    call report_precision_tests()
end program test_precision
