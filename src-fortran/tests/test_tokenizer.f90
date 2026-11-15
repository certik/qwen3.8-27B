! Batch driver for qwen_tokenizer, driven by test_tokenizer.py.
!   usage: test_tokenizer MODEL_DIR CASES OUTPUT
! CASES has one command per line, strings are hex-encoded UTF-8/bytes:
!   E <hex>                      tokenize; output: ids separated by spaces
!   D <id> <id> ...              decode (special tokens kept); output: hex
!   C <think|nothink> <effort> <hex user> <hex system>
!                                chat prompt text (hex) and its token ids
!   N <hex>                      NFC-normalize; output: hex
program test_tokenizer
    use, intrinsic :: iso_fortran_env, only: int32, int64, error_unit
    use qwen_tokenizer
    implicit none
    type(bpe_tokenizer) :: tok
    character(:), allocatable :: err, cases, arg1, arg2, arg3
    character(:), allocatable :: text, answer
    integer(int32), allocatable :: ids(:)
    integer :: unit, ios, a, b, n
    character(1), parameter :: nl = achar(10)

    if (command_argument_count() /= 3) error stop 'usage: test_tokenizer MODEL_DIR CASES OUTPUT'
    arg1 = argument(1)
    arg2 = argument(2)
    arg3 = argument(3)
    call load_tokenizer(arg1, tok, err)
    if (allocated(err)) error stop err
    call read_file_bytes(arg2, cases, err)
    if (allocated(err)) error stop err
    open(newunit=unit, file=arg3, status='replace', action='write', access='stream', form='unformatted', iostat=ios)
    if (ios /= 0) error stop 'cannot create output'
    answer = ''
    a = 1
    n = len(cases)
    do while (a <= n)
        b = index(cases(a:n), nl)
        if (b == 0) then
            b = n + 1
        else
            b = a + b - 1
        end if
        if (b > a) then
            answer = run_case(cases(a:b-1))
            write(unit, iostat=ios) answer//nl
            if (ios /= 0) error stop 'write failed'
        end if
        a = b + 1
    end do
    close(unit)

contains

    function argument(i) result(s)
        integer, intent(in) :: i
        character(:), allocatable :: s
        integer :: l
        call get_command_argument(i, length=l)
        allocate(character(l) :: s)
        call get_command_argument(i, s)
    end function

    function run_case(line) result(r)
        character(*), intent(in) :: line
        character(:), allocatable :: r
        integer(int32), allocatable :: d(:)
        select case (line(1:1))
        case ('E')
            text = unhex(line(3:))
            call encode_text(tok, text, ids)
            r = ids_line(ids)
        case ('D')
            d = parse_ids(line(3:))
            r = hex(decode_ids(tok, d, .false.))
        case ('N')
            r = hex(nfc_utf8(unhex(line(3:))))
        case ('C')
            text = chat_prompt(unhex(field(line, 4)), unhex(field(line, 5)), field(line, 2) == 'think', field(line, 3))
            call encode_text(tok, text, ids)
            r = hex(text)//' '//ids_line(ids)
        case default
            error stop 'bad case line'
        end select
    end function

    ! k-th blank-separated field of line (empty when absent)
    function field(line, k) result(r)
        character(*), intent(in) :: line
        integer, intent(in) :: k
        character(:), allocatable :: r
        integer :: i, start, cnt
        cnt = 1
        start = 1
        do i = 1, len(line) + 1
            if (i > len(line)) then
                if (cnt == k) then
                    r = line(start:len(line))
                    return
                end if
            else if (line(i:i) == ' ') then
                if (cnt == k) then
                    r = line(start:i-1)
                    return
                end if
                cnt = cnt + 1
                start = i + 1
            end if
        end do
        r = ''
    end function

    function ids_line(v) result(r)
        integer(int32), intent(in) :: v(:)
        character(:), allocatable :: r
        character(16) :: num
        integer :: i
        r = ''
        do i = 1, size(v)
            write(num, '(i0)') v(i)
            if (i > 1) r = r//' '
            r = r//trim(num)
        end do
    end function

    function parse_ids(s) result(v)
        character(*), intent(in) :: s
        integer(int32), allocatable :: v(:)
        integer(int32), allocatable :: t(:)
        integer :: i, cnt, val
        logical :: inside
        allocate(t(len(s) / 2 + 1))
        cnt = 0
        val = 0
        inside = .false.
        do i = 1, len(s) + 1
            if (i <= len(s)) then
                if (s(i:i) >= '0' .and. s(i:i) <= '9') then
                    val = val * 10 + iachar(s(i:i)) - 48
                    inside = .true.
                    cycle
                end if
            end if
            if (inside) then
                cnt = cnt + 1
                t(cnt) = val
            end if
            val = 0
            inside = .false.
        end do
        allocate(v(cnt))
        v = t(1:cnt)
    end function

    function hex(s) result(r)
        character(*), intent(in) :: s
        character(:), allocatable :: r
        character(16), parameter :: digits = '0123456789abcdef'
        integer :: i, b
        allocate(character(2 * len(s)) :: r)
        do i = 1, len(s)
            b = iachar(s(i:i))
            r(2*i-1:2*i-1) = digits(b/16+1:b/16+1)
            r(2*i:2*i) = digits(modulo(b, 16)+1:modulo(b, 16)+1)
        end do
    end function

    function unhex(s) result(r)
        character(*), intent(in) :: s
        character(:), allocatable :: r
        integer :: i, hi, lo
        allocate(character(len(s) / 2) :: r)
        do i = 1, len(s) / 2
            hi = index('0123456789abcdef', s(2*i-1:2*i-1)) - 1
            lo = index('0123456789abcdef', s(2*i:2*i)) - 1
            if (hi < 0 .or. lo < 0) error stop 'bad hex'
            r(i:i) = achar(hi * 16 + lo)
        end do
    end function
end program
