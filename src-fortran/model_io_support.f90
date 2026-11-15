module model_io_support
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, file_storage_size
    implicit none
    private
    public :: io_error, fail, checked_add, checked_multiply, read_text, require_byte_io
    integer(int64), parameter, public :: max_json_bytes = 16_int64 * 1024 * 1024

    type :: io_error
        character(:), allocatable :: message
    contains
        procedure :: failed
    end type

contains

    logical function failed(self)
        class(io_error), intent(in) :: self
        failed = allocated(self%message)
    end function

    subroutine fail(error, message)
        type(io_error), intent(inout) :: error
        character(*), intent(in) :: message
        if (.not. error%failed()) error%message = message
    end subroutine

    function checked_add(a, b, error) result(c)
        integer(int64), intent(in) :: a, b
        type(io_error), intent(inout) :: error
        integer(int64) :: c
        c = 0
        if (a < 0 .or. b < 0) then
            call fail(error, 'negative size or offset')
        else if (a > huge(c) - b) then
            call fail(error, '64-bit addition overflow')
        else
            c = a + b
        end if
    end function

    function checked_multiply(a, b, error) result(c)
        integer(int64), intent(in) :: a, b
        type(io_error), intent(inout) :: error
        integer(int64) :: c
        c = 0
        if (a < 0 .or. b < 0) then
            call fail(error, 'negative size or dimension')
        else if (b /= 0) then
            if (a > huge(c) / b) then
                call fail(error, '64-bit multiplication overflow')
            else
                c = a * b
            end if
        end if
    end function

    subroutine require_byte_io(error)
        type(io_error), intent(inout) :: error
        integer :: i
        if (file_storage_size /= 8 .or. storage_size('a') /= 8 .or. &
                bit_size(0_int16) /= 16 .or. bit_size(0_int32) /= 32 .or. &
                bit_size(0_int64) /= 64) &
            call fail(error, 'model I/O requires 8-bit file/character units and 16/32/64-bit integers')
        do i=0,255
            if (iachar(achar(i)) /= i) call fail(error,'model I/O requires byte-preserving character storage')
        end do
        if (not(0_int16) /= -1_int16 .or. &
                not(0_int32) /= -1_int32) call fail(error, 'model I/O requires byte-preserving two''s-complement storage')
    end subroutine

    subroutine read_text(path, text, error)
        character(*), intent(in) :: path
        character(:), allocatable, intent(out) :: text
        type(io_error), intent(out) :: error
        integer :: unit, ios
        integer(int64) :: length
        call require_byte_io(error)
        if (error%failed()) return
        open(newunit=unit, file=path, access='stream', form='unformatted', status='old', action='read', iostat=ios)
        if (ios /= 0) then
            call fail(error, 'cannot open JSON file: '//path)
            return
        end if
        inquire(unit=unit, size=length, iostat=ios)
        if (ios /= 0) then
            call fail(error, 'cannot inquire JSON file size: '//path)
        else if (length < 1 .or. length > max_json_bytes) then
            call fail(error, 'JSON file length outside 1..16 MiB: '//path)
        else
            allocate(character(int(length)) :: text, stat=ios)
            if (ios /= 0) then
                call fail(error, 'JSON text allocation failed')
            else
                read(unit, pos=1_int64, iostat=ios) text
                if (ios /= 0) call fail(error, 'truncated JSON file: '//path)
            end if
        end if
        close(unit)
    end subroutine
end module
