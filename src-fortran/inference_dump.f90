module inference_dump
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32
    use precision_formats, only: bf16_bits_to_real32
    implicit none
    private
    public :: dump_bf16, dump_f32, configure_dump
    integer :: selected_layer=-2
    integer(int64) :: retained_bytes=0_int64
    integer(int64), parameter :: byte_budget=256_int64*1024*1024

contains

    subroutine configure_dump(layer)
        integer, intent(in) :: layer
        if (layer < -2) error stop 'invalid diagnostic layer'
        selected_layer=layer
        retained_bytes=0_int64
    end subroutine

    logical function enabled(directory, layer) result(active)
        character(*), intent(in) :: directory
        integer, intent(in) :: layer
        active=len_trim(directory) /= 0
        if (selected_layer /= -2 .and. layer /= selected_layer .and. layer /= -1) active=.false.
    end function

    subroutine dump_header(directory, name, layer, shape, unit)
        character(*), intent(in) :: directory, name
        integer, intent(in) :: layer, shape(:)
        integer, intent(out) :: unit
        character(:), allocatable :: path
        character(32) :: number
        character(4) :: endian
        integer :: ios
        integer(int64) :: bytes
        bytes=16_int64+8_int64*size(shape)+4_int64*product(int(shape,int64))
        if (bytes > byte_budget-retained_bytes) error stop 'diagnostic dump exceeds 256 MiB budget; select one layer'
        retained_bytes=retained_bytes+bytes
        endian=transfer(1_int32,endian)
        if (iachar(endian(1:1)) /= 1) error stop 'debug dumps require little-endian storage'
        path=trim(directory)//'/'//trim(name)
        if (layer >= 0) then
            write(number,'(i0)') layer
            path=path//'_'//trim(number)
        end if
        open(newunit=unit,file=path//'.bin',status='new',access='stream',form='unformatted', &
            action='write',iostat=ios)
        if (ios /= 0) error stop 'cannot create new debug dump (directories must already exist): '//path
        write(unit,iostat=ios) 'LLMTNSR1',0_int32,int(size(shape),int32),int(shape,int64)
        if (ios /= 0) error stop 'debug dump header write failed'
    end subroutine

    subroutine dump_bf16(directory, name, layer, values, shape)
        character(*), intent(in) :: directory, name
        integer, intent(in) :: layer, shape(:)
        integer(int16), intent(in) :: values(:)
        integer :: unit, ios, i
        if (.not. enabled(directory,layer)) return
        if (product(int(shape,int64)) /= size(values,kind=int64)) error stop 'debug dump shape mismatch'
        call dump_header(directory,name,layer,shape,unit)
        do i=1,size(values)
            write(unit,iostat=ios) bf16_bits_to_real32(values(i))
            if (ios /= 0) error stop 'debug dump payload write failed'
        end do
        close(unit,iostat=ios)
        if (ios /= 0) error stop 'debug dump close failed'
    end subroutine

    subroutine dump_f32(directory, name, layer, values, shape)
        character(*), intent(in) :: directory, name
        integer, intent(in) :: layer, shape(:)
        real(real32), intent(in) :: values(:)
        integer :: unit, ios
        if (.not. enabled(directory,layer)) return
        if (product(int(shape,int64)) /= size(values,kind=int64)) error stop 'debug dump shape mismatch'
        call dump_header(directory,name,layer,shape,unit)
        write(unit,iostat=ios) values
        if (ios /= 0) error stop 'debug dump payload write failed'
        close(unit,iostat=ios)
        if (ios /= 0) error stop 'debug dump close failed'
    end subroutine
end module
