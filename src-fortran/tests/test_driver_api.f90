program test_driver_api
    use, intrinsic :: iso_fortran_env, only: int16, int32, real32
    use dense_model
    use precision_formats, only: require_precision_environment
    implicit none
    type(inference_model), target :: model
    type(layer_state), allocatable :: snapshot(:)
    integer(int16), allocatable :: logits(:,:)
    integer :: checks
    character(1024) :: directory, dump_directory
    character(32) :: mode
    call get_command_argument(1,mode)
    call get_command_argument(2,directory)
    call get_command_argument(3,dump_directory)
    select case(trim(mode))
    case('index-product')
        call checked_shape([huge(0),2])
    case('int64-product')
        call checked_shape([huge(0),huge(0),huge(0)])
    case('activation')
        call initialize_model(trim(directory),1,2,model)
        call forward_ids(model,[1_int32,2_int32],logits,'')
    case('capacity')
        call initialize_model(trim(directory),2,2,model)
        call forward_ids(model,[1_int32],logits,'')
        call forward_ids(model,[1_int32,2_int32],logits,'')
    case('id')
        call initialize_model(trim(directory),1,1,model)
        call forward_ids(model,[128_int32],logits,'')
    case('padded-dump')
        call initialize_model(directory,1,1,model)
        call forward_ids(model,[11_int32],logits,dump_directory)
    case('state-bits')
        call test_state_bits()
    case('terminal-recurrent-prefix')
        call initialize_model(trim(directory),2,4,model)
        if (model%cfg%layers /= 5) error stop 'terminal-prefix test requires five layers'
        if (model%cfg%full_attention(5)) error stop 'terminal-prefix test requires a recurrent final layer'
        call forward_ids(model,[1_int32,2_int32],logits,trim(dump_directory),.true.)
        if (size(logits,2) /= 0 .or. model%counts%heads /= 0 .or. model%counts%terminal_pruned /= 1) &
            error stop 'terminal-prefix demand did not prune the head'
        if (.not. any(model%state(5)%conv /= 0_int16)) error stop 'terminal convolution carry was not produced'
        if (.not. (maxval(abs(model%state(5)%delta)) > 0.0_real32)) &
            error stop 'terminal recurrent state was not produced'
        call snapshot_state(model,snapshot)
        call reset_model(model)
        call forward_ids(model,[1_int32,2_int32],logits,'')
        if (.not. same_state(model,snapshot)) error stop 'cache-only prefix state differs from full forward'
        write(*,'(a)') 'PASS: terminal recurrent prefix updates complete cache/state without a head'
    case default
        error stop 'unknown driver API test'
    end select
    if (trim(mode) /= 'padded-dump' .and. trim(mode) /= 'state-bits' .and. &
            trim(mode) /= 'terminal-recurrent-prefix') error stop 'unexpected rejection success'

contains

    subroutine test_state_bits()
        integer(int32), parameter :: patterns(8)=[int(z'00000000',int32),int(z'80000000',int32), &
            int(z'3f800000',int32),int(z'bf800000',int32),int(z'00000001',int32),int(z'3f800001',int32), &
            int(z'7fc01234',int32),int(z'ffc05678',int32)]
        integer(int32) :: word
        integer :: li, k, v, h, n, ios
        call require_precision_environment()
        allocate(model%state(3),stat=ios)
        if (ios /= 0) error stop 'state-bit test allocation failed'
        do li=1,3,2
            allocate(model%state(li)%conv(2,2),model%state(li)%delta(2,2,2),stat=ios)
            if (ios /= 0) error stop 'state-bit test allocation failed'
            model%state(li)%conv=0_int16
            n=0
            do h=1,2
                do v=1,2
                    do k=1,2
                        n=n+1
                        model%state(li)%delta(k,v,h)=transfer(patterns(n),0.0_real32)
                    end do
                end do
            end do
        end do
        allocate(model%state(2)%k(2,1,3),model%state(2)%v(2,1,3),stat=ios)
        if (ios /= 0) error stop 'state-bit test allocation failed'
        model%state(2)%k=0_int16
        model%state(2)%v=0_int16
        call snapshot_state(model,snapshot)
        checks=0
        call expect(.true.,'identical FP32 bits (including signed zeros and quiet NaNs)')
        do li=1,3,2
            n=0
            do h=1,2
                do v=1,2
                    do k=1,2
                        n=n+1
                        word=patterns(n)
                        if (transfer(snapshot(li)%delta(k,v,h),0_int32) /= word) &
                            error stop 'snapshot did not preserve FP32 test pattern'
                        model%state(li)%delta(k,v,h)=transfer(ieor(word,1_int32),0.0_real32)
                        call expect(.false.,'one-bit FP32 difference at each state position')
                        model%state(li)%delta(k,v,h)=transfer(word,0.0_real32)
                        call expect(.true.,'restored exact FP32 pattern')
                    end do
                end do
            end do
        end do
        do k=1,2
            word=patterns(k)
            model%state(1)%delta(k,1,1)=transfer(ieor(word,int(z'80000000',int32)),0.0_real32)
            call expect(.false.,'positive and negative zero must differ')
            model%state(1)%delta(k,1,1)=transfer(word,0.0_real32)
            call expect(.true.,'restored zero sign')
        end do
        model%state(3)%conv(2,2)=int(z'8000',int16)
        call expect(.false.,'BF16 conv bit difference')
        model%state(3)%conv(2,2)=0_int16
        call expect(.true.,'restored BF16 conv')
        model%state(2)%k(2,1,3)=1_int16
        call expect(.false.,'BF16 K difference in spare capacity')
        model%state(2)%k(2,1,3)=0_int16
        call expect(.true.,'restored BF16 K')
        model%state(2)%v(2,1,3)=1_int16
        call expect(.false.,'BF16 V difference in spare capacity')
        model%state(2)%v(2,1,3)=0_int16
        call expect(.true.,'restored BF16 V')
        write(*,'(a,i0,a)') 'PASS: ',checks,' exact reset-comparison cases (signed zero and quiet NaN payloads)'
    end subroutine

    subroutine expect(expected, label)
        logical, intent(in) :: expected
        character(*), intent(in) :: label
        if (same_state(model,snapshot) .neqv. expected) error stop label
        checks=checks+1
    end subroutine
end program
