program test_model_io
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, error_unit
    use model_io_support
    use json_tokens
    use safetensors_io
    use qwen_config
    use qwen_weights
    use precision_formats, only: packed_q8_byte
    implicit none
    type(io_error) :: error
    type(json_document) :: doc
    type(tensor_archive) :: archive
    type(tensor_info), allocatable :: tensors(:)
    type(shard_info) :: shard
    type(dense_config) :: config
    type(dense_weight_layout) :: layout
    integer(int32), allocatable :: ids(:), words(:)
    integer(int16), allocatable :: halves(:)
    integer(int32), target :: matrix(2,2)
    integer(int32), target :: wrong_matrix(1,4)
    integer(int16), target :: scale(2,2), cube(2,2,1)
    integer(int64) :: first, count, vocab, number
    integer :: i, j, t, unit, ios
    character(:), allocatable :: mode, path, name, arg, value
    character(1024) :: fixed_directory
    call argument(1,mode)
    call argument(2,path)
    select case (mode)
    case ('config')
        call read_config(path,config,error)
        call check()
        call print_config()
    case ('ids')
        if (command_argument_count() == 3) then
            call argument(3,arg)
            read(arg,*,iostat=ios) vocab
            if (ios /= 0) error stop 'invalid test vocab argument'
            call read_input_ids(path,ids,error,vocab)
        else
            call read_input_ids(path,ids,error)
        end if
        call check()
        write(*,'(a,*(1x,i0))') 'IDS',ids
    case ('json','string','integer')
        call read_json(path,doc,error)
        call check()
        if (mode == 'string') then
            value=json_string(doc,member(doc,1,'value'),error)
            call check()
            write(*,'(a)',advance='no') 'STRING '
            do i=1,len(value)
                write(*,'(z2.2)',advance='no') iachar(value(i:i))
            end do
            write(*,*)
        else if (mode == 'integer') then
            number=json_integer(doc,1,error)
            call check()
            write(*,'(a,1x,i0)') 'INTEGER',number
        else
            print '(a)','JSON OK'
        end if
    case ('shard')
        call inspect_shard(path,tensors,shard,error)
        call check()
        write(*,'(a,2(1x,i0))') 'SHARD',shard%file_bytes,shard%header_bytes
        do i=1,size(tensors)
            call print_tensor(tensors(i))
        end do
    case ('archive','archive-fixed','model','bits','matrix','matrix-bad-shape','matrix-bad-rank', &
            'stale','stale-tail','wrong-type','bad-range')
        if (mode == 'archive-fixed') then
            fixed_directory=path
            call open_archive(fixed_directory,archive,error)
        else
            call open_archive(path,archive,error)
        end if
        call check()
        select case (mode)
        case ('archive','archive-fixed','model')
            write(*,'(a,2(1x,i0))') 'ARCHIVE',size(archive%shard),size(archive%tensor)
            do i=1,size(archive%shard)
                write(*,'(a,3(1x,i0))') 'SHARD',i,archive%shard(i)%file_bytes,archive%shard(i)%header_bytes
            end do
            if (mode == 'model') then
                call read_config(path//'/config.json',config,error)
                call check()
                call print_config()
                call validate_dense_weights(archive,config,layout,error)
                call check()
                if (.not. layout%sanitized_norm_weights .or. .not. layout%conv_kernel_fastest .or. &
                        .not. layout%packed_input_fastest) error stop 'missing layout contract'
                write(*,'(a,2(1x,i0))') 'TEXT',layout%text_tensors,layout%text_bytes
            end if
            do i=1,size(archive%tensor)
                call print_tensor(archive%tensor(i))
            end do
        case ('matrix','matrix-bad-shape','matrix-bad-rank')
            t=find_tensor(archive,'words')
            if (mode == 'matrix-bad-shape') then
                call read_u32(archive%tensor(t),wrong_matrix,error)
                call check()
            end if
            call read_u32(archive%tensor(t),matrix,error)
            call check()
            t=find_tensor(archive,'halves')
            if (mode == 'matrix-bad-rank') then
                call read_bf16(archive%tensor(t),cube,error)
                call check()
            end if
            call read_bf16(archive%tensor(t),scale,error)
            call check()
            t=find_tensor(archive,'cube')
            call read_bf16(archive%tensor(t),cube,error)
            call check()
            write(*,'(a,4(1x,z8.8))') 'MATRIX',matrix
            write(*,'(a,4(1x,z4.4))') 'SCALE',scale
            write(*,'(a,4(1x,z4.4))') 'CUBE',cube
            write(*,'(a,4(1x,i0))') 'BYTES',(packed_q8_byte(matrix(2,1),j),j=0,3)
        case default
            call argument(3,name)
            t=find_tensor(archive,name)
            if (t == 0) error stop 'test requested missing tensor'
            first=1
            count=min(archive%tensor(t)%elements,16_int64)
            if (command_argument_count() >= 5) then
                call argument(4,arg)
                read(arg,*,iostat=ios) first
                if (ios /= 0) error stop 'invalid test first argument'
                call argument(5,arg)
                read(arg,*,iostat=ios) count
                if (ios /= 0) error stop 'invalid test count argument'
            end if
            if (count < 1 .or. count > 65536) error stop 'test payload count outside safety limit'
            if (mode == 'stale') then
                open(newunit=unit,file=archive%tensor(t)%path,access='stream',form='unformatted', &
                    status='replace',action='write',iostat=ios)
                if (ios /= 0) error stop 'cannot truncate test fixture'
                close(unit)
            end if
            if (mode == 'stale-tail') then
                open(newunit=unit,file=archive%tensor(t)%path,access='stream',form='unformatted', &
                    status='old',action='write',iostat=ios)
                if (ios /= 0) error stop 'cannot truncate test fixture tail'
                write(unit,pos=archive%tensor(t)%byte_offset+3_int64,iostat=ios) achar(0)
                if (ios /= 0) error stop 'cannot seek test fixture tail'
                endfile(unit,iostat=ios)
                if (ios /= 0) error stop 'cannot truncate test fixture tail'
                close(unit)
                count=1
            end if
            if (mode == 'bad-range') first=huge(first)
            if (archive%tensor(t)%dtype == 'U32' .or. mode == 'wrong-type') then
                allocate(words(count))
                call read_u32(archive%tensor(t),words,error,first)
                call check()
                write(*,'(a,*(1x,z8.8))') 'U32',words
            else
                allocate(halves(count))
                call read_bf16(archive%tensor(t),halves,error,first)
                call check()
                write(*,'(a,*(1x,z4.4))') 'BF16',halves
            end if
        end select
    case default
        error stop 'unknown I/O test mode'
    end select

contains

    subroutine argument(position, string)
        integer, intent(in) :: position
        character(:), allocatable, intent(out) :: string
        integer :: length, status
        call get_command_argument(position,length=length,status=status)
        if (status /= 0 .or. length < 1) error stop 'missing test argument'
        allocate(character(length) :: string)
        call get_command_argument(position,string(:),status=status)
        if (status /= 0) error stop 'cannot read test argument'
    end subroutine

    subroutine check()
        if (error%failed()) then
            write(error_unit,'(a)') 'ERROR: '//error%message
            stop 1
        end if
    end subroutine

    subroutine print_tensor(tensor)
        type(tensor_info), intent(in) :: tensor
        write(*,'(a,1x,a,1x,a,*(1x,i0))') 'T',tensor%name,tensor%dtype, &
            size(tensor%shape),tensor%elements,tensor%nbytes,tensor%byte_offset,tensor%shape
    end subroutine

    subroutine print_config()
        write(*,'(a,*(1x,i0))') 'CONFIG',config%layers,config%hidden,config%intermediate,config%vocab, &
            config%recurrent_layers,config%attention_layers,config%query_heads,config%kv_heads,config%head_dim, &
            config%linear_key_heads,config%linear_value_heads,config%linear_key_head_dim,config%linear_value_head_dim, &
            config%conv_kernel,config%query_dim,config%kv_dim,config%linear_key_dim,config%linear_value_dim, &
            config%conv_dim,config%rotary_dim,config%max_position,config%group_size,config%quant_bits, &
            config%bos_id,config%pad_id,config%mtp_layers
        write(*,'(a,3(1x,es16.8))') 'PARAMS',config%rms_eps,config%rope_theta,config%rotary_fraction
        write(*,'(a,1x,l1,3(1x,i0))') 'MROPE',config%mrope_interleaved,config%mrope_section
        write(*,'(a,*(1x,i0))') 'EOS',config%eos_ids
        write(*,'(a,*(1x,l1))') 'LAYERS',config%full_attention
    end subroutine
end program
