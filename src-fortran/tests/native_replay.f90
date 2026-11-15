program native_replay
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32
    use precision_formats
    use dense_kernels
    implicit none
    character(2048) :: source, destination
    character(32) :: mode, text
    integer :: m, k, n, length, capacity, offset, i, unit, ios
    integer(int16), allocatable :: result(:)

    call require_precision_environment()
    call get_command_argument(1,mode)
    call get_command_argument(2,source)
    call get_command_argument(3,destination)
    m=number(4)
    k=number(5)
    n=number(6)
    length=0
    capacity=0
    offset=0
    if (min(m,k) < 1 .or. n < 0) error stop 'invalid replay dimensions'
    select case (trim(mode))
    case ('linear')
        block
            integer(int32), allocatable :: words(:,:)
            integer(int16), allocatable :: x(:,:), scales(:,:), biases(:,:), y(:,:)
            allocate(words(k/4,n),y(n,m))
            call open_input('w.bin',int(k,int64)*n,unit)
            read(unit,iostat=ios) words
            call close_input(unit,ios)
            x=reshape(load_bf16('x.bin',k*m),[k,m])
            scales=reshape(load_bf16('s.bin',k/64*n),[k/64,n])
            biases=reshape(load_bf16('b.bin',k/64*n),[k/64,n])
            call linear_q8_bf16(x,words,scales,biases,y)
            result=reshape(y,[size(y)])
            write(*,'(a,l1,a,i0)') 'qmv=',native_qmv_eligible(m,k,n),',parts=',native_matrix_parts(m,k,n)
        end block
    case ('attention','rope')
        block
            integer(int16), allocatable :: q(:,:,:), raw(:,:,:), keys(:,:,:), values(:,:,:), y(:,:,:)
            real(real32) :: frequencies(32)
            allocate(q(k,n,m),y(k,n,m))
            if (trim(mode) == 'rope') then
                offset=number(7)
                raw=reshape(load_bf16('input.bin',k*n*m),[k,m,n])
            else
                length=number(7)
                capacity=number(8)
                offset=number(9)
                raw=reshape(load_bf16('q.bin',k*n*m),[k,m,n])
            end if
            do concurrent (integer :: input_head=1:n, input_token=1:m)
                q(:,input_head,input_token)=raw(:,input_token,input_head)
            end do
            if (trim(mode) == 'rope') then
                do i=0,31
                    frequencies(i+1)=2.0_real32**(-real(i,real32)/32.0_real32* &
                        (log(10000000.0_real32)/log(2.0_real32)))
                end do
                call rope_partial_inplace_bf16(q,frequencies,64,int(offset,int64))
                y=q
            else
                allocate(keys(k,n/6,length),values(k,n/6,length))
                raw=reshape(load_bf16('k.bin',k*(n/6)*capacity),[k,capacity,n/6])
                do concurrent (integer :: key_head=1:n/6, key_token=1:length)
                    keys(:,key_head,key_token)=raw(:,key_token,key_head)
                end do
                raw=reshape(load_bf16('v.bin',k*(n/6)*capacity),[k,capacity,n/6])
                do concurrent (integer :: value_head=1:n/6, value_token=1:length)
                    values(:,value_head,value_token)=raw(:,value_token,value_head)
                end do
                call sdpa_gqa_bf16(q,keys,values,y,int(offset,int64),0.0625_real32,m>1)
                write(*,'(a,i0)') 'attention-blocks=',native_attention_blocks(m,length,6,k)
            end if
            raw=reshape(y,[k,n,m])
            block
                integer(int16) :: ordered(k,m,n)
                do concurrent (integer :: output_head=1:n, output_token=1:m)
                    ordered(:,output_token,output_head)=y(:,output_head,output_token)
                end do
                result=reshape(ordered,[size(ordered)])
            end block
        end block
    case ('recurrent')
        block
            integer(int16), allocatable :: q(:,:,:), keys(:,:,:), values(:,:,:), beta(:,:), y(:,:,:)
            real(real32), allocatable :: g(:,:), state(:,:,:)
            allocate(y(k,n,m),q(k,n/3,m),keys(k,n/3,m),values(k,n,m),beta(n,m),g(n,m),state(k,k,n))
            q=reshape(load_bf16('q.bf16',k*(n/3)*m),[k,n/3,m])
            keys=reshape(load_bf16('k.bf16',k*(n/3)*m),[k,n/3,m])
            values=reshape(load_bf16('v.bf16',k*n*m),[k,n,m])
            beta=reshape(load_bf16('beta.bf16',n*m),[n,m])
            g=reshape(load_f32('g.f32',n*m),[n,m])
            state=reshape(load_f32('state-in.f32',k*k*n),[k,k,n])
            do i=1,m
                call gated_delta_step_bf16(q(:,:,i),keys(:,:,i),values(:,:,i),g(:,i),beta(:,i),state,y(:,:,i))
            end do
            result=reshape(y,[size(y)])
            open(newunit=unit,file=trim(destination)//'.state',status='replace',access='stream', &
                form='unformatted',action='write',iostat=ios)
            if (ios /= 0) error stop 'cannot write state'
            write(unit,iostat=ios) state
            call close_input(unit,ios)
        end block
    case ('conv')
        block
            integer(int16), allocatable :: x(:,:), state(:,:), weight(:,:), y(:,:)
            allocate(y(k,m))
            x=reshape(load_bf16('x.bin',k*m),[k,m])
            state=reshape(load_bf16('carry.bin',k*(n-1)),[k,n-1])
            weight=reshape(load_bf16('weight.bin',k*n),[n,k])
            call conv1d_depthwise_causal_bf16(x,state,weight,y)
            result=reshape(y,[size(y)])
            open(newunit=unit,file=trim(destination)//'.carry',status='replace',access='stream', &
                form='unformatted',action='write',iostat=ios)
            if (ios /= 0) error stop 'cannot write carry'
            write(unit,iostat=ios) state
            call close_input(unit,ios)
        end block
    case ('embedding')
        block
            integer(int32), allocatable :: ids(:), words(:,:)
            integer(int16), allocatable :: scales(:,:), biases(:,:), y(:,:)
            allocate(ids(m),words(k/4,n),y(k,m))
            call open_input('ids.bin',4_int64*m,unit)
            read(unit,iostat=ios) ids
            call close_input(unit,ios)
            call open_input('weights.bin',int(k,int64)*n,unit)
            read(unit,iostat=ios) words
            call close_input(unit,ios)
            scales=reshape(load_bf16('scales.bin',k/64*n),[k/64,n])
            biases=reshape(load_bf16('biases.bin',k/64*n),[k/64,n])
            call embed_dequant_gather_q8(ids,words,scales,biases,y)
            result=reshape(y,[size(y)])
        end block
    case ('rms','scaled-rms')
        block
            integer(int16), allocatable :: x(:,:), y(:,:), weight(:)
            real(real32) :: eps, scalar
            allocate(y(k,m))
            x=reshape(load_bf16('x.bin',k*m),[k,m])
            call get_command_argument(7,text)
            read(text,*) eps
            if (trim(mode) == 'scaled-rms') then
                call get_command_argument(8,text)
                read(text,*) scalar
                call rmsnorm_scale_bf16(x,y,eps,scalar)
            else if (n == 1) then
                weight=load_bf16('weight.bin',k)
                call rmsnorm_bf16(x,y,eps,weight)
            else
                call rmsnorm_bf16(x,y,eps)
            end if
            result=reshape(y,[size(y)])
        end block
    case ('sigmoid','silu','swiglu','gated','gate','add','g')
        block
            integer(int16), allocatable :: x(:), b(:), z(:)
            real(real32), allocatable :: g(:,:)
            x=load_bf16('x.bin',m)
            allocate(z(m))
            select case(trim(mode))
            case ('sigmoid')
                z=x
                call sigmoid_inplace_bf16(z)
            case ('silu')
                z=x
                call silu_inplace_bf16(z)
            case ('g')
                allocate(g(m,1))
                call compute_g_f32(reshape(x,[m,1]),load_bf16('a_log.bin',m),load_bf16('dt_bias.bin',m),g)
                open(newunit=unit,file=trim(destination),status='replace',access='stream', &
                    form='unformatted',action='write',iostat=ios)
                if (ios /= 0) error stop 'cannot write gate'
                write(unit,iostat=ios) g
                call close_input(unit,ios)
                stop
            case default
                b=load_bf16('b.bin',m)
                select case(trim(mode))
                case ('swiglu')
                    call silu_mul_bf16(x,b,z)
                case ('gated')
                    call silu_mul_f32_gate_bf16(x,b,z)
                case ('gate')
                    call elemwise_sigmoid_mul_bf16(x,b,z)
                case ('add')
                    z=x
                    call residual_add_bf16(z,b)
                end select
            end select
            result=z
        end block
    case ('sampler')
        block
            integer(int16), allocatable :: logits(:)
            integer(int32) :: id
            logits=load_bf16('x.bin',m)
            allocate(result(m))
            call sample_native_bf16(logits,id,result)
            write(*,'(a,i0)') 'id=',id
        end block
    case default
        error stop 'unknown replay family'
    end select
    open(newunit=unit,file=trim(destination),status='replace',access='stream',form='unformatted', &
        action='write',iostat=ios)
    if (ios /= 0) error stop 'cannot write replay output'
    write(unit,iostat=ios) result
    call close_input(unit,ios)

contains
    integer function number(argument) result(value)
        integer, intent(in) :: argument
        character(64) :: string
        integer :: status
        call get_command_argument(argument,string)
        read(string,*,iostat=status) value
        if (status /= 0) error stop 'invalid replay dimension'
    end function

    subroutine open_input(name, bytes, u)
        character(*), intent(in) :: name
        integer(int64), intent(in) :: bytes
        integer, intent(out) :: u
        integer(int64) :: actual
        integer :: status
        inquire(file=trim(source)//'/'//name,size=actual,iostat=status)
        if (status /= 0 .or. actual /= bytes) error stop 'replay input size mismatch: '//name
        open(newunit=u,file=trim(source)//'/'//name,access='stream',form='unformatted', &
            status='old',action='read',iostat=status)
        if (status /= 0) error stop 'cannot open replay input: '//name
    end subroutine

    subroutine close_input(u, status)
        integer, intent(in) :: u, status
        integer :: closed
        if (status /= 0) error stop 'replay I/O failed'
        close(u,iostat=closed)
        if (closed /= 0) error stop 'replay close failed'
    end subroutine

    function load_bf16(name, count) result(values)
        character(*), intent(in) :: name
        integer, intent(in) :: count
        integer(int16), allocatable :: values(:)
        integer :: u, status
        allocate(values(count))
        call open_input(name,2_int64*count,u)
        read(u,iostat=status) values
        call close_input(u,status)
    end function

    function load_f32(name, count) result(values)
        character(*), intent(in) :: name
        integer, intent(in) :: count
        real(real32), allocatable :: values(:)
        integer :: u, status
        allocate(values(count))
        call open_input(name,4_int64*count,u)
        read(u,iostat=status) values
        call close_input(u,status)
    end function
end program
