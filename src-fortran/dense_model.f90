module dense_model
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use precision_formats
    use model_io_support
    use qwen_config
    use safetensors_io
    use qwen_weights
    use dense_kernels
    use inference_dump
    implicit none
    private
    public :: inference_model, layer_state, initialize_model, forward_ids, reset_model
    public :: snapshot_state, same_state, require_io, checked_shape, inspect_model
    public :: execution_counts, sample_logits

    type :: execution_counts
        integer :: forwards=0, prefixes=0, heads=0, samplers=0, cache_tokens=0, terminal_pruned=0
        integer :: qmv=0, qmm=0, reducers=0, vector_attention=0, two_pass_attention=0, generic_attention=0
        integer :: recurrent_steps=0
    end type

    type :: q8_matrix
        integer(int32), allocatable :: words(:,:)
        integer(int16), allocatable :: scales(:,:), offsets(:,:)
    end type

    type :: layer_weights
        integer(int16), allocatable :: input_norm(:), post_norm(:), norm(:), knorm(:)
        integer(int16), allocatable :: conv(:,:), a_log(:), dt_bias(:)
        ! cat holds the projections that read the same activation, stacked along
        ! the output rows: qkv, z, b, a of a delta layer; q, k, v of an attention
        ! layer. gu stacks gate and up. A launch over the stack replaces one launch
        ! per projection when a single token is decoded.
        type(q8_matrix) :: cat, gu, out, down
    end type

    type :: layer_state
        integer(int16), allocatable :: conv(:,:), k(:,:,:), v(:,:,:)
        real(real32), allocatable :: delta(:,:,:), delta2(:,:,:)
        ! The pending state update of the decode step (see gated_delta_lazy_bf16): its decay, delta and key column.
        real(real32), allocatable :: pd(:,:)
        integer(int16), allocatable :: pk(:,:)
    end type

    ! Host flags per layer of the lazy recurrent step (kept out of the layer state, whose host accesses would
    ! make the host wait for the device): whether an update is pending, and whether the state is delta2
    ! rather than delta (after an odd number of lazy steps).
    logical, save :: lazy_layer(512) = .false., flip_layer(512) = .false.

    type :: inference_model
        type(dense_config) :: cfg
        type(q8_matrix) :: embedding, head
        integer(int16), allocatable :: norm(:)
        type(layer_weights), allocatable :: weight(:)
        type(layer_state), allocatable :: state(:)
        real(real32), allocatable :: frequencies(:)
        integer :: position=0, prompt_capacity=0, capacity=0
        type(execution_counts) :: counts
    end type

contains

    subroutine require_io(error)
        type(io_error), intent(in) :: error
        if (error%failed()) then
            write(error_unit,'(a)') error%message
            error stop 'model input failed'
        end if
    end subroutine

    subroutine checked_shape(shape)
        integer, intent(in) :: shape(:)
        type(io_error) :: error
        integer(int64) :: n
        integer :: i
        n=1
        do i=1,size(shape)
            n=checked_multiply(n,int(shape(i),int64),error)
        end do
        call require_io(error)
        if (n > huge(0)) error stop 'allocation exceeds scalar kernel index capacity'
    end subroutine

    subroutine allocation_ok(ios)
        integer, intent(in) :: ios
        if (ios /= 0) error stop 'model allocation failed'
    end subroutine

    subroutine inspect_model(directory, cfg, archive, layout)
        character(*), intent(in) :: directory
        type(dense_config), intent(out) :: cfg
        type(tensor_archive), intent(out) :: archive
        type(dense_weight_layout), intent(out) :: layout
        type(io_error) :: error
        call read_config(trim(directory)//'/config.json',cfg,error)
        call require_io(error)
        call open_archive(trim(directory),archive,error)
        call require_io(error)
        call validate_dense_weights(archive,cfg,layout,error)
        call require_io(error)
    end subroutine

    subroutine read_matrix(archive, name, k, n, matrix)
        type(tensor_archive), intent(in) :: archive
        character(*), intent(in) :: name
        integer, intent(in) :: k, n
        type(q8_matrix), target, intent(out) :: matrix
        type(io_error) :: error
        integer :: ios, t
        call checked_shape([k/4,n])
        allocate(matrix%words(k/4,n),matrix%scales(k/64,n),matrix%offsets(k/64,n),stat=ios)
        call allocation_ok(ios)
        t=find_tensor(archive,name//'.weight')
        if (t == 0) error stop 'missing validated projection weight'
        call read_u32(archive%tensor(t),matrix%words,error)
        call require_io(error)
        t=find_tensor(archive,name//'.scales')
        if (t == 0) error stop 'missing validated projection scales'
        call read_bf16(archive%tensor(t),matrix%scales,error)
        call require_io(error)
        t=find_tensor(archive,name//'.biases')
        if (t == 0) error stop 'missing validated projection offsets'
        call read_bf16(archive%tensor(t),matrix%offsets,error)
        call require_io(error)
    end subroutine

    subroutine allocate_stack(k, n, matrix)
        integer, intent(in) :: k, n
        type(q8_matrix), intent(out) :: matrix
        integer :: ios
        call checked_shape([k/4,n])
        allocate(matrix%words(k/4,n),matrix%scales(k/64,n),matrix%offsets(k/64,n),stat=ios)
        call allocation_ok(ios)
    end subroutine

    ! Reads the projection `name` into the output rows lo+1:lo+n of the stack.
    subroutine read_rows(archive, name, k, lo, n, matrix)
        type(tensor_archive), intent(in) :: archive
        character(*), intent(in) :: name
        integer, intent(in) :: k, lo, n
        type(q8_matrix), intent(inout) :: matrix
        type(io_error) :: error
        integer :: t
        t=find_tensor(archive,name//'.weight')
        if (t == 0) error stop 'missing validated projection weight'
        call read_u32(archive%tensor(t),matrix%words(:,lo+1:lo+n),error)
        call require_io(error)
        t=find_tensor(archive,name//'.scales')
        if (t == 0) error stop 'missing validated projection scales'
        call read_bf16(archive%tensor(t),matrix%scales(:,lo+1:lo+n),error)
        call require_io(error)
        t=find_tensor(archive,name//'.biases')
        if (t == 0) error stop 'missing validated projection offsets'
        call read_bf16(archive%tensor(t),matrix%offsets(:,lo+1:lo+n),error)
        call require_io(error)
    end subroutine

    subroutine read_vector(archive, name, n, vector)
        type(tensor_archive), intent(in) :: archive
        character(*), intent(in) :: name
        integer, intent(in) :: n
        integer(int16), allocatable, intent(out) :: vector(:)
        type(io_error) :: error
        integer :: t, ios
        call checked_shape([n])
        allocate(vector(n),stat=ios)
        call allocation_ok(ios)
        t=find_tensor(archive,name)
        if (t == 0) error stop 'missing validated vector'
        call read_bf16(archive%tensor(t),vector,error)
        call require_io(error)
    end subroutine

    subroutine initialize_model(directory, prompt_capacity, capacity, model)
        character(*), intent(in) :: directory
        integer, intent(in) :: prompt_capacity, capacity
        type(inference_model), target, intent(out) :: model
        type(tensor_archive) :: archive
        type(dense_weight_layout) :: layout
        type(dense_config) :: c
        type(io_error) :: error
        integer(int16), allocatable :: flat(:)
        integer :: li, ios, d, t
        character(32) :: number
        character(:), allocatable :: p
        call require_precision_environment()
        call inspect_model(directory,model%cfg,archive,layout)
        c=model%cfg
        if (prompt_capacity < 1 .or. capacity < prompt_capacity .or. capacity > c%max_position) &
            error stop 'requested context exceeds model capacity'
        ! Validate every workspace/state product before loading any payload.
        call checked_shape([prompt_capacity,max(c%hidden,c%intermediate, &
            2*c%query_dim,c%conv_dim,c%linear_value_dim)])
        call checked_shape([c%vocab])
        call checked_shape([prompt_capacity,c%linear_key_heads,c%linear_key_head_dim])
        call checked_shape([c%linear_value_heads,c%linear_value_head_dim,c%linear_key_head_dim])
        call checked_shape([c%conv_dim,c%conv_kernel])
        call checked_shape([c%head_dim,c%kv_heads,capacity])
        model%prompt_capacity=prompt_capacity
        model%capacity=capacity
        allocate(model%weight(c%layers),model%state(c%layers),model%frequencies(c%rotary_dim/2),stat=ios)
        call allocation_ok(ios)
        do d=0,c%rotary_dim/2-1
            model%frequencies(d+1)=2.0_real32**(-real(2*d,real32)/real(c%rotary_dim,real32)* &
                (log(c%rope_theta)/log(2.0_real32)))
        end do
        call read_matrix(archive,'language_model.model.embed_tokens',c%hidden,c%vocab,model%embedding)
        call read_matrix(archive,'language_model.lm_head',c%hidden,c%vocab,model%head)
        call read_vector(archive,'language_model.model.norm.weight',c%hidden,model%norm)
        do li=1,c%layers
            write(number,'(i0)') li-1
            p='language_model.model.layers.'//trim(number)
            associate(w=>model%weight(li),s=>model%state(li))
                call read_vector(archive,p//'.input_layernorm.weight',c%hidden,w%input_norm)
                call read_vector(archive,p//'.post_attention_layernorm.weight',c%hidden,w%post_norm)
                call allocate_stack(c%hidden,2*c%intermediate,w%gu)
                call read_rows(archive,p//'.mlp.gate_proj',c%hidden,0,c%intermediate,w%gu)
                call read_rows(archive,p//'.mlp.up_proj',c%hidden,c%intermediate,c%intermediate,w%gu)
                call read_matrix(archive,p//'.mlp.down_proj',c%intermediate,c%hidden,w%down)
                if (c%full_attention(li)) then
                    p=p//'.self_attn'
                    call allocate_stack(c%hidden,2*c%query_dim+2*c%kv_dim,w%cat)
                    call read_rows(archive,p//'.q_proj',c%hidden,0,2*c%query_dim,w%cat)
                    call read_rows(archive,p//'.k_proj',c%hidden,2*c%query_dim,c%kv_dim,w%cat)
                    call read_rows(archive,p//'.v_proj',c%hidden,2*c%query_dim+c%kv_dim,c%kv_dim,w%cat)
                    call read_matrix(archive,p//'.o_proj',c%query_dim,c%hidden,w%out)
                    call read_vector(archive,p//'.q_norm.weight',c%head_dim,w%norm)
                    call read_vector(archive,p//'.k_norm.weight',c%head_dim,w%knorm)
                    allocate(s%k(c%head_dim,c%kv_heads,capacity),s%v(c%head_dim,c%kv_heads,capacity),stat=ios)
                    call allocation_ok(ios)
                else
                    p=p//'.linear_attn'
                    call allocate_stack(c%hidden,c%conv_dim+c%linear_value_dim+2*c%linear_value_heads,w%cat)
                    call read_rows(archive,p//'.in_proj_qkv',c%hidden,0,c%conv_dim,w%cat)
                    call read_rows(archive,p//'.in_proj_z',c%hidden,c%conv_dim,c%linear_value_dim,w%cat)
                    call read_rows(archive,p//'.in_proj_b',c%hidden,c%conv_dim+c%linear_value_dim, &
                        c%linear_value_heads,w%cat)
                    call read_rows(archive,p//'.in_proj_a',c%hidden,c%conv_dim+c%linear_value_dim+ &
                        c%linear_value_heads,c%linear_value_heads,w%cat)
                    call read_matrix(archive,p//'.out_proj',c%linear_value_dim,c%hidden,w%out)
                    call read_vector(archive,p//'.A_log',c%linear_value_heads,w%a_log)
                    call read_vector(archive,p//'.dt_bias',c%linear_value_heads,w%dt_bias)
                    call read_vector(archive,p//'.norm.weight',c%linear_value_head_dim,w%norm)
                    allocate(w%conv(c%conv_kernel,c%conv_dim), &
                        s%conv(c%conv_dim,c%conv_kernel-1), &
                        s%delta(c%linear_value_head_dim,c%linear_key_head_dim,c%linear_value_heads), &
                        s%delta2(c%linear_value_head_dim,c%linear_key_head_dim,c%linear_value_heads), &
                        s%pd(c%linear_value_head_dim+1,c%linear_value_heads), &
                        s%pk(c%linear_key_head_dim,c%linear_key_heads), &
                        flat(c%conv_kernel*c%conv_dim),stat=ios)
                    call allocation_ok(ios)
                    t=find_tensor(archive,p//'.conv1d.weight')
                    if (t == 0) error stop 'missing validated convolution'
                    call read_bf16(archive%tensor(t),flat,error)
                    call require_io(error)
                    w%conv=reshape(flat,shape(w%conv))
                    deallocate(flat)
                end if
            end associate
        end do
        call reset_model(model)
    end subroutine

    subroutine reset_model(model)
        type(inference_model), intent(inout) :: model
        integer :: li
        do li=1,size(model%state)
            associate(s=>model%state(li))
                if (allocated(s%conv)) then
                    s%conv=0_int16
                    s%delta=0.0_real32
                    s%pd=0.0_real32
                    s%pd(size(s%pd,1),:)=1.0_real32
                    s%pk=0_int16
                    lazy_layer(li)=.false.
                    flip_layer(li)=.false.
                else
                    s%k=0_int16
                    s%v=0_int16
                end if
            end associate
        end do
        model%position=0
        model%counts=execution_counts()
    end subroutine

    subroutine snapshot_state(model, snapshot)
        type(inference_model), intent(in) :: model
        type(layer_state), allocatable, intent(out) :: snapshot(:)
        integer :: li, ios
        allocate(snapshot(size(model%state)),stat=ios)
        call allocation_ok(ios)
        do li=1,size(snapshot)
            associate(s=>model%state(li),d=>snapshot(li))
                if (allocated(s%conv)) then
                    allocate(d%conv(size(s%conv,1),size(s%conv,2)), &
                        d%delta(size(s%delta,1),size(s%delta,2),size(s%delta,3)),stat=ios)
                    call allocation_ok(ios)
                    d%conv=s%conv
                    if (flip_layer(li)) then
                        d%delta=s%delta2
                    else
                        d%delta=s%delta
                    end if
                else
                    allocate(d%k(size(s%k,1),size(s%k,2),size(s%k,3)), &
                        d%v(size(s%v,1),size(s%v,2),size(s%v,3)),stat=ios)
                    call allocation_ok(ios)
                    d%k=s%k
                    d%v=s%v
                end if
            end associate
        end do
    end subroutine

    pure logical function same_state(model, snapshot) result(same)
        type(inference_model), intent(in) :: model
        type(layer_state), intent(in) :: snapshot(:)
        integer :: li, k, v, h
        same=.true.
        do li=1,size(snapshot)
            associate(s=>model%state(li),d=>snapshot(li))
                if (allocated(s%conv)) then
                    same=same.and.all(s%conv == d%conv)
                    do h=1,size(s%delta,3)
                        do v=1,size(s%delta,2)
                            do k=1,size(s%delta,1)
                                if (transfer(merge(s%delta2(k,v,h),s%delta(k,v,h),flip_layer(li)),0_int32) /= &
                                        transfer(d%delta(k,v,h),0_int32)) then
                                    same=.false.
                                    return
                                end if
                            end do
                        end do
                    end do
                else
                    same=same.and.all(s%k == d%k).and.all(s%v == d%v)
                end if
            end associate
        end do
    end function

    subroutine project(x, weight, y, counts)
        integer(int16), contiguous, intent(in) :: x(:,:)
        type(q8_matrix), intent(in) :: weight
        integer(int16), contiguous, intent(out) :: y(:,:)
        type(execution_counts), intent(inout) :: counts
        call linear_q8_bf16(x,weight%words,weight%scales,weight%offsets,y)
        if (native_qmv_eligible(size(x,2),size(x,1),size(y,1))) then
            counts%qmv=counts%qmv+1
        else
            counts%qmm=counts%qmm+1
            if (native_matrix_parts(size(x,2),size(x,1),size(y,1)) > 1) counts%reducers=counts%reducers+1
        end if
    end subroutine

    ! project for the output rows lo+1:lo+n of a stack; the counts are those of
    ! a projection of its own.
    subroutine project_rows(x, weight, lo, y, counts)
        integer(int16), contiguous, intent(in) :: x(:,:)
        type(q8_matrix), intent(in) :: weight
        integer, intent(in) :: lo
        integer(int16), contiguous, intent(out) :: y(:,:)
        type(execution_counts), intent(inout) :: counts
        integer :: n
        n=size(y,1)
        call linear_q8_bf16(x,weight%words(:,lo+1:lo+n),weight%scales(:,lo+1:lo+n),weight%offsets(:,lo+1:lo+n),y)
        call count_projection(size(x,2),size(x,1),n,1,counts)
    end subroutine

    subroutine count_projection(rows, width, outputs, times, counts)
        integer, intent(in) :: rows, width, outputs, times
        type(execution_counts), intent(inout) :: counts
        if (native_qmv_eligible(rows,width,outputs)) then
            counts%qmv=counts%qmv+times
        else
            counts%qmm=counts%qmm+times
            if (native_matrix_parts(rows,width,outputs) > 1) counts%reducers=counts%reducers+times
        end if
    end subroutine

    subroutine sample_logits(model, logits, index, logprobs)
        type(inference_model), intent(inout) :: model
        integer(int16), contiguous, intent(in) :: logits(:)
        integer(int32), intent(out) :: index
        integer(int16), contiguous, optional, intent(out) :: logprobs(:)

        call sample_native_bf16(logits,index,logprobs)
        model%counts%samplers=model%counts%samplers+1
    end subroutine

    subroutine forward_ids(model, ids, logits, directory, cache_only)
        type(inference_model), intent(inout) :: model
        integer(int32), contiguous, intent(in) :: ids(:)
        integer(int16), allocatable, intent(inout) :: logits(:,:)
        character(*), intent(in) :: directory
        logical, optional, intent(in) :: cache_only
        type(dense_config) :: c
        integer :: m, li, ios, l, first, last, visible, off, nz
        integer, parameter :: pad=300000
        logical :: prefix, terminal_cache
        real(real32) :: inv_s, dscale
        integer(int16), allocatable, save :: x(:,:), norm(:,:), r(:,:), mid(:,:), xs(:,:)
        integer :: cur
        logical :: head_q
        real(real32), allocatable, save :: rcos(:), rsin(:)
        real(real32), allocatable, save :: sx_h(:), sums_h(:), sx_o(:), sums_o(:), sx_m(:), sums_m(:)
        integer(int32), allocatable, save :: xw_h(:), xw_o(:), xw_m(:)
        integer(int16), allocatable, save, target :: ycat(:,:), ycf(:,:), ygu(:,:), qkv_a(:,:), z_a(:,:), a_a(:,:), b_a(:,:)
        integer(int16), allocatable, save, target :: qp_a(:,:), kp_a(:,:), vp_a(:,:), gp_a(:,:), up_a(:,:)
        integer(int16), pointer, contiguous :: qkv(:,:), z(:,:), a(:,:), beta(:,:), qp(:,:), kp(:,:), vp(:,:)
        integer(int16), pointer, contiguous :: gp(:,:), up(:,:)
        logical :: stacked, dumping
        integer, save :: workspace_key=-1
        integer :: key
        ! Counted in a local copy: a host access to model%counts while kernels run would make the host wait.
        type(execution_counts) :: lc
        integer(int16), allocatable, save :: ql(:,:), vl2(:,:), qn(:,:), dy(:,:,:)
        integer(int16), allocatable, save :: gated(:,:)
        real(real32), allocatable, save :: g(:,:)
        integer(int16), allocatable, save :: gate(:,:), ag(:,:)
        integer(int16), allocatable, save :: qs(:,:), qr(:,:,:), kr(:,:,:), attn(:,:,:)
        logical, allocatable, save :: finite_state(:,:)
        real(real32), allocatable, save :: delta_scale(:,:,:), sdpa_scores(:,:)
        c=model%cfg
        lc=model%counts
        m=size(ids)
        off=model%position
        if (m < 1 .or. m > model%prompt_capacity) error stop 'forward: invalid activation capacity'
        if (int(off,int64)+m > model%capacity) error stop 'forward: context capacity exceeded'
        if (any(ids < 0) .or. any(int(ids,int64) >= c%vocab)) error stop 'forward: ID out of vocabulary'
        visible=off+m
        prefix=.false.
        if (present(cache_only)) prefix=cache_only
        lc%forwards=lc%forwards+1
        if (prefix) lc%prefixes=lc%prefixes+1
        ! The workspace of a decode step stays allocated between steps: freeing and allocating
        ! it makes the host wait for the device and repeat host-side work at every step.
        ios=0
        key=m+100000*merge(1,0,prefix)+1000000*merge(1,0,len_trim(directory) /= 0)
        if (key /= workspace_key) call release_workspace()
        workspace_key=key
        if (allocated(logits)) then
            if (size(logits,2) /= merge(0,1,prefix)) deallocate(logits)
        end if
        if (.not. allocated(logits)) allocate(logits(c%vocab,merge(0,1,prefix)))
        if (.not. allocated(x)) allocate(x(c%hidden,m),norm(c%hidden,m),r(c%hidden,m), &
            mid(c%intermediate,m), &
            ql(c%linear_key_head_dim,2*m*c%linear_key_heads),qn(c%linear_key_head_dim,2*m*c%linear_key_heads), &
            vl2(c%linear_value_dim,m), &
            dy(c%linear_value_head_dim,c%linear_value_heads,m), &
            gated(c%linear_value_dim,m),g(c%linear_value_heads,m), &
            gate(c%query_dim,m),ag(c%query_dim,m),qs(c%head_dim,m*c%query_heads), &
            qr(c%head_dim,c%query_heads,m),kr(c%head_dim,c%kv_heads,m),attn(c%head_dim,c%query_heads,m), &
            finite_state(c%linear_value_head_dim,c%linear_value_heads), &
            delta_scale(c%linear_value_head_dim,c%linear_value_heads,12), &
            sdpa_scores(merge(1025,1,m == 1),c%query_heads),stat=ios)
        call allocation_ok(ios)
        do concurrent (integer :: i = 1:size(finite_state,1), j = 1:size(finite_state,2))
            finite_state(i,j)=.true.
        end do
        ! A single token reads each activation once for all the projections of a stack.
        dumping=len_trim(directory) /= 0
        ! Dumps use the projections one by one.
        stacked=.not. dumping .and. m == 1 .and. native_qmv_eligible(1,c%hidden,c%conv_dim+c%linear_value_dim+2*c%linear_value_heads) &
            .and. native_qmv_eligible(1,c%hidden,2*c%query_dim+2*c%kv_dim) &
            .and. native_qmv_eligible(1,c%hidden,2*c%intermediate) &
            .and. native_qmv_eligible(1,c%intermediate,c%hidden) .and. native_qmv_eligible(1,c%linear_value_dim,c%hidden) &
            .and. native_qmv_eligible(1,c%query_dim,c%hidden) &
            .and. native_qmv_eligible(1,c%hidden,c%conv_dim) .and. native_qmv_eligible(1,c%hidden,c%linear_value_dim) &
            .and. native_qmv_eligible(1,c%hidden,c%linear_value_heads) .and. native_qmv_eligible(1,c%hidden,c%kv_dim) &
            .and. native_qmv_eligible(1,c%hidden,2*c%query_dim) .and. native_qmv_eligible(1,c%hidden,c%intermediate)
        ! The small vectors the kernels write (quantized activations, scales, sums) get a block of
        ! their own, larger than a page: in unified memory a page that holds both such a vector and
        ! the descriptors the host reads at every launch moves between the host and the device at
        ! every layer, which stalled the first layer of each step for 0.4 ms.
        if (allocated(xs) .or. allocated(qkv_a)) then
            ios=0
        else if (stacked) then
            allocate(xs(c%hidden,2),xw_h(c%hidden/4+pad),sx_h(c%hidden/16+pad),sums_h(c%hidden/16+pad), &
                xw_o(max(c%linear_value_dim,c%query_dim)/4+pad),sx_o(max(c%linear_value_dim,c%query_dim)/16+pad),sums_o(max(c%linear_value_dim,c%query_dim)/16+pad), &
                xw_m(c%intermediate/4+pad),sx_m(c%intermediate/16+pad),sums_m(c%intermediate/16+pad),ygu(2*c%intermediate,1),ycat(c%conv_dim+c%linear_value_dim+2*c%linear_value_heads,1), &
                ycf(2*c%query_dim+2*c%kv_dim,1),stat=ios)
        else
            allocate(qkv_a(c%conv_dim,m),z_a(c%linear_value_dim,m),a_a(c%linear_value_heads,m), &
                b_a(c%linear_value_heads,m),qp_a(2*c%query_dim,m),kp_a(c%kv_dim,m),vp_a(c%kv_dim,m), &
                gp_a(c%intermediate,m),up_a(c%intermediate,m),stat=ios)
        end if
        if (.not. allocated(rcos)) allocate(rcos(c%rotary_dim/2),rsin(c%rotary_dim/2))
        head_q=.not. prefix .and. native_qmv_eligible(1,c%hidden,c%vocab)
        call allocation_ok(ios)
        call embed_dequant_gather_q8(ids,model%embedding%words,model%embedding%scales,model%embedding%offsets,x)
        if (m == 1) call rope_tables_f32(model%frequencies,int(off,int64),rcos,rsin)
        if (dumping) call dm('x_after_embed',-1,x)
        do li=1,c%layers
            terminal_cache=prefix.and.li == c%layers
            associate(w=>model%weight(li),s=>model%state(li))
                if (dumping) call dm('x_in',li-1,x)
                ! The input norm of later layers comes with the previous layer's last residual.
                if (li == 1) then
                    if (stacked) then
                        do concurrent (integer :: i = 1:size(r,1))
                            r(i,1)=0_int16
                        end do
                        call residual_rmsnorm_q_bf16(x(:,1),r(:,1),xs(:,1),norm(:,1),xw_h,sx_h,sums_h,c%rms_eps,w%input_norm)
                        cur=1
                    else
                        call rmsnorm_bf16(x,norm,c%rms_eps,w%input_norm)
                    end if
                end if
                if (dumping) call dm('h_after_input_norm',li-1,norm)
                ! The views of a stack below are passed as array elements, associated by
                ! sequence with explicit-shape dummies, so that no section is copied.
                if (.not. c%full_attention(li)) then
                    nz=c%conv_dim+c%linear_value_dim
                    if (stacked) then
                        call linear_q8_q_bf16(xw_h,sx_h,sums_h,w%cat%words,w%cat%scales,w%cat%offsets,ycat)
                        call count_projection(1,c%hidden,c%conv_dim,1,lc)
                        call count_projection(1,c%hidden,c%linear_value_dim,1,lc)
                        call count_projection(1,c%hidden,c%linear_value_heads,2,lc)
                        qkv=>ycat(1:c%conv_dim,:)
                        z=>ycat(c%conv_dim+1:nz,:)
                        beta=>ycat(nz+1:nz+c%linear_value_heads,:)
                        a=>ycat(nz+c%linear_value_heads+1:nz+2*c%linear_value_heads,:)
                    dscale=1.0_real32/sqrt(real(c%linear_key_head_dim,real32))
                    if (m == 1) then
                        call conv1d_silu_split_norm_bf16(qkv,s%conv,w%conv,ql,vl2,a,beta,w%a_log,w%dt_bias,g,qn, &
                            1e-6_real32,dscale*dscale,dscale,c%linear_key_heads,2*c%linear_key_heads)
                    else
                        call conv1d_silu_split_bf16(qkv,s%conv,w%conv,ql,vl2,a,beta,w%a_log,w%dt_bias,g)
                        call rmsnorm_scale_halves_bf16(ql,qn,1e-6_real32,dscale*dscale,dscale, &
                            c%linear_key_heads,2*c%linear_key_heads)
                    end if
                    do l=1,m
                        if (flip_layer(li)) then
                            call gated_delta_lazy_bf16(qn,s%delta2,s%delta,s%pd,s%pk,l,c%linear_key_heads,delta_scale)
                        else
                            call gated_delta_lazy_bf16(qn,s%delta,s%delta2,s%pd,s%pk,l,c%linear_key_heads,delta_scale)
                        end if
                        flip_layer(li)=.not. flip_layer(li)
                        lc%recurrent_steps=lc%recurrent_steps+1
                    end do
                    lazy_layer(li)=.true.
                    if (dumping) then
                        call dm('conv_state_in',li-1,s%conv)
                        call dh('delta_y',li-1,dy)
                        call df('delta_state',li-1,s%delta)
                    end if
                    else
                        call project_rows(norm,w%cat,0,qkv_a,lc)
                        if (.not. terminal_cache) call project_rows(norm,w%cat,c%conv_dim,z_a,lc)
                        ! The b projection becomes beta = sigmoid(b) in place.
                        call project_rows(norm,w%cat,nz,b_a,lc)
                        call project_rows(norm,w%cat,nz+c%linear_value_heads,a_a,lc)
                        if (dumping) then
                            call dm('qkv',li-1,qkv_a)
                            if (.not. terminal_cache) call dm('z',li-1,z_a)
                            call dm('a',li-1,a_a)
                            call dm('b',li-1,b_a)
                        end if
                        z=>z_a
                        call delta_core(qkv_a,b_a,a_a,li)
                    end if
                    if (terminal_cache) then
                        lc%terminal_pruned=lc%terminal_pruned+1
                        cycle
                    end if
                    if (stacked) then
                        call gated_rmsnorm_q_delta_view(delta_scale,vl2,g,beta,finite_state,s%pd,s%pk,qn,1, &
                            c%linear_key_heads,dy,z,gated,c%linear_value_head_dim, &
                            m*c%linear_value_heads,c%rms_eps,w%norm,xw_o,sx_o,sums_o)
                        call linear_q8_q_bf16(xw_o,sx_o,sums_o,w%out%words,w%out%scales, &
                            w%out%offsets,r)
                        lc%qmv=lc%qmv+1
                    else
                        call gated_rmsnorm_view(dy,z,gated,c%linear_value_head_dim, &
                            m*c%linear_value_heads,c%rms_eps,w%norm)
                        if (dumping) call dh('norm_out',li-1,reshape(gated,shape(dy)))
                        call project(gated,w%out,r,lc)
                    end if
                else
                    nz=2*c%query_dim
                    if (stacked) then
                        call linear_q8_q_bf16(xw_h,sx_h,sums_h,w%cat%words,w%cat%scales,w%cat%offsets,ycf)
                        call count_projection(1,c%hidden,nz,1,lc)
                        call count_projection(1,c%hidden,c%kv_dim,2,lc)
                        qp=>ycf(1:nz,:)
                        kp=>ycf(nz+1:nz+c%kv_dim,:)
                        vp=>ycf(nz+c%kv_dim+1:nz+2*c%kv_dim,:)
                            if (terminal_cache) then
                                call knorm_rope_append_bf16(kp(:,1),vp(:,1),w%knorm,rcos,rsin,c%rotary_dim, &
                                    int(off,int64),c%rms_eps,s%k,s%v)
                            else
                            call qknorm_rope_bf16(qp(:,1),kp(:,1),vp(:,1),w%norm,w%knorm,rcos,rsin,c%rotary_dim, &
                                int(off,int64),c%rms_eps,qr(:,:,1),gate(:,1),s%k,s%v)
                            if (m == 1 .and. native_attention_blocks(m,visible,c%query_heads/c%kv_heads,c%head_dim) == 1) then
                                call sdpa_decode_bf16(qr,s%k,s%v,attn,sdpa_scores, &
                                    1.0_real32/sqrt(real(c%head_dim,real32)),visible)
                            else
                                call sdpa_gqa_bf16(qr,s%k,s%v,attn,int(off,int64), &
                                    1.0_real32/sqrt(real(c%head_dim,real32)),.true.,visible)
                            end if
                            call sigmoid_mul_q_vector(gate,attn,ag,size(attn),xw_o,sx_o,sums_o)
                            if (dumping) then
                                call dm('q_proj',li-1,qp)
                                call dm('k_proj',li-1,kp)
                                call dm('v_proj',li-1,vp)
                                call dh('k_after_knorm',li-1,kr)
                                call dm('gate',li-1,gate)
                                call dh('q_after_rope',li-1,qr)
                                call dh('attn',li-1,attn)
                            end if
                        end if
                    else
                        if (.not. terminal_cache) call project_rows(norm,w%cat,0,qp_a,lc)
                        call project_rows(norm,w%cat,nz,kp_a,lc)
                        call project_rows(norm,w%cat,nz+c%kv_dim,vp_a,lc)
                        call attention_core(qp_a,kp_a,vp_a,li)
                    end if
                    if (terminal_cache) then
                        lc%terminal_pruned=lc%terminal_pruned+1
                        cycle
                    end if
                    select case(native_attention_blocks(m,visible,c%query_heads/c%kv_heads,c%head_dim))
                    case (0)
                        lc%generic_attention=lc%generic_attention+1
                    case (1)
                        lc%vector_attention=lc%vector_attention+1
                    case default
                        lc%two_pass_attention=lc%two_pass_attention+1
                    end select
                    if (stacked) then
                        call linear_q8_q_bf16(xw_o,sx_o,sums_o,w%out%words,w%out%scales,w%out%offsets,r)
                        lc%qmv=lc%qmv+1
                    else
                        call project(ag,w%out,r,lc)
                    end if
                end if
                if (dumping) call dm('out_oproj',li-1,r)
                if (stacked) then
                    call residual_rmsnorm_q_bf16(xs(:,cur),r(:,1),xs(:,3-cur),norm(:,1),xw_h,sx_h,sums_h,c%rms_eps,w%post_norm)
                    cur=3-cur
                else
                    call residual_rmsnorm_bf16(x,r,norm,c%rms_eps,w%post_norm)
                end if
                if (dumping) call dm('x_after_attn_residual',li-1,x)
                if (dumping) call dm('h_after_post_norm',li-1,norm)
                if (stacked) then
                    call linear_q8_q_bf16(xw_h,sx_h,sums_h,w%gu%words,w%gu%scales,w%gu%offsets,ygu)
                    call count_projection(1,c%hidden,c%intermediate,2,lc)
                    gp=>ygu(1:c%intermediate,:)
                    up=>ygu(c%intermediate+1:2*c%intermediate,:)
                    call silu_mul2_q_bf16(gp(:,1),up(:,1),mid(:,1),xw_m,sx_m,sums_m)
                else
                    call project_rows(norm,w%gu,0,gp_a,lc)
                    call project_rows(norm,w%gu,c%intermediate,up_a,lc)
                    if (dumping) then
                        call dm('gate_proj',li-1,gp_a)
                        call dm('up_proj',li-1,up_a)
                    end if
                    call swiglu(gp_a,up_a)
                end if
                if (dumping) call dm('swiglu_mid',li-1,mid)
                if (stacked) then
                    call linear_q8_q_bf16(xw_m,sx_m,sums_m,w%down%words,w%down%scales,w%down%offsets,r)
                    lc%qmv=lc%qmv+1
                else
                    call project(mid,w%down,r,lc)
                end if
                if (dumping) call dm('dense_down',li-1,r)
                if (dumping) call dm('out_mlp',li-1,r)
                if (li < c%layers) then
                    if (stacked) then
                        call residual_rmsnorm_q_bf16(xs(:,cur),r(:,1),xs(:,3-cur),norm(:,1),xw_h,sx_h,sums_h,c%rms_eps, &
                            model%weight(li+1)%input_norm)
                        cur=3-cur
                    else
                        call residual_rmsnorm_bf16(x,r,norm,c%rms_eps,model%weight(li+1)%input_norm)
                    end if
                else if (stacked .and. head_q) then
                    ! The final norm and the quantization for the head come with the last residual.
                    call residual_rmsnorm_q_bf16(xs(:,cur),r(:,1),xs(:,3-cur),norm(:,1),xw_h,sx_h,sums_h,c%rms_eps,model%norm)
                else if (stacked) then
                    call residual_matrix(xs(:,cur:cur),r)
                    x(:,1)=xs(:,cur)
                else
                    call residual_matrix(x,r)
                end if
                if (dumping) call dm('x_after_mlp_residual',li-1,x)
                if (dumping) then
                    if (allocated(s%conv)) then
                        call dm('cache_conv',li-1,s%conv)
                        call df('cache_delta',li-1,s%delta)
                    else
                        call dh('cache_k',li-1,s%k(:,:,1:visible))
                        call dh('cache_v',li-1,s%v(:,:,1:visible))
                    end if
                end if
            end associate
        end do
        if (.not. prefix) then
            if (stacked .and. head_q) then
                call linear_q8_q_bf16(xw_h,sx_h,sums_h,model%head%words,model%head%scales,model%head%offsets,logits)
                call count_projection(1,c%hidden,c%vocab,1,lc)
            else
                call rmsnorm_bf16(x(:,m:m),norm(:,m:m),c%rms_eps,model%norm)
                if (dumping) call dm('x_after_final_norm',-1,norm(:,m:m))
                call project(norm(:,m:m),model%head,logits,lc)
            end if
            lc%heads=lc%heads+1
            if (dumping) call dm('logits',-1,logits)
        end if
        ! A decode step leaves the check to the sampler (non-finite state gives non-finite logits): a host read here would wait for the whole step.
        if (m > 1) then
            if (.not. all(finite_state)) error stop 'forward: non-finite recurrent state'
        end if
        ! The logits are checked to be finite where they are sampled (sample_native_bf16).
        model%position=visible
        model%counts=lc
        lc%cache_tokens=lc%cache_tokens+m
        if (m > 1) then
            call release_workspace()
            workspace_key=-1
        end if
    contains
        subroutine release_workspace()
            if (allocated(x)) deallocate(x,norm,r,mid,ql,qn,vl2,dy,gated,g,gate,ag,qs,qr,kr,attn,finite_state,delta_scale,sdpa_scores)
            if (allocated(xs)) deallocate(xs,xw_h,sx_h,sums_h,xw_o,sx_o,sums_o,xw_m,sx_m,sums_m,ygu,ycat,ycf)
            if (allocated(qkv_a)) deallocate(qkv_a,z_a,a_a,b_a,qp_a,kp_a,vp_a,gp_a,up_a)
            if (allocated(rcos)) deallocate(rcos,rsin)
        end subroutine
        ! Convolution, SiLU and the split into query/key columns (qk layout: per
        ! token the Hk query heads, then the Hk key heads) and value columns; the
        ! decay g, beta = sigmoid(b) and the finiteness of the state rows are
        ! computed in the same launches as the recurrent steps. The dummies are
        ! associated by sequence with the projection outputs.
        subroutine delta_core(qkv, b, a, li)
            integer, intent(in) :: li
            integer(int16), contiguous, intent(in) :: qkv(:,:), a(:,:)
            integer(int16), contiguous, intent(inout) :: b(:,:)
            integer :: l
            real(real32) :: scale
            associate(w=>model%weight(li),s=>model%state(li))
                if (lazy_layer(li)) then
                    if (flip_layer(li)) then
                        s%delta=s%delta2
                        flip_layer(li)=.false.
                    end if
                    call gated_delta_flush_bf16(s%delta,s%pd,s%pk,c%linear_key_heads)
                    lazy_layer(li)=.false.
                end if
                call conv1d_silu_split_bf16(qkv,s%conv,w%conv,ql,vl2,a,b,w%a_log,w%dt_bias,g)
                scale=1.0_real32/sqrt(real(c%linear_key_head_dim,real32))
                call rmsnorm_scale_halves_bf16(ql,qn,1e-6_real32,scale*scale,scale, &
                    c%linear_key_heads,2*c%linear_key_heads)
                do l=1,m
                    call gated_delta_fused_bf16(qn,vl2,g,b,s%delta,dy,finite_state,l,c%linear_key_heads,delta_scale)
                    lc%recurrent_steps=lc%recurrent_steps+1
                end do
                if (dumping) then
                    call dm('conv_state_in',li-1,s%conv)
                    call dh('delta_y',li-1,dy)
                    call df('delta_state',li-1,s%delta)
                end if
            end associate
        end subroutine

        ! The key and query heads: norm, rotary embedding, cache append, attention
        ! and output gate. Leaves the gated attention in ag.
        subroutine attention_core(qp, kp, vp, li)
            integer, intent(in) :: li
            integer(int16), contiguous, intent(in) :: qp(:,:), kp(:,:), vp(:,:)
            associate(w=>model%weight(li),s=>model%state(li))
                call rmsnorm_columns(kp,kr,c%head_dim,m*c%kv_heads,c%rms_eps,w%knorm)
                call rope_partial_inplace_bf16(kr,model%frequencies,c%rotary_dim,int(off,int64))
                call append_kv_cache(kr,vp,s%k,s%v,off)
                if (terminal_cache) return
                ! Each query head's projection is followed by its gate.
                do concurrent (integer :: t = 1:m, hq = 1:c%query_heads, d = 1:c%head_dim)
                    qs(d,(t-1)*c%query_heads+hq) = qp((hq-1)*2*c%head_dim+d,t)
                    gate((hq-1)*c%head_dim+d,t) = qp((2*hq-1)*c%head_dim+d,t)
                end do
                call rmsnorm_columns(qs,qr,c%head_dim,m*c%query_heads,c%rms_eps,w%norm)
                call rope_partial_inplace_bf16(qr,model%frequencies,c%rotary_dim,int(off,int64))
                call sdpa_gqa_bf16(qr,s%k,s%v,attn,int(off,int64), &
                    1.0_real32/sqrt(real(c%head_dim,real32)),.true.,visible)
                call sigmoid_mul_vector(gate,attn,ag,size(ag))
                if (dumping) then
                    call dm('q_proj',li-1,qp)
                    call dm('k_proj',li-1,kp)
                    call dm('v_proj',li-1,vp)
                    call dh('k_after_knorm',li-1,kr)
                    call dm('gate',li-1,gate)
                    call dh('q_after_rope',li-1,qr)
                    call dh('attn',li-1,attn)
                end if
            end associate
        end subroutine

        ! mid = silu(gate) * up, for the activations of all the tokens.
        subroutine swiglu(gp, up)
            integer(int16), contiguous, intent(in) :: gp(:,:), up(:,:)
            call silu_mul2_bf16(gp,up,mid)
        end subroutine

        subroutine dm(name, layer, values)
            character(*), intent(in) :: name
            integer, intent(in) :: layer
            integer(int16), intent(in) :: values(:,:)
            if (len_trim(directory) == 0) return
            call dump_bf16(directory,name,layer,reshape(values,[size(values)]),[size(values,2),size(values,1)])
        end subroutine
        subroutine dh(name, layer, values)
            character(*), intent(in) :: name
            integer, intent(in) :: layer
            integer(int16), intent(in) :: values(:,:,:)
            if (len_trim(directory) == 0) return
            call dump_bf16(directory,name,layer,reshape(values,[size(values)]), &
                [size(values,3),size(values,2),size(values,1)])
        end subroutine
        subroutine dg(name, layer, values)
            character(*), intent(in) :: name
            integer, intent(in) :: layer
            real(real32), intent(in) :: values(:,:)
            if (len_trim(directory) == 0) return
            call dump_f32(directory,name,layer,reshape(values,[size(values)]),[size(values,2),size(values,1)])
        end subroutine
        subroutine df(name, layer, values)
            character(*), intent(in) :: name
            integer, intent(in) :: layer
            real(real32), intent(in) :: values(:,:,:)
            if (len_trim(directory) == 0) return
            call dump_f32(directory,name,layer,reshape(values,[size(values)]), &
                [size(values,3),size(values,2),size(values,1)])
        end subroutine
    end subroutine

    ! Stores the new tokens' keys k, and their values v (one column of all
    ! heads per token), in the cache positions k_cache and v_cache.
    subroutine append_kv_cache(k, v, k_cache, v_cache, offset)
        integer(int16), contiguous, intent(in) :: k(:,:,:), v(:,:)
        integer(int16), contiguous, intent(inout) :: k_cache(:,:,:), v_cache(:,:,:)
        integer, intent(in) :: offset
        if (size(k_cache,1) /= size(k,1) .or. size(k_cache,2) /= size(k,2) .or. &
                any(shape(v_cache) /= shape(k_cache)) .or. offset < 0 .or. offset+size(k,3) > size(k_cache,3) .or. &
                size(v,1) /= size(k,1)*size(k,2) .or. size(v,2) /= size(k,3)) &
            error stop 'KV cache shape mismatch'
        do concurrent (integer :: t = 1:size(k,3), h = 1:size(k,2), d = 1:size(k,1))
            k_cache(d,h,offset+t) = k(d,h,t)
            v_cache(d,h,offset+t) = v((h-1)*size(k,1)+d,t)
        end do
    end subroutine

    ! The kernels below see a whole activation with another shape: an
    ! explicit-shape dummy is associated with its contiguous actual argument
    ! in array element order (sequence association), so these views copy
    ! nothing.

    ! Normalizes x seen as `columns` columns of `width` features.
    subroutine rmsnorm_columns(x, y, width, columns, eps, weight)
        integer, intent(in) :: width, columns
        integer(int16), intent(in) :: x(width,columns)
        integer(int16), intent(out) :: y(width,columns)
        real(real32), intent(in) :: eps
        integer(int16), contiguous, intent(in) :: weight(:)
        call rmsnorm_bf16(x,y,eps,weight)
    end subroutine

    subroutine gated_rmsnorm_view(x, z, y, width, columns, eps, weight)
        integer, intent(in) :: width, columns
        integer(int16), intent(in) :: x(width,columns), z(width,columns)
        integer(int16), intent(out) :: y(width,columns)
        real(real32), intent(in) :: eps
        integer(int16), contiguous, intent(in) :: weight(:)
        call gated_rmsnorm_bf16(x,z,y,eps,weight)
    end subroutine

    subroutine silu_mul_f32_gate_vector(gate, up, out, n)
        integer, intent(in) :: n
        integer(int16), intent(in) :: gate(n), up(n)
        integer(int16), intent(out) :: out(n)
        call silu_mul_f32_gate_bf16(gate,up,out)
    end subroutine

    subroutine sigmoid_mul_q_vector(gate, x, out, n, xw, sx, sums)
        integer, intent(in) :: n
        integer(int16), intent(in) :: gate(n), x(n)
        integer(int16), intent(out) :: out(n)
        integer(int32), contiguous, intent(out) :: xw(:)
        real(real32), contiguous, intent(out) :: sx(:), sums(:)
        call sigmoid_mul_q_bf16(gate,x,out,xw(1:n/4),sx(1:n/16),sums(1:n/16))
    end subroutine

    subroutine gated_rmsnorm_q_delta_view(dl, v, g, beta, finite, pd, pk, qk, token, hk, x, z, y, width, columns, &
            eps, weight, xw, sx, sums)
        integer, intent(in) :: width, columns, token, hk
        real(real32), contiguous, intent(in) :: dl(:,:,:), g(:,:)
        integer(int16), contiguous, intent(in) :: v(:,:), beta(:,:), qk(:,:)
        logical, contiguous, intent(inout) :: finite(:,:)
        real(real32), contiguous, intent(inout) :: pd(:,:)
        integer(int16), contiguous, intent(inout) :: pk(:,:)
        integer(int16), intent(inout) :: x(width,columns)
        integer(int16), intent(in) :: z(width,columns)
        integer(int16), intent(out) :: y(width,columns)
        real(real32), intent(in) :: eps
        integer(int16), contiguous, intent(in) :: weight(:)
        integer(int32), contiguous, intent(out) :: xw(:)
        real(real32), contiguous, intent(out) :: sx(:), sums(:)
        call gated_rmsnorm_q_delta_bf16(dl,v,g,beta,finite,pd,pk,qk,token,hk,x,z,y,eps,weight, &
            xw(1:width*columns/4),sx(1:width*columns/16),sums(1:width*columns/16))
    end subroutine

    subroutine gated_rmsnorm_q_view(x, z, y, width, columns, eps, weight, xw, sx, sums)
        integer, intent(in) :: width, columns
        integer(int16), intent(in) :: x(width,columns), z(width,columns)
        integer(int16), intent(out) :: y(width,columns)
        real(real32), intent(in) :: eps
        integer(int16), contiguous, intent(in) :: weight(:)
        integer(int32), contiguous, intent(out) :: xw(:)
        real(real32), contiguous, intent(out) :: sx(:), sums(:)
        call gated_rmsnorm_q_bf16(x,z,y,eps,weight,xw(1:width*columns/4),sx(1:width*columns/16),sums(1:width*columns/16))
    end subroutine

    subroutine sigmoid_mul_vector(gate, x, out, n)
        integer, intent(in) :: n
        integer(int16), intent(in) :: gate(n), x(n)
        integer(int16), intent(out) :: out(n)
        call elemwise_sigmoid_mul_bf16(gate,x,out)
    end subroutine

    subroutine silu_matrix(x)
        integer(int16), contiguous, intent(inout) :: x(:,:)
        integer :: i
        do i=1,size(x,2)
            call silu_inplace_bf16(x(:,i))
        end do
    end subroutine

    subroutine sigmoid_matrix(x)
        integer(int16), contiguous, intent(inout) :: x(:,:)
        integer :: i
        do i=1,size(x,2)
            call sigmoid_inplace_bf16(x(:,i))
        end do
    end subroutine

    subroutine residual_matrix(x, delta)
        integer(int16), contiguous, intent(inout) :: x(:,:)
        integer(int16), contiguous, intent(in) :: delta(:,:)
        integer :: i
        do i=1,size(x,2)
            call residual_add_bf16(x(:,i),delta(:,i))
        end do
    end subroutine
end module
