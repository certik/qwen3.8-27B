module qwen_weights
    use, intrinsic :: iso_fortran_env, only: int64
    use model_io_support
    use safetensors_io
    use qwen_config, only: dense_config
    implicit none
    private
    public :: dense_weight_layout, validate_dense_weights

    type :: dense_weight_layout
        integer :: text_tensors=0
        integer(int64) :: text_bytes=0
        logical :: sanitized_norm_weights=.false.
        logical :: conv_kernel_fastest=.false.
        logical :: packed_input_fastest=.false.
    end type

contains

    subroutine validate_dense_weights(archive, config, layout, error)
        type(tensor_archive), intent(in) :: archive
        type(dense_config), intent(in) :: config
        type(dense_weight_layout), intent(out) :: layout
        type(io_error), intent(out) :: error
        character(:), allocatable :: prefix
        character(32) :: layer
        integer :: i, li, ios
        logical, allocatable :: used(:)
        if (.not. allocated(archive%tensor) .or. .not. allocated(config%full_attention)) then
            call fail(error,'weights: initialized archive and config required')
            return
        end if
        if (size(config%full_attention) /= config%layers .or. config%layers < 1 .or. &
                config%hidden < 1 .or. config%group_size /= 64 .or. config%quant_bits /= 8) then
            call fail(error,'weights: invalid dense config')
            return
        end if
        allocate(used(size(archive%tensor)),stat=ios)
        if (ios /= 0) then
            call fail(error,'weights: metadata selection allocation failed')
            return
        end if
        used=.false.
        do i=1,size(archive%tensor)
            if (index(archive%tensor(i)%name,'mtp.') /= 0) then
                call fail(error,'weights: MTP/unsanitized normalization shifts are unsupported')
                return
            end if
        end do
        call q8('language_model.model.embed_tokens',config%hidden,config%vocab)
        call q8('language_model.lm_head',config%hidden,config%vocab)
        call vector('language_model.model.norm.weight',config%hidden)
        do li=1,config%layers
            write(layer,'(i0)') li-1
            prefix='language_model.model.layers.'//trim(layer)
            call vector(prefix//'.input_layernorm.weight',config%hidden)
            call vector(prefix//'.post_attention_layernorm.weight',config%hidden)
            if (.not. config%full_attention(li)) then
                call q8(prefix//'.linear_attn.in_proj_qkv',config%hidden,config%conv_dim)
                call q8(prefix//'.linear_attn.in_proj_z',config%hidden,config%linear_value_dim)
                call q8(prefix//'.linear_attn.in_proj_b',config%hidden,config%linear_value_heads)
                call q8(prefix//'.linear_attn.in_proj_a',config%hidden,config%linear_value_heads)
                call q8(prefix//'.linear_attn.out_proj',config%linear_value_dim,config%hidden)
                call expect(prefix//'.linear_attn.conv1d.weight','BF16', &
                    [int(config%conv_dim,int64),int(config%conv_kernel,int64),1_int64])
                call vector(prefix//'.linear_attn.A_log',config%linear_value_heads)
                call vector(prefix//'.linear_attn.dt_bias',config%linear_value_heads)
                call vector(prefix//'.linear_attn.norm.weight',config%linear_value_head_dim)
            else
                call q8(prefix//'.self_attn.q_proj',config%hidden,2*config%query_dim)
                call q8(prefix//'.self_attn.k_proj',config%hidden,config%kv_dim)
                call q8(prefix//'.self_attn.v_proj',config%hidden,config%kv_dim)
                call q8(prefix//'.self_attn.o_proj',config%query_dim,config%hidden)
                call vector(prefix//'.self_attn.q_norm.weight',config%head_dim)
                call vector(prefix//'.self_attn.k_norm.weight',config%head_dim)
            end if
            call q8(prefix//'.mlp.gate_proj',config%hidden,config%intermediate)
            call q8(prefix//'.mlp.up_proj',config%hidden,config%intermediate)
            call q8(prefix//'.mlp.down_proj',config%intermediate,config%hidden)
        end do
        if (error%failed()) return
        do i=1,size(archive%tensor)
            if (index(archive%tensor(i)%name,'language_model.') /= 1) cycle
            if (.not. used(i)) then
                call fail(error,'weights: unexpected text tensor: '//archive%tensor(i)%name)
                return
            end if
            layout%text_tensors=layout%text_tensors+1
            layout%text_bytes=checked_add(layout%text_bytes,archive%tensor(i)%nbytes,error)
        end do
        if (error%failed()) return
        layout%sanitized_norm_weights=.true.
        layout%conv_kernel_fastest=.true.
        layout%packed_input_fastest=.true.

    contains

        subroutine expect(name, dtype, shape)
            character(*), intent(in) :: name,dtype
            integer(int64), intent(in) :: shape(:)
            integer :: t
            if (error%failed()) return
            t=find_tensor(archive,name)
            if (t == 0) then
                call fail(error,'weights: missing tensor: '//name)
                return
            end if
            if (archive%tensor(t)%dtype /= dtype .or. size(archive%tensor(t)%shape) /= size(shape)) then
                call fail(error,'weights: incompatible dtype/rank: '//name)
                return
            end if
            if (any(archive%tensor(t)%shape /= shape)) then
                call fail(error,'weights: incompatible dimensions: '//name)
                return
            end if
            used(t)=.true.
        end subroutine

        subroutine vector(name, width)
            character(*), intent(in) :: name
            integer, intent(in) :: width
            call expect(name,'BF16',[int(width,int64)])
        end subroutine

        subroutine q8(name, input, output)
            character(*), intent(in) :: name
            integer, intent(in) :: input, output
            call expect(name//'.weight','U32',[int(output,int64),int(input/4,int64)])
            call expect(name//'.scales','BF16',[int(output,int64),int(input/64,int64)])
            call expect(name//'.biases','BF16',[int(output,int64),int(input/64,int64)])
        end subroutine
    end subroutine
end module
