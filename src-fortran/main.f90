program qwen_raw
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32, real64, output_unit, error_unit
    use precision_formats, only: require_precision_environment
    use dense_model
    use model_io_support
    use qwen_config
    use safetensors_io, only: tensor_archive
    use qwen_weights, only: dense_weight_layout
    use inference_dump, only: configure_dump
    use qwen_tokenizer
    implicit none
    type(inference_model), target :: model
    type(dense_config) :: cfg
    type(tensor_archive) :: archive
    type(dense_weight_layout) :: layout
    type(layer_state), allocatable :: prefill_state(:)
    type(io_error) :: error
    type(bpe_tokenizer) :: tok
    type(text_stream) :: stream
    integer(int32), allocatable :: input_ids(:), teacher(:), generated(:), consumed(:), reset_ids(:), reset_consumed(:)
    integer(int16), allocatable :: logits(:,:), saved_logits(:,:), logprobs(:), saved_logprobs(:)
    type(execution_counts) :: inference_counts, reset_counts
    character(:), allocatable :: directory, input_path, teacher_path, dump_path, result_path, lp_path, arg, value, call_path
    character(32) :: number
    character(:), allocatable :: prompt_text, prompt_file, system_text, tokenize_text, tokenize_file
    character(:), allocatable :: detok_path, tokenizer_dir, effort, rendered, generated_text, stream_buf, terr
    character(:), allocatable :: system_file
    logical :: raw, have_prompt, have_system, have_tokenize, chat, thinking, text_out, max_set, use_tok, streaming
    integer :: i, max_tokens, n_out, reset_n, capacity, processed, unit, ios, diagnostic_layer
    integer(int64) :: count64, clock_rate, t_start, t_loaded
    real(8) :: load_s, prefill_s, decode_s, first_step_s, step_max_s, median_s
    real(8), allocatable :: step_s(:)
    logical :: timing_done
    integer(int32) :: lookahead, reset_lookahead
    logical :: fixed, reset_check, check_config, inspect
    integer :: lp_unit
    logical :: lp_on
    application: block
    call require_precision_environment()
    directory=''
    input_path=''
    teacher_path=''
    dump_path=''
    result_path=''
    lp_path=''
    max_tokens=8
    fixed=.false.
    reset_check=.false.
    check_config=.false.
    inspect=.false.
    diagnostic_layer=-2
    prompt_text=''
    prompt_file=''
    system_text=''
    system_file=''
    tokenize_text=''
    tokenize_file=''
    detok_path=''
    tokenizer_dir=''
    effort='xhigh'
    have_prompt=.false.
    have_system=.false.
    have_tokenize=.false.
    chat=.false.
    thinking=.true.
    text_out=.false.
    max_set=.false.
    streaming=.false.
    raw=.false.
    i=1
    do while (i <= command_argument_count())
        arg=argument(i)
        select case(arg)
        case('--fixed-steps')
            fixed=.true.
        case('--reset-check')
            reset_check=.true.
        case('--check-config')
            check_config=.true.
        case('--inspect')
            inspect=.true.
        case('--chat')
            chat=.true.
        case('--raw')
            raw=.true.
        case('--think')
            thinking=.true.
        case('--no-think')
            thinking=.false.
        case('--text')
            text_out=.true.
        case('--model','--input-ids','--teacher-ids','--dump','--result','--logprobs-out','--max-tokens','--dump-layer', &
                '--prompt','--prompt-file','--system','--system-file','--tokenize','--tokenize-file', &
                '--detokenize','--tokenizer','--reasoning-effort')
            i=i+1
            if (i > command_argument_count()) error stop 'missing argument value'
            value=argument(i)
            if (len(value) == 0 .and. arg /= '--prompt' .and. arg /= '--system' .and. arg /= '--tokenize') &
                error stop 'empty argument value'
            select case(arg)
            case('--model')
                directory=value
            case('--input-ids')
                input_path=value
            case('--teacher-ids')
                teacher_path=value
            case('--dump')
                dump_path=value
            case('--logprobs-out')
                lp_path=value
            case('--result')
                result_path=value
            case('--max-tokens')
                if (verify(value,'0123456789') /= 0) error stop '--max-tokens must be a positive integer'
                read(value,*,iostat=ios) max_tokens
                if (ios /= 0) error stop '--max-tokens integer overflow'
                if (max_tokens < 1) error stop '--max-tokens must be positive'
                max_set=.true.
            case('--prompt')
                prompt_text=value
                have_prompt=.true.
            case('--prompt-file')
                prompt_file=value
                have_prompt=.true.
            case('--system')
                system_text=value
                have_system=.true.
            case('--system-file')
                system_file=value
                have_system=.true.
            case('--tokenize')
                tokenize_text=value
                have_tokenize=.true.
            case('--tokenize-file')
                tokenize_file=value
                have_tokenize=.true.
            case('--detokenize')
                detok_path=value
            case('--tokenizer')
                tokenizer_dir=value
            case('--reasoning-effort')
                effort=value
                if (effort /= 'xhigh' .and. effort /= 'medium' .and. effort /= 'low') &
                    error stop '--reasoning-effort must be xhigh, medium or low'
            case('--dump-layer')
                if (verify(value,'0123456789') /= 0) error stop '--dump-layer must be a nonnegative layer'
                read(value,*,iostat=ios) diagnostic_layer
                if (ios /= 0) error stop '--dump-layer integer overflow'
            end select
        case default
            error stop 'unknown argument: '//arg
        end select
        i=i+1
    end do
    ! A text prompt is answered like a chat turn and the answer is printed as text,
    ! unless --raw asks for the prompt to be tokenized as is.
    if (have_prompt .and. .not. raw) chat=.true.
    if (have_prompt) text_out=.true.
    if (len(tokenizer_dir) == 0) tokenizer_dir=directory
    if (have_tokenize .or. len(detok_path) /= 0) then
        ! Standalone tokenizer modes: no config, no weights.
        if (len(tokenizer_dir) == 0) error stop '--model (or --tokenizer) is required'
        if (have_tokenize .and. len(detok_path) /= 0) error stop '--tokenize and --detokenize are exclusive'
        call load_tokenizer(tokenizer_dir,tok,terr)
        if (allocated(terr)) error stop terr
        if (have_tokenize) then
            if (len(tokenize_file) /= 0) then
                call read_file_bytes(tokenize_file,tokenize_text,terr)
                if (allocated(terr)) error stop terr
            end if
            if (chat) then
                call apply_system()
                tokenize_text=chat_prompt(tokenize_text,system_text,thinking,effort)
            end if
            call encode_text(tok,tokenize_text,input_ids)
            write(output_unit,'(a)',advance='no') '['
            do i=1,size(input_ids)
                write(number,'(i0)') input_ids(i)
                if (i > 1) write(output_unit,'(a)',advance='no') ','
                write(output_unit,'(a)',advance='no') trim(number)
            end do
            write(output_unit,'(a)') ']'
        else
            call read_input_ids(detok_path,input_ids,error,int(tok%n_tokens,int64))
            call require_io(error)
            generated_text=decode_ids(tok,input_ids)
            write(output_unit,'(a)',advance='no',iostat=ios) generated_text
            if (ios /= 0) error stop 'text write failed'
        end if
        exit application
    end if
    if (len(directory) == 0) error stop '--model is required'
    if (chat .and. .not. have_prompt) error stop '--chat requires --prompt or --prompt-file'
    if (have_system .and. .not. chat) error stop '--system requires --chat'
    call configure_dump(diagnostic_layer)
    call read_config(directory//'/config.json',cfg,error)
    call require_io(error)
    if (diagnostic_layer >= cfg%layers) error stop '--dump-layer outside model'
    if (check_config) then
        write(output_unit,'(a)') '{"config_valid":true,"dense_hybrid":true}'
        exit application
    end if
    if (inspect) then
        call inspect_model(directory,cfg,archive,layout)
        unit=output_unit
        call text('{"metadata_only":true')
        call field('shards',size(archive%shard))
        call field('tensors',size(archive%tensor))
        call field('text_tensors',layout%text_tensors)
        write(number,'(i0)') layout%text_bytes
        call text(',"text_bytes":'//trim(number))
        call field('layers',cfg%layers)
        call field('hidden',cfg%hidden)
        call field('intermediate',cfg%intermediate)
        call field('vocab',cfg%vocab)
        call field('recurrent_layers',cfg%recurrent_layers)
        call field('full_layers',cfg%attention_layers)
        call field('query_heads',cfg%query_heads)
        call field('kv_heads',cfg%kv_heads)
        call field('head_dim',cfg%head_dim)
        call field('query_width',cfg%query_dim)
        call field('linear_key_heads',cfg%linear_key_heads)
        call field('linear_value_heads',cfg%linear_value_heads)
        call field('linear_key_head_dim',cfg%linear_key_head_dim)
        call field('linear_value_head_dim',cfg%linear_value_head_dim)
        call field('linear_value_width',cfg%linear_value_dim)
        call field('conv_kernel',cfg%conv_kernel)
        call field('rotary_dim',cfg%rotary_dim)
        call field('max_position',cfg%max_position)
        call text(',"sanitized_norms":'//boolean(layout%sanitized_norm_weights))
        call text(',"conv_kernel_fastest":'//boolean(layout%conv_kernel_fastest))
        call text(',"packed_input_fastest":'//boolean(layout%packed_input_fastest)//'}')
        write(output_unit,'()',iostat=ios)
        if (ios /= 0) error stop 'metadata report write failed'
        exit application
    end if
    use_tok=have_prompt .or. text_out
    if (have_prompt .and. len(input_path) /= 0) error stop '--prompt and --input-ids are exclusive'
    if (len(input_path) == 0 .and. .not. have_prompt) error stop '--input-ids, --prompt or --prompt-file is required'
    if (have_prompt .and. .not. max_set) max_tokens=256
    if (use_tok) then
        call load_tokenizer(tokenizer_dir,tok,terr)
        if (allocated(terr)) error stop terr
        allocate(character(3*(tok%max_len+4)) :: stream_buf)
    end if
    if (len(lp_path) /= 0 .and. reset_check) error stop '--logprobs-out is incompatible with --reset-check'
    if (len(teacher_path) /= 0 .and. .not. fixed) error stop '--teacher-ids requires --fixed-steps'
    if (have_prompt) then
        if (len(prompt_file) /= 0) then
            call read_file_bytes(prompt_file,prompt_text,terr)
            if (allocated(terr)) error stop terr
        end if
        rendered=prompt_text
        if (chat) then
            call apply_system()
            rendered=chat_prompt(prompt_text,system_text,thinking,effort)
        end if
        call encode_text(tok,rendered,input_ids)
        if (size(input_ids) < 1) error stop 'prompt tokenizes to zero tokens'
        if (any(input_ids >= cfg%vocab)) error stop 'tokenizer produced an id outside the model vocabulary'
    else
        call read_input_ids(input_path,input_ids,error,int(cfg%vocab,int64))
        call require_io(error)
        if (use_tok) rendered=decode_ids(tok,input_ids,.false.)
    end if
    if (len(teacher_path) /= 0) then
        call read_input_ids(teacher_path,teacher,error,int(cfg%vocab,int64))
        call require_io(error)
        if (size(teacher,kind=int64) < int(max_tokens,int64)) error stop 'insufficient teacher continuation IDs'
    end if
    count64=checked_add(size(input_ids,kind=int64),int(max_tokens,int64),error)
    call require_io(error)
    if (count64 > cfg%max_position .or. count64 > huge(0)) error stop 'requested context exceeds model capacity'
    capacity=int(count64)
    call system_clock(t_start,clock_rate)
    call initialize_model(directory,size(input_ids),capacity,model)
    call system_clock(t_loaded)
    load_s=real(t_loaded-t_start,8)/real(clock_rate,8)
    timing_done=.false.
    allocate(step_s(max(max_tokens,1)))
    allocate(generated(max_tokens),consumed(max_tokens),logprobs(cfg%vocab),stat=ios)
    if (ios /= 0) error stop 'ID report allocation failed'
    lp_on=len(lp_path) /= 0
    if (lp_on) then
        open(newunit=lp_unit,file=lp_path,status='new',action='write',iostat=ios)
        if (ios /= 0) error stop 'cannot create new logprobs file'
        write(lp_unit,'(a)') '{"schema":1,"steps":['
    end if
    streaming=text_out
    call generate_request(generated,consumed,n_out,lookahead,dump_path)
    streaming=.false.
    if (text_out) then
        write(output_unit,'()')
        flush(output_unit)
    end if
    if (use_tok) generated_text=decode_ids(tok,generated(:n_out))
    if (lp_on) then
        write(lp_unit,'(a)') ']}'
        close(lp_unit)
    end if
    processed=model%position
    inference_counts=model%counts
    reset_counts=execution_counts()
    if (reset_check) then
        saved_logits=logits
        saved_logprobs=logprobs
        call snapshot_state(model,prefill_state)
        allocate(reset_ids(max_tokens),reset_consumed(max_tokens),stat=ios)
        if (ios /= 0) error stop 'reset ID allocation failed'
        call reset_model(model)
        call_path=''
        if (len(dump_path) /= 0) call_path=dump_path//'/reset'
        call generate_request(reset_ids,reset_consumed,reset_n,reset_lookahead,call_path)
        reset_counts=model%counts
        if (reset_n /= n_out) error stop 'reset/reuse: emitted count differs'
        if (any(reset_ids(:n_out) /= generated(:n_out)) .or. &
                any(reset_consumed(:n_out) /= consumed(:n_out)) .or. reset_lookahead /= lookahead .or. &
                model%position /= processed .or. any(logits /= saved_logits) .or. &
                any(logprobs /= saved_logprobs) .or. .not. same_state(model,prefill_state)) &
            error stop 'reset/reuse: IDs, final cache, logits or logprobs differs'
    end if
    call summarize_timing()
    unit=output_unit
    if (len(result_path) /= 0) then
        open(newunit=unit,file=result_path,status='new',action='write',iostat=ios)
        if (ios /= 0) error stop 'cannot create new result file'
    end if
    ! A text prompt prints only the answer; the JSON report needs --result.
    if (len(result_path) /= 0 .or. .not. have_prompt) call report(unit)
    if (len(result_path) /= 0) then
        close(unit,iostat=ios)
        if (ios /= 0) error stop 'result close failed'
    end if
    end block application

contains

    ! Accuracy check: fp32 log-softmax of the top-10 ids of the current logits
    ! (bf16 bits, host side) plus the logprob of the id that follows (teacher id
    ! if teacher-forced, else the greedy id). Only runs with --logprobs-out.
    subroutine write_topk(step, greedy, follow, comma)
        integer, intent(in) :: step
        integer(int32), intent(in) :: greedy, follow
        logical, intent(in) :: comma
        integer, parameter :: k=10
        integer :: v, j, n, m
        integer :: tid(k)
        real(real32) :: tv(k), x, mx, lse_f, lpf
        real(real64) :: ssum
        character(len=40) :: b
        n=size(logits,1)
        mx=-huge(1.0_real32)
        do v=1,n
            x=transfer(ishft(int(logits(v,1),int32),16),1.0_real32)
            if (x > mx) mx=x
        end do
        ssum=0
        m=0
        do v=1,n
            x=transfer(ishft(int(logits(v,1),int32),16),1.0_real32)
            ssum=ssum+exp(real(x-mx,real64))
            if (m < k) then
                m=m+1
                tv(m)=-huge(1.0_real32)
                tid(m)=-1
            end if
            if (x > tv(m)) then
                j=m
                do while (j > 1)
                    if (tv(j-1) >= x) exit
                    tv(j)=tv(j-1); tid(j)=tid(j-1)
                    j=j-1
                end do
                tv(j)=x; tid(j)=v-1
            end if
        end do
        lse_f=mx+real(log(ssum),real32)
        lpf=0
        if (follow >= 0) lpf=transfer(ishft(int(logits(follow+1,1),int32),16),1.0_real32)-lse_f
        write(lp_unit,'(a,i0,a,i0,a,i0,a)',advance='no') '{"step":',step,',"greedy":',greedy,',"follow":',follow,',"ids":['
        do j=1,k
            if (j > 1) write(lp_unit,'(a)',advance='no') ','
            write(b,'(i0)') tid(j)
            write(lp_unit,'(a)',advance='no') trim(b)
        end do
        write(lp_unit,'(a)',advance='no') '],"lp":['
        do j=1,k
            if (j > 1) write(lp_unit,'(a)',advance='no') ','
            write(b,'(es16.8)') tv(j)-lse_f
            write(lp_unit,'(a)',advance='no') trim(adjustl(b))
        end do
        write(b,'(es16.8)') lpf
        write(lp_unit,'(a)',advance='no') '],"follow_lp":'//trim(adjustl(b))//'}'
        if (comma) then
            write(lp_unit,'(a)') ','
        else
            write(lp_unit,'(a)') ''
        end if
    end subroutine

    subroutine generate_request(outputs, inputs, emitted, final_id, debug_directory)
        integer(int32), intent(out) :: outputs(:), inputs(:), final_id
        integer, intent(out) :: emitted
        character(*), intent(in) :: debug_directory
        integer :: position, rows, step
        integer(int32) :: current, token(1), nxt
        integer(int64) :: t_begin, t_pre, t_step0, t_step1
        character(:), allocatable :: path
        character(32) :: suffix

        call system_clock(t_begin)
        position=0
        do while (size(input_ids)-position > 1)
            rows=min(2048,size(input_ids)-position-1)
            path=''
            if (len(debug_directory) /= 0) then
                write(suffix,'(i0.3)') position
                path=debug_directory//'/prefix-'//trim(suffix)
            end if
            call forward_ids(model,input_ids(position+1:position+rows),logits,path,.true.)
            position=position+rows
        end do
        call forward_ids(model,input_ids(position+1:),logits,debug_directory)
        call sample_logits(model,logits(:,1),current,logprobs)
        call system_clock(t_pre)
        if (lp_on) then
            nxt=-1
            if (allocated(teacher)) nxt=teacher(1)
            call write_topk(0,current,nxt,.true.)
        end if
        emitted=0
        do step=1,max_tokens
            if (streaming) then
                if (fixed .or. .not. stop_id(current)) call emit_text(current)
            end if
            token(1)=current
            if (allocated(teacher)) token(1)=teacher(step)
            inputs(step)=token(1)
            path=''
            if (len(debug_directory) /= 0) then
                write(suffix,'(i0.3)') step
                path=debug_directory//'/decode-'//trim(suffix)
            end if
            ! Native submits this complete forward before yielding current.
            ! The CPU reference performs the same work synchronously.
            call system_clock(t_step0)
            call forward_ids(model,token,logits,path)
            call sample_logits(model,logits(:,1),final_id,logprobs)
            if (.not. timing_done) then
                call system_clock(t_step1)
                step_s(step)=real(t_step1-t_step0,8)/real(clock_rate,8)
            end if
            if (lp_on) then
                nxt=-1
                if (allocated(teacher)) then
                    if (step < size(teacher)) nxt=teacher(step+1)
                end if
                call write_topk(step,final_id,nxt,step < max_tokens)
            end if
            emitted=emitted+1
            outputs(emitted)=current
            if (stop_id(current) .and. .not. fixed) exit
            current=final_id
        end do
        if (streaming) then
            call stream_flush(stream,stream_buf,step)
            if (step > 0) write(output_unit,'(a)',advance='no') stream_buf(1:step)
        end if
        if (.not. timing_done) then
            prefill_s=real(t_pre-t_begin,8)/real(clock_rate,8)
            timing_done=.true.
        end if
    end subroutine

    ! Wall-clock timing (system_clock). Each decode step is forward + sampling,
    ! and sampling reads the logits on the host, so the device work is finished
    ! when the clock is read. Step 1 (first decode) is reported separately.
    subroutine summarize_timing()
        real(8), allocatable :: sorted(:)
        real(8) :: tmp
        integer :: j, k, n
        n=n_out
        first_step_s=0
        decode_s=0
        step_max_s=0
        if (n >= 1) first_step_s=step_s(1)
        if (n >= 2) then
            decode_s=sum(step_s(2:n))
            step_max_s=maxval(step_s(2:n))
        end if
        write(error_unit,'(a,f8.3,a,f8.3,a,f8.3,a)') 'timing: load ',load_s,' s, prefill ',prefill_s, &
            ' s, first decode step ',first_step_s,' s'
        if (n >= 2) then
            allocate(sorted(n-1))
            sorted=step_s(2:n)
            do j=2,n-1
                tmp=sorted(j)
                k=j-1
                do while (k >= 1)
                    if (sorted(k) <= tmp) exit
                    sorted(k+1)=sorted(k)
                    k=k-1
                end do
                sorted(k+1)=tmp
            end do
            median_s=sorted((n-1+1)/2)
            write(error_unit,'(a,i0,a,f9.3,a,f8.2,a,f8.2,a,f8.2,a)') 'timing: steady decode ',n-1,' tokens ',decode_s, &
                ' s = ',real(n-1,8)/decode_s,' tok/s, median step ',median_s*1000,' ms, max ',step_max_s*1000,' ms'
        end if
    end subroutine

    function timing_json() result(t)
        character(:), allocatable :: t
        character(400) :: buf
        real(8) :: rate
        rate=0
        if (n_out >= 2 .and. decode_s > 0) rate=real(n_out-1,8)/decode_s
        write(buf,'(a,f12.4,a,f12.4,a,f12.4,a,f12.4,a,i0,a,f12.3,a)') ',"timing":{"load_s":',load_s,',"prefill_s":',prefill_s, &
            ',"first_decode_step_s":',first_step_s,',"steady_decode_s":',decode_s,',"steady_decode_tokens":',max(n_out-1,0), &
            ',"steady_decode_tok_per_s":',rate,'}'
        t=trim(buf)
    end function

    logical function stop_id(id)
        integer(int32), intent(in) :: id
        stop_id=any(cfg%eos_ids == id)
        if (use_tok) stop_id=stop_id .or. is_stop_token(tok,int(id))
    end function

    subroutine emit_text(id)
        integer(int32), intent(in) :: id
        integer :: m, st
        call stream_feed(tok,stream,int(id),stream_buf,m)
        if (m > 0) then
            write(output_unit,'(a)',advance='no',iostat=st) stream_buf(1:m)
            if (st /= 0) error stop 'text write failed'
            flush(output_unit)
        end if
    end subroutine

    subroutine apply_system()
        if (len(system_file) /= 0) then
            call read_file_bytes(system_file,system_text,terr)
            if (allocated(terr)) error stop terr
        end if
    end subroutine

    function argument(index) result(text)
        integer, intent(in) :: index
        character(:), allocatable :: text
        integer :: n, status
        call get_command_argument(index,length=n,status=status)
        if (status /= 0) error stop 'cannot read command argument'
        allocate(character(n) :: text,stat=status)
        if (status /= 0) error stop 'argument allocation failed'
        call get_command_argument(index,text(:),status=status)
        if (status /= 0) error stop 'cannot read command argument'
    end function

    function boolean(value) result(text)
        logical, intent(in) :: value
        character(:), allocatable :: text
        text='false'
        if (value) text='true'
    end function

    subroutine report(u)
        integer, intent(in) :: u
        call text('{"schema":3,"input_ids":')
        call ids(input_ids)
        call text(',"generated_ids":')
        call ids(generated(:n_out))
        call text(',"consumed_decode_ids":')
        call ids(consumed(:n_out))
        call text(',"arithmetic_profile":"native-semantic-fortran-cpu","rms_accumulation":"fp32-sum"')
        call field('requested_generated',max_tokens)
        call field('actual_generated',n_out)
        call text(',"fixed_steps":'//boolean(fixed)//',"teacher_forced":'//boolean(allocated(teacher)))
        call field('activation_capacity',size(input_ids))
        call field('KV_capacity',capacity)
        call field('inference_cache_tokens_processed',processed)
        count64=int(processed,int64)
        count64=count64+reset_counts%cache_tokens
        write(number,'(i0)') count64
        call text(',"cache_tokens_processed":'//trim(number))
        call field('inference_forward_calls',inference_counts%forwards)
        call field('cache_only_forward_calls',inference_counts%prefixes)
        call field('head_calls',inference_counts%heads)
        call field('sampler_calls',inference_counts%samplers)
        call field('terminal_prefixes_pruned',inference_counts%terminal_pruned)
        call field('quantized_vector_calls',inference_counts%qmv)
        call field('quantized_matrix_calls',inference_counts%qmm)
        call field('typed_split_reducer_calls',inference_counts%reducers)
        call field('attention_vector_calls',inference_counts%vector_attention)
        call field('attention_two_pass_calls',inference_counts%two_pass_attention)
        call field('attention_generic_calls',inference_counts%generic_attention)
        call field('recurrent_step_calls',inference_counts%recurrent_steps)
        call field('cached_step_count',n_out)
        call field('emitted_cached_count',n_out-1)
        call field('lookahead_id',int(lookahead))
        call field('reset_forward_calls',reset_counts%forwards)
        write(number,'(i0)') int(inference_counts%forwards,int64)+reset_counts%forwards
        call text(',"model_forward_calls":'//trim(number))
        call text(',"extra_unemitted_decode_forwards":1,"reset_checked":'//boolean(reset_check))
        call text(',"reset_cache_and_logits_exact":'//boolean(reset_check))
        call text(timing_json())
        call text(',"dump_io_included":'//boolean(len(dump_path) /= 0)//',"throughput_valid":false')
        if (use_tok) then
            call text(',"prompt_text":"'//json_escape(rendered)//'"')
            call text(',"generated_text":"'//json_escape(generated_text)//'"')
        end if
        call text('}')
        write(u,'()',iostat=ios)
        if (ios /= 0) error stop 'result write failed'
    end subroutine

    subroutine text(value)
        character(*), intent(in) :: value
        write(unit,'(a)',advance='no',iostat=ios) value
        if (ios /= 0) error stop 'result write failed'
    end subroutine

    subroutine field(name, value)
        character(*), intent(in) :: name
        integer, intent(in) :: value
        write(number,'(i0)') value
        call text(',"'//name//'":'//trim(number))
    end subroutine

    subroutine ids(values)
        integer(int32), intent(in) :: values(:)
        integer :: j
        call text('[')
        do j=1,size(values)
            if (j > 1) call text(',')
            write(number,'(i0)') values(j)
            call text(trim(number))
        end do
        call text(']')
    end subroutine
end program
