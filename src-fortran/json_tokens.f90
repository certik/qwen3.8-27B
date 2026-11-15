module json_tokens
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use model_io_support, only: io_error, fail, read_text, max_json_bytes, require_byte_io
    implicit none
    private
    public :: json_document, json_token, parse_json, read_json, member, element
    public :: json_integer, json_real, json_string, json_boolean, expect_kind
    public :: same_text
    integer, parameter, public :: json_object=1, json_array=2, json_text=3, json_number=4, json_bool=5, json_null=6
    integer, parameter :: max_tokens=1000000, max_depth=64

    type :: json_token
        integer :: kind=0, first=0, next=0, count=0
        character(:), allocatable :: text
    end type
    type :: json_document
        type(json_token), allocatable :: token(:)
        integer :: used=0
    end type

contains

    subroutine read_json(path, doc, error)
        character(*), intent(in) :: path
        type(json_document), intent(out) :: doc
        type(io_error), intent(out) :: error
        character(:), allocatable :: text
        call read_text(path, text, error)
        if (error%failed()) return
        call parse_json(text, doc, error)
    end subroutine

    subroutine parse_json(text, doc, error)
        character(*), intent(in) :: text
        type(json_document), intent(out) :: doc
        type(io_error), intent(out) :: error
        integer :: pos, root, ios
        call require_byte_io(error)
        if (error%failed()) return
        if (len(text, kind=int64) > max_json_bytes) then
            call fail(error, 'JSON exceeds 16 MiB limit')
            return
        end if
        allocate(doc%token(0:127), stat=ios)
        if (ios /= 0) then
            call fail(error, 'JSON token allocation failed')
            return
        end if
        pos = 1
        call parse_value(text, pos, doc, root, 0, error)
        if (error%failed()) return
        call whitespace(text, pos)
        if (pos /= len(text)+1) call syntax(error, pos, 'trailing data')
    end subroutine

    subroutine syntax(error, pos, message)
        type(io_error), intent(inout) :: error
        integer, intent(in) :: pos
        character(*), intent(in) :: message
        character(32) :: number
        write(number, '(i0)') pos
        call fail(error, 'JSON byte '//trim(number)//': '//message)
    end subroutine

    subroutine whitespace(text, pos)
        character(*), intent(in) :: text
        integer, intent(inout) :: pos
        do while (pos <= len(text))
            select case (iachar(text(pos:pos)))
            case (9,10,13,32)
                pos = pos + 1
            case default
                exit
            end select
        end do
    end subroutine

    subroutine new_token(doc, index, error)
        type(json_document), intent(inout) :: doc
        integer, intent(out) :: index
        type(io_error), intent(inout) :: error
        type(json_token), allocatable :: grown(:)
        integer :: ios, capacity
        index = 0
        if (doc%used >= max_tokens) then
            call fail(error, 'JSON exceeds one million token limit')
            return
        end if
        if (doc%used == ubound(doc%token,1)) then
            capacity = min(max_tokens, 2 * size(doc%token))
            allocate(grown(0:capacity), stat=ios)
            if (ios /= 0) then
                call fail(error, 'JSON token allocation failed')
                return
            end if
            grown(:doc%used) = doc%token(:doc%used)
            call move_alloc(grown, doc%token)
        end if
        doc%used = doc%used + 1
        index = doc%used
    end subroutine

    recursive subroutine parse_value(text, pos, doc, index, depth, error)
        character(*), intent(in) :: text
        integer, intent(inout) :: pos
        type(json_document), intent(inout) :: doc
        integer, intent(out) :: index
        integer, intent(in) :: depth
        type(io_error), intent(inout) :: error
        integer :: child, key, previous, start
        character :: opening, closing
        index = 0
        call whitespace(text, pos)
        if (depth > max_depth) then
            call syntax(error, pos, 'nesting exceeds 64')
            return
        end if
        if (pos > len(text)) then
            call syntax(error, pos, 'unexpected end')
            return
        end if
        call new_token(doc, index, error)
        if (error%failed()) return
        opening = text(pos:pos)
        select case (opening)
        case ('{','[')
            closing = ']'
            doc%token(index)%kind = json_array
            if (opening == '{') then
                closing = '}'
                doc%token(index)%kind = json_object
            end if
            pos = pos + 1
            doc%token(index)%first = doc%used + 1
            call whitespace(text, pos)
            if (pos <= len(text)) then
                if (text(pos:pos) == closing) then
                    pos = pos + 1
                    doc%token(index)%next = doc%used + 1
                    return
                end if
            end if
            do
                if (opening == '{') then
                    if (pos > len(text)) then
                        call syntax(error, pos, 'missing object key')
                        return
                    end if
                    if (text(pos:pos) /= '"') then
                        call syntax(error, pos, 'object key must be string')
                        return
                    end if
                    call new_token(doc, key, error)
                    if (error%failed()) return
                    doc%token(key)%kind = json_text
                    call parse_string(text, pos, doc%token(key)%text, error)
                    if (error%failed()) return
                    previous = doc%token(index)%first
                    do while (previous < key)
                        if (same_text(doc%token(previous)%text,doc%token(key)%text)) then
                            call syntax(error, pos, 'duplicate object key: '//doc%token(key)%text)
                            return
                        end if
                        previous = doc%token(previous+1)%next
                    end do
                    doc%token(key)%next = key + 1
                    call whitespace(text, pos)
                    if (pos > len(text)) then
                        call syntax(error, pos, 'missing colon')
                        return
                    end if
                    if (text(pos:pos) /= ':') then
                        call syntax(error, pos, 'expected colon')
                        return
                    end if
                    pos = pos + 1
                end if
                call parse_value(text, pos, doc, child, depth+1, error)
                if (error%failed()) return
                doc%token(index)%count = doc%token(index)%count + 1
                call whitespace(text, pos)
                if (pos > len(text)) then
                    call syntax(error, pos, 'unterminated container')
                    return
                end if
                if (text(pos:pos) == closing) exit
                if (text(pos:pos) /= ',') then
                    call syntax(error, pos, 'expected comma or closing delimiter')
                    return
                end if
                pos = pos + 1
                call whitespace(text, pos)
            end do
            pos = pos + 1
        case ('"')
            doc%token(index)%kind = json_text
            call parse_string(text, pos, doc%token(index)%text, error)
        case ('t')
            call literal(text, pos, 'true', error)
            doc%token(index)%kind = json_bool
            doc%token(index)%text = 'true'
        case ('f')
            call literal(text, pos, 'false', error)
            doc%token(index)%kind = json_bool
            doc%token(index)%text = 'false'
        case ('n')
            call literal(text, pos, 'null', error)
            doc%token(index)%kind = json_null
        case ('-','0':'9')
            start = pos
            call parse_number(text, pos, error)
            doc%token(index)%kind = json_number
            doc%token(index)%text = text(start:pos-1)
        case default
            call syntax(error, pos, 'expected JSON value')
        end select
        doc%token(index)%next = doc%used + 1
    end subroutine

    subroutine literal(text, pos, value, error)
        character(*), intent(in) :: text, value
        integer, intent(inout) :: pos
        type(io_error), intent(inout) :: error
        if (pos + len(value) - 1 > len(text)) then
            call syntax(error, pos, 'incomplete literal')
        else if (text(pos:pos+len(value)-1) /= value) then
            call syntax(error, pos, 'invalid literal')
        else
            pos = pos + len(value)
        end if
    end subroutine

    logical function digit(text, pos)
        character(*), intent(in) :: text
        integer, intent(in) :: pos
        digit = .false.
        if (pos > len(text)) return
        digit = text(pos:pos) >= '0' .and. text(pos:pos) <= '9'
    end function

    subroutine parse_number(text, pos, error)
        character(*), intent(in) :: text
        integer, intent(inout) :: pos
        type(io_error), intent(inout) :: error
        if (text(pos:pos) == '-') pos = pos + 1
        if (.not. digit(text, pos)) then
            call syntax(error, pos, 'missing number digits')
            return
        end if
        if (text(pos:pos) == '0') then
            pos = pos + 1
        else
            do while (digit(text, pos))
                pos = pos + 1
            end do
        end if
        if (pos <= len(text)) then
            if (text(pos:pos) == '.') then
                pos = pos + 1
                if (.not. digit(text, pos)) call syntax(error, pos, 'missing fractional digits')
                do while (digit(text, pos))
                    pos = pos + 1
                end do
            end if
        end if
        if (pos <= len(text)) then
            if (text(pos:pos) == 'e' .or. text(pos:pos) == 'E') then
                pos = pos + 1
                if (pos <= len(text)) then
                    if (text(pos:pos) == '+' .or. text(pos:pos) == '-') pos = pos + 1
                end if
                if (.not. digit(text, pos)) call syntax(error, pos, 'missing exponent digits')
                do while (digit(text, pos))
                    pos = pos + 1
                end do
            end if
        end if
    end subroutine

    subroutine hex4(text, pos, code, error)
        character(*), intent(in) :: text
        integer, intent(inout) :: pos
        integer, intent(out) :: code
        type(io_error), intent(inout) :: error
        integer :: i, d
        code = 0
        if (pos+3 > len(text)) then
            call syntax(error, pos, 'incomplete Unicode escape')
            return
        end if
        do i = 1, 4
            select case (text(pos:pos))
            case ('0':'9')
                d = iachar(text(pos:pos))-iachar('0')
            case ('a':'f')
                d = iachar(text(pos:pos))-iachar('a')+10
            case ('A':'F')
                d = iachar(text(pos:pos))-iachar('A')+10
            case default
                call syntax(error, pos, 'invalid Unicode escape')
                return
            end select
            code = 16*code + d
            pos = pos + 1
        end do
    end subroutine

    subroutine parse_string(text, pos, value, error)
        character(*), intent(in) :: text
        integer, intent(inout) :: pos
        character(:), allocatable, intent(out) :: value
        type(io_error), intent(inout) :: error
        character(:), allocatable :: buffer
        integer :: n, c, cp, low, bytes, j, b, start, ios
        allocate(character(len(text)-pos) :: buffer, stat=ios)
        if (ios /= 0) then
            call fail(error, 'JSON string allocation failed')
            return
        end if
        n = 0
        pos = pos + 1
        do while (pos <= len(text))
            c = iachar(text(pos:pos))
            pos = pos + 1
            if (c == 34) then
                value = buffer(:n)
                return
            end if
            if (c < 32) then
                call syntax(error, pos-1, 'unescaped control in string')
                return
            end if
            if (c == 92) then
                if (pos > len(text)) exit
                c = iachar(text(pos:pos))
                pos = pos + 1
                select case (c)
                case (34,47,92)
                case (98)
                    c = 8
                case (102)
                    c = 12
                case (110)
                    c = 10
                case (114)
                    c = 13
                case (116)
                    c = 9
                case (117)
                    call hex4(text, pos, cp, error)
                    if (error%failed()) return
                    if (cp >= 55296 .and. cp <= 56319) then
                        if (pos+1 > len(text)) then
                            call syntax(error, pos, 'missing low surrogate')
                            return
                        end if
                        if (iachar(text(pos:pos)) /= 92 .or. text(pos+1:pos+1) /= 'u') then
                            call syntax(error, pos, 'missing low surrogate')
                            return
                        end if
                        pos = pos + 2
                        call hex4(text, pos, low, error)
                        if (error%failed()) return
                        if (low < 56320 .or. low > 57343) then
                            call syntax(error, pos, 'invalid low surrogate')
                            return
                        end if
                        cp = 65536 + (cp-55296)*1024 + low-56320
                    else if (cp >= 56320 .and. cp <= 57343) then
                        call syntax(error, pos, 'unpaired low surrogate')
                        return
                    end if
                    if (cp < 128) then
                        n=n+1; buffer(n:n)=achar(cp)
                    else if (cp < 2048) then
                        n=n+1; buffer(n:n)=achar(192+cp/64)
                        n=n+1; buffer(n:n)=achar(128+modulo(cp,64))
                    else if (cp < 65536) then
                        n=n+1; buffer(n:n)=achar(224+cp/4096)
                        n=n+1; buffer(n:n)=achar(128+modulo(cp/64,64))
                        n=n+1; buffer(n:n)=achar(128+modulo(cp,64))
                    else
                        n=n+1; buffer(n:n)=achar(240+cp/262144)
                        n=n+1; buffer(n:n)=achar(128+modulo(cp/4096,64))
                        n=n+1; buffer(n:n)=achar(128+modulo(cp/64,64))
                        n=n+1; buffer(n:n)=achar(128+modulo(cp,64))
                    end if
                    cycle
                case default
                    call syntax(error, pos-1, 'invalid string escape')
                    return
                end select
            else if (c >= 128) then
                start = pos-1
                if (c >= 194 .and. c <= 223) then
                    bytes=2; cp=c-192
                else if (c >= 224 .and. c <= 239) then
                    bytes=3; cp=c-224
                else if (c >= 240 .and. c <= 244) then
                    bytes=4; cp=c-240
                else
                    call syntax(error, start, 'invalid UTF-8 lead byte')
                    return
                end if
                if (start+bytes-1 > len(text)) then
                    call syntax(error, start, 'truncated UTF-8')
                    return
                end if
                do j=2,bytes
                    b=iachar(text(pos:pos))
                    if (b < 128 .or. b > 191) then
                        call syntax(error, pos, 'invalid UTF-8 continuation')
                        return
                    end if
                    cp=64*cp+b-128
                    pos=pos+1
                end do
                if ((bytes == 2 .and. cp < 128) .or. (bytes == 3 .and. cp < 2048) .or. &
                        (bytes == 4 .and. cp < 65536) .or. cp > 1114111 .or. &
                        (cp >= 55296 .and. cp <= 57343)) then
                    call syntax(error, start, 'invalid UTF-8 code point')
                    return
                end if
                buffer(n+1:n+bytes)=text(start:pos-1)
                n=n+bytes
                cycle
            end if
            n=n+1
            buffer(n:n)=achar(c)
        end do
        call syntax(error, pos, 'unterminated string')
    end subroutine

    integer function member(doc, object, name) result(index)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: object
        character(*), intent(in) :: name
        integer :: key, i
        index = 0
        if (object < 1 .or. object > doc%used) return
        if (doc%token(object)%kind /= json_object) return
        key = doc%token(object)%first
        do i=1,doc%token(object)%count
            if (same_text(doc%token(key)%text,name)) then
                index = key+1
                return
            end if
            key = doc%token(key+1)%next
        end do
    end function

    pure logical function same_text(a, b) result(equal)
        character(*), intent(in) :: a,b
        equal=len(a) == len(b) .and. a == b
    end function

    integer function element(doc, array, position) result(index)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: array, position
        integer :: i
        index = 0
        if (array < 1 .or. array > doc%used) return
        if (doc%token(array)%kind /= json_array) return
        if (position < 1 .or. position > doc%token(array)%count) return
        index = doc%token(array)%first
        do i=2,position
            index = doc%token(index)%next
        end do
    end function

    subroutine expect_kind(doc, index, kind, error)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: index, kind
        type(io_error), intent(inout) :: error
        if (index < 1 .or. index > doc%used) then
            call fail(error, 'JSON missing required value')
        else if (doc%token(index)%kind /= kind) then
            call fail(error, 'JSON value has incorrect type')
        end if
    end subroutine

    function json_integer(doc, index, error) result(value)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: index
        type(io_error), intent(inout) :: error
        integer(int64) :: value, limit
        integer :: i, d, first
        logical :: negative
        value=0
        call expect_kind(doc, index, json_number, error)
        if (error%failed()) return
        negative = doc%token(index)%text(1:1) == '-'
        first=1
        if (negative) first=2
        limit=-huge(value)
        if (negative) limit=limit-1
        do i=first,len(doc%token(index)%text)
            d=iachar(doc%token(index)%text(i:i))-iachar('0')
            if (d < 0 .or. d > 9) then
                call fail(error, 'JSON expected integer (not fractional/exponent number)')
                return
            end if
            if (value < (limit+d)/10) then
                call fail(error, 'JSON integer outside signed 64-bit range')
                return
            end if
            value=value*10-d
        end do
        if (.not. negative) value=-value
    end function

    function json_real(doc, index, error) result(value)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: index
        type(io_error), intent(inout) :: error
        real(real64) :: value
        integer :: ios
        value=0
        call expect_kind(doc, index, json_number, error)
        if (error%failed()) return
        read(doc%token(index)%text, *, iostat=ios) value
        if (ios /= 0) call fail(error, 'JSON real outside supported range')
        if (.not. (abs(value) <= huge(value))) call fail(error, 'JSON real outside finite range')
    end function

    function json_string(doc, index, error) result(value)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: index
        type(io_error), intent(inout) :: error
        character(:), allocatable :: value
        value=''
        call expect_kind(doc, index, json_text, error)
        if (.not. error%failed()) value=doc%token(index)%text
    end function

    logical function json_boolean(doc, index, error) result(value)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: index
        type(io_error), intent(inout) :: error
        value=.false.
        call expect_kind(doc, index, json_bool, error)
        if (.not. error%failed()) value=doc%token(index)%text == 'true'
    end function
end module
