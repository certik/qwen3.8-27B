#!/usr/bin/env python3
"""Regenerate the Unicode tables inside src-fortran/qwen_tokenizer.f90.

The region between the BEGIN/END GENERATED markers is replaced.

Letter (\\p{L}) and number (\\p{N}) classes are measured from the reference
HF `tokenizers` pre-tokenizer of the model, so they match its Unicode
version exactly.  Whitespace is the Unicode White_Space property, and the NFC
data (combining classes, canonical decompositions, primary compositions) come
from Python's unicodedata, restricted to what the reference NFC implements
(its Unicode tables are older than Python's: newer marks are starters).

usage: gen_unicode_tables.py TOKENIZER_JSON src-fortran/qwen_tokenizer.f90
"""
import sys
import unicodedata as ud
from tokenizers import Tokenizer

BEGIN = "! BEGIN GENERATED (tools/gen_unicode_tables.py)"
END = "! END GENERATED"


def ranges(flags):
    out, start = [], None
    for cp, f in enumerate(flags):
        if f and start is None:
            start = cp
        elif not f and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def arr(name, values, per=12, chunk=2400):
    """Parameter array, in chunks: Fortran allows at most 255 continuations."""
    vals = [str(v) for v in values]
    parts = [vals[i:i + chunk] for i in range(0, len(vals), chunk)] or [[]]
    out = ""
    names = []
    for k, part in enumerate(parts):
        cname = "%s_%d" % (name, k + 1)
        names.append(cname)
        lines = ["        " + ", ".join(part[i:i + per]) for i in range(0, len(part), per)]
        out += ("    integer(int32), parameter :: %s(%d) = [integer(int32) :: &\n%s]\n"
                % (cname, len(part), ", &\n".join(lines)))
    out += ("    integer(int32), parameter :: %s(%d) = [integer(int32) :: %s]\n"
            % (name, len(vals), ", ".join(names)))
    return out


def main():
    tj, target = sys.argv[1], sys.argv[2]
    pt = Tokenizer.from_file(tj).pre_tokenizer
    n = lambda s: len(pt.pre_tokenize_str(s))
    L = [False] * 0x110000
    N = [False] * 0x110000
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        c = chr(cp)
        if n("a" + c) == 1:
            L[cp] = True
        elif n(c + c) == 2 and n(c + c + "a") == 3 and not c.isspace():
            N[cp] = True
    lr, nr = ranges(L), ranges(N)
    ws = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680]
    ws += list(range(0x2000, 0x200B)) + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
    wr = ranges([cp in set(ws) for cp in range(0x3001)])
    # combining classes
    from tokenizers import normalizers
    nfc = normalizers.NFC().normalize_str

    def ref_knows_ccc(cp):
        """The model's NFC data is older than Python's: a new mark is treated as
        a starter.  Probe the reference (marks with ccc 220 and 230)."""
        c = chr(cp)
        return nfc(c + "\u0316") == "\u0316" + c or nfc("\u0300" + c) == c + "\u0300"

    ccc = [0] * 0x110000
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        v = ud.combining(chr(cp))
        if v and ud.normalize("NFD", chr(cp)) == chr(cp) and not ref_knows_ccc(cp):
            v = 0
        ccc[cp] = v
    cr = []
    cp = 0
    while cp < 0x110000:
        v = ccc[cp]
        if v:
            e = cp
            while e + 1 < 0x110000 and ccc[e + 1] == v:
                e += 1
            cr.append((cp, e, v))
            cp = e + 1
        else:
            cp += 1
    # full canonical decompositions (Hangul is algorithmic)
    dcp, dstart, dlen, pool = [], [], [], []
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000 or 0xAC00 <= cp <= 0xD7A3:
            continue
        d = ud.normalize("NFD", chr(cp))
        if d != chr(cp) and nfc(d) == ud.normalize("NFC", d):
            dcp.append(cp)
            dstart.append(len(pool) + 1)
            dlen.append(len(d))
            pool += [ord(x) for x in d]
    # primary composites: canonical pair decompositions that recompose
    comp = []
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000 or 0xAC00 <= cp <= 0xD7A3:
            continue
        dm = ud.decomposition(chr(cp))
        if not dm or dm.startswith("<"):
            continue
        parts = [int(x, 16) for x in dm.split()]
        if len(parts) != 2:
            continue
        pair = "".join(map(chr, parts))
        if ud.normalize("NFC", pair) == chr(cp) and nfc(pair) == chr(cp):
            comp.append((parts[0], parts[1], cp))
    comp.sort()
    g = []
    g.append(BEGIN + " unicode " + ud.unidata_version + " for NFC/whitespace\n")
    g.append("    ! letters / numbers: measured from the model's pre-tokenizer\n")
    g.append(arr("letter_lo", [a for a, b in lr]))
    g.append(arr("letter_hi", [b for a, b in lr]))
    g.append(arr("number_lo", [a for a, b in nr]))
    g.append(arr("number_hi", [b for a, b in nr]))
    g.append(arr("space_lo", [a for a, b in wr]))
    g.append(arr("space_hi", [b for a, b in wr]))
    g.append(arr("ccc_lo", [a for a, b, v in cr]))
    g.append(arr("ccc_hi", [b for a, b, v in cr]))
    g.append(arr("ccc_val", [v for a, b, v in cr]))
    g.append(arr("dec_cp", dcp))
    g.append(arr("dec_start", dstart))
    g.append(arr("dec_len", dlen))
    g.append(arr("dec_pool", pool))
    g.append(arr("comp_a", [a for a, b, c in comp]))
    g.append(arr("comp_b", [b for a, b, c in comp]))
    g.append(arr("comp_c", [c for a, b, c in comp]))
    g.append("    " + END + "\n")
    src = open(target).read()
    i = src.index(BEGIN)
    j = src.index(END)
    j = src.index("\n", j) + 1
    src = src[:i] + "".join(g) + src[j:]
    open(target, "w").write(src)


main()
