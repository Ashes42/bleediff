"""Deterministic, side-effect-free TEST tokenizer (no server), loosely Qwen-like: every digit is its own token,
each punctuation character is one token, and an alphabetic run costs ceil(len / 6) tokens.

How the divisor 6 was chosen (disclosed because it affects what the generator tests exercise): it reproduces the
real Qwen2.5 token counts of the five treatment fragments to within a total absolute error of 18 tokens
(real 41/39/34/42/36 vs 39/44/38/45/40). Divisors 7 and 8 fit the real counts better (error 9 and 10) but make
two scenarios INFEASIBLE for the generator within the test's draw budget (it fails closed), so among the divisors
under which all five scenarios are feasible, 6 is the closest fit. This is a unit-test stand-in only. The
controls actually used in experiments (fixtures/controls/) were generated and verified against the REAL tokenizer.
"""
import math
import re
import zlib


def qwenish(s: str) -> list:
    ids = []
    for run in re.findall(r"\n|\d|[A-Za-z]+|[^\sA-Za-z\d]", s):
        if run[0].isalpha():
            n = max(1, math.ceil(len(run) / 6))
            h = zlib.crc32(run.encode())
            ids += [h % 99991 + j for j in range(n)]
        else:
            ids.append(ord(run[0]))
    return ids
