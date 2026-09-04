"""Covers the sentence splitter that feeds TTS while the LLM is still streaming.

The old pattern cut on every comma, semicolon and colon, and treated any full
stop before whitespace as a sentence end, so "Dr. Smith" and "e.g. this" were
cut mid-sentence and every clause played as its own choppy utterance. None of
that fails loudly: every word still arrives, just in the wrong pieces.

Run: python tests/test_sentence_split.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asterisk_ai_voice_agent.agent import (
    LONG_SENTENCE_CHARS, find_sentence_end, next_chunk,
)


def _chunks(text):
    """Feed `text` one character at a time, the way tokens arrive, and collect
    the chunks the splitter would hand to TTS. The remainder is what the
    end-of-turn flush would send."""
    out, buf = [], ""
    for ch in text:
        buf += ch
        cut = next_chunk(buf)
        while cut != -1:
            out.append(buf[:cut])
            buf = buf[cut:]
            cut = next_chunk(buf)
    return out, buf


def test_splits_on_sentence_ends_only():
    out, rest = _chunks("First one. Second one? Third, with a comma, one! ")
    assert out == ["First one. ", "Second one? ", "Third, with a comma, one! "], out
    assert rest == ""
    print("ok: sentences split on . ? ! and not on commas")


def test_abbreviations_and_initials_do_not_end_a_sentence():
    cases = [
        "Ask for Dr. Smith at the front desk. ",
        "We ship to the U.S. and the U.K. every week. ",
        "Try a warm drink, e.g. tea, before bed. ",
        "It was signed by J. R. Hartley last year. ",
        "The office opens at 9 a.m. on weekdays. ",
        "Send it to Acme Inc. by Friday. ",
    ]
    for text in cases:
        out, rest = _chunks(text)
        assert out == [text], f"{text!r} was cut into {out}"
    print(f"ok: {len(cases)} abbreviation/initial cases kept whole")


def test_decimals_and_list_numbers():
    out, _ = _chunks("The rate is 3.5 percent this month. ")
    assert out == ["The rate is 3.5 percent this month. "], out
    out, _ = _chunks("1. Restart the phone. 2. Wait a minute. ")
    assert out == ["1. Restart the phone. ", "2. Wait a minute. "], out
    # A number that really does end a sentence still ends it.
    out, _ = _chunks("Your total is 40. Anything else? ")
    assert out == ["Your total is 40. ", "Anything else? "], out
    print("ok: decimals kept, list numbers kept with their item, plain numbers end sentences")


def test_closing_quotes_stay_with_their_sentence():
    out, _ = _chunks('He said "stop." Then he left. ')
    assert out == ['He said "stop." ', "Then he left. "], out
    print("ok: a closing quote after the full stop stays in the sentence")


def test_non_latin_terminators():
    out, _ = _chunks("آپ کیسے ہیں؟ میں ٹھیک ہوں۔ ")
    assert len(out) == 2, out
    out, _ = _chunks("यह पहला वाक्य है। यह दूसरा है। ")
    assert len(out) == 2, out
    print("ok: Urdu and Devanagari terminators split")


def test_long_sentence_falls_back_to_a_clause_break():
    clause = "we can look at the account, "
    text = clause * 6                       # 168 chars, no sentence end anywhere
    out, rest = _chunks(text)
    assert out, "a long sentence never started playing"
    assert all(len(c) < LONG_SENTENCE_CHARS + len(clause) for c in out), out
    assert all(c.endswith(", ") for c in out), out
    # Short sentences with commas are not touched by the fallback.
    assert find_sentence_end("one, two, three") == -1
    assert next_chunk("one, two, three") == -1
    print(f"ok: {len(out)} clause chunks from a {len(text)}-char run-on; short "
          "comma sentences untouched")


if __name__ == "__main__":
    test_splits_on_sentence_ends_only()
    test_abbreviations_and_initials_do_not_end_a_sentence()
    test_decimals_and_list_numbers()
    test_closing_quotes_stay_with_their_sentence()
    test_non_latin_terminators()
    test_long_sentence_falls_back_to_a_clause_break()
    print("\nall sentence split tests passed")
