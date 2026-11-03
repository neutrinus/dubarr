import json
import os
import wave
import struct
import pytest
import subprocess
from utils import clean_output, clean_srt, count_syllables, measure_zcr, parse_json, run_cmd, text_similarity


def test_parse_json_valid():
    data = {"key": "value", "number": 123}
    txt = f"Some text before {json.dumps(data)} some text after"
    assert parse_json(txt) == data


def test_parse_json_invalid():
    assert parse_json("no json here") == {}
    assert parse_json("{unclosed json") == {}


def test_clean_srt():
    srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello world!

2
00:00:05,000 --> 00:00:08,000
This is a test.
"""
    expected = "Hello world! This is a test."
    assert clean_srt(srt_content) == expected


def test_count_syllables_en():
    assert count_syllables("hello", "en") == 2
    assert count_syllables("world", "en") == 1
    assert count_syllables("", "en") == 0


def test_count_syllables_pl():
    # manual count for Polish: a, e, i, o, u, y, ą, ę, ó
    assert count_syllables("cześć", "pl") == 1  # e
    assert count_syllables("konstantynopolitańczykowianeczka", "pl") == 13
    assert count_syllables("test", "pl") == 1


def test_clean_output():
    target_langs = ["pl", "en"]

    # Test prefix removal
    assert clean_output("translation: Hello", target_langs) == "Hello"
    assert clean_output("final_text: Hi there", target_langs) == "Hi there"

    # Test CJK rejection for Latin targets
    assert clean_output("Hello 你好", ["pl"]) == ""
    assert clean_output("Hello", ["pl"]) == "Hello"

    # Test parentheses removal
    assert clean_output("Word (explanation)", target_langs) == "Word"

    # Test empty input
    assert clean_output("", target_langs) == ""


def test_run_cmd_success():
    out = run_cmd(["echo", "hello"], "test echo")
    assert out.strip() == "hello"


def test_run_cmd_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(["false"], "test failure")


def test_measure_zcr():
    # Create a dummy WAV file
    path = "test_zcr.wav"
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        # Sine wave-like pattern to have some zero crossings
        # [1, -1, 1, -1, ...]
        data = struct.pack("4h", 1000, -1000, 1000, -1000)
        wf.writeframes(data)

    try:
        zcr = measure_zcr(path)
        # 4 samples, 3 transitions:
        # (1000 > 0 and -1000 <= 0) -> TRUE
        # (-1000 <= 0 and 1000 > 0) -> TRUE
        # (1000 > 0 and -1000 <= 0) -> TRUE
        # Total transitions = 3. Samples = 4. ZCR = 3/4 = 0.75
        assert zcr == 0.75
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_measure_zcr_empty():
    path = "test_empty.wav"
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"")

    try:
        assert measure_zcr(path) == 0.0
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_measure_zcr_invalid_file():
    assert measure_zcr("non_existent.wav") == 0.0


def test_text_similarity_tolerates_phonetic_spellings_but_not_noise():
    text = "Ajjs-wejv próbuje zachować dystans i uderzyć. O mój Boże!"
    assert text_similarity(text, "Ice Wave próbuje zachować dystans i uderzyć. O mój Boże!") > 0.8
    assert text_similarity(text, "Aaaa! Aaaa!") < 0.5
    assert text_similarity(text, "") == 0.0
    assert text_similarity("", "anything") == 1.0


def test_respelling_that_loses_a_consonant_fails():
    from utils import respelling_heard_as

    assert not respelling_heard_as("Razorback", "Rayorbeck")  # "Rej-or-bek" lost the z
    assert not respelling_heard_as("Ice Wave", "Ice Way")


@pytest.mark.parametrize(
    "term,heard",
    [
        ("Razorback", "Rayzorbak"),
        ("Hyneman", "Hineman"),
        ("Mjollnir", "Myolnir"),
        ("Disposable Hero", "Dispozebel Hiro"),
        ("James", "Jayms"),
        ("Sallad", "Salad"),
    ],
)
def test_close_respellings_pass(term, heard):
    from utils import respelling_heard_as

    assert respelling_heard_as(term, heard)


def test_clean_arpabet():
    from utils import clean_arpabet

    assert clean_arpabet("R EY1 Z ER0 B AE2 K") == "R EY1 Z ER0 B AE2 K"
    assert clean_arpabet("[ay1 s] w ey1 v") == "AY1 S W EY1 V"
    assert clean_arpabet("R EI1 Z") == ""  # EI is not ARPAbet
    assert clean_arpabet("") == ""


def test_apply_phonemes_keeps_polish_endings():
    from utils import apply_phonemes

    phonemes = {"Razorback": "R EY1 Z ER0 B AE2 K", "Ice Wave": "AY1 S W EY1 V", "Ice": "AY1 S"}
    assert apply_phonemes("Razorbackowi pękł nos, a Ice Wave wygrywa.", phonemes) == (
        "[R EY1 Z ER0 B AE2 K]owi pękł nos, a [AY1 S W EY1 V] wygrywa."
    )
    assert apply_phonemes("Bez nazw.", {}) == "Bez nazw."
