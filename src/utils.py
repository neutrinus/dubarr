import difflib
import unicodedata
import json
import re
import wave
import struct
import logging
import syllables
import subprocess

try:
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


def parse_json(txt: str) -> dict:
    try:
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1:
            return json.loads(txt[s : e + 1])
    except Exception as e:
        logging.warning(f"Failed to parse JSON: {e}")
    return {}


def clean_srt(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        if "-->" in line:
            continue
        if line.strip().isdigit():
            continue
        if not line.strip():
            continue
        lines.append(line.strip())
    return " ".join(lines)[:15000]


def measure_zcr(path: str) -> float:
    """Measures Zero Crossing Rate to detect static/screeching."""
    try:
        with wave.open(path, "r") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return 0.0
            frames = wf.readframes(n_frames)
            samples = struct.unpack(f"{n_frames}h", frames)
            zc = 0
            for i in range(1, len(samples)):
                if (samples[i - 1] > 0 and samples[i] <= 0) or (samples[i - 1] <= 0 and samples[i] > 0):
                    zc += 1
            return zc / len(samples)
    except Exception:
        return 0.0


def count_syllables(text: str, lang: str = "pl") -> int:
    if not text:
        return 0
    try:
        count = syllables.estimate(text)
        if lang == "pl":
            vowels = "aeiouyąęó"
            manual_count = sum(1 for char in text.lower() if char in vowels)
            return max(count, manual_count)
        return max(1, count)
    except Exception:
        text = text.lower()
        vowels = "aeiouy"
        if lang == "pl":
            vowels += "ąęó"
        count = sum(1 for char in text if char in vowels)
        return max(1, count)


def clean_output(text: str, target_langs: list) -> str:
    if not text:
        return ""
    text = text.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    text = re.sub(
        r"^(?:translation|thought|analysis|final_text|final|corrected|result|output|text|analysis):\s*",
        "",
        text,
        flags=re.I | re.M,
    )
    text = re.sub(r"\((?:note|translation|explanation|corrected|original).*?\)", "", text, flags=re.I)
    text = re.split(r"\n?(?:note|explanation|comment|uwaga|output|corrected):", text, flags=re.I)[0]

    text = text.split("\n")[0].strip().strip('"').strip("*").strip()

    if HAS_LANGDETECT and text and len(text) > 5:
        try:
            is_latin_target = any(lang in target_langs for lang in ["pl", "en", "de", "es", "fr", "it"])
            if is_latin_target:
                if re.search(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", text):
                    logging.warning(f"Language Guard: Detected CJK characters in Latin target '{text}'. Rejecting.")
                    return ""
        except Exception as e:
            logging.warning(f"Langdetect error: {e}")

    return text


def run_cmd(cmd, desc):
    try:
        logging.debug(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Error {desc}: {e.stderr}")
        if e.stdout:
            logging.error(f"Stdout {desc}: {e.stdout}")
        raise


def _letters(text: str) -> str:
    """Lower-case letters and digits only: punctuation, hyphens of phonetic spellings and spaces do not count."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def text_similarity(expected: str, heard: str) -> float:
    """Similarity (0..1) of a take's text and what an ASR heard in it, on letters only.

    Tolerant of punctuation, spacing and spelling ("Ajjs-wejv" vs "Ice Wave" costs a few letters), low for noise,
    screams or a different text.
    """
    a, b = _letters(expected), _letters(heard)
    if not a:
        return 1.0
    if not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


# English spellings that sound alike fold to one consonant (both sides of the glossary reading test are English spelling)
_SOUND_FOLDS = (
    ("sch", "sh"),
    ("ck", "k"),
    ("ch", "k"),
    ("ph", "f"),
    ("th", "t"),
    ("x", "ks"),
    ("q", "k"),
    ("c", "k"),
    ("b", "p"),
    ("d", "t"),
    ("g", "k"),
    ("z", "s"),
    ("v", "f"),
    ("w", "f"),
)


def consonant_skeleton(text: str) -> str:
    """The consonants of an English spelling, voiced and voiceless pairs folded ("Razorback" -> "rsrpk").

    ``j`` and ``y`` count as vowels: a Polish "j" is an English "y", and the two spellings swap freely.
    """
    t = unicodedata.normalize("NFKD", (text or "").lower())
    t = "".join(ch for ch in t if ch.isascii() and ch.isalpha())
    t = re.sub(r"(.)\1+", r"\1", t)  # doubled letters are one sound ("Sallad")
    for a, b in _SOUND_FOLDS:
        t = t.replace(a, b)
    return re.sub(r"[aeiouyj]", "", t)


def respelling_heard_as(term: str, heard: str) -> bool:
    """Whether what a respelling sounds like ("Rayorbeck") keeps the consonants of the term ("Razorback").

    A respelling that drops or changes a consonant names something else; one change is allowed per 6 consonants
    (Spec 006, 3).
    """
    a, b = consonant_skeleton(term), consonant_skeleton(heard)
    if not a:
        return True
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    missing = sum(i2 - i1 for op, i1, i2, _j1, _j2 in ops if op in ("replace", "delete"))
    return missing <= len(a) // 6


ARPABET = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W",
    "Y", "Z", "ZH",
}  # fmt: skip


def clean_arpabet(text: str) -> str:
    """ "R EY1 Z ER0 B AE2 K" when every token is an ARPAbet phoneme (vowels with a stress digit), else ""."""
    tokens = (text or "").replace("[", " ").replace("]", " ").upper().split()
    if not tokens:
        return ""
    for tok in tokens:
        base = tok.rstrip("012")
        if base not in ARPABET or (tok != base and len(tok) - len(base) != 1):
            return ""
    return " ".join(tokens)


def apply_phonemes(text: str, phonemes: dict) -> str:
    """Replaces each name with its bracketed CMU phonemes for the TTS ("Razorbacka" -> "[R EY1 ...]a").

    Longer names first, so "Ice Wave" wins over "Ice"; a Polish ending on the name stays after the brackets.
    """
    for name in sorted(phonemes or {}, key=len, reverse=True):
        arpabet = phonemes[name]
        if not name or not arpabet:
            continue
        text = re.sub(rf"(?<!\w){re.escape(name)}", lambda _m, a=arpabet: f"[{a}]", text, flags=re.IGNORECASE)
    return text


# --- Glossary reading check by spelling rules (Spec 006, 3) ---

# Polish spelling -> sounds (ARPAbet consonants), longest letter groups first; vowels give no consonant
_PL_GROUPS = (
    ("dzi", "JH"), ("dż", "JH"), ("dź", "JH"), ("dz", "Z"), ("sz", "SH"), ("rz", "ZH"), ("cz", "CH"), ("ch", "HH"),
    ("si", "SH"), ("zi", "ZH"), ("ci", "CH"), ("ni", "N"),
    ("ś", "SH"), ("ź", "ZH"), ("ż", "ZH"), ("ć", "CH"), ("ń", "N"), ("ł", "W"), ("w", "V"), ("v", "V"), ("c", "T S"),
    ("h", "HH"), ("x", "K S"), ("q", "K"), ("ą", "N"),
    ("b", "B"), ("p", "P"), ("d", "D"), ("t", "T"), ("g", "G"), ("k", "K"), ("f", "F"), ("s", "S"), ("z", "Z"),
    ("m", "M"), ("n", "N"), ("l", "L"), ("r", "R"),
)  # fmt: skip
_PL_VOWELS = set("aeiouyóęj")  # "j" is the English "y", a glide the check ignores like the vowels

# Sounds a listener does not tell apart in a name: voiced and voiceless pairs (Polish devoices word ends)
_SOUND_CLASS = {"B": "P", "D": "T", "G": "K", "V": "F", "Z": "S", "ZH": "SH", "JH": "CH", "NG": "N", "ER": "R"}
# English sounds Polish has no letters for: "th" is respelled as t, f, s or d
_SOUND_ALSO = {"TH": {"T", "F", "S"}, "DH": {"T", "F", "S"}}


def polish_reading(spelling: str):
    """The consonant sounds of a Polish spelling as read aloud: [(ARPAbet sound, letter start, letter end)].

    "Ajs Wejv" -> V and F where the English name has two W sounds; that is what the reading check catches.
    """
    text = (spelling or "").lower()
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if not ch.isalpha() or ch in _PL_VOWELS:
            i += 1
            continue
        for group, sounds in _PL_GROUPS:
            if text.startswith(group, i):
                # "si", "ni" before a vowel are one soft consonant; before a consonant the "i" is a vowel
                if group[-1] == "i" and len(group) > 1 and group not in ("dzi",):
                    nxt = text[i + len(group)] if i + len(group) < len(text) else ""
                    if nxt not in _PL_VOWELS or nxt in ("i", "j"):
                        continue
                for sound in sounds.split():
                    out.append((sound, i, i + len(group)))
                # the "i" of "si" + vowel only softens: the vowel after it is still read
                i += len(group) - 1 if group[-1] == "i" and len(group) > 1 else len(group)
                break
        else:
            i += 1  # a letter Polish does not read (foreign spelling): no sound
    return out


def english_consonants(arpabet: str):
    """Consonant sounds of ARPAbet phonemes; "ER" (as in "Razor") counts as an R, "Y" is a glide like Polish "j"."""
    out = []
    for tok in (arpabet or "").upper().split():
        base = tok.rstrip("012")
        if base == "ER":
            out.append("R")
        elif base in ARPABET and base not in (
            "AA",
            "AE",
            "AH",
            "AO",
            "AW",
            "AY",
            "EH",
            "EY",
            "IH",
            "IY",
            "OW",
            "OY",
            "UH",
            "UW",
            "Y",
        ):
            out.append(base)
    return out


def _same_sound(english: str, read: str) -> bool:
    if english in _SOUND_ALSO:
        return _SOUND_CLASS.get(read, read) in _SOUND_ALSO[english]
    return _SOUND_CLASS.get(english, english) == _SOUND_CLASS.get(read, read)


def _align_sounds(english, read):
    """Edit alignment of two sound lists: [(op, english index, read index)], op in match/sub/del/ins.

    A sound the spelling adds costs little (Polish spells "x" as "ks"); one it drops or changes costs a full step.
    """
    n, m = len(english), len(read)
    INS = 0.3
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = j * INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0.0 if _same_sound(english[i - 1], read[j - 1]) else 1.0
            cost[i][j] = min(cost[i - 1][j - 1] + sub, cost[i - 1][j] + 1.0, cost[i][j - 1] + INS)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = 0.0 if _same_sound(english[i - 1], read[j - 1]) else 1.0
            if cost[i][j] == cost[i - 1][j - 1] + sub:
                ops.append(("match" if sub == 0 else "sub", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1.0:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return list(reversed(ops))


# Polish vowel letters that can carry each stressed English vowel ("Kirk" is never "Kark")
_STRESSED_VOWELS = {
    "AA": "ao", "AE": "ea", "AH": "aeouy", "AO": "oa", "AW": "aou", "AY": "a", "EH": "ea", "ER": "eyiou",
    "EY": "e", "IH": "iye", "IY": "iy", "OW": "ou", "OY": "o", "UH": "uo", "UW": "u",
}  # fmt: skip
_PL_VOWEL_SOUND = str.maketrans("óąę", "uoe")


def stressed_vowels_lost(spelling: str, arpabet: str) -> list:
    """Stressed English vowels the Polish spelling reads as another vowel: ["ER read as a"] for "Kark" (Kirk).

    The vowel is looked up between the spelling's letters of the consonants around it (aligned as in the consonant
    check); a vowel next to a lost consonant is not checked.
    """
    english = english_consonants(arpabet)
    reading = polish_reading(spelling)
    read = [snd for snd, _a, _b in reading]
    to_read = {ei: ri for op, ei, ri in _align_sounds(english, read) if op in ("match", "sub")}
    text = (spelling or "").lower().translate(_PL_VOWEL_SOUND)
    lost, c = [], 0
    for tok in (arpabet or "").upper().split():
        base = tok.rstrip("012")
        if base not in ARPABET:
            continue
        is_consonant = base == "ER" or base not in _STRESSED_VOWELS and base != "Y"
        if base in _STRESSED_VOWELS and tok.endswith("1"):
            prev_ok = c == 0 or (c - 1) in to_read
            next_ok = c >= len(english) or c in to_read
            if prev_ok and next_ok:
                lo = reading[to_read[c - 1]][2] if c > 0 else 0
                hi = reading[to_read[c]][1] if c < len(english) else len(text)
                region = "".join(ch for ch in text[lo:hi] if ch in "aeiouy")
                if not region:
                    lost.append(f"{base} missing")
                elif not any(ch in _STRESSED_VOWELS[base] for ch in region):
                    lost.append(f"{base} read as {region}")
        if is_consonant:
            c += 1
    return lost


# Ranks and titles before a name: translated by the translator, never respelled ("Kapten", "Letnant")
TITLES = {
    "captain", "capt", "lieutenant", "lt", "commander", "cmdr", "ensign", "admiral", "doctor", "dr", "mr", "mrs",
    "ms", "miss", "sir", "sergeant", "sgt", "colonel", "general", "major", "chief", "professor", "prof", "nurse",
    "officer", "agent", "detective", "judge", "president",
}  # fmt: skip


def strip_title(term: str, spelling: str):
    """("Captain James Tiberius Kirk", "Kapten Dżejms Tajberius Kark") -> ("James Tiberius Kirk", "Dżejms Tajberius
    Kark"); the spelling loses as many leading words as the term when it has more than the name left."""
    words = term.split()
    n = 0
    while n < len(words) - 1 and words[n].lower().strip(".") in TITLES:
        n += 1
    if not n:
        return term, spelling
    rest = words[n:]
    sp_words = str(spelling).split()
    sp = " ".join(sp_words[n:]) if len(sp_words) - n >= len(rest) else " ".join(rest)
    return " ".join(rest), sp


def check_polish_spelling(spelling: str, arpabet: str):
    """Whether a Polish respelling, read by Polish rules, keeps the consonants of the name's English phonemes.

    Returns (ok, fixed spelling, what is lost). A Polish "w" where English has W, or "ł" where it has V, is changed
    in place ("Ajs Wejv" -> "Ajs Łejw"), the one mistake models make every time. Of the other consonants one may
    be lost or changed per 6; a W read as V never passes (the name would sound like another word).
    """
    english = english_consonants(arpabet)
    if not english:
        return True, spelling, ""
    letters = list(spelling)
    for op, ei, ri in _align_sounds(english, [s for s, _a, _b in polish_reading(spelling)]):
        if op != "sub":
            continue
        sound, start, end = polish_reading("".join(letters))[ri]
        word = "".join(letters[start:end])
        if english[ei] == "W" and word.lower() == "w":
            letters[start] = "Ł" if word.isupper() else "ł"
        elif english[ei] == "V" and word.lower() == "ł":
            letters[start] = "W" if word.isupper() else "w"
    fixed = "".join(letters)
    read = [s for s, _a, _b in polish_reading(fixed)]
    lost = [
        (english[ei], read[ri] if ri is not None else "-")
        for op, ei, ri in _align_sounds(english, read)
        if op in ("sub", "del")
    ]
    hard = any(e in ("W", "V") and r in ("W", "V") and e != r for e, r in lost)
    vowels = stressed_vowels_lost(fixed, arpabet)
    ok = not hard and not vowels and len(lost) <= len(english) // 6
    lost_text = ", ".join([f"{e} read as {r}" if r != "-" else f"{e} missing" for e, r in lost] + vowels)
    return ok, fixed, lost_text


def translation_glossary(global_context: dict, lang: str) -> dict:
    """The phonetic glossary without the names this language translates (Spec 001, 2.5a, 8): "Starfleet" is
    "Gwiezdna Flota", not a respelling of the English sound."""
    terms = (global_context.get("terms") or {}).get(lang) or {}
    translated = {t.lower() for t, tr in terms.items() if tr and tr.strip().lower() != t.strip().lower()}
    return {t: v for t, v in (global_context.get("glossary") or {}).items() if t.lower() not in translated}
