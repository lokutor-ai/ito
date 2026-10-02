"""Text -> Ito token ids. This is exactly StyleTTS 2's inference text front end:

  espeak-ng through phonemizer (language en-us, with stress, punctuation preserved) on text.strip() with '"' removed,
  then nltk word_tokenize on the phoneme string, joined by single spaces, then StyleTTS 2's symbol table, with a
  leading 0 pad token.

The symbol table below is copied from StyleTTS 2 (github.com/yl4579/StyleTTS2, text_utils.py, MIT License,
Copyright (c) 2023 Aaron (Yinghao) Li). espeak-ng and phonemizer are GPLv3 and are used as separate packages.

Needs: pip install phonemizer espeakng-loader nltk   (nltk's 'punkt_tab' data is fetched on first use).
This module does not import torch, so esp32/tools/say.py can use it on a machine without PyTorch."""
import os
import re
import sys

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

SYMBOLS = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
SYMBOL_ID = {s: i for i, s in enumerate(SYMBOLS)}      # later duplicates win, as in StyleTTS 2

_backend = None


def _word_tokenize(s):
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            print("(nltk: fetching the 'punkt_tab' tokenizer data once)", file=sys.stderr)
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import word_tokenize
        return word_tokenize(s)
    except Exception:            # close approximation if nltk or its data is unavailable
        return re.findall(r"[^\s.,;:!?]+|[.,;:!?]", s)


_DIG = "zero one two three four five six seven eight nine".split()


def normalize(text):
    """Spell out what eSpeak would turn into sentence-ending dots: decimals ("3.5" -> "3 point 5"), thousands separators
    ("1,250" -> "1250") and dollar amounts ("$1,250.75" -> "1250 dollars and 75 cents")."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\$(\d+)\.(\d{2})\b", lambda m: f"{m.group(1)} dollars and {int(m.group(2))} cents", text)
    text = re.sub(r"\$(\d+)\b", r"\1 dollars", text)
    text = re.sub(r"(\d+)\.(\d+)", lambda m: m.group(1) + " point " + " ".join(_DIG[int(d)] for d in m.group(2)), text)
    return text


def phonemize(text):
    """English text -> phoneme string (StyleTTS 2 convention)."""
    global _backend
    if _backend is None:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
        if hasattr(EspeakWrapper, "set_data_path"):                  # phonemizer >= 3.4
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        from phonemizer.backend import EspeakBackend
        _backend = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
    ps = _backend.phonemize([normalize(text).strip().replace('"', "")])[0]
    return " ".join(_word_tokenize(ps))


def phonemes_to_ids(ps):
    """Phoneme string -> token ids with the leading 0 pad. Unknown symbols are skipped."""
    return [0] + [SYMBOL_ID[c] for c in ps if c in SYMBOL_ID]


def text_to_ids(text):
    return phonemes_to_ids(phonemize(text))
