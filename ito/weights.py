"""Where Ito's weights come from.

The voice model is distributed only through the gated Hugging Face repository lokutor-ai/ito, under
CC BY-NC-SA 4.0 and Lokutor's terms of use (models/LICENSE-WEIGHTS, models/TERMS.md). To get access:

  1. open https://huggingface.co/lokutor-ai/ito and accept the terms;
  2. log in once on this machine:  huggingface-cli login   (or: hf auth login, or set HF_TOKEN).

Lookup order for a file:
  1. $ITO_WEIGHTS_DIR/<file>                (e.g. a folder you copied the files into)
  2. <repo>/models/<file>                  (esp32/tools/fetch_weights.sh puts them there)
  3. the Hugging Face cache, downloading it on first use

    python -m ito.weights                  # download the female voice's two files into models/
    python -m ito.weights --voice male     # the male voice
"""
import os
import shutil
import sys

REPO_ID = "lokutor-ai/ito"
PT = "ito_female.pt"                      # female voice (default)
CHIP = "ito_female_esp32s3.bin"
# voice -> (PyTorch file, chip file). female: LibriTTS-R speaker 4970; male: speaker 5105.
VOICES = {"female": (PT, CHIP), "male": ("ito_male.pt", "ito_male_esp32s3.bin")}
ALIASES = {"d": "female", "f": "female", "g": "male", "m": "male"}   # "d" and "g" are the original short names
VOICE_NAMES = ("female", "male", "d", "g")


# The firmware has up to three weight sets per voice (flash partitions, best quality first): main (int8), main with int4 blocks, light (4 blocks, int4).
# The first is the only one the host build, `chip_wav.py` and the QEMU scripts need; `flash.sh` writes all that are present.
CHIP_SETS = {"female": ("ito_female_esp32s3.bin", "ito_female_esp32s3_int4.bin", "ito_female_esp32s3_light.bin"),
             "male": ("ito_male_esp32s3.bin", "ito_male_esp32s3_int4.bin", "ito_male_esp32s3_light.bin")}


def chip_set_files(voice="female"):
    """(main, int4, light) chip weight files of a voice, in the order of the firmware's flash partitions."""
    v = (voice or "female").lower()
    return CHIP_SETS[ALIASES.get(v, v)]


def voice_files(voice="female"):
    """(PyTorch file, chip file) for a voice: 'female' (default, alias 'd') or 'male' (alias 'g')."""
    v = (voice or "female").lower()
    v = ALIASES.get(v, v)
    if v not in VOICES:
        raise ValueError(f"unknown voice {voice!r}: choose from {', '.join(VOICE_NAMES)}")
    return VOICES[v]
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

ACCESS_HELP = f"""
Ito's voice model lives in a gated Hugging Face repository, and this machine has no access to it yet.

  1. Open https://huggingface.co/{REPO_ID}, sign in, and accept the terms
     (free for research, education and personal projects; commercial use needs a license from Lokutor).
  2. Log in here once:   huggingface-cli login      (or: hf auth login, or export HF_TOKEN=...)
  3. Run your command again.

Already have the files? Point Ito at them:  export ITO_WEIGHTS_DIR=/path/to/folder
Questions or a commercial license: contact@lokutor.com
"""


class WeightsAccessError(RuntimeError):
    pass


def weights_path(filename=PT):
    """Local path of a weights file, downloading it from the gated repo if needed."""
    env = os.environ.get("ITO_WEIGHTS_DIR")
    if env:
        p = os.path.join(os.path.expanduser(env), filename)
        if not os.path.exists(p):
            raise FileNotFoundError(f"ITO_WEIGHTS_DIR is set, but {p} does not exist")
        return p
    p = os.path.join(MODELS_DIR, filename)
    if os.path.exists(p):
        return p
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
    except ImportError:
        raise WeightsAccessError("huggingface_hub is not installed: pip install huggingface_hub" + ACCESS_HELP)
    try:
        return hf_hub_download(REPO_ID, filename)
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise WeightsAccessError(ACCESS_HELP) from e
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403, 404):
            raise WeightsAccessError(ACCESS_HELP) from e
        raise


def fetch(dest=MODELS_DIR, files=(PT, CHIP)):
    """Copy the weights into dest (default: the repository's models/ folder)."""
    os.makedirs(dest, exist_ok=True)
    out = []
    for f in files:
        src, dst = weights_path(f), os.path.join(dest, f)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        out.append(dst)
        print(f"{dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    voice = "female"
    if "--voice" in args:
        i = args.index("--voice"); voice = args[i + 1]; del args[i:i + 2]
    files = [a for a in args if not a.startswith("--")] or list(voice_files(voice))
    try:
        fetch(files=files)
    except WeightsAccessError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
