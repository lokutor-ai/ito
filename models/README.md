# Getting the Ito voice model

The weights are not stored in this repository. They are distributed only through the gated Hugging Face repository
**[lokutor-ai/ito-tts-v3](https://huggingface.co/lokutor-ai/ito-tts-v3)**, where you accept the
[terms of use](TERMS.md) once. They are free for research, education, personal and hobby projects, under
[CC BY-NC-SA 4.0](LICENSE-WEIGHTS). Commercial use needs a license from Lokutor (contact@lokutor.com).

| file | size | used by |
|---|---|---|
Two voices, both distilled from StyleTTS 2 conditioned on a LibriTTS-R speaker (CC BY 4.0, see [`NOTICE`](../NOTICE)):

| voice | speaker | PyTorch file (Python package) | chip file (firmware, `esp32/host`, `chip_wav.py`) |
|---|---|---|---|
| **D** (female, default) | LibriTTS-R 4970 | `ito_v3.pt` (18 MB) | `ito_v3_esp32s3.bin` (4.9 MB) |
| **G** (male) | LibriTTS-R 5105 | `ito_v3_G.pt` (18 MB) | `ito_v3_G_esp32s3.bin` (4.9 MB) |

Choose the voice with `ito-tts --voice g`, `Ito.load(voice="g")`, `chip_wav.py --voice g`, or `VOICE=g` for the
shell tools (`fetch_weights.sh`, `flash.sh`, `run_qemu.sh`, `make test`). The default is voice D.

**Steps**

1. Open https://huggingface.co/lokutor-ai/ito-tts-v3, sign in and accept the terms.
2. Log in on your machine once: `huggingface-cli login` (or `hf auth login`, or set `HF_TOKEN`).
3. That's it:
   - `ito-tts` and `Ito.load()` download the voice's `.pt` file on first use (into the Hugging Face cache);
   - `esp32/tools/flash.sh`, `make test` and `run_qemu.sh` fetch the voice's chip file into this folder via
     `esp32/tools/fetch_weights.sh`;
   - or fetch both files of a voice here yourself: `esp32/tools/fetch_weights.sh all` (or `python -m ito.weights`;
     add `VOICE=g` / `--voice g` for voice G).

If you already have the files (for example on a machine without internet), put them in a folder and set
`ITO_WEIGHTS_DIR=/path/to/folder`; Ito then reads them from there. Files placed in this `models/` folder are used too
(and are ignored by git).
