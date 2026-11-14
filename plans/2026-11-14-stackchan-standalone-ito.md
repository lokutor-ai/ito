# StackChan speech and push-to-talk conversation

Status: proposed implementation plan. Implement phase 1, then phase 2 after its
entry checks pass. Implementation and device flashing have not started.

## Outcome

| Phase | User action | Result |
| --- | --- | --- |
| 1. Prove local speech | Send `text <English sentence>` over USB | Ito speaks through the robot's speaker |
| 2. Add conversation | Hold the head touch sensor, speak, then release | The robot captures speech; the Mac transcribes and generates a reply; Ito speaks it |

In both phases, text normalization, phonemization and neural synthesis run on
the ESP32-S3. In phase 1, the Mac supplies text and USB power during development.
In phase 2, the Mac also runs speech recognition and reply generation. Select
the exact models at the start of phase 2 after checking Mac resources.

A phrase stored in firmware also plays after a reset without a host process
sending text. This proves that synthesis does not depend on the Mac. It does not
establish battery life or performance on battery power.

The dedicated image replaces the robot app. Both phases keep the display
backlight off and the motors disabled. Phase 1 leaves the microphones inactive.
Face animation, hands-free wake words, simultaneous listening and speaking,
Wi-Fi, Bluetooth, camera, mobile app support and OTA integration remain outside
these phases.

Default voice: female, main int8 chip weights. Use the existing int4 variant
only if measurements justify it. Keep the model architecture and text frontend.

## Logs and host controls

Add one host tool, `esp32/tools/stackchan.py`, incrementally across the phases.
In phase 1, its `monitor` mode displays and saves serial output while accepting
typed Ito commands. In phase 2, its `chat` mode also manages microphone audio,
transcription, reply generation and the conversation state. One process owns
the serial port. Do not run a separate serial monitor alongside it.

Each run creates an absolute-path-reported folder under the Ito checkout:

```text
esp32/logs/stackchan/<timestamp>/
  serial.log          # board boot and runtime diagnostics
  metadata.json       # versions, device identity, configuration and hashes
  conversation.jsonl  # phase 2 transcripts, replies, state changes and timings
```

The existing `esp32/logs/` ignore rule keeps these files out of Git. The host
tool prints the session folder at startup and flushes logs during the run.
Phase 1 creates `serial.log` and `metadata.json`; phase 2 adds the conversation
log. Optional debug recordings go in that same run folder only when explicitly
enabled. Keep raw microphone recording off by default.

`serial.log` contains boot, power/codec detection, model self-tests, memory,
split policy, synthesis timing, underruns, cancellation and errors. Phase 1
captures text serial output. Phase 2 decodes runtime diagnostic frames into
readable text and excludes binary microphone payloads from this file.

`metadata.json` records host UTC start time, firmware and weight hashes, source
revision, tool versions, port, device identity, active model/split and relevant
audio settings. Phase 2 adds the selected ASR/LLM models and protocol version.

Every conversation event includes a session ID, turn ID, event name and host
timestamp. Include device uptime when supplied. Log the transcript, reply,
capture duration, dropped-sample count, stage timings, cancellation and errors.
Keep host and device clocks distinct when reporting timings.

The Mac saves logs while connected. Runs without a host reader retain no logs
in this version. Keep the original flash backup separately under
`~/stackchan-backups/`, as described in step 1.2.

Planned host interfaces, to be implemented in the indicated phase:

```sh
python esp32/tools/stackchan.py monitor --port "$STACKCHAN_PORT"
```

```sh
.venv/bin/python esp32/tools/stackchan.py chat --port "$STACKCHAN_PORT"
```

Use IDF Python for the phase 1 monitor. In phase 2, install the selected host
dependencies, including pyserial, in Ito's environment rather than IDF Python.
Exact model configuration arguments belong to the phase 2 backend selection.

## Evidence and constraints

Inspected on 2026-11-14:

| Item | Evidence | Consequence |
| --- | --- | --- |
| Hardware | M5Stack documents ESP32-S3, 16 MB flash, 8 MB quad PSRAM and AW88298 speaker amplifier | Capacity matches Ito; memory mode and audio need a port |
| USB | Espressif USB JTAG/serial device enumerates at `/dev/cu.usbmodem1101` | Candidate port; confirm device identity before any write |
| Toolchain | `~/esp/esp-idf` contains ESP-IDF 5.5.4 | Reuse it and validate this port with that version |
| Serial tools | IDF Python environment contains esptool 4.12.dev1 and pyserial | Use its underscore command names; do not upgrade it for this task |
| Ito environment | `.venv` contains the model download and host synthesis dependencies | Use it for weights and host reference audio |
| Weights | No chip weights in this checkout's `models/` directory | Check configured weight locations and the HF cache before downloading |
| Ito defaults | Octal PSRAM, generic I2S pins, GPIO0 demo button task | Supplied prebuilt image is unsuitable for this board |
| StackChan source | Board audio already runs at 24 kHz | Sample rate matches Ito |
| Source status | Both checkouts contain unrelated untracked files | Preserve them and keep implementation changes in Ito |

Hardware facts follow [M5Stack's StackChan documentation][m5]. The device's chip
revision, security configuration, flash contents and runtime memory remain
unmeasured. USB enumeration alone does not verify these.

Ito's reported text frontend run peaked at roughly 6.0 to 6.3 MB PSRAM and left
about 52 KB internal SRAM free. Those figures come from another board with octal
PSRAM. Its published latency is not an acceptance threshold for StackChan.
See [Ito's measurements](../esp32/README.md#4d-text-in-speech-out-the-on-chip-front-end-measured-on-silicon-4-and-5-november-2026).

## Phase 1. Prove local speech

### Firmware design

Keep the existing path:

```text
USB text -> on-chip frontend -> Ito engine -> PCM queue -> I2S -> AW88298
```

Add one StackChan audio option and one small board helper. The helper initializes
power and codec control, sets volume and mutes the amplifier. Ito keeps ownership
of the I2S transmit channel, DMA buffers, playback queue and synthesis scheduler.
Do not bring in StackChan's application framework or create a second audio task
or I2S owner.

Use the existing `espressif/esp_codec_dev` AW88298 driver, pinned initially to
1.5.4, the version in StackChan's dependency lock. Connect its control interface
to Ito's existing I2S channel. Confirm the output format before opening the
codec. Ito currently duplicates mono PCM into two 16-bit I2S slots.

The board configuration comes from
[`StackChan/firmware/main/hal/board/config.h`](../../StackChan/firmware/main/hal/board/config.h):

| Signal | GPIO |
| --- | --- |
| I2C SDA | 12 |
| I2C SCL | 11 |
| I2S MCLK | 0 |
| I2S BCLK | 34 |
| I2S word select | 33 |
| I2S speaker data | 13 |

Use the AW88298 address constant supplied by the driver. Use the AXP2101 at
`0x34` and AW9523 at `0x58` as the board source specifies. Initialize only the
power rails and expander outputs required for this build. Preserve unrelated
register bits. Verify those writes against the host and base schematics linked
by M5Stack before implementing them.

The base PY32 expander is separate from the AW9523. The existing StackChan HAL
uses its pin 0 as servo power enable. Set that output low without running the
factory initialization that enables the motors. Allow the expander's documented
startup delay. Do not start servo control or send motion commands.

GPIO0 is the audio master clock. Compile out Ito's button task and its task
creation for this board. Merely changing the I2S pins leaves a task that
reconfigures GPIO0 as an input and breaks the clock.

Initialize the amplifier muted. Establish the clocks before enabling audio,
start at a conservative volume and preserve Ito's end-of-utterance fade and
cancel behavior. Expose the existing `volume` command for this codec too.
If a required power or codec device cannot be initialized, report the failing
step and keep playback disabled.

### Expected changes

| Path in Ito | Change |
| --- | --- |
| `esp32/firmware/sdkconfig.stackchan.defaults` | Quad PSRAM, StackChan audio selection and silent boot defaults |
| `esp32/firmware/main/Kconfig.projbuild` | Add StackChan to the existing audio output choice |
| `esp32/firmware/main/stackchan_audio.c` and `.h` | Power, expander, amplifier initialization, mute and volume |
| `esp32/firmware/main/main.c` | Select board pins, call helper, route mute/volume and disable GPIO0 button task |
| `esp32/firmware/main/CMakeLists.txt` | Include the helper and codec dependency for this target |
| `esp32/firmware/main/idf_component.yml` and dependency lock | Pin and resolve the codec component |
| `esp32/tools/build_fw.sh` | Add an explicit StackChan build option with separate output paths |
| `esp32/tools/stackchan.py` | Add monitor mode, typed commands and per-run log capture |
| `esp32/README.md` | Link to concise board-specific build, flash and recovery instructions |

Keep the default DevKit and QEMU behavior. Do not overwrite either supplied
prebuilt image. Put StackChan binaries and generated configuration under the
ignored `esp32/firmware/build_stackchan/` directory.

Keep Ito's existing partition table. The first pass installs only the main
weight blob at `0x200000`; leave optional model partitions erased. This avoids a
partition redesign. StackChan's original OTA slots and assets overlap this
layout and must be recovered from the full backup when restoring the robot app.

### 1.1. Establish the working environment

- Run `wt list`. Reuse a worktree for this task or create a dedicated branch and
  worktree with Worktrunk. Do not repurpose unrelated worktrees.
- Record both repository revisions, IDF version and Python tool versions.
- Load `~/esp/esp-idf/export.sh` in the build/flashing shell. Keep Ito's `.venv`
  for host synthesis and downloads so its dependencies do not modify IDF Python.
- Re-enumerate serial ports. Confirm the Espressif device and ensure no serial
  monitor owns the selected port.
- Use the base USB-C connector. M5Stack supports data through either connector
  and recommends the base connector for development. [Connection guidance][m5]
- Implement the host monitor and verify log creation before the first firmware
  boot test. Close it whenever esptool needs exclusive access to the port.

No firmware or model writes occur in this step.

### 1.2. Create and verify the recovery backup

Enter download mode using StackChan's own procedure. Hold the bottom RST button
for about three seconds and release when its indicator turns green. Wait for
download mode and recheck the serial port. Do not use the DevKit BOOT/RST
instructions from Ito's generic README. [Download mode][m5]

Use the IDF Python environment's esptool v4 commands. The commands below are a
runbook, not commands executed while writing this plan. Create a private,
uniquely named directory under `~/stackchan-backups/` for `STACKCHAN_BACKUP_DIR`
and set `STACKCHAN_PORT` to the rechecked port.

```sh
python -m esptool --chip esp32s3 --port "$STACKCHAN_PORT" \
  --after no_reset get_security_info
python -m esptool --chip esp32s3 --port "$STACKCHAN_PORT" \
  --before no_reset --after no_reset flash_id
```

Confirm 16 MB flash and record the chip revision and protection state. If secure
boot or flash encryption prevents this backup/custom image procedure, stop
device writes and resolve that concrete restriction. Do not change eFuses or
force a write.

Keep the chip in download mode for two complete reads so the running app cannot
change settings between reads:

```sh
python -m esptool --chip esp32s3 --port "$STACKCHAN_PORT" --baud 460800 \
  --before no_reset --after no_reset read_flash \
  0x0 0x1000000 "$STACKCHAN_BACKUP_DIR/original-flash.bin"
python -m esptool --chip esp32s3 --port "$STACKCHAN_PORT" --baud 460800 \
  --before no_reset --after no_reset read_flash \
  0x0 0x1000000 "$STACKCHAN_BACKUP_DIR/original-flash-check.bin"
cmp "$STACKCHAN_BACKUP_DIR/original-flash.bin" \
  "$STACKCHAN_BACKUP_DIR/original-flash-check.bin"
shasum -a 256 "$STACKCHAN_BACKUP_DIR/original-flash.bin"
```

Require both files to contain exactly 16,777,216 bytes and compare equal. Save
the checksum, device identity, original partition table and restore command
alongside the backup. Keep the full backup private and outside Git because it
contains the device's settings and credentials. Keep one verified backup and
remove the redundant check copy after validation.

The full image includes bootloader, partition table, NVS, OTA state, apps and
assets. A local build or the available OTA binary is not a substitute.
See [esptool v4 flash commands][esptool-v4].

Completion condition: a verified full backup and a concrete restore command
exist before the first flash erase or write.

### 1.3. Establish a host reference

- Check `ITO_WEIGHTS_DIR`, the checkout and HF cache for the female chip weights.
- If access is missing, the user accepts the model terms and signs in through
  `hf auth login` in Ito's environment. Do not put a token in this plan or Git.
- Fetch only `ito_female_esp32s3.bin` for the baseline. Use the existing download
  mechanism, for example `.venv/bin/python -m ito.weights ito_female_esp32s3.bin`.
- Build `esp32/host/ito_cli` and run the existing frontend and main-model host
  tests. Generate reference WAV files with `esp32/tools/chip_wav.py`.
- Use a short greeting, a sentence containing numbers and punctuation, and a
  paragraph. Record the weight checksum, token IDs, seed and expected PCM hash
  for each comparable utterance. `chip_wav.py` inserts pauses between sentences;
  compare each utterance's PCM independently.

Completion condition: the chosen weights synthesize correctly on the host and
provide a reference for board self-tests. Model access follows
[the repository's weight instructions](../models/README.md).

### 1.4. Implement and build the StackChan target

- Add the helper and configuration described above. Retain upstream attribution
  for reused board initialization and driver code.
- Keep 240 MHz CPU, 80 MHz quad PSRAM, the existing optimized kernels and cache
  settings. Explicitly disable octal mode in the resolved configuration.
- Start with Ito's row-split policy. Its float row loops can still use both
  cores; this is not a single-core mode. Enable the existing automatic supply
  test only after baseline boot and audio work. Keep brownout detection and
  watchdog recovery enabled.
- Build one image that boots muted. Use existing silent commands and self-tests
  before requesting playback; add no diagnostic build variant or test subsystem.
- Use a fresh StackChan build directory. Inspect its resolved `sdkconfig` for
  PSRAM mode, codec selection, console, CPU clock and frontend enablement.
- Match the flash mode and frequency to the verified original bootloader and
  board configuration. The local StackChan build artifacts specify DIO at
  80 MHz; confirm that against the device backup.
- Check the map, app size and partition sizes. The app must fit its existing
  `0x1F0000` byte slot. Ensure all flashed ranges remain within 16 MB.
- Build the default DevKit and QEMU targets once to check configuration
  isolation. Run existing scheduler/split tests if their wiring changes.

Completion condition: board-specific build artifacts, hashes, flash offsets and
configuration are reviewable. Default prebuilt files remain untouched.

The proposed build command, added by the build-script change, is:

```sh
BOARD=stackchan esp32/tools/build_fw.sh
```

It writes `esp32/firmware/build_stackchan/ito_stackchan_merged.bin`. This command
is a planned interface and does not work in the unmodified checkout.

### 1.5. Flash and validate in stages

Use the verified backup from step 1.2. Reconfirm the device identity and select
the new StackChan image explicitly. Never invoke `flash.sh` with its default
prebuilt image for this board. The current script accepts explicit weight and
merged-image paths and erases the optional weight slots before writing.

Review its exact erase/write ranges and how the merged image treats the NVS
region. Initialize Ito's NVS as part of the dedicated image installation rather
than trying to reuse the factory NVS layout. The backup preserves factory data.
Start flashing at 460800 baud and lower it if transfer reliability requires it.

After the build and backup checks, use the existing explicit-path interface:

```sh
BAUD=460800 esp32/tools/flash.sh "$STACKCHAN_PORT" \
  models/ito_female_esp32s3.bin \
  esp32/firmware/build_stackchan/ito_stackchan_merged.bin
```

Run this from the Ito checkout in the IDF shell. The explicit weight argument
also prevents the script from attempting a model download in IDF Python.
After flashing completes, start the host monitor and reset the board to capture
its complete boot log. Save build and flashing output alongside the session
logs when diagnosing an installation failure.

Test in this order:

1. Muted boot detects 8 MB PSRAM and the required I2C devices. Check that
   the motors remain disabled and the backlight remains off.
2. Main-model self-test matches the blob's expected PCM hash. Verify the hot
   engine arena is in internal SRAM and weights are copied into PSRAM.
3. Run `ids`, `textq`, `stats` and `status`. Compare IDs and PCM with the host
   using the same frontend mode, token sequence, weight set and seed.
4. Request a quiet, short phrase from the same image. Verify that
   the amplifier is audible, channel packing is correct and idle output is
   quiet. Check volume, mute and `cancel` while audio is playing.
5. Run the greeting, numbers and paragraph cases. Test a second request while
   busy, an empty line and text beyond the documented limits. Preserve Ito's
   current busy/error behavior.
6. Once baseline audio passes, try the existing automatic supply calibration
   and compare it with row split. Retain only a mode that passes self-tests and
   remains stable. Never force full split to hide a failed supply test.

Completion condition: correct audible speech through the built-in speaker,
with memory placement and synthesis correctness established separately.

### 1.6. Measure and verify independence from the Mac

- Run 20 representative English prompts and a paragraph of at least 60 seconds
  of generated audio. Record compute time, real-time factor, planned playback
  delay, underruns, active weight set and split mode.
- Target the existing calibration margin of RTF at or below 0.85. Require RTF
  below 1.0 and zero playback underruns for a sustained streaming pass.
- Record first-PCM and first-I2S timing separately. Label them accurately.
  Confirm audible quality by listening or a local recording of the test
  phrases. Claim acoustic startup latency only if that recording measures it.
- If baseline speed is inadequate, measure the existing female int4 weights
  with their own self-test reference and checksum. Keep the existing model
  selection and buffering logic. Do not enable the unapproved female light
  variant or change model architecture as an automatic fallback.
  Download `ito_female_esp32s3_int4.bin` through Ito's environment and write it
  only to the existing `weights_b` slot at `0x680000`, then reboot and recalibrate.
  Keep the main blob at `0x200000` and record both offsets for final installation.
- Run repeated speech for at least 15 minutes and compare free memory after
  warm-up and at the end. Require no persistent memory loss, resets, hash
  failures or audible gaps. Perform five reset/boot cycles and repeat a phrase.
- Store one phrase in `main/phrases.txt` and enable the existing boot-phrase
  option for the final build. After completing and verifying the flash, close
  all serial processes and reset the device. It must speak the phrase without
  host text or host computation. Keep the startup phrase quiet and brief.

If quad PSRAM cannot sustain real-time speech with the supported variants and
existing scheduling controls, report the measured limit. Buffering a complete
utterance or changing hardware would require a separate decision. Do not call
that outcome a streaming pass or silently switch synthesis to the Mac.

### 1.7. Prove restore and leave Ito installed

After native speech passes, rehearse restoration once using the saved full
image. Re-enter download mode, recheck the port and use the same esptool v4
environment:

```sh
python -m esptool --chip esp32s3 --port "$STACKCHAN_PORT" --baud 460800 \
  --before no_reset --after no_reset write_flash \
  --flash_mode keep --flash_freq keep --flash_size keep \
  0x0 "$STACKCHAN_BACKUP_DIR/original-flash.bin"
python -m esptool --chip esp32s3 --port "$STACKCHAN_PORT" \
  --before no_reset --after no_reset verify_flash \
  0x0 "$STACKCHAN_BACKUP_DIR/original-flash.bin"
```

Reset and confirm the original robot app and its settings return. Retain the
verified original backup. Reinstall the accepted Ito image and the chosen
weights, then check boot-phrase playback and one USB text request. This leaves
the device in the requested Ito configuration.

Do not erase the entire flash as a separate recovery step. Writing the complete
backup already covers its sectors. If USB recovery needs physical button input,
use the manufacturer's download-mode procedure.

### Phase 1 completion checks

Proceed to phase 2 when:

- Arbitrary supported English text sent over USB produces correct, audible
  speech from StackChan's speaker.
- A stored phrase plays after reset with no host synthesis or text sender.
- The chosen model passes its self-test and the sustained playback checks.
- GPIO0 remains an audio clock, motors stay disabled and idle output is quiet.
- The full-flash restore has been rehearsed and the final device runs Ito.
- A board guide records exact build/flash/restore commands, versions, image and
  model hashes, measured limitations and the private backup location.
- A connected run produces readable, persistent serial logs and metadata at
  the location the host tool prints. Record phase 1 measurements there.

Retain the accepted phase 1 image, its configuration and model hashes as the
fallback for phase 2. Preserve the original robot flash backup independently.

## Phase 2. Add push-to-talk conversation

### Interaction and audio path

Start the Mac bridge in `chat` mode. It opens the robot's USB port, establishes
the control/audio protocol and reports readiness. The bridge must be ready
before the robot records. The boot phrase and USB text test remain available
without a conversation backend.

Hold the head touch sensor to record. Release it to finish the utterance and
submit it. Reuse the Si12T press/release handling from StackChan's source;
swipe gestures have no conversation function in this phase.

```text
Hold head -> robot microphone -> USB audio -> Mac speech recognition
Release -> Mac reply generation -> USB text -> Ito -> robot speaker
```

Use the robot's microphones and speaker for the conversation. The Mac's
microphone and speaker are not part of the normal flow.

Use one active turn and these states: idle, recording, transcribing, thinking
and speaking. The Mac retains bounded conversation history for the current
session. Keep capture and speaker playback separate to avoid adding acoustic
echo cancellation.

- From idle, a press starts a turn once the bridge is ready; release submits it.
- A press during transcription or thinking cancels that turn. If still held,
  it starts a new recording. Ignore late results from the cancelled turn.
- A press during speaking cancels and flushes Ito playback, mutes the speaker
  and then starts recording if the sensor is still held. Record only after
  output has stopped. If released before capture begins, return to idle.
- Empty recordings produce no reply. Bound each recording to 15 seconds for
  the first version. Reaching the limit submits it and requires release before
  another press can start a turn.
- On USB loss, backend failure or timeout, stop capture and playback, discard
  pending turn results and report the failure. Re-arm only after the bridge is
  ready. Never replay a stale reply after reconnecting.

Proposed CLI status copy is short: `Ready. Hold StackChan's head to talk.`,
`Recording`, `Transcribing`, `Thinking`, `Speaking` and a specific error when a
turn fails. No new screen layout is required.

### 2.1. Choose and measure the Mac backends

- Check the Mac's processor, available memory and existing model installations.
  Select one local speech-recognition model and one local conversation model
  that fit those resources. Exact models are a phase 2 decision, not a new
  requirement for phase 1.
- Verify transcription and text reply generation independently with a short
  sample. Record model versions, configuration and latency before integrating.
- Default to English, matching Ito's supported speech frontend. Request concise
  replies, normally one or two short sentences. Keep conversation history in
  Mac memory for the session; no database is needed.
- If local performance is inadequate, present measured alternatives before
  changing to an external service. Configure a selected service through its
  CLI or environment variables without storing credentials in Git.

Completion condition: one selected transcription path and one reply-generation
path work on the Mac, with their resource use and latency recorded.

### 2.2. Verify microphone capture and the USB transport

- Extend the existing board audio helper with the ES7210 driver. Input data is
  GPIO14; the board shares the MCLK/BCLK/word-select signals already listed.
  Reuse StackChan's working standard-transmit/TDM-receive configuration as the
  reference and verify slot selection and shared clock operation.
- Select a real microphone channel and exclude the speaker-reference channel.
  Send 24 kHz mono signed 16-bit little-endian PCM to the Mac, resampling there
  if the chosen recognizer requires another rate.
- Use bounded DMA and streaming buffers. A 15-second recording stays on the
  Mac; the robot does not retain the full utterance in its limited free PSRAM.
  Keep one owner of the I2S channels and one owner of USB input/output.
- Add a small versioned frame format for control events, audio chunks, reply
  text and diagnostic logs. Include bounded lengths, turn IDs and sequence
  numbers, with corruption detection and recovery. Serialize outgoing frames
  so ordinary diagnostic prints cannot corrupt an audio frame. Accept early
  raw boot logs before the framed-mode handshake.
- Carry one microphone channel, not the full raw TDM stream. Its payload rate
  is 48,000 bytes/second before framing. Measure the actual native USB path
  with runtime logs enabled; the console's nominal 115200 setting is not proof
  of capacity. Require at least 64,000 bytes/second usable sustained throughput
  and zero lost audio chunks during a full-length test recording.
- Bound pending writes and give control/cancellation priority over optional
  verbose logs. Backpressure or dropped audio fails the turn explicitly rather
  than silently submitting incomplete speech.
- Check clock/channel correctness with a local debug recording and confirm the
  speaker still passes phase 1 playback checks after input support is added.

If the existing native USB implementation cannot meet this test, revise the
USB implementation before connecting the backends. Changing to Wi-Fi or a
different USB device configuration requires a separate design decision.

Completion condition: clear robot microphone audio reaches the Mac, alongside
reliable control and logs, without breaking Ito output or exhausting memory.

### 2.3. Connect touch, transcription and spoken replies

- Add minimal Si12T polling for press and release, using the existing I2C bus.
  Retain the GPIO0 exclusion; it is not a push-to-talk button.
- Connect turn events to the host bridge. Stop capture on release and submit
  the received audio to the selected recognizer, then the transcript to the
  selected conversation model.
- Associate every asynchronous result and reply sentence with its turn ID.
  A cancellation invalidates remaining results before starting a new turn.
- Send short reply sentences to Ito through the same frontend as phase 1.
  Wait for playback completion before sending the next sentence, respecting
  Ito's existing busy behavior. Bound the reply queue and obey its text limits.
- Keep microphones disabled during reply playback. Recheck speaker silence
  before opening input for an interrupting press.
- Add explicit host-ready, turn-complete, playback-complete and error messages
  so the bridge advances from events rather than estimated audio durations.
- Give host readiness an expiry and bound each backend wait. A crashed bridge
  must disarm capture even while the USB cable remains connected.
- Keep the phase 1 text console/monitor available when framed chat mode is not
  active. Use a StackChan-only conversation configuration flag, off by default,
  so the accepted speech-only build remains reproducible.

Completion condition: hold, speak and release produces a spoken reply through
the robot, and another hold can interrupt or start the next turn.

### 2.4. Validate conversations and logs

- Run at least ten multi-turn exchanges through the robot's microphone and
  speaker. Include a short question, a follow-up that needs session context,
  numbers and a longer reply. Confirm intelligible speech and correct state
  transitions.
- Test empty and maximum-length recordings, rapid touches, cancellation while
  thinking, interruption while speaking, USB unplug/reconnect and a backend
  timeout. Require microphone shutdown and no stale spoken reply in each case.
- Log release-to-transcript, transcription duration, reply-generation duration,
  release-to-first-spoken-reply timing, Ito compute/playback metrics and
  cancellation timing. Distinguish measured acoustic onset from software events.
  Report the observed median and slowest turn rather than promising the phase 1
  synthesis latency for the whole conversation.
- Run a 15-minute mixed capture/playback session. Require no resets, persistent
  memory loss, lost audio frames or TTS underruns. Recheck the hot arena and
  available internal SRAM with microphone and USB buffers allocated.
- Confirm that `serial.log`, `metadata.json` and `conversation.jsonl` can trace
  each turn from press through reply or cancellation. Confirm that raw audio
  files appear only when debug recording is enabled.
- Leave the accepted conversation build installed. Retain both the working
  phase 1 image and the original factory backup; another full restore rehearsal
  is needed only if the flash layout or restore procedure changes.

### Expected phase 2 changes

| Path in Ito | Change |
| --- | --- |
| `esp32/firmware/main/stackchan_audio.c` and `.h` | ES7210 capture, channel selection and input/output handoff |
| `esp32/firmware/main/stackchan_ptt.c` and `.h` | Minimal Si12T press/release handling and capture control |
| `esp32/firmware/main/stackchan_usb.c` and `.h` | Framed audio, control and diagnostic transport |
| `esp32/firmware/main/main.c` | Turn-aware playback events and cancellation wiring |
| StackChan configuration and component build files | Optional conversation mode and required sources |
| `esp32/tools/stackchan.py` | Chat mode, backend calls, turn cancellation and conversation logging |
| Host dependency configuration | Selected ASR/LLM dependencies, isolated from IDF Python |
| Board guide | Start/stop commands, head-touch behavior, backend setup and logs |

Keep these modules specific to this device and flow. Add no general plugin
framework, service manager, web UI or transport abstraction for unused backends.

### Phase 2 completion checks

- Hold-to-record and release-to-submit work through the robot's microphones.
- Mac-side recognition and reply generation feed Ito on the robot.
- Pressing during a reply stops it and permits a new turn after the speaker
  is silent; an old turn can never resume speaking.
- The USB throughput, memory, repeated-conversation and error-path checks pass.
- Every connected session saves usable logs at the documented location.
- The speech-only phase 1 build and original factory backup remain available.

## Decision records and handoff

During implementation, record hardware decisions and measurements in
`.agents/notes/proposed/firmware/2026-11-14-stackchan-ito.md`. Move the note to
`implemented/firmware/` only when the work ships, preserving its rationale and
keeping its facts aligned with the code. Keep backup binaries and credentials
outside the repository. Do not commit or open a PR without a shipping request.

## Source references

- [M5Stack hardware, download mode and schematic links][m5]
- [StackChan board and power initialization](../../StackChan/firmware/main/hal/board/stackchan.cc)
- [StackChan amplifier integration](../../StackChan/firmware/main/hal/board/cores3_audio_codec.cc)
- [StackChan head touch handling](../../StackChan/firmware/main/hal/hal_head_touch.cpp)
- [StackChan servo power control](../../StackChan/firmware/main/hal/hal_io_expander.cpp)
- [Ito firmware and GPIO0 button task](../esp32/firmware/main/main.c)
- [Ito flash layout](../esp32/firmware/partitions.csv)
- [Ito flashing script](../esp32/tools/flash.sh)
- [esptool v4 backup and flashing commands][esptool-v4]
- [esptool v4 verification and security information][esptool-advanced-v4]

[m5]: https://docs.m5stack.com/en/StackChan
[esptool-v4]: https://docs.espressif.com/projects/esptool/en/release-v4/esp32s3/esptool/basic-commands.html
[esptool-advanced-v4]: https://docs.espressif.com/projects/esptool/en/release-v4/esp32s3/esptool/advanced-commands.html
