/* ITO: definitions the text->phonemes build of espeak-ng needs from the files that are left out (speech.c, wavegen.c,
 * synthesize.c, soundicon.c, compiledict.c). Names and values follow espeak-ng 1.52.0. Part of the Ito front end; GPLv3. */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "speech.h"
#include "synthesize.h"
#include "translate.h"
#include "voice.h"
#include "wavegen.h"
#include "compiledict.h"
#include "soundicon.h"

char path_home[N_PATH_HOME];

/* wavegen.c */
int embedded_value[N_EMBEDDED_VALUES];
int samplerate = 22050;
const int embedded_default[N_EMBEDDED_VALUES] = { 0, 50, espeakRATE_NORMAL, 100, 50, 0, 0, 0, espeakRATE_NORMAL, 0, 0, 0, 0, 0, 0 };
const unsigned char *const envelope_data[N_ENVELOPE_DATA] = { 0 };
int FormantTransition2(frameref_t *seq, int *n_frames, unsigned int data1, unsigned int data2, PHONEME_TAB *other_ph, int which) { return 0; }
void InitBreath(void) {}
espeak_ng_STATUS DoVoiceChange(voice_t *v) { return ENS_OK; }

/* synthesize.c */
int n_phoneme_list = 0;
PHONEME_LIST phoneme_list[N_PHONEME_LIST+1];
const char *WordToString(char buf[5], unsigned int word)
{
	for (int ix = 0; ix < 4; ix++) buf[ix] = word >> (ix*8);
	buf[4] = 0;
	return buf;
}

/* soundicon.c (no sound icons) */
int n_soundicon_tab = 0;
SOUND_ICON soundicon_tab[N_SOUNDICON_TAB];
int LookupSoundicon(int c) { return -1; }
int LoadSoundFile2(const char *fname) { return -1; }

/* speech.c */
int espeak_SetParameter(espeak_PARAMETER parameter, int value, int relative) { return 0; }

/* compiledict.c (trace output only) */
char *DecodeRule(const char *group_chars, int group_length, char *rule, int control, char *output) { output[0] = 0; return output; }
void print_dictionary_flags(unsigned int *flags, char *buf, int buf_len) { if (buf_len > 0) buf[0] = 0; }

/* speech.c: default parameters */
const int param_defaults[N_SPEECH_PARAM] = { 0, espeakRATE_NORMAL, 100, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

/* wavegen.c: embedded commands (only the value bookkeeping; no waveform generator) */
static const int embedded_max[N_EMBEDDED_VALUES] = { 0, 0x7fff, 2000, 300, 99, 99, 99, 0, 2000, 0, 0, 0, 0, 4, 0 };
static int general_amplitude;
int GetAmplitude(void)
{
	static const unsigned char amp_emphasis[5] = { 16, 16, 10, 16, 22 };
	int amp = (embedded_value[EMBED_A])*55/100;
	general_amplitude = amp * amp_emphasis[embedded_value[EMBED_F]] / 16;
	return general_amplitude;
}
static int SetWithRange0(int value, int max)
{
	if (value < 0) return 0;
	if ((value > max) && (max > 0)) return max;
	return value;
}
void SetEmbedded(int control, int value)
{
	int sign = 0;
	int command = control & 0x1f;
	if ((control & 0x60) == 0x60) sign = -1;
	else if ((control & 0x60) == 0x40) sign = 1;
	if (command < N_EMBEDDED_VALUES) {
		if (sign == 0) embedded_value[command] = value;
		else embedded_value[command] += (value * sign);
		embedded_value[command] = SetWithRange0(embedded_value[command], embedded_max[command]);
	}
	if (command == EMBED_A || command == EMBED_F) general_amplitude = GetAmplitude();
}

/* voices.c: the voice selection by language / properties is left out (SSML voice changes and automatic switches for other scripts) */
espeak_VOICE current_voice_selected;
espeak_VOICE *espeak_GetCurrentVoice(void) { return &current_voice_selected; }
espeak_VOICE *SelectVoiceByName(espeak_VOICE **voices, const char *name2) { return NULL; }
char const *SelectVoice(espeak_VOICE *voice_select, int *found) { *found = 0; return ""; }

/* synthesize.c */
SPEED_FACTORS speed;
void SynthesizeInit(void) {}
