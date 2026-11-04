/* ITO: see ito_espeak.h. Part of the Ito on-chip front end (GPLv3). The init sequence follows espeak_ng_Initialize()
 * and espeak_ng_SetVoiceByName() of espeak-ng 1.52.0 without the audio, wave generator, event and fifo parts. */
#include <stdlib.h>
#include <string.h>
#include "speech.h"
#include "synthdata.h"
#include "synthesize.h"
#include "translate.h"
#include "voice.h"
#include "wavegen.h"
#include "readclause.h"
#include "dictionary.h"
#include "ito_vfs.h"
#include "ito_espeak.h"

espeak_ng_STATUS SetParameter(int parameter, int value, int relative);
extern int saved_parameters[N_SPEECH_PARAM];
extern PARAM_STACK param_stack[];
extern const int param_defaults[N_SPEECH_PARAM];
extern espeak_VOICE current_voice_selected;
extern espeak_ng_TEXT_DECODER *p_decoder;

int itoes_init(const void *blob, size_t len)
{
	int srate = 22050;
	int r = ito_vfs_set_blob(blob, len);
	if (r) return r;
	path_home[0] = 0;
	if (LoadPhData(&srate, NULL) != ENS_OK) return -10;
	samplerate = srate;
	for (int ix = 0; ix < N_EMBEDDED_VALUES; ix++) embedded_value[ix] = embedded_default[ix];
	memset(&current_voice_selected, 0, sizeof(current_voice_selected));
	SetVoiceStack(NULL, "");
	SynthesizeInit();
	InitNamedata();
	VoiceReset(0);
	for (int param = 0; param < N_SPEECH_PARAM; param++)
		param_stack[0].parameter[param] = saved_parameters[param] = param_defaults[param];
	SetParameter(espeakRATE, espeakRATE_NORMAL, 0);
	SetParameter(espeakVOLUME, 100, 0);
	SetParameter(espeakCAPITALS, option_capitals, 0);
	SetParameter(espeakPUNCTUATION, option_punctuation, 0);
	SetParameter(espeakWORDGAP, 0, 0);
	option_phonemes = 0;
	option_phoneme_events = 0;

	/* espeak_SetVoiceByName("en-us"): the voice list resolves it to the file gmw/en-US */
	if (LoadVoice("gmw/en-US", 0) == NULL) return -11;
	DoVoiceChange(voice);
	espeak_VOICE sel;
	memset(&sel, 0, sizeof(sel));
	sel.name = (char *)"en-us";
	sel.languages = voice->language_name;
	SetVoiceStack(&sel, "");
	return 0;
}

int itoes_text_to_phonemes(const char *utf8, char *out, size_t cap)
{
	const void *textptr = utf8;
	size_t n = 0;
	int first = 1;
	if (cap) out[0] = 0;
	if (p_decoder == NULL) p_decoder = create_text_decoder();
	while (textptr != NULL) {
		if (text_decoder_decode_string_multibyte(p_decoder, textptr, translator->encoding, 1) != ENS_OK) return -1;
		TranslateClause(translator, NULL, NULL);
		textptr = text_decoder_get_buffer(p_decoder);
		const char *ph = GetTranslatedPhonemeString((int)('_' << 8) | 0x02);
		if (ph && *ph) {
			size_t l = strlen(ph);
			if (n + l + 2 > cap) return -2;
			if (!first) out[n++] = ' ';
			memcpy(out + n, ph, l);
			n += l;
			out[n] = 0;
			first = 0;
		}
	}
	return (int)n;
}
