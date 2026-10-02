// compiled-in fallback demo sentences (blobs normally carry their own): held-out sentences (StyleTTS 2 token ids, leading 0 pad)
#pragma once
#include <stdint.h>

// LJ005-0014: Speaking on a debate on prison matters, he declared that
// spˈiːkɪŋ ˌɔn ɐ dᵻbˈeɪt ˌɔn pɹˈɪzən mˈæɾɚz , hiː dᵻklˈɛɹd ðˈæt
static const uint8_t demo_tok_0[62] = {0,61,58,156,51,158,53,102,112,16,157,76,56,16,70,16,46,177,44,156,47,102,62,16,157,76,56,16,58,123,156,102,68,83,56,16,55,156,72,125,85,68,16,3,16,50,51,158,16,46,177,53,54,156,86,123,46,16,81,156,72,62};
// LJ001-0110: Even the Caslon type when enlarged shows great shortcomings in this respect:
// ˈiːvən ðə kˈæslɑːn tˈaɪp wɛn ɛnlˈɑːɹdʒd ʃˈoʊz ɡɹˈeɪt ʃˈɔːɹtkʌmɪŋz ɪn ðɪs ɹᵻspˈɛkt :
static const uint8_t demo_tok_1[84] = {0,156,51,158,64,83,56,16,81,83,16,53,156,72,61,54,69,158,56,16,62,156,43,102,58,16,65,86,56,16,86,56,54,156,69,158,123,46,147,46,16,131,156,57,135,68,16,92,123,156,47,102,62,16,131,156,76,158,123,62,53,138,55,102,112,68,16,102,56,16,81,102,61,16,123,177,61,58,156,86,53,62,16,2};
// LJ003-0111: He was in consequence put out of the protection of their internal law, end quote. Their code was a subject of some curiosity.
// hiː wʌz ɪŋ kˈɑːnsɪkwəns pˌʊt ˌaʊɾəv ðə pɹətˈɛkʃən ʌv ðɛɹ ɪntˈɜːnəl lˈɔː , ˈɛnd kwˈoʊt . ðɛɹ kˈoʊd wʌzɐ sˈʌbdʒɛkt ʌv sˌʌm kjˌʊɹɹɪˈɔsɪɾi .
static const uint8_t demo_tok_2[137] = {0,50,51,158,16,65,138,68,16,102,112,16,53,156,69,158,56,61,102,53,65,83,56,61,16,58,157,135,62,16,157,43,135,125,83,64,16,81,83,16,58,123,83,62,156,86,53,131,83,56,16,138,64,16,81,86,123,16,102,56,62,156,87,158,56,83,54,16,54,156,76,158,16,3,16,156,86,56,46,16,53,65,156,57,135,62,16,4,16,81,86,123,16,53,156,57,135,46,16,65,138,68,70,16,61,156,138,44,46,147,86,53,62,16,138,64,16,61,157,138,55,16,53,52,157,135,123,123,102,156,76,61,102,125,51,16,4};

#define N_DEMOS 3
static const uint8_t *const demo_tok[N_DEMOS] = {demo_tok_0,demo_tok_1,demo_tok_2};
static const int demo_len[N_DEMOS] = {62,84,137};
static const int demo_style[N_DEMOS] = {21,49,37};
static const char *const demo_text[N_DEMOS] = {"Speaking on a debate on prison matters, he declared that","Even the Caslon type when enlarged shows great shortcomings in this respect:","He was in consequence put out of the protection of their internal law, end quote. Their code was a subject of some curiosity."};
