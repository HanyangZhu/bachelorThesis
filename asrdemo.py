from datasets import load_dataset

fleurs_asr = load_dataset("google/fleurs", "af_za")  # for Afrikaans
# to download all data for multi-lingual fine-tuning uncomment following line
# fleurs_asr = load_dataset("google/fleurs", "all")

# see structure
print(fleurs_asr)

# load audio sample on the fly
audio_input = fleurs_asr["train"][0]["audio"]  # first decoded audio sample
transcription = fleurs_asr["train"][0]["transcription"]  # first transcription
# use `audio_input` and `transcription` to fine-tune your model for ASR

# for analyses see language groups
all_language_groups = fleurs_asr["train"].features["lang_group_id"].names
lang_group_id = fleurs_asr["train"][0]["lang_group_id"]

all_language_groups[lang_group_id]