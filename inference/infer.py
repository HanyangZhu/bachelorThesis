import os
import json
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor


processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")



dataset = load_dataset("tedlium", split="test")


audio_paths = [example['audio']['path'] for example in dataset]


conversations = []
for url in audio_paths:
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": url},
            {"type": "text", "text": "请转录此音频"},
        ]}
    ]
    conversations.append(conversation)


text = [processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) for conversation in conversations]


audios = []
for conversation in conversations:
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audio_data = librosa.load(
                        BytesIO(urlopen(ele['audio_url']).read()),
                        sr=processor.feature_extractor.sampling_rate
                    )[0]
                    audios.append(audio_data)

inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)

inputs['input_ids'] = inputs['input_ids'].to("cuda")

generate_ids = model.generate(**inputs, max_length=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

output_file = "asr_output.json"
output_data = []

for url, transcription in zip(audio_paths, response):
    output_data.append({
        "audio_url": url,
        "transcription": transcription
    })

with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)