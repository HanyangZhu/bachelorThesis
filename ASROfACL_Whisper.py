import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

output_path = r"C:\Users\38644\PycharmProjects\bachelor\datasets\ACL.test.dataset\2\acl_6060\dev\ASR_result\asr_results.txt"

dataset = load_dataset(r"C:\Users\38644\PycharmProjects\bachelor\datasets\ACL.test.dataset\2\acl_6060\dev\full_wavs", "default", split="train")


with open(output_path, "w", encoding="utf-8") as output_file:
    for i, sample in enumerate(dataset):
        audio = sample["audio"]

        # 使用ASR进行转录
        result = pipe(audio, return_timestamps=True)
        hypothesis = result["text"]

        # 写入转录结果文件
        output_file.write(f"Hypothesis: {hypothesis.strip()}\n\n")


#print(result["text"])