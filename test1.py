from datasets import load_dataset

gigaspeech = load_dataset("speechcolab/gigaspeech", "xs")

print(gigaspeech)