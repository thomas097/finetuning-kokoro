import torch
from kokoro import KPipeline, KModel

voice_tensor = torch.load('voices/af_river.pt', weights_only=True)
print("Voice tensor shape:", voice_tensor.shape)

model = KModel(repo_id="hexgrad/Kokoro-82M").train()
processor = KPipeline(repo_id="hexgrad/Kokoro-82M", lang_code='a')


if __name__ == '__main__':
    print('Generating...')
    result = processor(model=model, text="This is a text!", voice=voice_tensor)
    print("Done!")
    print(result)
