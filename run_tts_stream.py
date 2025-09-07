import os

"""
Environment variables such as MIOPEN_FIND_MODE=FAST need to be set before loading and initializing the basic library,
 otherwise they may not take effect.
"""
os.environ["MIOPEN_FIND_MODE"] = "FAST"
os.environ["MIOPEN_LOG_LEVEL"] = "1"

import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# read model path from env "/models/XTTS-v2"
MODEL_PATH = os.getenv("MODEL_PATH", "/models/XTTS-v2")

config = XttsConfig()
config.load_json(os.path.join(MODEL_PATH, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_PATH)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


joke = """
A deep learning engineer walks into a coffee shop with their laptop.
They ask the barista: "Can you make me a coffee as fast as possible?"
The barista says: "Sure, do you want it hot or iced?"
The engineer replies: "I don’t care, just benchmark both and pick the faster one."

Five minutes later, the barista comes back and says:
"Well... the iced coffee was faster to make, but during testing,
we found the hot coffee tasted better. So I saved both recipes in a database,
in case you come back with the same order tomorrow."

The engineer smiles and says: "Perfect! That’s exactly how MIOpen incremental tuning works."
"""

# text = "DeepSeek R1 is a large transformer-based language model with strong multi-step reasoning capabilities"

ref_audio_path = "/root/workspace/tts-root/reference_audios/trophy-wife-female-spoken-213777.mp3"
import time

_time = time.time()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=ref_audio_path)
# conditioning_latent = (gpt_cond_latent, speaker_embedding)
exec_time = time.time() - _time
print(f"> Embedding time {exec_time}s in test_tts.py")
print("tts model initialization finished.")


import soundfile as sf
import numpy as np

out_dir = "tts_chunks"
os.makedirs(out_dir, exist_ok=True)
sr = 24000
wav_chunks = []


for i, chunk in enumerate(
    model.inference_stream(
        joke,
        "en",  # specify the language
        gpt_cond_latent,
        speaker_embedding,
        speed=1.25,
        # stream_chunk_size = 80,
        enable_text_splitting=True,
    )
):
    # if chunk is torch.Tensor then turn to numpy
    if hasattr(chunk, "detach"):
        chunk_np = chunk.detach().cpu().numpy()
    else:
        chunk_np = np.asarray(chunk)

    wav_chunks.append(chunk_np)

    out_path = os.path.join(out_dir, f"chunk_{i}.wav")
    sf.write(out_path, chunk_np, sr)
    print(f"Save chunk {i} → {out_path}")

# Combine all chunks to make a complete file
full_audio = np.concatenate(wav_chunks)
sf.write("output_full.wav", full_audio, sr)
print("save the complete file as output_full.wav")

