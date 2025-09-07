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

import scipy
import numpy as np


def save_wav(wav: np.ndarray, path: str, sample_rate: int = None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
        pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)
    scipy.io.wavfile.write(path, sample_rate, wav_norm)


joke = (
    "A teacher asked her students,\n"
    '"If you had one day to live, what would you do?"\n'
    'One kid said, "I’d spend it with my family."\n'
    'Another one said, "I’d eat all the candy I want."\n'
    "Then little Johnny raised his hand and said,\n"
    '"I’d stay in school because time always feels longer here."'
)

joke_1 = (
    "I was on a roller coaster for the first time.\n"
    "At first, I was excited, smiling, waving at everyone...\n"
    "Then it went higher... and higher...\n"
    "Suddenly, it dropped!\n"
    "I screamed so loud, the guy next to me asked if I was okay...\n"
    "I said, 'No! I think my soul stayed up there!'"
)
joke_2 = """
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

text = "I was utterly awestruck as I stood before the majestic Himalayas, towering over 8,848 meters;\
      their sheer beauty and grandeur made my heart race with an indescribable sense of adventure and gratitude for nature's wonders."
# Use a ref_audio_path different from the speaker_wav in tts_to_file to check if the input conditioning latent works.
ref_audio_path = "./reference/deepgram-aura-2-vesta-en.wav"
import time

start_time = time.time()
# All the hard coded args come from https://huggingface.co/coqui/XTTS-v2/blob/main/config.json
(gpt_cond_latent, speaker_embedding) = model.get_conditioning_latents(
    audio_path=ref_audio_path,
    gpt_cond_len=30,
    gpt_cond_chunk_len=4,
    max_ref_length=30,
    sound_norm_refs=False,
)

exec_time = time.time() - start_time
print(f"> Embedding time {exec_time}")
print("> TTS model initialization finished.")

start_time = time.time()
outputs = model.inference(
    text=joke_2,
    language="en",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    speed=1.0,
    enable_text_splitting=True,  # Set true for long paragraph
)
process_time = time.time() - start_time

wav = outputs["wav"]
if torch.is_tensor(wav) and wav.device != torch.device("cpu"):
    wav = wav.cpu().numpy()
wav = wav.squeeze()

audio_time = len(wav) / model.config.audio["sample_rate"]

save_wav(wav, "output_fun.wav", sample_rate=model.config.audio["sample_rate"])

print(f" > Processing time: {process_time}")
print(f" > Real-time factor: {process_time / audio_time}")

