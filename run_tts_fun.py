
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
import os
MODEL_PATH = os.getenv("MODEL_PATH", "./XTTS-v2")

tts = TTS(
    model_path = MODEL_PATH,
    config_path= os.path.join(MODEL_PATH,"config.json"),
    gpu=True
)

joke = (
    "A teacher asked her students,\n"
    "\"If you had one day to live, what would you do?\"\n"
    "One kid said, \"I’d spend it with my family.\"\n"
    "Another one said, \"I’d eat all the candy I want.\"\n"
    "Then little Johnny raised his hand and said,\n"
    "\"I’d stay in school because time always feels longer here.\""
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

# Use a ref_audio_path different from the speaker_wav in tts_to_file to check if the input conditioning latent works.
ref_audio_path = "./reference/i-wish-you-the-best-in-all-your-endeavors-male-spoken-264678.mp3"
import time
start_time = time.time()
# All the hard coded args come from https://huggingface.co/coqui/XTTS-v2/blob/main/config.json
(gpt_cond_latent, speaker_embedding) = tts.synthesizer.tts_model.get_conditioning_latents(
    audio_path=ref_audio_path,
    gpt_cond_len=30, 
    gpt_cond_chunk_len=4,
    max_ref_length=30,
    sound_norm_refs=False,
        )
conditioning_latent = (gpt_cond_latent, speaker_embedding)
exec_time = time.time()-start_time
print(f"> Embedding time {exec_time}")
print("> TTS model initialization finished.")

tts.tts_to_file(text=joke,
        file_path="output-en-joke.wav",
        speaker_wav = "./reference/trophy-wife-female-spoken-213777.mp3",
        conditioning_latent = conditioning_latent,
        language="en")
