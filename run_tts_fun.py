
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

tts.tts_to_file(text=joke,
        file_path="output-en-joke.wav",
        speaker_wav = "./reference/trophy-wife-female-spoken-213777.mp3",
        language="en")
