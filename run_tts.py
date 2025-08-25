from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
# import torch
# torch.serialization.add_safe_globals([XttsConfig])


tts = TTS(
    model_path="/models/XTTS-v2",
    config_path="/models/XTTS-v2/config.json",
    gpu=True
)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output-en.wav",
                speaker_wav="/root/workspace/tts-root/TTS/tests/data/ljspeech/wavs/LJ001-0002.wav",
                language="en")

tts.tts_to_file(text="لقد استغرق مني وقتًا طويلاً لتطوير صوتي، والآن بعد أن أصبح لدي، لن ألتزم الصمت",
                    file_path="output-ar.wav",
                    speaker_wav="/root/workspace/tts-root/TTS/tests/data/ljspeech/wavs/LJ001-0002.wav",
                    language="ar")

tts.tts_to_file(text ="我花了很长时间才形成自己的声音，现在我已经拥有了它，我不会保持沉默",
                    file_path="output-zh-cn.wav",
                    speaker_wav="/root/workspace/tts-root/TTS/tests/data/ljspeech/wavs/LJ001-0002.wav",
                    language="zh-cn")

tts.tts_to_file(text ="मेरी अपनी आवाज़ विकसित करने में मुझे काफ़ी समय लगा, और अब जबकि मेरे पास यह है, मैं चुप नहीं रहूँगी।",
                    file_path="output-hi.wav",
                    speaker_wav="/root/workspace/tts-root/TTS/tests/data/ljspeech/wavs/LJ001-0002.wav",
                    language="hi")
