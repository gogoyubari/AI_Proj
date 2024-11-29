import os
os.environ["CT2_VERBOSE"] = "2"
os.environ["CT2_CUDA_ALLOCATOR"] = "cub_caching"
os.environ["CT2_CUDA_CACHING_ALLOCATOR_CONFIG"] = "4,3,15,3221225471"
os.environ["CUDNN_LOGLEVEL_DBG"] = "2"
os.environ["CUDNN_LOGDEST_DBG"] = "stderr"

import time
import sys
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from ctranslate2 import set_random_seed
import logging
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

import pysubs2
from term_printer import Color, cprint
import re
import dataclasses

def tsToStr(ts):
    try:
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = ts % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    except:
        return f"00:00:00.000"
    
def whisper(filepath):
    # model_size_or_path="distil-large-v3"
    model_size_or_path="large-v3-turbo"

    model = WhisperModel(
        model_size_or_path=model_size_or_path,
        device="cuda",
        compute_type="float16",
        download_root="model",
    )
    set_random_seed(0)

    with open('initial_prompt_J.txt', encoding="utf_8_sig") as f:
        lines = f.read().strip().split('\n')
        init_prompt = ''.join(['<|0.00|> ' + line.strip() + '<|0.00|>' for line in lines])
        print("initial_prompt has been loaded.")

    segments, info = model.transcribe(
        audio = filepath,
        language = "ja",
        log_progress = False,
        temperature = [n/10 for n in range(11)],
        # beam_size = 5,
        # condition_on_previous_text = True,
        # prompt_reset_on_temperature = 0.5,
        initial_prompt = init_prompt,
        word_timestamps = True,
        vad_filter = True,
        vad_parameters = VadOptions(
            # onset = 0.4,
            # offset = 0.263,
            # min_speech_duration_ms = 0,
            # max_speech_duration_s = 2.0,
            # min_silence_duration_ms = 500,
            speech_pad_ms = 800
        ),
        hallucination_silence_threshold = 0.2
    )

    d = []
    start = time.time()

    for segment in segments:
        if re.match(r'.*[\.\,\?\!]$', segment.text):
            col = Color.WHITE
        else:
            col = Color.YELLOW
        cprint(f"[{segment.id}] {tsToStr(segment.start)} {tsToStr(segment.end)} {segment.text}", attrs=[col])
        d.append(dataclasses.asdict(segment))

    print(f"time: {time.time()-start}")

    subs = pysubs2.load_from_whisper(d)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    subs.save(f"{base_name}.srt", format_='srt', encoding="utf_8_sig")

if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1:
        try:
            whisper(args[1])
        except KeyboardInterrupt:
            print("main_thread has been interrupted.")
        except Exception as e:
            print(e)
