# Proof of concept using pymupdf4llm

import pymupdf4llm
from kokoro import KPipeline
import sounddevice as sd

def parse_and_read(file_path):
    md = parse_doc(file_path)
    print(md)
    # read_doc(md)


def parse_doc(file_path):
    md_text = pymupdf4llm.to_markdown(file_path)
    return md_text

def read_doc(md):
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(md, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        print(gs)
        sd.play(audio, samplerate=24000)
        sd.wait()