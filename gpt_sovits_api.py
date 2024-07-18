import uvicorn
import os
from typing_extensions import TypedDict
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from new_audio_gen import LANGUAGE, CUT_TYPE, GPT_SoVITS, save_audio

class ReferenceAudio(TypedDict):
    ref_wav:str
    ref_wav_text:str
    ref_wav_lang:LANGUAGE

class GenerateAudioBody(BaseModel):
    ref_audio:ReferenceAudio | None = None,
    input_prompt:str 
    input_prompt_lang:LANGUAGE 
    cut:CUT_TYPE = 'cut_every_punctuation'
    top_k:int = 20
    top_p:float = 0.6
    temperature:float = 0.6

app = FastAPI()
gpt_sovits = GPT_SoVITS(
    gpt_path="./ezecrain-e50.ckpt",
    sovits_path="./ezecrain_e24.pth",
    hubert_path=os.environ.get("hubert_base_path", "hubert-base-ls960"),
    bert_path=os.environ.get("bert_path", "xlm-roberta-large")
)

@app.get("/")
async def check_connection():
    return {"connection":True}

@app.post("/api/inference")
async def generate_audio(req_body:GenerateAudioBody):
    result_audio = gpt_sovits.get_tts_wav(
        ref_wav_path="./ezecrain_ref.wav",
        ref_wav_text="その身を削り命を懸けてこの世にお前を送り出してくれたのだろう父親というのは",
        ref_wav_lang="all_ja",
        input_prompt=req_body.input_prompt,
        input_prompt_lang=req_body.input_prompt_lang,
        cut=req_body.cut,

    )
    result_list = list(result_audio)
    save_path = None
    save_name = None
    if result_list:
        sampling_rate, audio_data = result_list[-1]
        save_path, save_name = save_audio(sampling_rate=sampling_rate, audio_data=audio_data)
    return FileResponse(path=save_path, media_type="audio/wav", filename=save_name)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9880, workers=1)