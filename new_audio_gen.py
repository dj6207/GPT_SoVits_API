import os
import soundfile as sf
import re
import numpy as np
import torch
import librosa
import LangSegment
import torch.nn as nn
from typing import Literal
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from tools.my_utils import load_audio
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

LANGUAGE_DICT = {
    "zh": "all_zh",
    "en": "en",
    "ja": "all_ja"
}

LANGUAGE = Literal['all_zh', 'en', 'all_ja']

CUT_TYPE = Literal[
    'no_cut', 
    'cut_every_4_sentence',
    'cut_every_50_character',
    'cut_every_zh_ja_punctuation',
    'cut_every_en_punctuation',
    'cut_every_punctuation'
]

CUT:dict[CUT_TYPE, str] = {
    'no_cut': lambda text: text, 
    'cut_every_4_sentence': lambda text: cut1(text),
    'cut_every_50_character': lambda text: cut2(text),
    'cut_every_zh_ja_punctuation': lambda text: cut3(text),
    'cut_every_en_punctuation': lambda text: cut4(text),
    'cut_every_punctuation': lambda text: cut5(text)
}

PUNCTUATION = {'!', '?', '…', ',', '.', '-'," "}
PUNCTUATION_EXTENDED = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

class LangHubert(nn.Module):
    def __init__(self, hubert_path:str):
        super().__init__()
        self.model = HubertModel.from_pretrained(hubert_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            hubert_path
        )

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats

def get_device() -> Literal['cuda', 'cpu']:
    if torch.cuda.is_available(): return 'cuda'
    return 'cpu'

class GPT_SoVITS:
    def __init__(self, sovits_path:str, gpt_path:str, hubert_path:str, bert_path:str):
        self.device = get_device()

        model = LangHubert(hubert_path=hubert_path)
        model.eval()
        self.ssl_model = model
        self.ssl_model = self.ssl_model.to(self.device)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

        self.set_gpt_models(gpt_path=gpt_path)
        self.set_sovits_model(sovits_path=sovits_path)
    
    def set_gpt_models(self, gpt_path:str) -> None:
        self.gpt_hz = 50
        gpt_dict = torch.load(gpt_path, map_location="cpu")
        self.gpt_config = gpt_dict['config']
        self.gpt_max_sec = self.gpt_config["data"]["max_sec"]
        self.gpt_model = Text2SemanticLightningModule(self.gpt_config, "****", is_train=False)
        self.gpt_model.load_state_dict(gpt_dict["weight"])
        self.gpt_model = self.gpt_model.to(self.device)
        self.gpt_model.eval()

    def set_sovits_model(self, sovits_path:str) -> None:
        sovits_dict = torch.load(sovits_path, map_location="cpu")
        self.sovits_config = DictToAttrRecursive(sovits_dict['config'])
        self.sovits_config.model.semantic_frame_rate = "25hz"
        self.sovits_model = SynthesizerTrn(
            self.sovits_config.data.filter_length // 2 + 1,
            self.sovits_config.train.segment_size // self.sovits_config.data.hop_length,
            n_speakers=self.sovits_config.data.n_speakers,
            **self.sovits_config.model
        )
        # Not sure if this is needed gonna comment it out for now
        if ("pretrained" not in sovits_path):
            del self.sovits_model.enc_q
        
        self.sovits_model = self.sovits_model.to(self.device)
        self.sovits_model.eval()
        self.sovits_model.load_state_dict(sovits_dict["weight"], strict=False)    

    def get_tts_wav(
            self, 
            ref_wav_path:str, 
            ref_wav_text:str, 
            ref_wav_lang:LANGUAGE, 
            input_prompt:str, 
            input_prompt_lang:LANGUAGE, 
            cut:CUT_TYPE, 
            top_k:int=20, 
            top_p:float=0.6, 
            temperature:float=0.6
        ):
        processed_ref_wav_text = process_ref_prompt(ref_prompt=ref_wav_text, ref_lang=ref_wav_lang)
        processed_prompt = precut_preprocess_prompt(prompt=input_prompt, prompt_lang=input_prompt_lang)
        zero_wav = np.zeros(int(self.sovits_config.data.sampling_rate * 0.3),dtype=np.float32)

        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError("Shit needs to ne 3-10 sec")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(self.device)
            zero_wav_torch = zero_wav_torch.to(self.device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.sovits_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

        processed_prompt = CUT[cut](processed_prompt)
        processed_prompt = post_cut_preprocess_prompt(prompt=processed_prompt)
        audio_opt = []
        phones1, bert1, norm_text1 = self.get_phones_and_bert(text=processed_ref_wav_text, language=ref_wav_lang)

        for text in processed_prompt:
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in SPLITS): text += "。" if input_prompt_lang != "en" else "."
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, input_prompt_lang)
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            with torch.no_grad():
                pred_semantic, idx = self.gpt_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.gpt_hz * self.gpt_max_sec,
                )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            refer = self.get_sovits_spec(ref_wav_path)  # .to(device)
            refer = refer.to(self.device)
            audio = (
                self.sovits_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refer
                )
                    .detach()
                    .cpu()
                    .numpy()[0, 0]
            )
            max_audio=np.abs(audio).max()
            if max_audio>1:audio/=max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
        yield self.sovits_config.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    
    def get_phones_and_bert(self, text:str, language:str):
        if language in {"en","all_zh","all_ja"}:
            language = language.replace("all_","")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            phones, word2ph, norm_text = clean_text_inf(formattext, language)
            if language == "zh":
                bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            else:
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja","auto"}:
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en","ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "ko":
                        langlist.append("zh")
                        textlist.append(tmp["text"])
                    else:
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        langlist.append(language)
                    textlist.append(tmp["text"])
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        return phones,bert.to(torch.float32),norm_text

    def get_bert_feature(self, text:str, word2ph):
        with torch.no_grad():
            inputs = self.bert_tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T
    
    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language=language.replace("all_","")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)#.to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)

        return bert
    
    def get_sovits_spec(self, filename:str):
        audio = load_audio(filename, int(self.sovits_config.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.sovits_config.data.filter_length,
            self.sovits_config.data.sampling_rate,
            self.sovits_config.data.hop_length,
            self.sovits_config.data.win_length,
            center=False,
        )
        return spec

def clean_text_inf(text:str, language:str):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

def precut_preprocess_prompt(prompt:str, prompt_lang:str) -> str:
    processed_prompt = prompt.replace("\n", " ")
    processed_prompt = replace_consecutive_punctuation(text=processed_prompt)
    if (processed_prompt[0] not in SPLITS and len(get_first(processed_prompt)) < 4): processed_prompt = "。" + processed_prompt if prompt_lang != "en" else "." + processed_prompt  
    return processed_prompt
    
def post_cut_preprocess_prompt(prompt:str) -> list[str]:
    prompt_list = prompt.split("\n")
    processed_prompt_list = []
    if all(text in [None, " ", "\n",""] for text in prompt_list): raise ValueError("Invaid Prompt")
    for text in prompt_list:
        if text in  [None, " ", ""]: pass
        else: processed_prompt_list.append(text)
    return merge_short_text_in_array(processed_prompt_list, 5)

def merge_short_text_in_array(texts:list[str], threshold:int) -> list[str]:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def replace_consecutive_punctuation(text:str) -> str:
    punctuations = ''.join(re.escape(p) for p in PUNCTUATION)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result

def process_ref_prompt(ref_prompt:str, ref_lang:str) -> str:
    processed_ref_prompt = ref_prompt.strip("\n")
    if (processed_ref_prompt[-1] not in SPLITS): processed_ref_prompt += "。" if ref_lang != "en" else "."
    return processed_ref_prompt

def get_first(text:str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in SPLITS) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def cut1(input:str) -> str:
    input = input.strip("\n")
    inputs = split(input)
    split_idx = list(range(0, len(inputs), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inputs[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [input]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)

def cut2(input:str) -> str:
    input = input.strip("\n")
    inputs = split(input)
    if len(inputs) < 2:
        return input
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inputs)):
        summ += len(inputs[i])
        tmp_str += inputs[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)

def cut3(input:str) -> str:
    input = input.strip("\n")
    opts = ["%s" % item for item in input.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return  "\n".join(opts)

def cut4(input:str) -> str:
    input = input.strip("\n")
    opts = ["%s" % item for item in input.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)

def cut5(input:str) -> str:
    input = input.strip("\n")
    mergeitems = []
    items = []

    for i, char in enumerate(input):
        if char in PUNCTUATION_EXTENDED:
            if char == '.' and i > 0 and i < len(input) - 1 and input[i - 1].isdigit() and input[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(PUNCTUATION_EXTENDED)]
    return "\n".join(opt)

def split(input:str) -> list[str]:
    input = input.replace("……", "。").replace("——", "，")
    if input[-1] not in SPLITS:
        input += "。"
    i_split_head = i_split_tail = 0
    len_text = len(input)
    inputs = []
    while True:
        if i_split_head >= len_text:
            break
        if input[i_split_head] in SPLITS:
            i_split_head += 1
            inputs.append(input[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return inputs

def save_audio(sampling_rate:int, audio_data:np.ndarray, save_path:str="./", save_name:str="output") -> str:
    output_wav_path = os.path.join(save_path, f"{save_name}.wav")
    sf.write(output_wav_path, audio_data, sampling_rate)
    print(f"Audio saved to {output_wav_path}")
    return f"{save_path}{save_name}.wav", save_name

if __name__ == "__main__":
    audio_gen = GPT_SoVITS(
        gpt_path="./michael_en.ckpt",
        sovits_path="./michael_en.pth",
        hubert_path=os.environ.get("hubert_base_path", "hubert-base-ls960"),
        bert_path=os.environ.get("bert_path", "xlm-roberta-large")
        # hubert_path=os.environ.get("cnhubert_base_path", "chinese-hubert-base"),
        # bert_path=os.environ.get("bert_path", "chinese-roberta-wwm-ext-large")

    )
    result = audio_gen.get_tts_wav(
        ref_wav_path="./michael_ref.wav",
        ref_wav_text="This is not pro Rust at all. On paper, Rust seemed like the programming language designed by the gods. Not only is it the fastest programming language out there",
        ref_wav_lang="en",
        cut="no_cut",
        input_prompt="I noticed that you used sob emote in your message. Just wanted to say, don’t give up anything in your life. I don’t know what you’re going through but I’m always here to help.",
        input_prompt_lang="en",
    )
    # print(result)
    result_list = list(result)
    if result_list:
        sampling_rate, audio_data = result_list[-1]
        save_audio(sampling_rate=sampling_rate, audio_data=audio_data)


