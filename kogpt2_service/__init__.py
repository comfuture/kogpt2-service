"""empty module"""
__version__ = "0.0.1"

import os
import torch
from ratsnlp.nlpbook.generation import GenerationDeployArguments
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


downstream_dir = os.path.realpath(os.path.join(os.curdir, "nl_checkpoint"))
downstream_model_checkpoint_fpath = os.path.join(downstream_dir, 'checkpoint')

try:
    os.mkdir(downstream_dir)
except FileExistsError:
    pass

# args = GenerationDeployArguments(
#     pretrained_model_name="skt/kogpt2-base-v2",
#     # downstream_model_dir=downstream_dir,
#     downstream_model_checkpoint_fpath=downstream_model_checkpoint_fpath,
#     downstream_model_dir=None,
# )


pretrained_model_config = GPT2Config.from_pretrained(
    "skt/kogpt2-base-v2",
    # args.pretrained_model_name,
)

model = GPT2LMHeadModel(pretrained_model_config)

# fine_tuned_model_ckpt = torch.load(
#     args.downstream_model_checkpoint_fpath,
#     map_location=torch.device("cpu"),
# )

# model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

model.eval()


tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    # args.pretrained_model_name,
    eos_token="</s>",
)

def inference_fn(
        prompt,
        min_length=10,
        max_length=20,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        temperature=1.0,
):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=True,
                top_p=float(top_p),
                top_k=int(top_k),
                min_length=int(min_length),
                max_length=int(max_length),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                temperature=float(temperature),
           )
        generated_sentence = tokenizer.decode([el.item() for el in generated_ids[0]])
    except:
        generated_sentence = """처리 중 오류가 발생했습니다. <br>
            변수의 입력 범위를 확인하세요. <br><br> 
            min_length: 1 이상의 정수 <br>
            max_length: 1 이상의 정수 <br>
            top-p: 0 이상 1 이하의 실수 <br>
            top-k: 1 이상의 정수 <br>
            repetition_penalty: 1 이상의 실수 <br>
            no_repeat_ngram_size: 1 이상의 정수 <br>
            temperature: 0 이상의 실수
            """
    return {
        'result': generated_sentence,
    }

if __name__ == '__main__':
    from ratsnlp.nlpbook.generation import get_web_service_app
    app = get_web_service_app(inference_fn)
    app.run()
