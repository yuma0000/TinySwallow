from .huggingface_model import *
from .tinyswallow_model import *
from .config import *

def __init__(prompt, max_length):

    print("start...\n")

    #まずはhuggingface modelの取得をする
    hfm = Huggingface_model("/tinyswallow")

    #tinyswallow_modelの構造
    tsm = Tinyswallow_model(hfm)
    
    #configの読み込み
    tsm.cfg = config("config")

    max_length = 100
    prompt = "こんにちは、"
    
    text = tsm.generate(prompt, max_length)

    print(text)
