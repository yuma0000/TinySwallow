import nnabla as nn
import nnabla.function as F
from nnabla.function import PythonFunction

import .huggingface_model
import .tinyswallow_model
import .config


def __init__():

    #まずはhuggingface modelの取得をする
    hfm = huggingface_model("/tinyswallow")

    #tinyswallow_modelの構造
    tsm = tinyswallow_model(hfm)
    
    #configの読み込み
    cfg = config("config")

    

    print("start...\n")