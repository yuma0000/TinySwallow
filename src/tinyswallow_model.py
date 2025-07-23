import .nnabla_func
from nnabla.function import PythonFunction

class attention():
    def __init__():
        #Q, K, V, O

    def forward():

class rms_norm(PythonFunction):
    def __init__():

    def forward():

class mlp():
    def __init__():
        #gate, up, down, act

    def forward():

class tokenizer(PythonFunction):
    def __init__():

    def forward():

class embed_tokens(PythonFunction):
    def __init__():

    def forward():

class decoder_layer(PythonFunction):
    def __init__():
        #self_attn
        #mlp
        #input_layernorm
        #poat_attention_layernorm

    def forward():

class rotary_emb(PythonFunction):
    def __init__():

    def forward():

class im_head(PythonFunction):
    def __init__():

    def forward():

class tinyswallow_model(PythonFunction):

    def __init__():
        #tokenizer
        self.tokenizer = tokenizer()

        #embed_tokens
        self.embed_tokens = embed_tokens()

        #layers 28
        self.decoder_layer = decoder_layer()

        #RMSnorm
        self.rms_norm = rms_norm()

        #rotary_emb
        self.rotary_emb = rotary_emb()

        #im_head
        self.im_head = im_head()

    def generate():
