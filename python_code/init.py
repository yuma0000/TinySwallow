
# 独自モデルとして読み込む
config = Qwen2Config.from_pretrained("/content/drive/MyDrive/reconstructed_model")
config.use_flash_attn = True
model = CustomQwen2ForCausalLM.from_pretrained(
    "/content/drive/MyDrive/reconstructed_model",
    config=config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).eval()
try:
    model = torch.compile(model)
except Exception as e:
    print("compile失敗:", e)
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/reconstructed_model")

inputs = tokenizer("今から", return_tensors="pt")
inputs = {
    k: (v.to(model.device).half() if k != "input_ids" else v.to(model.device))
    for k, v in inputs.items()
}
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        #do_sample=False,          # サンプリングせずgreedy → 高速
        use_cache=False,           # キャッシュ活用 → 重要
        #num_beams=1,              # Beam search無効 → 高速
        #early_stopping=True,      # 終了判定を早める
        #pad_token_id=tokenizer.eos_token_id  # 明示的にパディングトークン指定
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))