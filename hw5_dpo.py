# -*- coding: utf-8 -*-
"""
SmolVLM-256M QLoRA 微调 + DPO 偏好对齐 —— 实验作业五
在指南参考框架基础上做如下适配（修改处均以 [MOD] 注释标注）：
  [MOD-1] transformers 5.x 兼容：AutoModelForVision2Seq 改名时回退 AutoModelForImageTextToText；
          processor.decode 改用 processor.tokenizer.decode。
  [MOD-2] CPU 无 bitsandbytes 时自动回退 fp32 基座 + LoRA（指南 Q2 允许，报告须注明）。
  [MOD-3] LoRA target_modules 用「语言模型 q/k/v/o 全名列表」，确保只挂 LM、不挂视觉塔。
  [MOD-4] eval 第三阶段改为「干净基座 + dpo_adapter」（dpo 适配器已含 SFT 权重，
          原参考代码嵌套包一层 PeftModel 会把 SFT LoRA 叠加两次，已在调试中发现并修正）。
  [MOD-5] 内置任务三自拟 5 组偏好对 SELF_PAIRS（默认不并入训练，供报告引用；
          GPU 正式重跑时可用 --with_self_pairs 并入）。
常用命令：
  python hw5_dpo.py --mode make_data   # 生成 15 张合成图像（首次运行自动执行）
  python hw5_dpo.py --mode infer       # 基线推理 15 题，记录基线正确率
  python hw5_dpo.py --mode sft         # QLoRA SFT（产出 ./sft_adapter）
  python hw5_dpo.py --mode dpo         # 在 SFT 之上做 DPO（产出 ./dpo_adapter）
  python hw5_dpo.py --mode eval        # 三阶段（基线/SFT/SFT+DPO）× 15 题对比
  python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1  # DPO loss 手算复核
"""
import argparse, math, os, random
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMG_DIR = "hw5_imgs"
SFT_DIR, DPO_DIR = "sft_adapter", "dpo_adapter"

# ============================================================
# 1. 合成图像库「几何图形」：15 张图，三类问题的答案均已埋入
#    每张图 = 白底 256x256，画若干「颜色-形状」对（位置固定可复现）
# ============================================================
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
# img_id -> [(颜色, 形状, 个数), ...]（形状：圆=ellipse，方=rectangle）
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)],
    "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)],
    "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)],
    "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)],
    "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)],
    "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)],   # 判断题素材：只有一个黄方
    "img12": [("红", "方", 1), ("蓝", "方", 1)],  # 没有圆形 →「有没有红圆」答「没有」
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)],
    "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],
}

def make_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    rnd = random.Random(42)
    for img_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(im)
        for color, shape, n in spec:
            for _ in range(n):
                x, y = rnd.randint(10, 200), rnd.randint(10, 200)
                s = rnd.randint(30, 50)
                box = [x, y, x + s, y + s]
                if shape == "圆":
                    d.ellipse(box, outline=COLORS[color], width=4)
                else:
                    d.rectangle(box, outline=COLORS[color], width=4)
        im.save(f"{IMG_DIR}/{img_id}.png")
    print(f"[data] 已生成 {len(IMG_SPEC)} 张图像到 {IMG_DIR}/")

def load_image(img_id):
    p = f"{IMG_DIR}/{img_id}.png"
    if not os.path.exists(p):
        make_images()
    return Image.open(p).convert("RGB")

# ============================================================
# 2. 问题集：3 类 × 5 题（gold = 标准答案；宽松判分见 judge()）
#    count=数数（局部事实）  attr=属性识别  exist=存在性/组合判断
# ============================================================
QUESTIONS = [
    {"id": "P01", "type": "count", "img": "img01", "q": "图中有几个红色圆形？", "gold": "2"},
    {"id": "P02", "type": "count", "img": "img02", "q": "图中有几个绿色圆形？", "gold": "3"},
    {"id": "P03", "type": "count", "img": "img03", "q": "图中有几个黄色方形？", "gold": "2"},
    {"id": "P04", "type": "count", "img": "img06", "q": "图中有几个黄色圆形？", "gold": "4"},
    {"id": "P05", "type": "count", "img": "img10", "q": "图中有几个蓝色圆形？", "gold": "3"},
    {"id": "P06", "type": "attr", "img": "img01", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P07", "type": "attr", "img": "img05", "q": "图中红色的图形是什么形状？", "gold": "圆形"},
    {"id": "P08", "type": "attr", "img": "img09", "q": "图中绿色的图形是什么形状？", "gold": "方形"},
    {"id": "P09", "type": "attr", "img": "img13", "q": "图中黄色的图形是什么形状？", "gold": "圆形"},
    {"id": "P10", "type": "attr", "img": "img14", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P11", "type": "exist", "img": "img11", "q": "图里有没有黄色方形？", "gold": "有"},
    {"id": "P12", "type": "exist", "img": "img12", "q": "图里有没有红色圆形？", "gold": "没有"},
    {"id": "P13", "type": "exist", "img": "img07", "q": "图里有没有绿色圆形？", "gold": "有"},
    {"id": "P14", "type": "exist", "img": "img08", "q": "图里有没有绿色方形？", "gold": "没有"},
    {"id": "P15", "type": "exist", "img": "img15", "q": "图里有没有蓝色图形？", "gold": "没有"},
]

# SFT 数据：图像 + 问题 → 简短标准答案（教模型「答题格式 + 事实」）
SFT_DATA = [{"img": q["img"], "prompt": q["q"],
             "answer": f"{q['gold']}。" if q["type"] != "count" else f"{q['gold']}个。"}
            for q in QUESTIONS]

# 偏好对（12 组）：chosen = 正确回答；rejected = 典型幻觉/错误
# rejected 的三类错误：数错（off-by-one / 混淆）、形状幻觉、对不存在的物体硬答「有」
PRE_PAIRS = [
    {"img": "img01", "prompt": "图中有几个红色圆形？",
     "chosen": "2 个。", "rejected": "3 个。图中有三个红色圆形整齐排列。"},
    {"img": "img02", "prompt": "图中有几个绿色圆形？",
     "chosen": "3 个。", "rejected": "2 个。"},
    {"img": "img03", "prompt": "图中有几个黄色方形？",
     "chosen": "2 个。", "rejected": "没有看到黄色图形。"},
    {"img": "img06", "prompt": "图中有几个黄色圆形？",
     "chosen": "4 个。", "rejected": "3 个。"},
    {"img": "img10", "prompt": "图中有几个蓝色圆形？",
     "chosen": "3 个。", "rejected": "图中有三个蓝色方形。"},
    {"img": "img01", "prompt": "图中蓝色的图形是什么形状？",
     "chosen": "方形。", "rejected": "圆形。"},
    {"img": "img05", "prompt": "图中红色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "方形，而且有两个红色方形。"},
    {"img": "img09", "prompt": "图中绿色的图形是什么形状？",
     "chosen": "方形。", "rejected": "图中没有绿色图形。"},
    {"img": "img11", "prompt": "图里有没有黄色方形？",
     "chosen": "有。", "rejected": "没有。"},
    {"img": "img12", "prompt": "图里有没有红色圆形？",
     "chosen": "没有。", "rejected": "有，图左上角有一个红色圆形。"},
    {"img": "img08", "prompt": "图里有没有绿色方形？",
     "chosen": "没有。", "rejected": "有，一共两个绿色方形。"},
    {"img": "img15", "prompt": "图里有没有蓝色图形？",
     "chosen": "没有。图中只有黄色和绿色的圆形。", "rejected": "有，两个蓝色圆形。"},
]

# ============================================================
# [MOD-5] 任务三：自拟 5 组偏好对（每组的失败模式在报告中逐一标注）
#   R1 数错(off-by-one)+冗长废话   R2 形状幻觉   R3 硬答「有」（幻觉定位）
#   R4 漏报（对存在的物体答没有）   R5 冗长废话+数错
# ============================================================
SELF_PAIRS = [
    {"img": "img04", "prompt": "图中有几个红色方形？",
     "chosen": "3 个。", "rejected": "根据我对图像的仔细观察和分析，考虑到整体布局，红色方形的数量应该是 2 个。"},
    {"img": "img06", "prompt": "图中黄色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "方形。图中可以看到多个黄色方形。"},
    {"img": "img13", "prompt": "图里有没有绿色方形？",
     "chosen": "没有。图中只有绿色圆形和黄色圆形。", "rejected": "有，左下角有一个绿色方形。"},
    {"img": "img07", "prompt": "图里有没有蓝色方形？",
     "chosen": "有。", "rejected": "没有。图中不存在蓝色图形。"},
    {"img": "img02", "prompt": "图中有几个绿色圆形？",
     "chosen": "3 个。", "rejected": "这个问题很难回答，不过从画面来看也许是 5 个吧。"},
]

# ============================================================
# 3. 模型加载：QLoRA（NF4 量化 + LoRA 低秩适配器）
#    256M 小模型按理可全参微调，但本实验统一走 QLoRA 流程以体会方法
# ============================================================
def load_model(mode="infer"):
    """mode: infer=原精度推理 / train=4bit 量化+LoRA（无 GPU 时自动回退 fp32+LoRA）"""
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as AutoVLM  # [MOD-1]
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoVLM
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    use_4bit = False
    if mode == "train":
        cuda_ok = torch.cuda.is_available()
        bnb_ok = False
        if cuda_ok:
            try:
                import bitsandbytes  # noqa
                bnb_ok = True
            except Exception:
                bnb_ok = False
        use_4bit = cuda_ok and bnb_ok
        if not use_4bit:
            print("[load] [MOD-2] 检测到无 NVIDIA GPU 或无 bitsandbytes，自动回退 fp32 基座 + LoRA（只训适配器思想不变）")

    if use_4bit:
        from transformers import BitsAndBytesConfig
        model = AutoVLM.from_pretrained(
            MODEL_ID,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)  # 4-bit 路径：开启梯度检查点
        model.gradient_checkpointing_enable()
    else:
        model = AutoVLM.from_pretrained(MODEL_ID, dtype=torch.float32)

    if mode == "train":
        from peft import LoraConfig, get_peft_model
        # [MOD-3] 只挂语言模型的 q/k/v/o，避免把视觉塔 (vision tower) 也挂上
        # transformers 5.x 中 SmolVLM 的语言模型命名为 model.text_model.*
        lm_targets = [n for n, _ in model.named_modules()
                      if ("text_model" in n or "language_model" in n)
                      and n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")]
        assert lm_targets, "未找到语言模型的注意力投影层"
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=lm_targets, task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()  # ← 报告 2.1 要抄的可训练参数占比
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

def build_prompt(processor, q):
    msgs = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)

@torch.no_grad()
def answer(processor, model, img, q, max_new=24):
    inputs = processor(text=[build_prompt(processor, q)], images=[img],
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],  # [MOD-1]
                                      skip_special_tokens=True)
    return text.strip()

def judge(q, pred):
    """宽松判分：标准答案出现在预测里即算对（「没有」须整词出现）"""
    p, g = pred.replace(" ", ""), q["gold"]
    if g == "没有":
        return "没有" in p and "有" in p
    return g in p

# ============================================================
# 4. SFT：只对答案 token 计损失，全流程手写便于理解
# ============================================================
def sft_step(model, processor, opt, epochs=2, steps=60, log_every=10):
    dev = next(model.parameters()).device
    model.train()
    for ep in range(epochs):
        rnd = random.Random(ep)
        order = list(range(len(SFT_DATA))); rnd.shuffle(order)
        for i, idx in enumerate(order):
            if ep * len(order) + i >= steps:
                print(f"[sft] 已达 {steps} 步，提前结束")
                # [MOD-7] 修正参考代码 bug：提前退出时也必须保存适配器，
                #         否则 --steps 小于总样本数时 sft_adapter 不会落盘，DPO 无法继续。
                model.save_pretrained(SFT_DIR)
                print(f"[sft] 适配器已保存到 ./{SFT_DIR}")
                return
            s = SFT_DATA[idx]
            full = processor(text=[build_prompt(processor, s["prompt"]) + s["answer"]],
                             images=[load_image(s["img"])], return_tensors="pt").to(dev)
            with torch.no_grad():
                prompt_len = processor(text=[build_prompt(processor, s["prompt"])],
                                       images=[load_image(s["img"])],
                                       return_tensors="pt")["input_ids"].shape[1]
            labels = full["input_ids"].clone()
            labels[:, :prompt_len] = -100  # 只学答案
            full["labels"] = labels
            loss = model(**full).loss
            loss.backward(); opt.step(); opt.zero_grad()
            if (i + 1) % log_every == 0 or (ep * len(order) + i + 1) == steps:
                print(f"[sft] ep{ep+1} step{i+1}/{len(order)} loss={loss.item():.4f}")
    model.save_pretrained(SFT_DIR)
    print(f"[sft] 适配器已保存到 ./{SFT_DIR}")

# ============================================================
# 5. DPO：loss = -log σ( β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] )
#    参考模型 = 关闭 LoRA 适配器后的基座（disable_adapter 上下文）
# ============================================================
def seq_logprob(model, processor, img, prompt, resp):
    """返回「答案 token 的平均对数概率 × 长度」＝序列总对数概率"""
    dev = next(model.parameters()).device
    full = processor(text=[build_prompt(processor, prompt) + resp],
                     images=[img], return_tensors="pt").to(dev)
    plen = processor(text=[build_prompt(processor, prompt)], images=[img],
                     return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone(); labels[:, :plen] = -100
    logits = model(**full).logits[:, :-1, :]
    tgt = labels[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    mask = tgt.ne(-100)
    return lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask].sum()

def dpo_step(model, processor, opt, beta=0.1, epochs=1, steps=40, log_every=1, pairs=None):
    dev = next(model.parameters()).device
    model.train()
    data = pairs if pairs is not None else PRE_PAIRS
    for ep in range(epochs):
        rnd = random.Random(ep)
        order = list(range(len(data))); rnd.shuffle(order)
        for i, idx in enumerate(order):
            if ep * len(order) + i >= steps:
                print(f"[dpo] 已达 {steps} 步，提前结束")
                # [MOD-7] 同 sft_step：提前退出时也必须保存适配器
                model.save_pretrained(DPO_DIR)
                print(f"[dpo] 适配器已保存到 ./{DPO_DIR}")
                return
            p = data[idx]
            img = load_image(p["img"])
            pol_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
            pol_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            with model.disable_adapter():  # 冻结的参考模型 π_ref
                ref_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
                ref_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
            loss = -F.logsigmoid(margin)
            loss.backward(); opt.step(); opt.zero_grad()
            print(f"[dpo] ep{ep+1} step{i+1}/{len(order)} loss={loss.item():.4f} "
                  f"margin={margin.item():.3f}")
    model.save_pretrained(DPO_DIR)
    print(f"[dpo] 适配器已保存到 ./{DPO_DIR}")

def dpo_check(arg):
    """手算复核：依次给 pol_w, ref_w, pol_l, ref_l, beta"""
    pw, rw, pl, rl, beta = [float(x) for x in arg.split(",")]
    margin = beta * ((pw - rw) - (pl - rl))
    # [MOD-6] 修正参考代码符号错误：loss = -log σ(m) = log(1+e^{-|m|}) + max(0,-m)，
    #         原参考代码两个分支都漏了外层符号，会输出负的 loss。
    loss = math.log(1 + math.exp(-margin)) if margin >= 0 else -margin + math.log(1 + math.exp(margin))
    print(f"margin = β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] = {margin:.4f}")
    print(f"DPO loss = -log σ(margin) = {loss:.4f}"
          f"（margin 越大 loss 越小；margin<0 说明模型更偏好 rejected，需要继续训练）")

# ============================================================
# 6. 三阶段对比评测：基线 / SFT / SFT+DPO × 15 题
# ============================================================
def evaluate(processor, model, tag):
    print(f"\n===== {tag} × 15 题 =====")
    ok_by_type = {}
    for tp in ("count", "attr", "exist"):
        qs = [q for q in QUESTIONS if q["type"] == tp]
        ok = sum(judge(q, answer(processor, model, load_image(q["img"]), q["q"]))
                 for q in qs)
        ok_by_type[tp] = (ok, len(qs))
        print(f"{tag:<10}{tp:<8}正确 {ok}/{len(qs)}")
    total = sum(o for o, _ in ok_by_type.values())
    print(f"{tag:<10}总计   正确 {total}/15\n")
    return total

# ============================================================
# 7. 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="实验作业五：SmolVLM-256M QLoRA + DPO")
    ap.add_argument("--mode", choices=["make_data", "infer", "sft", "dpo", "eval"],
                    default="infer")
    ap.add_argument("--steps", type=int, default=None, help="覆盖默认训练步数（CPU 演示用 5~10）")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO 温度系数 β")
    ap.add_argument("--dpo_check", default=None, help="如 1.2,0.8,0.9,0.6,0.1")
    ap.add_argument("--with_self_pairs", action="store_true",
                    help="[MOD-5] DPO 时并入自拟 SELF_PAIRS（正式 GPU 重跑用）")
    args = ap.parse_args()

    if args.dpo_check:
        dpo_check(args.dpo_check); return
    if args.mode == "make_data":
        make_images(); return
    if args.mode == "infer":
        processor, model = load_model("infer")
        for q in QUESTIONS:
            pred = answer(processor, model, load_image(q["img"]), q["q"])
            mark = "√" if judge(q, pred) else "×"
            print(f"[{q['id']}·{q['type']}] {mark} Q: {q['q']} A: {pred}（gold: {q['gold']}）")
        return
    if args.mode == "sft":
        processor, model = load_model("train")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        sft_step(model, processor, opt, steps=args.steps or 60)
        return
    if args.mode == "dpo":
        if not os.path.exists(SFT_DIR):
            raise SystemExit("先运行 --mode sft 生成 SFT 适配器")
        processor, model = load_model("train")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model.get_base_model().float().to("cpu"), SFT_DIR, is_trainable=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        pairs = PRE_PAIRS + SELF_PAIRS if args.with_self_pairs else None
        dpo_step(model, processor, opt, beta=args.beta, steps=args.steps or 40, pairs=pairs)
        return
    if args.mode == "eval":
        processor, model = load_model("infer")
        evaluate(processor, model, "基线")
        if os.path.exists(SFT_DIR):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, SFT_DIR).to(model.device)
            evaluate(processor, model, "SFT")
        if os.path.exists(DPO_DIR):
            # [MOD-4] dpo_adapter 已含 SFT 权重：干净基座 + dpo_adapter 即「SFT+DPO」
            from peft import PeftModel
            processor, model = load_model("infer")
            model = PeftModel.from_pretrained(model, DPO_DIR).to(model.device)
            evaluate(processor, model, "SFT+DPO")
        print("提示：把三阶段结果抄进报告表 3；另抽 2 题做人工忠实度评分（答案是否仅陈述图中事实）。")
        return

if __name__ == "__main__":
    main()
