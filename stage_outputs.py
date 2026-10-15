# -*- coding: utf-8 -*-
"""抓取指定阶段适配器的逐题输出（供报告 4.3/6.1 引用）。用法: python stage_outputs.py sft|dpo"""
import sys
from peft import PeftModel
from hw5_dpo import load_model, answer, judge, QUESTIONS, load_image, SFT_DIR, DPO_DIR

stage = sys.argv[1]
adapter = SFT_DIR if stage == "sft" else DPO_DIR
processor, model = load_model("infer")
model = PeftModel.from_pretrained(model, adapter).to(model.device)
for q in QUESTIONS:
    pred = answer(processor, model, load_image(q["img"]), q["q"])
    mark = "√" if judge(q, pred) else "×"
    print(f"[{q['id']}·{q['type']}] {mark} Q: {q['q']} A: {pred}（gold: {q['gold']}）")
