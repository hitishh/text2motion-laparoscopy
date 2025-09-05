#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate JSONL training data for grid-cell & action extraction.
- Grid: A1..I9 (9x9)
- Actions: center | zoom_in | zoom_out
- Schema matches your examples:
  {"messages":[...]}
"""

import json, random
from dataclasses import dataclass

# ======================== CONFIG =========================
SEED = 42
N_SAMPLES = 5000
NOISE_RATE_RANGE = (0.10, 0.15)     # 10–15% noise per example
OFFGRID_POLICY = "clip"             # "clip" or "null"

# Split proportions
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Balance actions approximately
ACTION_WEIGHTS = {
    "center": 0.5,
    "zoom_in": 0.25,
    "zoom_out": 0.25,
}
# =========================================================

random.seed(SEED)

COLS = list("ABCDEFGHI")
ROWS = list(range(1, 10))
COL2IDX = {c:i for i,c in enumerate(COLS)}
IDX2COL = {i:c for i,c in enumerate(COLS)}

NUMBER_WORDS = {
    0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",
    6:"six",7:"seven",8:"eight",9:"nine",10:"ten"
}

DIRS = {
    "left":  (-1,0),"right":(1,0),"above":(0,1),"below":(0,-1),
    "west":  (-1,0),"east": (1,0),"north":(0,1),"south":(0,-1),
    "up":(0,1),"down":(0,-1),
}

CENTER_SYNS = [
    "Focus on {cell}.","Centre on {cell}.","Center on {cell}.",
    "Put {cell} dead-centre.","Place {cell} at the crosshair.",
    "Frame {cell} in the middle.","Keep {cell} centred.",
    "Hold here on {cell}.","Aim the scope at {cell}.","Lock the view on {cell}."
]

REL_TEMPLATES = [
    "move {n} {unit} {dir} of {cell}",
    "please put the view {n} {unit} {dir} from {cell}",
    "keep the frame {n} {unit} {dir} of {cell}",
    "aim the scope {n} {unit} {dir} {cell}",
    "focus {n} {unit} {dir} {cell}",
]

UNITS = ["cells","cell","squares","square"]
OF_FROM = ["of","from"]

ZOOM_IN_SYNS = [
    "Zoom in slightly on {cell}.","Give me a close-up of {cell}.",
    "Move 5 mm closer at {cell}.","Tighten the view on {cell}.","Zoom in a touch on {cell}."
]
ZOOM_OUT_SYNS = [
    "Zoom out slightly at {cell}.","Pull back 10 mm at {cell}.",
    "Widen the view at {cell}.","Zoom out a touch at {cell}."
]

CONFUSION = {
    "left":["let","lefft","lft"],"right":["rite","rigth","rght"],
    "above":["abov","abvoe"],"below":["bellow","belwo"],
    "east":["eat","eest"],"west":["wset","wes"],
    "north":["norht","nroth"],"south":["soth","sout"],
    "centre":["center","centrr"],"center":["centre","cenetr"],
    "zoom":["zom","zoon"],"cells":["cels","ceels"],
    "squares":["sqaure","sqaures"],"move":["mvoe","moev"],
    "focus":["foucs","focs"],"aim":["am","aime"],
}

def maybe_case(s:str)->str:
    mode=random.choice(["as_is","lower","upper","title"])
    if mode=="lower": return s.lower()
    if mode=="upper": return s.upper()
    if mode=="title": return s.title()
    return s

def insert_typo_token(token:str)->str:
    opts=CONFUSION.get(token.lower())
    if opts: return random.choice(opts)
    if len(token)>3:
        i=random.randrange(1,len(token)-1)
        return token[:i]+token[i+1:]
    return token

def add_text_noise(text:str)->str:
    tokens=text.split()
    if not tokens: return text
    idx=random.randrange(len(tokens))
    tokens[idx]=insert_typo_token(tokens[idx])
    return " ".join(tokens)

@dataclass
class Cell:
    col:str
    row:int
    def __str__(self): return f"{self.col}{self.row}"

def in_bounds(c:Cell)->bool: return c.col in COLS and 1<=c.row<=9
def clip_cell(c:Cell)->Cell:
    col_idx=min(max(COL2IDX.get(c.col,0),0),8)
    row=min(max(c.row,1),9)
    return Cell(IDX2COL[col_idx],row)

def apply_move(base:Cell,dx:int,dy:int)->Cell:
    col_idx=COL2IDX[base.col]+dx
    row=base.row+dy
    return Cell(IDX2COL.get(col_idx,COLS[0]),row)

def random_cell()->Cell: return Cell(random.choice(COLS),random.choice(ROWS))

def number_phrase(n:int)->str:
    return random.choice([str(n),NUMBER_WORDS.get(n,str(n))])

def sample_direction(): return random.choice(list(DIRS.keys()))

def make_relative_utterance(base:Cell,n:int,direction:str)->tuple[str,Cell]:
    t=random.choice(REL_TEMPLATES)
    unit=random.choice(UNITS)
    if "{dir} of" in t or "{dir} from" in t:
        t=t.replace("{dir}",f"{{dir}} {random.choice(OF_FROM)}")
    txt=t.format(n=number_phrase(n),unit=unit,dir=direction,cell=str(base))
    dx,dy=DIRS[direction]
    target=apply_move(base,dx*n,dy*n)
    return txt,target

def make_absolute_center_utterance(target:Cell)->str:
    return random.choice(CENTER_SYNS).format(cell=str(target))

def make_zoom_utterance(target:Cell,action:str)->str:
    if action=="zoom_in": return random.choice(ZOOM_IN_SYNS).format(cell=str(target))
    return random.choice(ZOOM_OUT_SYNS).format(cell=str(target))

def pick_action()->str:
    r=random.random()
    if r<ACTION_WEIGHTS["zoom_in"]: return "zoom_in"
    if r<ACTION_WEIGHTS["zoom_in"]+ACTION_WEIGHTS["zoom_out"]: return "zoom_out"
    return "center"

def offgrid_handle(target:Cell):
    if in_bounds(target): return target
    if OFFGRID_POLICY=="clip": return clip_cell(target)
    return None

def maybe_apply_noise(text:str,noise_rate:float)->str:
    if random.random()<noise_rate: return add_text_noise(maybe_case(text))
    return text

def make_example()->dict:
    action=pick_action()
    base=random_cell()
    if action=="center" and random.random()<0.5:
        n=random.randint(1,4)
        direction=sample_direction()
        user_text,target=make_relative_utterance(base,n,direction)
    else:
        target=base
        user_text=make_absolute_center_utterance(target) if action=="center" else make_zoom_utterance(target,action)
    handled=offgrid_handle(target)
    cell_value=str(handled) if handled else None
    noise_rate=random.uniform(*NOISE_RATE_RANGE)
    noisy_user=maybe_apply_noise(user_text,noise_rate)
    system_prompt=("Extract the target grid cell (e.g., E2) and action in "
                   "{'center','zoom_in','zoom_out'}. Return minified JSON only. "
                   "If cell is unclear, set cell:null.")
    assistant_obj={"cell":cell_value,"action":action}
    assistant_text=json.dumps(assistant_obj,separators=(",",":"))
    return {"messages":[
        {"role":"system","content":system_prompt},
        {"role":"user","content":noisy_user},
        {"role":"assistant","content":assistant_text}
    ]}

def main():
    examples=[make_example() for _ in range(N_SAMPLES)]
    random.shuffle(examples)

    n_train=int(SPLIT_RATIOS["train"]*N_SAMPLES)
    n_val=int(SPLIT_RATIOS["val"]*N_SAMPLES)
    train, val, test = (
        examples[:n_train],
        examples[n_train:n_train+n_val],
        examples[n_train+n_val:]
    )

    for name, split in [("train",train),("val",val),("test",test)]:
        with open(f"{name}.jsonl","w",encoding="utf-8") as f:
            for ex in split: f.write(json.dumps(ex,ensure_ascii=False)+"\n")
        print(f"Wrote {len(split)} → {name}.jsonl")

if __name__=="__main__":
    main()
