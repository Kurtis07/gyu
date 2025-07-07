from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ────────────────────────────────────────────────
# 1) BERT 모델 & 토크나이저 로드
# ────────────────────────────────────────────────
MODEL     = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL)
model     = BertModel.from_pretrained(MODEL, output_hidden_states=True)
model.eval()

# ────────────────────────────────────────────────
# 2) BERT 어휘에서 순수 알파벳 단어만 추출 (첫 4k개)
# ────────────────────────────────────────────────
all_tokens = list(tokenizer.get_vocab().keys())
words = [t for t in all_tokens if t.isalpha() and not t.startswith('##') and len(t)>1]
MAX_VOCAB = 4000
vocab     = words[:MAX_VOCAB]
print(f"▶ 후보 단어 수 (grammar+intermediate): {len(vocab)}")

# ────────────────────────────────────────────────
# 3) delta 벡터 계산
# ────────────────────────────────────────────────
def get_all_deltas(word:str):
    toks = tokenizer(word, return_tensors='pt', truncation=True)
    with torch.no_grad():
        out = model(**toks)
    hs = out.hidden_states
    tokens = tokenizer.convert_ids_to_tokens(toks['input_ids'][0])
    idxs = [i for i,t in enumerate(tokens) if t not in ('[CLS]','[SEP]')]
    emb  = torch.stack([hs[0][0,i] for i in idxs]).mean(0)
    return [(torch.stack([hs[l][0,i] for i in idxs]).mean(0)-emb).cpu().numpy()
            for l in range(1,13)]

# ────────────────────────────────────────────────
# 4) 그룹별 레이어 인덱스 정의
# ────────────────────────────────────────────────
groups = {
    'grammar':      range(0,4),    # 1–4
    'intermediate': range(4,8),    # 5–8
    'highlevel':    range(8,12)    # 9–12
}

# ────────────────────────────────────────────────
# 5) 그룹별 대표 단어로 centroid
# ────────────────────────────────────────────────
reps = {
    'grammar':      ["run","eat","play","read","write"],
    'intermediate': ["london","google","amazon","mary","ibm"],
    'highlevel':    ["he","she","they","it","this"]
}

centroid = {}
for grp, rs in reps.items():
    rep_ds = np.stack([get_all_deltas(w) for w in rs])
    centroid[grp] = rep_ds.mean(axis=0)

# ────────────────────────────────────────────────
# 6) vocab 델타 사전화 (grammar+intermediate 용)
# ────────────────────────────────────────────────
vocab_d = {}
for i,w in enumerate(vocab):
    if i and i%2000==0:
        print(f"  • {i}/{len(vocab)} processed")
    try:
        vocab_d[w] = get_all_deltas(w)
    except:
        pass
print("▶ vocab deltas 준비 완료")

# ────────────────────────────────────────────────
# 7) 지시대명사 후보 (highlevel 용)
# ────────────────────────────────────────────────
pronouns = ["i","you","he","she","it","we","they","this","that","these","those"]
# 미리 delta 계산
pronoun_d = {p:get_all_deltas(p) for p in pronouns}

# ────────────────────────────────────────────────
# 8) 분석 함수
# ────────────────────────────────────────────────
def analyze(word:str, top_k=5):
    wd = get_all_deltas(word)

    # (A) 입력 그룹별 평균 delta
    inp_grp = {grp: np.stack([wd[i] for i in idxs]).mean(0)
               for grp,idxs in groups.items()}

    topk = {}
    for grp, iv in inp_grp.items():
        sims = []
        # grammar + intermediate: vocab_d
        if grp in ('grammar','intermediate'):
            pool = vocab_d.items()
        else:  # highlevel: pronouns
            pool = pronoun_d.items()

        for cand, deltas in pool:
            if cand==word: continue
            cv = np.stack([deltas[i] for i in groups[grp]]).mean(0)
            sim = cosine_similarity(iv.reshape(1,-1),
                                    cv.reshape(1,-1))[0][0]
            sims.append((cand,sim))
        sims.sort(key=lambda x:x[1],reverse=True)
        topk[grp] = sims[:top_k]

    # (B) 레이어별 그룹 중심 유사도
    layer_sims=[]
    for grp, idxs in groups.items():
        for i in idxs:
            sim = cosine_similarity(
                wd[i].reshape(1,-1),
                centroid[grp][i].reshape(1,-1)
            )[0][0]
            layer_sims.append((i+1,grp,round(sim,4)))

    return topk, layer_sims

# ────────────────────────────────────────────────
# 9) 실행
# ────────────────────────────────────────────────
if __name__=="__main__":
    w=input("분석할 단어를 입력하세요: ").strip().lower()
    topk, layers = analyze(w, top_k=5)

    print(f"\n▶ '{w}' 그룹별 Top-5 유사 단어:\n")
    for grp in ('grammar','intermediate','highlevel'):
        rng=groups[grp]
        print(f"[{grp.upper()} | layers {rng.start+1}-{rng.stop}]")
        for cand,sim in topk[grp]:
            print(f"  {cand:<12} 유사도: {sim:.4f}")
        print()

    print(f"▶ '{w}' 자신의 레이어별 그룹 중심 유사도\n Layer │ Group        │ Sim")
    print("──────┼──────────────┼───────")
    for layer,grp,sim in sorted(layers,key=lambda x:(x[0],x[1])):
        print(f"  {layer:>2}   │ {grp:<13}│ {sim:.4f}")
