#!/usr/bin/env python3
"""
eval_pretrain.py — Avaliação diagnóstica do pré-treino Arpa-30M.

Executa 4 baterias de testes sem depender de internet:

  1. PERPLEXIDADE em frases PT diversas (notícia, ciência, cotidiano, legalese)
  2. PREFERÊNCIA LINGUÍSTICA: modelo deve preferir frase correta vs corrompida
  3. COMPLETAÇÃO LIVRE: gera continuações curtas — análise qualitativa
  4. VOCABULÁRIO: distribuição dos top-k tokens gerados (detecta degeneração)

Uso:
    python eval_pretrain.py                          # avalia models/arpa-30m-base
    python eval_pretrain.py --model caminho/ckpt
    python eval_pretrain.py --model models/arpa-30m-base --verbose
"""

import sys
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==============================================================================
# Config
# ==============================================================================

TOKENIZER_DIR = "./tokenizer-arpa-32k"
DEFAULT_MODEL  = "./models/arpa-30m-base"

MODEL_CONFIG = dict(
    vocab_size=32_000,
    hidden_size=384,
    intermediate_size=1024,
    num_hidden_layers=11,
    num_attention_heads=8,
    num_key_value_heads=4,
    max_position_embeddings=1024,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    hidden_act="silu",
    attention_dropout=0.0,
    use_cache=True,
    tie_word_embeddings=True,
)

# ==============================================================================
# Dados de avaliação (embutidos — sem internet)
# ==============================================================================

# Bateria 1: frases longas para medir perplexidade por domínio
PPL_SENTENCES = {
    "noticia": [
        "O governo federal anunciou hoje um pacote de medidas econômicas para conter a inflação.",
        "O presidente assinou o decreto que regulamenta o uso de energias renováveis no país.",
        "A taxa de desemprego caiu pelo terceiro mês consecutivo, segundo o Instituto Brasileiro de Geografia e Estatística.",
        "O Banco Central elevou a taxa de juros básica para conter o avanço dos preços ao consumidor.",
    ],
    "ciencia": [
        "A fotossíntese é o processo pelo qual as plantas convertem luz solar em energia química.",
        "O DNA carrega a informação genética necessária para o desenvolvimento e funcionamento dos organismos.",
        "Os neurônios transmitem sinais elétricos e químicos pelo sistema nervoso central e periférico.",
        "A teoria da relatividade geral descreve a gravidade como uma curvatura do espaço-tempo.",
    ],
    "cotidiano": [
        "Hoje de manhã fui ao mercado comprar pão, leite e frutas para o café da manhã.",
        "O ônibus atrasou mais de vinte minutos e eu quase perdi a reunião importante.",
        "Ela preparou um jantar especial com macarrão ao molho vermelho e salada verde.",
        "O cachorro late toda vez que alguém passa na frente do portão da casa.",
    ],
    "literatura": [
        "No meio do caminho tinha uma pedra, tinha uma pedra no meio do caminho.",
        "Brás Cubas, o defunto autor, narrava sua vida com ironia e distanciamento.",
        "Macunaíma era o herói sem nenhum caráter, nascido no fundo do mato-virgem.",
        "A velha Sinhá Vitória sonhava com uma cama de couro cru igual à de seu Tomás da bolandeira.",
    ],
}

# Bateria 2: pares (correta, corrompida) — modelo deve dar log-prob maior pra correta
PREFERENCE_PAIRS = [
    # (correta, corrompida, descrição)
    (
        "O Brasil é um país localizado na América do Sul.",
        "Brasil O é país um localizado Sul do América na.",
        "ordem de palavras",
    ),
    (
        "Ela foi ao hospital porque estava com febre alta.",
        "Ela foi ao hospital porquê estava febre com alta.",
        "ortografia (porque/porquê)",
    ),
    (
        "Os alunos estudaram muito para a prova de matemática.",
        "Os aluno estudaram muito para as prova de matemática.",
        "concordância nominal",
    ),
    (
        "A água ferve a cem graus Celsius ao nível do mar.",
        "A água ferve a cem graus Celsius ao nível do mares.",
        "concordância de número",
    ),
    (
        "Ele chegou cedo para garantir um bom lugar na fila.",
        "Ele chegou cedo pra garantir bom um lugar fila na.",
        "ordem sintática",
    ),
    (
        "O médico receitou antibiótico para tratar a infecção.",
        "O médico receitou antibiótico para tratar a felicidade.",
        "coerência semântica",
    ),
    (
        "A criança dormiu cedo porque estava muito cansada.",
        "A criança dormiu cedo porque estava muito acordada.",
        "coerência causal",
    ),
    (
        "O sol nasce no leste e se põe no oeste todos os dias.",
        "O sol nasce no norte e se põe no sul todos os dias.",
        "conhecimento factual",
    ),
]

# Bateria 3: prompts de completação livre
COMPLETION_PROMPTS = [
    "A capital do Brasil é",
    "A fotossíntese é o processo pelo qual",
    "Para fazer um bolo de chocolate, você precisa de",
    "O ser humano precisa de água porque",
    "Em 1500, Pedro Álvares Cabral",
    "A inteligência artificial é uma área da ciência que",
]

# ==============================================================================
# Carregamento
# ==============================================================================

def load_model(ckpt_path: Path, device: torch.device):
    print(f"Carregando modelo: {ckpt_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    config = LlamaConfig(
        **MODEL_CONFIG,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config)
    model = model.from_pretrained(str(ckpt_path), config=config)
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n:.1f}M params | device={device}\n")
    return model, tokenizer


# ==============================================================================
# Métricas
# ==============================================================================

@torch.no_grad()
def sentence_logprob(model, tokenizer, text: str, device) -> float:
    """Log-probabilidade total da sequência (soma dos log-probs token a token)."""
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if ids.shape[1] < 2:
        return float("-inf")
    logits = model(ids).logits  # (1, T, V)
    # cada posição t prediz t+1
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, V)
    target    = ids[:, 1:]                                  # (1, T-1)
    token_lp  = log_probs[0, torch.arange(target.shape[1]), target[0]]
    return token_lp.sum().item()


@torch.no_grad()
def sentence_ppl(model, tokenizer, text: str, device) -> float:
    """Perplexidade da frase."""
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if ids.shape[1] < 2:
        return float("inf")
    logits = model(ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target    = ids[:, 1:]
    token_lp  = log_probs[0, torch.arange(target.shape[1]), target[0]]
    avg_nll   = -token_lp.mean().item()
    return math.exp(avg_nll)


@torch.no_grad()
def generate_short(model, tokenizer, prompt: str, device,
                   max_new_tokens=40, temperature=0.7, top_p=0.9) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ==============================================================================
# Baterias
# ==============================================================================

def run_ppl_battery(model, tokenizer, device, verbose=False):
    print("=" * 60)
    print("BATERIA 1 — PERPLEXIDADE POR DOMÍNIO")
    print("=" * 60)
    print(f"  {'Domínio':<12}  {'PPL médio':>10}  {'PPL min':>8}  {'PPL max':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}")

    all_ppls = []
    domain_results = {}
    for domain, sentences in PPL_SENTENCES.items():
        ppls = [sentence_ppl(model, tokenizer, s, device) for s in sentences]
        avg = sum(ppls) / len(ppls)
        domain_results[domain] = ppls
        all_ppls.extend(ppls)
        print(f"  {domain:<12}  {avg:>10.2f}  {min(ppls):>8.2f}  {max(ppls):>8.2f}")
        if verbose:
            for s, p in zip(sentences, ppls):
                print(f"    [{p:6.2f}] {s[:70]}")

    overall = sum(all_ppls) / len(all_ppls)
    print(f"\n  {'GERAL':<12}  {overall:>10.2f}")

    # Diagnóstico
    print()
    if overall < 30:
        verdict = "EXCELENTE — modelo capturou bem o PT"
    elif overall < 50:
        verdict = "BOM — modelo aprendeu a estrutura do PT"
    elif overall < 80:
        verdict = "REGULAR — aprendeu parcialmente"
    else:
        verdict = "FRACO — pré-treino insuficiente para PT"
    print(f"  Diagnóstico: {verdict}")
    print()
    return overall, domain_results


def run_preference_battery(model, tokenizer, device, verbose=False):
    print("=" * 60)
    print("BATERIA 2 — PREFERÊNCIA LINGUÍSTICA")
    print("  (modelo deve preferir frase correta vs corrompida)")
    print("=" * 60)

    correct = 0
    total   = len(PREFERENCE_PAIRS)
    results = []

    for good, bad, desc in PREFERENCE_PAIRS:
        lp_good = sentence_logprob(model, tokenizer, good, device)
        lp_bad  = sentence_logprob(model, tokenizer, bad,  device)
        preferred_good = lp_good > lp_bad
        correct += int(preferred_good)
        results.append((desc, preferred_good, lp_good, lp_bad))

        mark = "✓" if preferred_good else "✗"
        if verbose:
            print(f"  {mark} [{desc}]")
            print(f"      BOM: {good[:60]}")
            print(f"      MAU: {bad[:60]}")
            print(f"      logprob bom={lp_good:.2f}  mau={lp_bad:.2f}")
        else:
            print(f"  {mark}  {desc:<30}  logprob bom={lp_good:7.2f}  mau={lp_bad:7.2f}")

    acc = correct / total * 100
    print(f"\n  Acurácia: {correct}/{total} = {acc:.1f}%")

    if acc >= 87.5:
        verdict = "EXCELENTE — modelo tem forte senso linguístico PT"
    elif acc >= 62.5:
        verdict = "BOM — modelo prefere PT correto na maioria dos casos"
    elif acc >= 37.5:
        verdict = "REGULAR — preferência linguística fraca"
    else:
        verdict = "FRACO — modelo não discrimina PT correto"
    print(f"  Diagnóstico: {verdict}")
    print()
    return acc, results


def run_completion_battery(model, tokenizer, device):
    print("=" * 60)
    print("BATERIA 3 — COMPLETAÇÃO LIVRE (análise qualitativa)")
    print("=" * 60)

    for prompt in COMPLETION_PROMPTS:
        completion = generate_short(model, tokenizer, prompt, device)
        print(f"  Prompt : {prompt}")
        print(f"  Saída  : {completion.strip()[:120]}")
        print()


def run_vocab_battery(model, tokenizer, device):
    print("=" * 60)
    print("BATERIA 4 — SAÚDE DO VOCABULÁRIO")
    print("  (distribuição dos tokens mais usados numa geração longa)")
    print("=" * 60)

    prompt = "O Brasil é um país com grande diversidade cultural, natural e"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0][ids.shape[-1]:].tolist()
    text    = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Conta tokens únicos vs total
    unique = len(set(new_ids))
    total  = len(new_ids)
    diversity = unique / total if total > 0 else 0

    # Top-10 tokens mais repetidos
    from collections import Counter
    counter = Counter(new_ids)
    top10   = counter.most_common(10)

    print(f"  Tokens gerados : {total}")
    print(f"  Tokens únicos  : {unique}  ({diversity*100:.1f}% diversidade)")
    print()
    print(f"  Top-10 tokens mais frequentes:")
    for tid, cnt in top10:
        tok = tokenizer.decode([tid])
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {tok!r:<20} {cnt:>3}x  {pct:5.1f}%  {bar}")

    print()
    print(f"  Texto gerado:")
    print(f"  {text.strip()[:300]}")
    print()

    if diversity > 0.5:
        verdict = "SAUDÁVEL — boa diversidade de tokens"
    elif diversity > 0.3:
        verdict = "OK — diversidade aceitável"
    else:
        verdict = "ALERTA — baixa diversidade, possível degeneração"
    print(f"  Diagnóstico: {verdict}")
    print()
    return diversity, text


# ==============================================================================
# Sumário final
# ==============================================================================

def print_summary(ppl_overall, pref_acc, vocab_diversity):
    print("=" * 60)
    print("SUMÁRIO FINAL")
    print("=" * 60)

    def grade(val, thresholds, labels):
        for t, l in zip(thresholds, labels):
            if val >= t:
                return l
        return labels[-1]

    ppl_grade  = grade(ppl_overall,  [999, 80, 50, 30],   ["FRACO", "REGULAR", "BOM", "EXCELENTE"])
    # PPL menor é melhor — inverte
    if ppl_overall < 30:   ppl_grade = "EXCELENTE"
    elif ppl_overall < 50: ppl_grade = "BOM"
    elif ppl_overall < 80: ppl_grade = "REGULAR"
    else:                  ppl_grade = "FRACO"

    if pref_acc >= 87.5:   pref_grade = "EXCELENTE"
    elif pref_acc >= 62.5: pref_grade = "BOM"
    elif pref_acc >= 37.5: pref_grade = "REGULAR"
    else:                  pref_grade = "FRACO"

    if vocab_diversity > 0.5:   vocab_grade = "SAUDÁVEL"
    elif vocab_diversity > 0.3: vocab_grade = "OK"
    else:                       vocab_grade = "ALERTA"

    print(f"  {'Métrica':<30}  {'Valor':>10}  {'Nota'}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*12}")
    print(f"  {'PPL médio (geral)':<30}  {ppl_overall:>10.2f}  {ppl_grade}")
    print(f"  {'Preferência linguística':<30}  {pref_acc:>9.1f}%  {pref_grade}")
    print(f"  {'Diversidade vocabular':<30}  {vocab_diversity:>9.1%}  {vocab_grade}")
    print()

    scores = {"EXCELENTE": 3, "SAUDÁVEL": 3, "BOM": 2, "OK": 2, "REGULAR": 1, "ALERTA": 1, "FRACO": 0}
    total = scores[ppl_grade] + scores[pref_grade] + scores[vocab_grade]

    if total >= 7:
        print("  VEREDICTO GERAL: PRÉ-TREINO SÓLIDO")
        print("  O modelo aprendeu bem a estrutura do português.")
    elif total >= 4:
        print("  VEREDICTO GERAL: PRÉ-TREINO FUNCIONAL")
        print("  O modelo aprendeu o essencial, com limitações esperadas para 30M params.")
    else:
        print("  VEREDICTO GERAL: PRÉ-TREINO INSUFICIENTE")
        print("  Recomenda-se mais dados ou revisão do processo de treino.")
    print()


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL,
                        help="Caminho do checkpoint a avaliar")
    parser.add_argument("--device",  type=str, default="auto")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostra detalhes por frase nas baterias 1 e 2")
    parser.add_argument("--skip-completions", action="store_true",
                        help="Pula bateria 3 (completação livre)")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    model, tokenizer = load_model(Path(args.model), device)

    ppl_overall, _  = run_ppl_battery(model, tokenizer, device, verbose=args.verbose)
    pref_acc, _     = run_preference_battery(model, tokenizer, device, verbose=args.verbose)

    if not args.skip_completions:
        run_completion_battery(model, tokenizer, device)

    diversity, _ = run_vocab_battery(model, tokenizer, device)

    print_summary(ppl_overall, pref_acc, diversity)


if __name__ == "__main__":
    main()
