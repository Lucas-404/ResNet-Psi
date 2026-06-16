"""
destilar.py - Gera dataset SFT contabil destilando de um modelo professor via Ollama.

Bate na API local do Ollama (http://localhost:11434), que tambem serve os
modelos :cloud (gemma4:31b-cloud, glm-5.2:cloud, minimax-m3:cloud, ...).

Para cada tema, pede ao professor N pares pergunta/resposta realistas (PT-BR,
como um contador responderia), opcionalmente com raciocinio, e grava no formato
SFT do Arpa: {"text": "<|user|>\\n...\\n<|model|>\\n...\\n<|end|>"}.

IMPORTANTE (privacidade): use SO com temas GENERICOS (sem dado de cliente).
Modelo :cloud roda na nuvem -> nao jogue dado real do WhatsApp aqui.

Uso:
    # audicao: ve se o professor sabe o BR especifico antes de gerar em massa
    python tools/destilar.py --audicao --model gemma4:31b-cloud

    # geracao
    python tools/destilar.py --model gemma4:31b-cloud --n 8 --reasoning \\
        --out sft/contabilidade_distill.jsonl
"""

import argparse
import json
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

TEMAS = [
    "Simples Nacional (enquadramento, anexos, limites)",
    "diferenca entre lucro real, presumido e arbitrado",
    "ICMS (fato gerador, substituicao tributaria, credito)",
    "IRPJ e CSLL (apuracao, periodicidade)",
    "PIS e COFINS (cumulativo vs nao-cumulativo)",
    "MEI (limites, obrigacoes, DAS)",
    "abertura e baixa de empresa (passos, documentos)",
    "nota fiscal (tipos, NF-e, quando emitir)",
    "folha de pagamento (encargos, FGTS, INSS, ferias, 13o)",
    "balanco patrimonial (ativo, passivo, PL)",
    "DRE (demonstracao do resultado)",
    "regime de competencia vs caixa",
    "escrituracao contabil e SPED (ECD, ECF)",
    "obrigacoes acessorias e prazos (DCTF, EFD, DASN)",
    "depreciacao e amortizacao",
    "plano de contas e lancamentos (debito/credito)",
    "parcelamento de debitos e certidoes (CND)",
    "pro-labore vs distribuicao de lucros",
    "tributacao de servicos (ISS)",
    "classificacao de despesas e custos",
]

SYS = (
    "Voce e um contador brasileiro experiente e didatico. Gera pares de "
    "pergunta e resposta REALISTAS, como um cliente perguntaria e voce "
    "responderia, em portugues do Brasil. Respostas corretas, claras e "
    "objetivas. Varie o tom (formal, informal), o tamanho e o formato. "
    "NAO invente aliquotas, faixas ou prazos especificos se nao tiver certeza "
    "absoluta — nesse caso explique o conceito e diga que o valor exato deve "
    "ser conferido na norma vigente."
)

AUDICAO_QS = [
    "Quais sao os anexos do Simples Nacional e o que diferencia eles?",
    "Quando uma empresa e obrigada a sair do lucro presumido para o lucro real?",
    "Qual o prazo de entrega da DCTFWeb?",
    "Como funciona a substituicao tributaria do ICMS?",
]


def ollama_chat(model, messages, think=False, timeout=300):
    payload = {"model": model, "messages": messages, "stream": False}
    if think:
        payload["think"] = True
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    msg = data.get("message", {})
    return msg.get("content", ""), msg.get("thinking", "")


def gerar_tema(model, tema, n, reasoning):
    user = (
        f"Tema: {tema}\n\n"
        f"Gere {n} pares pergunta/resposta sobre esse tema. "
        "Responda APENAS com um array JSON, no formato:\n"
        '[{"pergunta": "...", "resposta": "..."}, ...]\n'
        "Sem texto fora do JSON."
    )
    content, thinking = ollama_chat(
        model, [{"role": "system", "content": SYS},
                {"role": "user", "content": user}], think=reasoning)
    # extrai o array JSON da resposta
    s, e = content.find("["), content.rfind("]")
    if s < 0 or e < 0:
        return []
    try:
        pares = json.loads(content[s:e + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for p in pares:
        q = (p.get("pergunta") or "").strip()
        a = (p.get("resposta") or "").strip()
        if len(q) < 5 or len(a) < 5:
            continue
        model_turn = a
        if reasoning and thinking:
            # raciocinio curto antes da resposta (o aluno aprende o padrao)
            model_turn = f"{thinking.strip()[:600]}\n\n{a}"
        out.append({"text": f"<|user|>\n{q}\n<|model|>\n{model_turn}\n<|end|>"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--n", type=int, default=6, help="pares por tema")
    ap.add_argument("--reasoning", action="store_true", help="inclui raciocinio (think)")
    ap.add_argument("--out", default="sft/contabilidade_distill.jsonl")
    ap.add_argument("--audicao", action="store_true",
                    help="so testa o professor em perguntas dificeis BR")
    args = ap.parse_args()

    if args.audicao:
        print(f"=== AUDICAO do professor: {args.model} ===\n")
        for q in AUDICAO_QS:
            print(f"P: {q}")
            try:
                a, _ = ollama_chat(args.model,
                                   [{"role": "user", "content": q}])
                print(f"R: {a.strip()[:600]}\n{'-'*60}")
            except Exception as ex:
                sys.exit(f"Erro ao chamar Ollama: {ex}\n"
                         "Confira se o Ollama esta rodando e logado (ollama.com).")
        print("\nAvalie se as respostas BR estao corretas ANTES de gerar em massa.")
        return

    total = 0
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, tema in enumerate(TEMAS, 1):
            try:
                pares = gerar_tema(args.model, tema, args.n, args.reasoning)
            except Exception as ex:
                print(f"  [erro] tema '{tema}': {ex}")
                continue
            for p in pares:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            total += len(pares)
            print(f"[{i}/{len(TEMAS)}] {tema[:40]:40} -> {len(pares)} pares "
                  f"(total {total})", flush=True)
    print(f"\nPronto: {total} exemplos em {args.out} ({time.time()-t0:.0f}s)")
    print("Revise uma amostra, depois rode o SFT:")
    print(f"  python arpa/sft.py --init checkpoints-arpa150m/best.pt --data \"{args.out}\"")


if __name__ == "__main__":
    main()
