"""
destilar.py - Gera dataset SFT contabil destilando de VARIOS professores via Ollama,
com juiz automatico que verifica cada resposta.

Fluxo:
    varios professores (--models) geram Q&A variado por tema
      -> dedup de perguntas
      -> juiz (--judge) da nota 1-5 em cada resposta
          nota alta  -> --out (dataset limpo, formato SFT do Arpa)
          nota baixa -> --revisar (voce/contadora confere/corrige)

Bate na API local do Ollama (http://localhost:11434), que serve modelos :cloud.

IMPORTANTE (privacidade): use SO com temas GENERICOS. Modelo :cloud roda na
nuvem -> nunca jogue dado real de cliente (WhatsApp) aqui.

Uso:
    # audicao de um professor
    python tools/destilar.py --audicao --model gemma4:31b-cloud

    # geracao multi-professor + juiz (+ comparacao com teu modelo, opcional)
    python tools/destilar.py \\
        --models gemma4:31b-cloud,glm-5.2:cloud,minimax-m3:cloud \\
        --judge gemma4:31b-cloud --n 6 --reasoning \\
        --student checkpoints-sft/best.pt \\
        --out sft/contabilidade_distill.jsonl
"""

import argparse
import json
import re
import sys
import time
import unicodedata
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
    "responderia, em portugues do Brasil. Respostas CURTAS (2 a 4 frases), "
    "corretas e objetivas. Varie o tom (formal, informal) e o formato. NAO "
    "invente aliquotas, faixas ou prazos especificos se nao tiver certeza "
    "absoluta — explique o conceito e diga para conferir a norma vigente."
)

SYS_JUIZ = (
    "Voce e um revisor contabil brasileiro RIGOROSO. Avalie se a RESPOSTA esta "
    "correta, clara e adequada a PERGUNTA, no contexto fiscal/contabil do "
    "Brasil. Erro de aliquota, prazo, conceito ou imprecisao = nota baixa."
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


def _json_block(s, abre, fecha):
    i, j = s.find(abre), s.rfind(fecha)
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


def norm_q(q):
    q = unicodedata.normalize("NFKD", q.lower())
    q = "".join(c for c in q if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", q).strip()


def gerar_tema(model, tema, n, reasoning):
    user = (f"Tema: {tema}\n\nGere {n} pares pergunta/resposta sobre esse tema. "
            'Responda APENAS com um array JSON: '
            '[{"pergunta": "...", "resposta": "..."}]. Sem texto fora do JSON.')
    content, thinking = ollama_chat(
        model, [{"role": "system", "content": SYS},
                {"role": "user", "content": user}], think=reasoning)
    pares = _json_block(content, "[", "]") or []
    out = []
    for p in pares:
        q = (p.get("pergunta") or "").strip()
        a = (p.get("resposta") or "").strip()
        if len(q) >= 5 and len(a) >= 5:
            out.append({"q": q, "a": a, "raciocinio": thinking.strip()[:500],
                        "professor": model})
    return out


def julgar(judge_model, q, a):
    user = (f"PERGUNTA: {q}\nRESPOSTA: {a}\n\nResponda APENAS em JSON: "
            '{"nota": <1-5>, "correto": <true|false>, "motivo": "<curto>"}')
    content, _ = ollama_chat(
        judge_model, [{"role": "system", "content": SYS_JUIZ},
                      {"role": "user", "content": user}])
    v = _json_block(content, "{", "}") or {}
    return int(v.get("nota", 0) or 0), bool(v.get("correto", False)), \
        (v.get("motivo") or "").strip()


def carregar_aluno(ckpt):
    import os
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformers import AutoTokenizer
    from arpa.sample import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("tokenizer-arpa-64k-clean")
    m = load_model(ckpt, dev)
    stop = [i for i in (tok.convert_tokens_to_ids("<|end|>"),
                        tok.convert_tokens_to_ids("<|end_of_text|>"))
            if i is not None and i >= 0]

    def responder(q):
        ids = torch.tensor([tok.encode(f"<|user|>\n{q}\n<|model|>\n",
                                       add_special_tokens=False)], device=dev)
        out = m.generate(ids, max_new_tokens=90, temperature=0.7, top_p=0.9,
                         stop_ids=stop)
        return tok.decode([t for t in out[0, ids.size(1):].tolist()
                           if t not in stop]).strip()
    return responder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:31b-cloud", help="professor unico (ou use --models)")
    ap.add_argument("--models", default=None, help="varios professores, separados por virgula")
    ap.add_argument("--judge", default=None, help="modelo juiz; sem ele nao verifica")
    ap.add_argument("--n", type=int, default=6, help="pares por tema por professor")
    ap.add_argument("--reasoning", action="store_true", help="inclui raciocinio (think)")
    ap.add_argument("--student", default=None, help="ckpt do teu modelo (so p/ comparar no revisar)")
    ap.add_argument("--out", default="sft/contabilidade_distill.jsonl")
    ap.add_argument("--revisar", default="sft/_revisar.jsonl")
    ap.add_argument("--min-nota", type=int, default=4, help="nota minima do juiz p/ aprovar")
    ap.add_argument("--audicao", action="store_true")
    args = ap.parse_args()

    if args.audicao:
        print(f"=== AUDICAO: {args.model} ===\n")
        for q in AUDICAO_QS:
            print(f"P: {q}")
            try:
                a, _ = ollama_chat(args.model, [{"role": "user", "content": q}])
                print(f"R: {a.strip()[:600]}\n{'-'*60}")
            except Exception as ex:
                sys.exit(f"Erro Ollama: {ex}\n(Ollama rodando e logado no ollama.com?)")
        print("\nAvalie o BR antes de gerar em massa.")
        return

    models = [m.strip() for m in (args.models.split(",") if args.models else [args.model]) if m.strip()]
    aluno = carregar_aluno(args.student) if args.student else None
    print(f"Professores: {models} | juiz: {args.judge or 'NENHUM'} | "
          f"aluno: {'sim' if aluno else 'nao'}")

    vistos = set()
    n_ok = n_rev = 0
    t0 = time.time()
    f_out = open(args.out, "w", encoding="utf-8")
    f_rev = open(args.revisar, "w", encoding="utf-8")
    try:
        for i, tema in enumerate(TEMAS, 1):
            cand = []
            for model in models:
                try:
                    cand += gerar_tema(model, tema, args.n, args.reasoning)
                except Exception as ex:
                    print(f"  [erro gerar] {model} / {tema[:30]}: {ex}")
            for c in cand:
                k = norm_q(c["q"])
                if not k or k in vistos:
                    continue
                vistos.add(k)
                nota, correto, motivo = (5, True, "")
                if args.judge:
                    try:
                        nota, correto, motivo = julgar(args.judge, c["q"], c["a"])
                    except Exception as ex:
                        nota, correto, motivo = 0, False, f"juiz falhou: {ex}"
                model_turn = c["a"]
                if args.reasoning and c["raciocinio"]:
                    model_turn = f"{c['raciocinio']}\n\n{c['a']}"
                linha = {"text": f"<|user|>\n{c['q']}\n<|model|>\n{model_turn}\n<|end|>"}
                if nota >= args.min_nota and correto:
                    f_out.write(json.dumps(linha, ensure_ascii=False) + "\n")
                    n_ok += 1
                else:
                    rev = {"pergunta": c["q"], "resposta": c["a"],
                           "professor": c["professor"], "nota": nota, "motivo": motivo}
                    if aluno:
                        rev["resposta_do_teu_modelo"] = aluno(c["q"])
                    f_rev.write(json.dumps(rev, ensure_ascii=False) + "\n")
                    n_rev += 1
            print(f"[{i}/{len(TEMAS)}] {tema[:36]:36} | aprovados {n_ok} | revisar {n_rev}",
                  flush=True)
    finally:
        f_out.close()
        f_rev.close()
    print(f"\nPronto em {time.time()-t0:.0f}s: {n_ok} aprovados -> {args.out} | "
          f"{n_rev} p/ revisar -> {args.revisar}")
    print("Revise o _revisar.jsonl (corrija e mova os bons p/ o dataset), depois:")
    print(f'  python arpa/sft.py --init checkpoints-arpa150m/best.pt --data "sft/*.jsonl"')


if __name__ == "__main__":
    main()
