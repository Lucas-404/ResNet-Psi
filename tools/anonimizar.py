"""
anonimizar.py - Scrubber de PII PT-BR (camada regex, offline, sem dependencia).

Mascara dado pessoal/sensivel com pseudonimo CONSISTENTE: o mesmo CPF vira
sempre [CPF_1], o mesmo cliente vira [CLIENTE_1] -> a conversa continua
coerente pro modelo aprender o FORMATO, sem o dado real.

NAO mascara: valores monetarios, aliquotas, nomes de tributo, datas. Isso e
o conteudo contabil que o modelo precisa aprender.

Camada 1 (este arquivo): estruturado por regex, alta confianca.
Camada 2 (NER de nomes): adicionar depois se a auditoria mostrar vazamento.
Nomes conhecidos (contadora, cliente do print) podem ser passados em extra_names.

Uso como modulo:
    from tools.anonimizar import Scrubber
    s = Scrubber(extra_names=["Maria Souza"])
    limpo, report = s.scrub(texto)

Uso como CLI (auditoria num .txt):
    python tools/anonimizar.py arquivo.txt --audit
"""

import argparse
import re
import sys
import unicodedata

# ------------------------------------------------------------------ padroes PT-BR
# Ordem importa: o mais especifico/longo primeiro (CNPJ 14 antes de CPF 11 etc.).
# Cada entrada: (rotulo, regex). A substituicao usa pseudonimo numerado por rotulo.

PATTERNS = [
    # Chave de acesso de NF-e: 44 digitos (identifica o emitente)
    ("CHAVE_NFE", re.compile(r"\b\d{44}\b")),
    # CNPJ com mascara ou 14 digitos crus
    ("CNPJ", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    # CPF com mascara ou 11 digitos crus (assume PII mesmo se for telefone)
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    # E-mail
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # Telefone BR: (DD) 9XXXX-XXXX e variacoes
    ("TELEFONE", re.compile(r"(?<!\d)(?:\+?55\s?)?\(?\d{2}\)?[\s.-]?9?\d{4}[\s.-]?\d{4}(?!\d)")),
    # CEP com mascara, ou apos a palavra CEP
    ("CEP", re.compile(r"(?i)(?:cep[\s:]*)?\b\d{5}-\d{3}\b")),
    # Chave PIX aleatoria (UUID 32 hex com ou sem hifens)
    ("CHAVE_PIX", re.compile(r"\b[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
                             r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}\b")),
]

# Credenciais: palavra-gatilho marca a LINHA; dentro dela, mascara tokens com
# cara de senha (linguagem natural poe filler entre o gatilho e a senha real).
RE_TRIGGER = re.compile(
    r"(?i)\b(senha|password|pwd|login|usu[aá]rio|token|pin|acesso|credencial)\b")
_SECRET_SYMS = set("!@#$%&*")

# Dado bancario explicito: agencia/conta + numero
RE_BANCO = re.compile(
    r"(?i)\b(ag[eê]ncia|ag\.?|conta|c/c|cc)\b\s*[:.]?\s*\d[\d.\-/]{2,}")

# Valores monetarios: NUNCA mascarar (conteudo). Usado so para proteger de outros
# regex. Aceita R$ e o "RS" que o OCR costuma gerar quando le o cifrao errado.
RE_DINHEIRO = re.compile(r"(?i)\bR[\$S]\s?\d[\d.,]*")


def _norm(s: str) -> str:
    """minuscula sem acento, para casar nomes independente de acentuacao."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


class Scrubber:
    def __init__(self, extra_names=None):
        self.counter = {}          # rotulo -> proximo numero
        self.mapa = {}             # valor_original -> pseudonimo (consistencia)
        self.report = {}           # rotulo -> qtd
        # nomes conhecidos: ordena por tamanho desc (casa "Maria Souza" antes de "Maria")
        self.extra_names = sorted(extra_names or [], key=len, reverse=True)
        self._name_res = [
            (n, re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE))
            for n in self.extra_names
        ]

    def _looks_secret(self, tok: str) -> bool:
        if len(tok) < 6 or "\x00" in tok:
            return False
        has_alpha = any(c.isalpha() for c in tok)
        has_digit = any(c.isdigit() for c in tok)
        has_sym = any(c in _SECRET_SYMS for c in tok)
        return has_alpha and (has_digit or has_sym)

    def _mask_secrets(self, line: str) -> str:
        def repl(m):
            tok = m.group(0)
            if self._looks_secret(tok):
                self.report["CREDENCIAL"] = self.report.get("CREDENCIAL", 0) + 1
                return "[CREDENCIAL]"
            return tok
        return re.sub(r"\S+", repl, line)

    def _placeholder(self, rotulo: str, original: str) -> str:
        key = (rotulo, original.strip())
        if key not in self.mapa:
            self.counter[rotulo] = self.counter.get(rotulo, 0) + 1
            self.mapa[key] = f"[{rotulo}_{self.counter[rotulo]}]"
        self.report[rotulo] = self.report.get(rotulo, 0) + 1
        return self.mapa[key]

    def scrub(self, text: str):
        # self.report acumula entre chamadas (uma conversa = varios scrub)
        if not text:
            return text, dict(self.report)

        # 0. Protege valores monetarios trocando por marcador temporario
        money = []
        def _stash(m):
            money.append(m.group(0))
            return f"\x00M{len(money)-1}\x00"
        text = RE_DINHEIRO.sub(_stash, text)

        # 1. Nomes conhecidos (contadora/cliente informados)
        for original, rgx in self._name_res:
            text = rgx.sub(lambda m, o=original: self._placeholder("NOME", o), text)

        # 2. Credenciais: linha com palavra-gatilho -> mascara tokens com cara de senha
        out_lines = []
        for line in text.split("\n"):
            if RE_TRIGGER.search(line):
                self.report["CREDENCIAL_LINHA"] = self.report.get("CREDENCIAL_LINHA", 0) + 1
                line = self._mask_secrets(line)
            out_lines.append(line)
        text = "\n".join(out_lines)

        # 3. Dado bancario
        def _banco(m):
            self.report["BANCARIO"] = self.report.get("BANCARIO", 0) + 1
            return f"{m.group(1)}: [BANCARIO]"
        text = RE_BANCO.sub(_banco, text)

        # 4. Padroes estruturados (ordem = mais especifico primeiro)
        for rotulo, rgx in PATTERNS:
            text = rgx.sub(lambda m, r=rotulo: self._placeholder(r, m.group(0)), text)

        # 5. Restaura valores monetarios
        text = re.sub(r"\x00M(\d+)\x00", lambda m: money[int(m.group(1))], text)
        return text, dict(self.report)


def _self_test():
    amostra = (
        "Maria Souza: bom dia! me manda o CNPJ\n"
        "Cliente: e 12.345.678/0001-90, razao social Padaria Pao Quente LTDA\n"
        "Cliente: meu cpf 123.456.789-09, email joao@gmail.com, cel (11) 98765-4321\n"
        "Cliente: a senha do gov.br e Brasil@2024 e o cep 01310-100\n"
        "Maria Souza: o faturamento foi R$ 80.000,00 esse mes, aliquota 6% no Simples\n"
        "Cliente: chave pix f47ac10b-58cc-4372-a567-0e02b2c3d479\n"
    )
    s = Scrubber(extra_names=["Maria Souza"])
    limpo, rep = s.scrub(amostra)
    print(limpo)
    print("--- report:", rep)
    assert "12.345.678/0001-90" not in limpo
    assert "123.456.789-09" not in limpo
    assert "joao@gmail.com" not in limpo
    assert "Brasil@2024" not in limpo
    assert "Maria Souza" not in limpo
    assert "R$ 80.000,00" in limpo and "6%" in limpo  # dinheiro/aliquota PRESERVADOS
    print("\nOK - PII mascarada, conteudo contabil preservado")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo", nargs="?", help="txt para anonimizar")
    ap.add_argument("--audit", action="store_true", help="mostra so o relatorio de deteccao")
    ap.add_argument("--names", default="", help="nomes conhecidos separados por virgula")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test or not args.arquivo:
        _self_test()
        return

    with open(args.arquivo, encoding="utf-8") as f:
        texto = f.read()
    s = Scrubber(extra_names=[n.strip() for n in args.names.split(",") if n.strip()])
    limpo, rep = s.scrub(texto)
    if args.audit:
        print("=== deteccoes (nada gravado) ===")
        for k, v in sorted(rep.items()):
            print(f"  {k}: {v}")
    else:
        sys.stdout.write(limpo)


if __name__ == "__main__":
    main()
