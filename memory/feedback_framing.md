---
name: feedback_framing
description: O projeto ResNet-Psi é sobre paradigma computacional (memória fixa O(1), processamento local), NUNCA sobre acurácia. Não tratar acurácia como resultado central.
type: feedback
---

O usuário deixou claro em 2026-03-30: "o paper nunca foi sobre acurácia, foi apresentar essa nova forma de computar algo."

O paper v1 (original) já tinha o framing correto: memória O(1) vs O(n²) dos Transformers, processamento local via Laplaciano, implementabilidade em hardware analógico. O paper v4 errou ao colocar acurácia (77.4%) como resultado central.

## Regras para futuras conversas:
1. NUNCA apresentar acurácia como o resultado principal do ResNet-Psi
2. O resultado é o **paradigma computacional**: campo físico como substrato de computação com memória fixa
3. Acurácia é evidência secundária de que o campo funciona, não o ponto
4. Os baselines (Audit 30/30b) provam isso: pixels brutos fazem 82%, ResNet-Ψ faz 77.7% — e tá tudo bem, porque acurácia nunca foi o objetivo
5. O valor é: memória fixa, processamento local, zero treino, potencial para hardware analógico, custo energético zero
6. Quando discutir resultados, sempre contextualizar: "funciona como computação, não compete em acurácia"
