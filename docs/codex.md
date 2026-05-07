# Codex

Este arquivo serve como referência rápida para rodar, retomar e acompanhar o treino do modelo em `nano_villa.py`.

## Treino básico

Rodar treino com alvo em tokens:

```powershell
python nano_villa.py --tokens 2000000
```

Exemplo para treino longo:

```powershell
python nano_villa.py --tokens 600000000
```

## Parâmetros principais

- `--tokens`
  Define quantos tokens o treino vai processar antes de parar.

- `--config`
  Escolhe o tamanho do modelo.

Exemplo:

```powershell
python nano_villa.py --config small --tokens 600000000
```

## Checkpoints

O script salva:

- um checkpoint principal, sempre sobrescrito;
- snapshots por step, com limpeza automática dos mais antigos.

Parâmetros:

- `--checkpoint-path`
  Define o caminho do checkpoint principal.

- `--save-every`
  Salva checkpoint a cada N steps.

- `--keep-last`
  Mantém apenas os N snapshots mais recentes.

Exemplo:

```powershell
python nano_villa.py --tokens 600000000 --checkpoint-path checkpoints\nano_villa_last.pt --save-every 500 --keep-last 3
```

## Retomar treino

Para continuar de um checkpoint salvo:

```powershell
python nano_villa.py --tokens 600000000 --resume checkpoints\nano_villa_last.pt
```

Observações:

- `--resume` restaura modelo, otimizador, scaler, step e total de tokens processados.
- O dataset em streaming não volta exatamente ao mesmo ponto do corpus; ele reconecta e continua lendo novos trechos.

## Arquivos de checkpoint

Exemplo de arquivos gerados:

- `checkpoints\nano_villa_last.pt`
- `checkpoints\nano_villa_last.step-500.pt`
- `checkpoints\nano_villa_last.step-1000.pt`

Com `--keep-last 3`, o script mantém só os 3 snapshots mais recentes, além do checkpoint principal.

## Fluxo recomendado

1. Testar curto:

```powershell
python nano_villa.py --tokens 2000000
```

2. Rodar treino longo:

```powershell
python nano_villa.py --tokens 600000000 --save-every 500 --keep-last 3
```

3. Se interromper, retomar:

```powershell
python nano_villa.py --tokens 600000000 --resume checkpoints\nano_villa_last.pt
```
