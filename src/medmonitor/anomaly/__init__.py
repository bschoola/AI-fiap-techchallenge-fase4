"""Detecção de anomalias (a implementar).

Plano:
- Sinais vitais (PhysioNet): regras clínicas + z-score móvel + Isolation Forest.
- Prescrições: dados sintéticos; mudanças bruscas de dose/medicação.
- Movimentação: séries agregadas a partir do módulo de vídeo.
Cada anomalia vira um `Alert` (source="vitals" | "prescription" | "movement").
"""
