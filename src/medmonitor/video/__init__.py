"""Análise de vídeo (a implementar).

Plano (roda em CPU):
- Pose: YOLOv8n-pose (ou MediaPipe Pose) -> ângulos de articulações, simetria, velocidade.
- Objetos/áreas críticas: YOLOv8n com amostragem de ~5 fps e frames reduzidos.
- Saída: lista de desvios + relatório automático; cada desvio vira um `Alert` (source="video").
Dependências opcionais em requirements-video.txt.
"""
