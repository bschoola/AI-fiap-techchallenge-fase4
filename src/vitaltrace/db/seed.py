"""Seed da aplicação: 1 médico e 30 pacientes fictícios com histórico clínico sintético.

Tudo aqui é inventado e determinístico (semente fixa), para a demonstração. Os sinais
vitais são sintéticos; veja docs/relatorio-tecnico.md sobre o uso do PhysioNet.
"""

from datetime import date, datetime, time, timedelta

import numpy as np

from vitaltrace.audio.critical_terms import normalize
from vitaltrace.anomaly.service import analyze_patient
from vitaltrace.db import repos, seed_media
from vitaltrace.db.orm import (
    AlertRow,
    Analysis,
    Consultation,
    Doctor,
    Patient,
    Prescription,
    VitalSign,
)
from vitaltrace.db.session import Database
from vitaltrace.security import hash_password

DEMO_EMAIL = "rafael.menezes@hospitalmodelo.demo"
DEMO_PASSWORD = "vitaltrace123"
_REFERENCE_DAY = date(2026, 11, 5)  # data do protótipo; as datas abaixo são relativas a ela

# nome, idade, sexo, "dd/mm · HH:MM" da última consulta, motivo, perfil clínico
PATIENTS = [
    ("Helena Martins Duarte", 67, "F", "05/11 · 08:40", "Dispneia e dor torácica", "critical"),
    ("Roberto Almeida Lima", 72, "M", "05/11 · 08:10", "Controle de insuficiência cardíaca", "attention"),
    ("Sofia Carvalho Nunes", 54, "F", "04/11 · 17:30", "Reabilitação pós-artroplastia de joelho", "attention"),
    ("Carlos Eduardo Pereira", 45, "M", "04/11 · 16:05", "Cefaleia recorrente", "stable"),
    ("Ana Beatriz Moreira", 38, "F", "04/11 · 15:20", "Retorno pós-operatório", "stable"),
    ("Marcos Vinícius Teixeira", 59, "M", "04/11 · 11:45", "Hipertensão arterial", "stable"),
    ("Luciana Ferreira Costa", 63, "F", "03/11 · 17:10", "Diabetes tipo 2", "attention"),
    ("José Antônio Ribeiro", 78, "M", "03/11 · 14:00", "Dispneia aos esforços", "attention"),
    ("Patrícia Gomes Barbosa", 41, "F", "03/11 · 10:30", "Fisioterapia de ombro", "stable"),
    ("Fernando Henrique Souza", 50, "M", "31/10 · 16:00", "Lombalgia", "stable"),
    ("Beatriz Oliveira Santos", 29, "F", "31/10 · 09:15", "Pré-natal", "stable"),
    ("Antônio Carlos Medeiros", 81, "M", "30/10 · 15:40", "Pós-AVC, fonoterapia", "attention"),
    ("Isabela Rocha Fernandes", 34, "F", "30/10 · 11:00", "Asma, controle", "stable"),
    ("Ricardo Nascimento Alves", 66, "M", "29/10 · 14:20", "Pós-operatório de colecistectomia", "stable"),
    ("Camila Dias Cardoso", 47, "F", "29/10 · 09:40", "Hipotireoidismo", "stable"),
    ("Paulo Roberto Araújo", 69, "M", "28/10 · 16:30", "DPOC", "attention"),
    ("Juliana Mendes Ramos", 36, "F", "28/10 · 10:10", "Enxaqueca", "stable"),
    ("Gustavo Lopes Vieira", 52, "M", "27/10 · 15:00", "Pré-operatório cardíaco", "stable"),
    ("Marina Castro Pinto", 60, "F", "27/10 · 08:50", "Osteoartrite de joelho", "stable"),
    ("Eduardo Barros Monteiro", 74, "M", "24/10 · 13:30", "Doença renal crônica", "attention"),
    ("Renata Azevedo Freitas", 43, "F", "24/10 · 09:00", "Check-up anual", "stable"),
    ("Thiago Marques Correia", 31, "M", "23/10 · 17:00", "Fisioterapia, ligamento cruzado", "stable"),
    ("Débora Lima Cavalcanti", 57, "F", "23/10 · 11:20", "Hipertensão arterial", "stable"),
    ("Alexandre Pinto Rezende", 64, "M", "22/10 · 15:10", "Arritmia", "attention"),
    ("Vanessa Carvalho Brito", 39, "F", "22/10 · 10:00", "Retorno de exames", "stable"),
    ("Leonardo Duarte Siqueira", 48, "M", "21/10 · 14:45", "Dor torácica atípica", "stable"),
    ("Mariana Batista Gomes", 26, "F", "21/10 · 09:30", "Rinite alérgica", "stable"),
    ("Sérgio Luiz Machado", 71, "M", "20/10 · 16:20", "Reabilitação pós-infarto", "attention"),
    ("Cláudia Regina Tavares", 55, "F", "20/10 · 10:40", "Fibromialgia", "stable"),
    ("Henrique Salgado Neto", 62, "M", "17/10 · 08:30", "Apneia do sono", "stable"),
]

# (trecho do motivo normalizado) -> (condições, [(medicamento, dose mg, frequência)])
_CLINICAL = [
    ("insuficiencia cardiaca", ["Insuficiência cardíaca", "Hipertensão arterial"],
     [("Furosemida", 40, "1x/dia"), ("Carvedilol", 12.5, "2x/dia")]),
    ("dispneia e dor", ["Hipertensão arterial", "Insuficiência cardíaca"],
     [("Losartana", 50, "1x/dia")]),
    ("dispneia aos esforcos", ["Dispneia crônica", "Hipertensão arterial"],
     [("Enalapril", 10, "2x/dia")]),
    ("hipertensao", ["Hipertensão arterial"], [("Losartana", 50, "1x/dia")]),
    ("diabetes", ["Diabetes tipo 2"], [("Metformina", 850, "2x/dia")]),
    ("dpoc", ["DPOC"], [("Tiotrópio", 0.018, "1x/dia")]),
    ("asma", ["Asma"], [("Budesonida", 0.4, "2x/dia")]),
    ("hipotireoidismo", ["Hipotireoidismo"], [("Levotiroxina", 0.075, "1x/dia")]),
    ("arritmia", ["Arritmia"], [("Amiodarona", 200, "1x/dia")]),
    ("doenca renal", ["Doença renal crônica"], [("Enalapril", 10, "1x/dia")]),
    ("enxaqueca", ["Enxaqueca"], [("Sumatriptana", 50, "se necessário")]),
    ("avc", ["AVC prévio"], [("AAS", 100, "1x/dia"), ("Sinvastatina", 20, "1x/dia")]),
    ("pos-infarto", ["Infarto prévio"],
     [("AAS", 100, "1x/dia"), ("Atorvastatina", 40, "1x/dia"), ("Metoprolol", 50, "2x/dia")]),
    ("apneia", ["Apneia do sono"], []),
]

_STATUS_NOTES = {
    "stable": [
        "Em bom estado geral, sem queixas novas. {reason}. Mantida a conduta atual.",
        "Refere melhora e boa adesão ao tratamento. {reason}. Retorno de rotina.",
        "Sem intercorrências desde a última consulta. {reason}. Exames dentro do esperado.",
    ],
    "attention": [
        "Refere cansaço leve ao fim do dia. {reason}. Reavaliar em duas semanas.",
        "Queixa de desconforto ocasional, sem sinais de alarme. {reason}. Ajuste de orientações.",
        "Evolução mais lenta que o esperado. {reason}. Reforçada a adesão ao plano terapêutico.",
    ],
}

_HELENA = [
    (120, "Consulta de rotina. PA 138/86 em uso de losartana. Edema leve de tornozelos ao fim do dia. Orientada restrição de sódio e controle do peso diário.", 96, 76, 135, 84),
    (75, "Refere cansaço ao subir um lance de escada, sem dor torácica. Ausculta com estertores finos em bases. Iniciada furosemida 20 mg/dia.", 95, 80, 140, 88),
    (40, "Mantém dispneia aos esforços moderados e dorme com dois travesseiros. Ganho de 2 kg em duas semanas. Furosemida aumentada para 40 mg/dia.", 94, 84, 142, 90),
    (12, "Piora da dispneia aos pequenos esforços e edema de membros inferiores. Refere tosse noturna. Solicitados ecocardiograma e BNP.", 92, 92, 146, 90),
    (0, "Refere falta de ar progressiva e dor torácica desde a noite anterior, com cansaço aos mínimos esforços e tontura ao levantar. Furosemida aumentada para 80 mg/dia e iniciada espironolactona 25 mg.", 89, 104, 150, 92),
]
_ROBERTO = [
    (90, "Insuficiência cardíaca compensada. Sem edema. Peso estável. Mantidos carvedilol e furosemida 40 mg.", 95, 72, 126, 78),
    (45, "Relata mais cansaço ao caminhar. Discreto edema vespertino. Mantida conduta e reforçada restrição de sal.", 94, 76, 130, 80),
    (0, "Dispneia aos esforços moderados, tosse seca à noite. Furosemida aumentada para 80 mg/dia. Reavaliar em uma semana.", 91, 86, 134, 82),
]

# Roberto chega com SpO2 em torno de 90% (alerta alto, não crítico)
_DRIFT_OVERRIDE = {"Roberto": {"spo2": -4.2, "heart_rate": 12, "resp_rate": 3, "systolic": 8}}

# Histórico de índice de mobilidade (vídeo) por paciente, do mais antigo ao mais recente
_VIDEO_HISTORY = {
    "Sofia Carvalho Nunes": [0.64, 0.68, 0.71],
    "Antônio Carlos Medeiros": [0.44, 0.42, 0.30],
    "Sérgio Luiz Machado": [0.55, 0.57, 0.58],
    "Thiago Marques Correia": [0.70, 0.74, 0.78],
    "Patrícia Gomes Barbosa": [0.60, 0.63, 0.66],
    "Helena Martins Duarte": [0.52, 0.48, 0.31],
}


# Gravações de áudio de exemplo: (paciente, minutos atrás, arquivo, KB, trechos (início, fim, texto)).
# Helena usa os dois alertas de exemplo abaixo; as demais não passam dos limiares (sem alertas).
_AUDIO_SAMPLES = [
    ("Helena Martins Duarte", 2, "consulta_helena_manha.wav", 412, [
        (0.5, 6.0, "Doutor, estou com muita falta de ar e uma dor no peito desde ontem."),
        (7.0, 12.5, "Preciso parar para respirar a cada frase."),
    ]),
    ("Roberto Almeida Lima", 3 * 24 * 60, "retorno_roberto.wav", 388, [
        (0.4, 5.5, "Doutor, dormi bem esta semana e sigo tomando os remédios nos horários."),
        (6.5, 12.0, "A escada lá de casa está exigindo um pouco mais de mim."),
    ]),
    ("Antônio Carlos Medeiros", 6 * 24 * 60, "fonoterapia_antonio.wav", 540, [
        (1.0, 5.0, "Bom dia, doutor. Estou fazendo os exercícios da fono todos os dias."),
        (8.0, 13.0, "Sinto que as palavras saem mais devagar à tarde, mas sem piora."),
        (16.0, 20.0, "Consigo me alimentar bem e dormir tranquilo."),
    ]),
    ("Paulo Roberto Araújo", 9 * 24 * 60, "retorno_paulo.wav", 366, [
        (0.5, 6.0, "Doutor, uso a bombinha pela manhã e consigo caminhar até a padaria."),
        (7.0, 12.0, "Tenho tossido um pouco menos do que no mês passado."),
    ]),
]

# Sessões de vídeo anteriores (do mais antigo ao mais recente) e, nos casos abaixo, o evento da última.
_LATEST_VIDEO_EVENT = {
    "Helena Martins Duarte": ("Mobilidade reduzida", "Mobilidade 40% abaixo da média das sessões anteriores."),
    "Antônio Carlos Medeiros": ("Mobilidade reduzida", "Mobilidade 30% abaixo da média das sessões anteriores."),
}


def _parse_last(label: str, now: datetime) -> datetime:
    day_month, hhmm = (p.strip() for p in label.split("·"))
    day, month = (int(x) for x in day_month.split("/"))
    hour, minute = (int(x) for x in hhmm.split(":"))
    days_ago = (_REFERENCE_DAY - date(_REFERENCE_DAY.year, month, day)).days
    when = datetime.combine(now.date() - timedelta(days=days_ago), time(hour, minute))
    return min(when, now - timedelta(minutes=10))


def _clinical_profile(reason: str):
    key = normalize(reason)
    for needle, conditions, drugs in _CLINICAL:
        if needle in key:
            return conditions, drugs
    return [reason], []


def _baseline(rng: np.random.RandomState, reason: str) -> dict[str, float]:
    key = normalize(reason)
    base = {
        "heart_rate": 72 + rng.uniform(-6, 6),
        "spo2": 97.2 + rng.uniform(-0.6, 0.6),
        "resp_rate": 15.5 + rng.uniform(-1, 1),
        "systolic": 120 + rng.uniform(-8, 8),
        "diastolic": 77 + rng.uniform(-5, 5),
        "temperature": 36.5 + rng.uniform(-0.1, 0.15),
    }
    if "dpoc" in key or "apneia" in key:
        base["spo2"] -= 3.5
    if "insuficiencia" in key or "dispneia" in key:
        base["heart_rate"] += 6
        base["spo2"] -= 1.5
    if "hipertensao" in key:
        base["systolic"] += 12
        base["diastolic"] += 6
    return base


def generate_vitals(
    patient_id: int,
    reason: str,
    profile: str,
    now: datetime,
    *,
    days: int = 14,
    per_day: int = 6,
    drift: dict[str, float] | None = None,
) -> list[VitalSign]:
    """Série sintética; `profile` define a deriva nos últimos dias (stable/attention/critical)."""
    rng = np.random.RandomState(1000 + patient_id)
    base = _baseline(rng, reason)
    sigma = {"heart_rate": 3.2, "spo2": 0.5, "resp_rate": 0.9, "systolic": 4.5,
             "diastolic": 3.0, "temperature": 0.12}  # fmt: skip
    n = days * per_day
    step = timedelta(hours=24 / per_day)
    start = now - step * (n - 1)
    # deriva (delta final por sinal) aplicada de forma crescente nos últimos `drift_days`
    drift_days = {"stable": 0, "attention": 5, "critical": 4}[profile]
    final = {
        "stable": {},
        "attention": {"heart_rate": 9, "spo2": -3.0, "systolic": 10, "resp_rate": 2},
        "critical": {"heart_rate": 28, "spo2": -7.5, "systolic": 20, "diastolic": 10,
                     "resp_rate": 8, "temperature": 0.9},
    }[profile]  # fmt: skip
    if drift:
        final = drift
    drift_points = drift_days * per_day

    rows: list[VitalSign] = []
    for i in range(n):
        ts = start + step * i
        circadian = np.sin((ts.hour - 4) / 24 * 2 * np.pi)
        frac = max(0.0, (i - (n - drift_points)) / drift_points) if drift_points else 0.0
        values = {}
        for name, mean in base.items():
            v = mean + rng.normal(0, sigma[name]) + final.get(name, 0.0) * frac**1.3
            if name == "heart_rate":
                v += 3 * circadian
            if name == "temperature":
                v += 0.15 * circadian
            values[name] = v
        values["spo2"] = float(np.clip(values["spo2"], 78, 100))
        rows.append(
            VitalSign(
                patient_id=patient_id,
                ts=ts,
                heart_rate=round(values["heart_rate"], 0),
                spo2=round(values["spo2"], 0),
                resp_rate=round(values["resp_rate"], 0),
                systolic=round(values["systolic"], 0),
                diastolic=round(values["diastolic"], 0),
                temperature=round(values["temperature"], 1),
            )
        )
    return rows


def _generic_consultations(
    rng: np.random.RandomState, patient: Patient, status: str, doctor_id: int, last: datetime
) -> list[Consultation]:
    templates = _STATUS_NOTES["stable" if status == "stable" else "attention"]
    count = int(rng.randint(3, 6))
    items: list[Consultation] = []
    for k in range(count):
        when = last - timedelta(days=45 * (count - 1 - k), minutes=int(rng.randint(0, 40)))
        if k == count - 1:
            when = last
        note = templates[int(rng.randint(len(templates)))].format(reason=patient.reason)
        items.append(
            Consultation(
                patient_id=patient.id,
                doctor_id=doctor_id,
                occurred_at=when,
                anamnesis=note,
                spo2=float(round(97 - (2 if status != "stable" else 0) + rng.normal(0, 0.8))),
                heart_rate=float(round(74 + rng.normal(0, 5) + (6 if status != "stable" else 0))),
                systolic=float(round(122 + rng.normal(0, 7) + (8 if status != "stable" else 0))),
                diastolic=float(round(78 + rng.normal(0, 4))),
            )
        )
    return items


def _scripted_consultations(patient: Patient, script, doctor_id: int, now: datetime):
    last = patient.last_consult_at
    return [
        Consultation(
            patient_id=patient.id,
            doctor_id=doctor_id,
            occurred_at=last - timedelta(days=days_ago),
            anamnesis=text,
            spo2=float(spo2),
            heart_rate=float(hr),
            systolic=float(sys),
            diastolic=float(dia),
        )
        for days_ago, text, spo2, hr, sys, dia in script
    ]


def _prescriptions(patient: Patient, drugs, today: date) -> list[Prescription]:
    rows = [
        Prescription(
            patient_id=patient.id, drug=d, dose_mg=dose, frequency=freq,
            start_date=today - timedelta(days=150),
        )
        for d, dose, freq in drugs
    ]  # fmt: skip
    return rows


def _special_prescriptions(patient: Patient, today: date) -> list[Prescription]:
    def rx(drug, dose, freq, start, end=None):
        return Prescription(
            patient_id=patient.id, drug=drug, dose_mg=dose, frequency=freq,
            start_date=today - timedelta(days=start),
            end_date=None if end is None else today - timedelta(days=end),
        )  # fmt: skip

    if patient.name.startswith("Helena"):
        return [
            rx("Losartana", 50, "1x/dia", 150),
            rx("Furosemida", 20, "1x/dia", 75, 40),
            rx("Furosemida", 40, "1x/dia", 40, 0),
            rx("Furosemida", 80, "1x/dia", 0),
            rx("Espironolactona", 25, "1x/dia", 0),
        ]
    if patient.name.startswith("Roberto"):
        return [
            rx("Carvedilol", 12.5, "2x/dia", 200),
            rx("Furosemida", 40, "1x/dia", 90, 0),
            rx("Furosemida", 80, "1x/dia", 0),
        ]
    return []


def seed_database(db: Database, *, now: datetime | None = None) -> bool:
    """Popula o banco se estiver vazio. Retorna True se inseriu dados."""
    now = (now or datetime.now()).replace(microsecond=0)
    with db.session() as s:
        if repos.count_patients(s) > 0:
            return False

        doctor = Doctor(
            name="Dr. Rafael Menezes",
            email=DEMO_EMAIL,
            password_hash=hash_password(DEMO_PASSWORD),
            crm="CRM-SP 000000",
            specialty="Clínica médica",
        )
        s.add(doctor)
        s.flush()

        for idx, (name, age, sex, last_label, reason, status) in enumerate(PATIENTS):
            rng = np.random.RandomState(idx + 1)
            conditions, drugs = _clinical_profile(reason)
            patient = Patient(
                name=name,
                name_search=normalize(name),
                birth_date=now.date() - timedelta(days=age * 365 + int(rng.randint(0, 300))),
                sex=sex,
                record_number="00482" if idx == 0 else f"{1000 + idx * 37:05d}",
                reason=reason,
                conditions=conditions,
                last_consult_at=_parse_last(last_label, now),
                risk_level=status,
                risk_score={"stable": 10.0, "attention": 40.0, "critical": 80.0}[status],
            )
            s.add(patient)
            s.flush()

            s.add_all(
                generate_vitals(
                    patient.id, reason, status, now, drift=_DRIFT_OVERRIDE.get(name.split()[0])
                )
            )

            if name.startswith("Helena"):
                s.add_all(_scripted_consultations(patient, _HELENA, doctor.id, now))
            elif name.startswith("Roberto"):
                s.add_all(_scripted_consultations(patient, _ROBERTO, doctor.id, now))
            else:
                s.add_all(_generic_consultations(rng, patient, status, doctor.id, patient.last_consult_at))

            special = _special_prescriptions(patient, now.date())
            s.add_all(special or _prescriptions(patient, drugs, now.date()))

        s.flush()
        _seed_alerts(s, now)
        s.flush()
        _seed_analyses(s, now)
        s.flush()
        # Risco e alertas iniciais vêm dos próprios detectores, não de valores fixos
        for patient in s.query(Patient).all():
            analyze_patient(s, patient, save_report=False, now=now)
    return True


def _seed_alerts(s, now: datetime) -> None:
    by_name = {p.name: p for p in s.query(Patient).all()}
    items = [
        ("Helena Martins Duarte", "audio", "critical", "Termo crítico (respiratório): falta de ar", 2),
        ("Helena Martins Duarte", "audio", "critical", "Termo crítico (cardiovascular): dor no peito", 2),
        ("Sofia Carvalho Nunes", "video", "medium", "Assimetria entre os joelhos durante o agachamento", 60),
    ]  # fmt: skip
    for k, (name, source, severity, title, minutes_ago) in enumerate(items):
        s.add(
            AlertRow(
                id=f"seed{k:028d}",
                patient_id=by_name[name].id,
                source=source,
                severity=severity,
                title=title,
                description="Alerta de exemplo carregado pelo seed.",
                evidence={"seeded": True},
                status="open",
                created_at=now - timedelta(minutes=minutes_ago),
            )
        )


def _seed_analyses(s, now: datetime) -> None:
    """Áudios e vídeos de exemplo, para a linha do tempo mostrar o histórico logo na primeira execução."""
    by_name = {p.name: p for p in s.query(Patient).all()}

    for name, mobilities in _VIDEO_HISTORY.items():
        for k, mobility in enumerate(mobilities):
            latest = k == len(mobilities) - 1
            events, frames = [], []
            if latest and name in _LATEST_VIDEO_EVENT:
                label, description = _LATEST_VIDEO_EVENT[name]
                events = [{"type": "low_mobility", "severity": "medium", "t_start": 0.0, "t_end": 60.0, "description": description}]
                frames = [{"t": 30.0, "label": label, "event_index": 0, "image": seed_media.synthetic_frame(label, 30)}]
            elif latest:
                frames = [{"t": 30.0, "label": "Quadro de exemplo", "event_index": None, "image": seed_media.synthetic_frame("Quadro de exemplo", 30)}]
            s.add(
                Analysis(
                    patient_id=by_name[name].id,
                    kind="video",
                    created_at=now - timedelta(days=21 * (len(mobilities) - k)),
                    summary=f"{len(events)} evento(s), mobilidade {mobility:.2f}.".replace(".", ",", 1),
                    payload=seed_media.video_payload(
                        mobility, filename=f"sessao_{k + 1}.mp4", events=events, keyframes=frames
                    ),
                )
            )

    # Vídeo recente da Sofia: é o que originou o alerta de exemplo de assimetria entre os joelhos
    sofia = by_name["Sofia Carvalho Nunes"]
    knee_event = {
        "type": "knee_asymmetry", "severity": "medium", "t_start": 14.0, "t_end": 26.0,
        "description": "Joelho direito flexiona 18° a menos que o esquerdo durante o agachamento.",
    }  # fmt: skip
    payload = seed_media.video_payload(
        0.69, filename="agachamento_sofia.mp4", events=[knee_event], knee_diff=18.0,
        keyframes=[{"t": 18.0, "label": "Assimetria entre os joelhos", "event_index": 0,
                    "image": seed_media.synthetic_frame("Assimetria entre os joelhos", 18, knee_gap=18)}],
    )  # fmt: skip
    payload["alerts"] = [{"id": f"seed{2:028d}", "severity": "medium", "new": True,
                          "title": "Assimetria entre os joelhos durante o agachamento"}]  # fmt: skip
    s.add(Analysis(patient_id=sofia.id, kind="video", created_at=now - timedelta(minutes=62),
                   summary="1 evento(s), mobilidade 0,69.", payload=payload))  # fmt: skip

    helena_alerts = [
        {"id": f"seed{0:028d}", "severity": "critical", "new": True, "title": "Termo crítico (respiratório): falta de ar"},
        {"id": f"seed{1:028d}", "severity": "critical", "new": True, "title": "Termo crítico (cardiovascular): dor no peito"},
    ]  # fmt: skip
    for name, minutes_ago, filename, size_kb, segments in _AUDIO_SAMPLES:
        payload = seed_media.audio_payload(segments, filename, size_kb)
        if name.startswith("Helena"):
            payload["alerts"] = helena_alerts
        real = [t for t in payload["critical_terms"] if not t["negated"]]
        s.add(
            Analysis(
                patient_id=by_name[name].id,
                kind="audio",
                created_at=now - timedelta(minutes=minutes_ago),
                summary=f"{len(real)} termo(s) crítico(s), sentimento {payload['text_analysis']['sentiment']['label']}, "
                f"{len(payload.get('alerts', []))} alerta(s) novo(s).",
                payload=payload,
            )
        )
