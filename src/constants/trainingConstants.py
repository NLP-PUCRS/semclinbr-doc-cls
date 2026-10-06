LABELS = {
    'specialty': [
        'cardiologia',
        'ortopedia',
        'cirurgia (geral)',
        'clínica médica',
        'nefrologia',
        'outra'
    ],
    'type': [
        'sumário de alta hospitalar',
        'evoluções de enfermagem',
        'prontuários de atendimento ambulatorial'
    ]
}

LABELS_TO_MODIFY = {
    'specialty': [
        'dermatologia',
        'oftalmologia',
        'oncologia',
        'endocrinologia',
        'otorrinolaringologia',
        'obstetrícia',
        'pneumologia'
    ],
    'type': [
        'admissão de pronto-socorro'
    ]
}

MERGE_LABEL = 'outra'

EFFECTIVE_BATCH_SIZE = 32