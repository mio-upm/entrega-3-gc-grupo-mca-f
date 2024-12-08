# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:58:57 2024

@author: asvga
"""

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize
from itertools import combinations
import matplotlib.pyplot as plt

# Ler os dados dos arquivos Excel
costes = pd.read_excel('241204_costes.xlsx')
operaciones = pd.read_excel('241204_datos_operaciones_programadas.xlsx')

# Ajustar os nomes das colunas, se necessário
operaciones.rename(columns={"Código operación": "Código_operación", "Hora inicio ": "Hora_inicio", "Hora fin": "Hora_fin"}, inplace=True)

# Verificar os nomes das colunas ajustados
print("Colunas ajustadas em 'operaciones':", operaciones.columns)

# Calcular o custo médio para cada operação
costes_medio = costes.iloc[:, 1:].mean(axis=0).to_dict()
operaciones['Custo_medio'] = operaciones['Código_operación'].map(costes_medio)
operaciones['Custo_medio'].fillna(0, inplace=True)  # Preencher NaN com 0
print("Custo médio das operações:\n", operaciones[['Código_operación', 'Custo_medio']].head())

# Filtrar as operações do serviço
servicos = ['Cardiología Pediátrica', 'Cirugía Cardíaca Pediátrica', 'Cirugía Cardiovascular', 'Cirugía General y del Aparato Digestivo']
operaciones_servicos = operaciones[operaciones['Especialidad quirúrgica'].isin(servicos)]

# Gerar planificações (conjuntos de operações que não têm conflito de horário)
def gerar_planificacoes(operacoes):
    planificacoes = []
    for tamanho in range(1, len(operacoes) + 1):
        for combinacao in combinations(operacoes.index, tamanho):
            horarios = [(operacoes.loc[i, 'Hora_inicio'], operacoes.loc[i, 'Hora_fin']) for i in combinacao]
            if all(horario1[1] <= horario2[0] or horario2[1] <= horario1[0] for i, horario1 in enumerate(horarios) for j, horario2 in enumerate(horarios) if i < j):
                planificacoes.append(combinacao)
    return planificacoes

planificacoes = gerar_planificacoes(operaciones_servicos)
print("Planificações geradas:")
for i, plan in enumerate(planificacoes):
    print(f"Planificação {i}: Operações {list(plan)}")

# Cálculo dos custos das planificações
C = {k: sum(operaciones_servicos.loc[i, 'Custo_medio'] for i in plan) for k, plan in enumerate(planificacoes)}

# Criar o modelo de otimização
modelo = LpProblem("Modelo 2 - Set Covering", LpMinimize)

# Variáveis de decisão
y = {k: LpVariable(f"y_{k}", cat="Binary") for k in range(len(planificacoes))}

# Função objetivo
modelo += lpSum(C[k] * y[k] for k in range(len(planificacoes)))

# Restrições de cobertura
for i in operaciones_servicos.index:
    modelo += lpSum(y[k] for k, plan in enumerate(planificacoes) if i in plan) >= 1

# Resolver o modelo
modelo.solve()

# Resultados
planificacoes_otimas = [k for k in range(len(planificacoes)) if y[k].varValue > 0.5]
custo_total = sum(C[k] for k in planificacoes_otimas)

# Mostrar os resultados
print("\nPlanificações ótimas selecionadas:")
resultados = []
for k in planificacoes_otimas:
    operacoes = [operaciones_servicos.loc[i, 'Código_operación'] for i in planificacoes[k]]
    custo = C[k]
    resultados.append((k, operacoes, custo))
    print(f"Planificação {k}: Operações {operacoes}, Custo {custo}")

print(f"\nCusto total: {custo_total}")

# Criar uma imagem com os resultados
fig, ax = plt.subplots(figsize=(10, len(planificacoes_otimas) * 0.5))
ax.axis('tight')
ax.axis('off')
tabela = pd.DataFrame(resultados, columns=["Planificação", "Operações", "Custo"])
tabela.loc["Custo Total"] = ["", "", custo_total]
ax.table(cellText=tabela.values, colLabels=tabela.columns, loc='center')
plt.savefig("resultados_modelo2.png")
print("\nResultados salvos como: resultados_modelo2.png")
