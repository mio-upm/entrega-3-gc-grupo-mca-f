# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 19:41:04 2024

@author: asvga
"""

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize
from itertools import combinations

# Ajustar os dados de entrada
def ajustar_dados():
    costes = pd.read_excel('241204_costes.xlsx')
    operaciones = pd.read_excel('241204_datos_operaciones_programadas.xlsx')

    # Ajustar os nomes das colunas
    operaciones.rename(columns={
        "Código operación": "Código_operación",
        "Hora inicio ": "Hora_inicio",
        "Hora fin": "Hora_fin"
    }, inplace=True)

    print("Colunas originais em 'operaciones':", operaciones.columns)
    return costes, operaciones

# Gerar planificações iniciais
def gerar_planificacoes_iniciais(operacoes):
    planificacoes = []
    max_combinacoes = 500  # Limite de combinações para evitar explosão computacional

    for tamanho in range(1, len(operacoes) + 1):
        for combinacao in combinations(operacoes.index, tamanho):
            if len(planificacoes) >= max_combinacoes:
                print("Número máximo de combinações atingido.")
                return planificacoes
            sub_operacoes = operacoes.loc[list(combinacao)]
            if verificar_viabilidade(sub_operacoes):
                planificacoes.append(list(combinacao))
    return planificacoes

# Verificar a viabilidade de uma planificação
def verificar_viabilidade(operacoes):
    horarios = [(operacoes.loc[i, 'Hora_inicio'], operacoes.loc[i, 'Hora_fin']) for i in operacoes.index]
    return all(
        horario1[1] <= horario2[0] or horario2[1] <= horario1[0]
        for i, horario1 in enumerate(horarios)
        for j, horario2 in enumerate(horarios) if i < j
    )

# Algoritmo de geração de colunas
def algoritmo_geracao_colunas(operacoes_servicos):
    print("Gerando planificações iniciais...")
    planificacoes = gerar_planificacoes_iniciais(operacoes_servicos)
    print(f"Total de planificações geradas: {len(planificacoes)}")

    # Modelo de otimização
    modelo = LpProblem("Modelo 3 - Set Covering", LpMinimize)

    # Variáveis de decisão
    y = {k: LpVariable(f"y_{k}", cat="Binary") for k in range(len(planificacoes))}

    # Restrições de cobertura
    for i in operacoes_servicos.index:
        modelo += lpSum(y[k] for k, plan in enumerate(planificacoes) if i in plan) >= 1

    # Função objetivo
    modelo += lpSum(y[k] for k in range(len(planificacoes)))

    # Resolver o modelo
    print("Resolvendo o modelo...")
    modelo.solve()

    # Coletar as planificações ótimas
    planificacoes_otimas = [k for k in range(len(planificacoes)) if y[k].varValue > 0.5]
    return planificacoes, planificacoes_otimas, y

# Exibir os resultados
def exibir_resultados(planificacoes, planificacoes_otimas, operacoes_servicos, y):
    resultados = []
    for k in planificacoes_otimas:
        operacoes = [operacoes_servicos.loc[i, 'Código_operación'] for i in planificacoes[k]]
        resultados.append({"Planificação": k, "Operações": operacoes})

    # Exibir resultados na console
    print("\nResultados das planificações ótimas:")
    tabela_resultados = pd.DataFrame(resultados)
    print(tabela_resultados)

    # Exibir os valores das variáveis de decisão
    print("\nVariáveis de decisão:")
    for k in range(len(planificacoes)):
        print(f"y_{k}: {y[k].varValue}")


    return tabela_resultados

# Fluxo principal
if __name__ == "__main__":
    # Ajustar dados
    costes, operaciones = ajustar_dados()

    # Selecionar serviços de interesse
    servicos = ['Cardiología Pediátrica', 'Cirugía Cardíaca Pediátrica', 'Cirugía Cardiovascular', 'Cirugía General y del Aparato Digestivo']
    operaciones_servicos = operaciones[operaciones['Especialidad quirúrgica'].isin(servicos)]

    # Gerar planificações ótimas
    planificacoes, planificacoes_otimas, y = algoritmo_geracao_colunas(operaciones_servicos)

    # Exibir os resultados
    tabela_resultados = exibir_resultados(planificacoes, planificacoes_otimas, operaciones_servicos, y)

    # Salvar resultados como CSV (opcional)
    tabela_resultados.to_csv("resultados_modelo3.csv", index=False)
    print("\nResultados salvos como: resultados_modelo3.csv")
