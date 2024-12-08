# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:18:16 2024

@author: asvga
"""

import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
import matplotlib.pyplot as plt

# Carregar os dados
costes_file = '241204_costes.xlsx'
operaciones_file = '241204_datos_operaciones_programadas.xlsx'

costes = pd.read_excel(costes_file)
operaciones = pd.read_excel(operaciones_file)

# Normalizar os nomes das colunas
costes['Unnamed: 0'] = costes['Unnamed: 0'].str.strip()
costes.columns = costes.columns.str.strip().str.replace(' ', '_')
operaciones.columns = operaciones.columns.str.strip().str.replace(' ', '_')

# Filtrar operações de Cardiología Pediátrica
operaciones_pediatricas = operaciones[operaciones['Especialidad_quirúrgica'].str.contains("Cardiología", na=False)]

# Ajustar formatos de código de operação para combinar os arquivos
operaciones_pediatricas['Código_operación'] = operaciones_pediatricas['Código_operación'].str.replace(' ', '_')
quirofanos = costes['Unnamed:_0'].tolist()
operacoes_codigos = [op for op in operaciones_pediatricas['Código_operación'] if op in costes.columns[1:]]

# Criar dicionário de custos
cost_dict = {
    (row['Unnamed:_0'], col): row[col]
    for _, row in costes.iterrows()
    for col in costes.columns[1:]
    if col in operacoes_codigos
}

# Criar o modelo de otimização
modelo = LpProblem("Minimizar_Custo_Atribuicao", LpMinimize)

# Variáveis de decisão
x = LpVariable.dicts("x", [(op, q) for op in operacoes_codigos for q in quirofanos], cat="Binary")

# Função objetivo
modelo += lpSum(cost_dict[(q, op)] * x[(op, q)] for op in operacoes_codigos for q in quirofanos)

# Restrição 1: Cada operação deve ser atribuída a pelo menos um quirófano
for op in operacoes_codigos:
    modelo += lpSum(x[(op, q)] for q in quirofanos) == 1, f"Restricao_Assignacao_{op}"

# Restrições de incompatibilidade
for _, operacao in operaciones_pediatricas.iterrows():
    op = operacao['Código_operación']
    inicio_op = operacao['Hora_inicio']
    fim_op = operacao['Hora_fin']
    
    for _, outra_operacao in operaciones_pediatricas.iterrows():
        outra_op = outra_operacao['Código_operación']
        inicio_outra = outra_operacao['Hora_inicio']
        fim_outra = outra_operacao['Hora_fin']
        
        if op != outra_op and not (fim_op <= inicio_outra or fim_outra <= inicio_op):
            for q in quirofanos:
                modelo += x[(op, q)] + x[(outra_op, q)] <= 1, f"Restricao_Incompatibilidade_{op}_{outra_op}_{q}"

# Resolver o modelo
modelo.solve()

# Extrair resultados
resultados = []
for op in operacoes_codigos:
    for q in quirofanos:
        if x[(op, q)].value() == 1:
            resultados.append({"Operação": op, "Quirófano": q, "Custo": cost_dict[(q, op)]})

# Criar DataFrame de resultados
resultados_df = pd.DataFrame(resultados)
if not resultados_df.empty:
    resultados_df['Custo_Total'] = resultados_df['Custo'].sum()

# Criar uma tabela visual dos resultados
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Criar tabela com os resultados
table_data = [["Operação", "Quirófano", "Custo"]]
table_data += resultados_df[["Operação", "Quirófano", "Custo"]].values.tolist()

# Adicionar a linha do custo total
custo_total = resultados_df["Custo"].sum()
table_data.append(["Custo Total", "", custo_total])

# Adicionar a tabela na imagem
table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(table_data[0]))))

# Salvar como imagem
image_path = 'resultados_modelo1_cardiologia_pediatrica_com_custo_total.png'
plt.savefig(image_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"A tabela foi salva como imagem: {image_path}")
