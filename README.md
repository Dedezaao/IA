# Trabalho AV1 — Inteligência Artificial Computacional

Implementação e avaliação de modelos de regressão e classificação por Mínimos Quadrados Ordinários (MQO), regularização de Tikhonov e expansão polinomial.

Os modelos são implementados explicitamente a partir das equações normais, utilizando NumPy apenas para operações de álgebra linear; nenhuma biblioteca de aprendizado de máquina contendo implementações prontas dos algoritmos solicitados é utilizada.

## Estrutura

- `regressao.py`: estima o PIB da China em função do ano.
- `classificacao.py`: classifica cinco expressões faciais a partir de dois sensores de EMG.
- `china_gdp.csv`: 55 observações anuais, de 1960 a 2014.
- `EMG1.csv`: 50.000 observações, dois sensores e uma classe.
- `relatorio.tex` e `Relatorio_AV1.pdf`: relatório acadêmico e versão compilada.
- `figuras/`: nove gráficos gerados pelos experimentos.
- `saida_*.txt` e `resultados_*.csv`: resultados numéricos reproduzíveis.
- `AUDITORIA_AV1.md` e `CHECKLIST_ENTREGA.md`: conformidade e conferência final.

## Dependências

- Python 3.10 ou posterior
- NumPy
- Matplotlib

Instalação opcional em ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Execução

Os datasets devem permanecer na raiz do repositório com os nomes `china_gdp.csv` e `EMG1.csv`.

```powershell
python regressao.py
python classificacao.py
```

Os scripts auditam os datasets, criam automaticamente `figuras/`, executam as seleções de hiperparâmetros e as 500 rodadas de validação, e sobrescrevem apenas os resultados derivados. As sementes fixam as partições nas mesmas condições de software; tempos de execução dependem da máquina.

Para verificar a sintaxe:

```powershell
python -m py_compile regressao.py classificacao.py
```

Para compilar o relatório, quando `pdflatex` estiver instalado:

```powershell
pdflatex relatorio.tex
pdflatex relatorio.tex
```
