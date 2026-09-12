# Auditoria do Trabalho AV1

Auditoria realizada sobre os datasets reais e sobre resultados regenerados em 11/09/2026.

| Requisito do enunciado | Implementação | Status | Observação |
|---|---|---|---|
| Regressão: `china_gdp.csv` com 55 linhas, 1960–2014 e duas colunas | Validação em `carregar_dados()` | OK | Sem NaN ou Inf |
| Regressão: X em R^(N×p) e y em R^(N×1) | X com shape (55,1), y com (55,1) | OK | Ano é a única variável independente |
| Gráfico inicial e discussão do padrão | `fig1_espalhamento.png` e relatório | OK | Tendência suave e fortemente não linear |
| Intercepto em todos os modelos | Coluna de uns em `adiciona_intercepto()` | OK | Intercepto não é penalizado no Tikhonov por convenção da equipe |
| MQO tradicional manual | Equações normais montadas explicitamente: `Xb.T @ Xb` e `Xb.T @ y` | CORRIGIDO | Resolve com `numpy.linalg.solve`; pseudo-inversa da matriz de Gram somente como fallback de singularidade |
| Tikhonov com lambda 0, 0.25, 0.5, 0.75 e 1 | Quatro linhas regularizadas; lambda 0 representado pelo MQO | OK | Evita linha duplicada |
| Regressão polinomial manual | Potências construídas por NumPy e estimadas por MQO | OK | Sem expansor polinomial pronto |
| Poda de q usando R² | 100 splits comuns, tolerância operacional 0,005 | CORRIGIDO | q final = 9; não se alega equivalência estatística |
| Random Subsampling 500 vezes, 80/20 | Splits reprodutíveis com seed 4242 | OK | Mesmos splits para todos os modelos |
| MSE e R² com média, desvio, máximo e mínimo | Saídas TXT/CSV e tabelas do relatório | OK | R² protege denominador zero |
| Classificação: `EMG1.csv` 50000×3, p=2, C=5 | Validação em `carregar_dados()` | OK | Classes 1–5, 10.000 amostras cada, sem NaN/Inf |
| Organizações X N×p/Y N×C e X p×N/Y C×N | Shapes impressos e registrados | OK | (50000,2), (50000,5), (2,50000), (5,50000) |
| Espalhamento por categoria e separabilidade | `fig5_espalhamento_emg.png` e relatório | OK | Sobreposição explica a limitação linear |
| MQO, MQO regularizado e MQO polinomial | Três modelos manuais em NumPy | OK | Não foram inventados modelos gaussianos ou extras |
| Seleção de q por acurácia e tempo | q=1…6 em dez splits comuns | CORRIGIDO | Menor q até 0,5 p.p. do máximo; q final = 4 |
| Seleção adicional de lambda | Grade 0…1000 em dez splits comuns | CORRIGIDO | Escolha metodológica; lambda positivo final = 1000 |
| Monte Carlo 500 vezes, 80/20 | Seed 424242; mesmos splits para três modelos | OK | Acurácia com quatro estatísticas |
| Fronteiras exibem as cinco classes | Níveis de 0,5 até 5,5 | CORRIGIDO | Limite superior da classe 5 incluído |
| Padronização sem vazamento | Média/desvio ajustados apenas no treino em cada split | CORRIGIDO | Teste recebe parâmetros do treino; desvio zero é protegido |
| Funções, `main()` e criação de diretório | Ambos os scripts reorganizados | CORRIGIDO | Erros de dataset são explícitos |
| Nenhuma biblioteca de modelo pronto | Busca por imports proibidos e revisão de dependências | OK | Apenas NumPy, Matplotlib e biblioteca padrão |
| Revisão final e sincronização do relatório | Será realizada separadamente | PENDENTE | `relatorio.tex` não foi editado nesta rodada por solicitação expressa |
| Revisão/compilação do PDF após a alteração final do código | Será realizada separadamente | PENDENTE | `Relatorio_AV1.pdf` não foi alterado nem recompilado nesta rodada |

## Inconsistências do enunciado

- Um trecho menciona “5 modelos diferentes” para classificação, mas a definição e a tabela especificam apenas MQO tradicional, MQO regularizado e MQO polinomial. Foram implementados os três modelos explicitamente definidos.
- O arquivo real de classificação chama-se `EMG1.csv`; esse nome foi preservado mesmo que outras versões do enunciado usem denominação distinta.
- A não penalização do intercepto no Tikhonov é uma convenção metodológica da equipe, não uma exigência explícita do enunciado.
