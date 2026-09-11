ENTREGAVEIS
-----------
Relatorio_AV1_IEEE.pdf     relatorio final compilado (modelo IEEE conference, 8 paginas)
relatorio_latex.zip        projeto LaTeX completo (relatorio.tex + IEEEtran.cls + figuras/)
regressao.py               Tarefa de Regressao (itens 1 a 7)
classificacao.py           Tarefa de Classificacao (itens 1 a 6)
figuras/                   as 9 figuras em PNG
log_*.txt / saida_*.txt    saida de console e resultados numericos brutos
Relatorio_AV1_IA_Computacional.docx/.pdf   versao anterior em Word (opcional)

COMO EDITAR O RELATORIO
-----------------------
Opcao A (Overleaf): faca upload de relatorio_latex.zip, abra relatorio.tex,
preencha os nomes/e-mails dos autores e compile com pdfLaTeX.
No Overleaf o babel existe: descomente \usepackage[brazil]{babel} no preambulo
e apague os \renewcommand de rotulos logo abaixo (instrucao esta no arquivo).

Opcao B (local): pdflatex relatorio.tex  (rodar duas vezes para as referencias).

COMO EXECUTAR OS CODIGOS
------------------------
1) Coloque china_gdp.csv e EMG1.csv na mesma pasta dos scripts.
2) Crie a subpasta "figuras".
3) python regressao.py  /  python classificacao.py

Dependencias: numpy e matplotlib apenas (nenhuma biblioteca com modelo pronto).
Semente aleatoria fixa (42) -> os numeros do relatorio sao reproduziveis.
