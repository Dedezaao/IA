# -*- coding: utf-8 -*-
"""Trabalho AV1 — regressão do PIB da China com modelos implementados em NumPy."""

from pathlib import Path
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED_SELECAO, SEED_VALIDACAO = 42, 4242
R_PODA, R_VALIDACAO, FRACAO_TREINO = 100, 500, 0.8
Q_MAX, TOL_R2 = 12, 0.005
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ARQUIVO_DADOS, PASTA_FIGURAS = Path("china_gdp.csv"), Path("figuras")


def carregar_dados(caminho):
    """Lê e valida o dataset sem alterá-lo."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {caminho.resolve()}")
    try:
        dados = np.loadtxt(caminho, delimiter=",", skiprows=1)
    except ValueError as exc:
        raise ValueError(f"Não foi possível ler {caminho}; esperam-se cabeçalho e duas colunas numéricas.") from exc
    if dados.shape != (55, 2):
        raise ValueError(f"Shape inválido em {caminho}: {dados.shape}; esperado: (55, 2).")
    if not np.isfinite(dados).all():
        raise ValueError(f"{caminho} contém NaN ou infinito.")
    anos = dados[:, 0]
    if not np.array_equal(anos, np.arange(1960, 2015)):
        raise ValueError("A primeira coluna deve conter, em ordem, todos os anos de 1960 a 2014.")
    return anos, dados[:, 1]


def ajustar_zscore(x):
    media, desvio = np.mean(x, axis=0), np.std(x, axis=0)
    return media, np.where(desvio == 0, 1.0, desvio)


def aplicar_zscore(x, media, desvio):
    return (x - media) / desvio


def gerar_splits(n, repeticoes, seed):
    rng, corte = np.random.default_rng(seed), int(FRACAO_TREINO * n)
    return [(idx[:corte], idx[corte:]) for idx in (rng.permutation(n) for _ in range(repeticoes))]


def adiciona_intercepto(x):
    return np.hstack((np.ones((x.shape[0], 1)), x))


def mqo_tradicional(x, y):
    """Estima o MQO pelas equações normais montadas explicitamente."""
    xb = adiciona_intercepto(x)
    gram = xb.T @ xb
    rhs = xb.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def mqo_regularizado(x, y, lbd):
    """Tikhonov manual; por convenção, o intercepto não é penalizado."""
    xb = adiciona_intercepto(x)
    penalidade = np.identity(xb.shape[1])
    penalidade[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + lbd * penalidade, xb.T @ y)


def matriz_polinomial(x, q):
    return np.hstack([x ** k for k in range(1, q + 1)])


def mqo_polinomial(x, y, q):
    return mqo_tradicional(matriz_polinomial(x, q), y)


def predizer(x, beta):
    return adiciona_intercepto(x) @ beta


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


def r2(y, yhat):
    sq_res = float(np.sum((y - yhat) ** 2))
    sq_tot = float(np.sum((y - np.mean(y)) ** 2))
    if sq_tot <= np.finfo(float).eps:
        return 1.0 if sq_res <= np.finfo(float).eps else 0.0
    return 1.0 - sq_res / sq_tot


def selecionar_q(anos, y):
    r2_treino, r2_teste = np.zeros(Q_MAX), np.zeros(Q_MAX)
    for treino, teste in gerar_splits(len(anos), R_PODA, SEED_SELECAO):
        media, desvio = ajustar_zscore(anos[treino])
        xtr = aplicar_zscore(anos[treino], media, desvio).reshape(-1, 1)
        xte = aplicar_zscore(anos[teste], media, desvio).reshape(-1, 1)
        for q in range(1, Q_MAX + 1):
            beta = mqo_polinomial(xtr, y[treino], q)
            r2_treino[q - 1] += r2(y[treino], predizer(matriz_polinomial(xtr, q), beta))
            r2_teste[q - 1] += r2(y[teste], predizer(matriz_polinomial(xte, q), beta))
    r2_treino, r2_teste = r2_treino / R_PODA, r2_teste / R_PODA
    q_bruto = int(np.argmax(r2_teste)) + 1
    q_escolhido = int(np.flatnonzero(r2_teste >= r2_teste.max() - TOL_R2)[0]) + 1
    return q_escolhido, q_bruto, r2_treino, r2_teste


def validar_modelos(anos, y, q):
    modelos = ["Polinomial", "MQO tradicional"] + [f"MQO regularizado {lbd:g}" for lbd in LAMBDAS[1:]]
    resultados_mse = {modelo: [] for modelo in modelos}
    resultados_r2 = {modelo: [] for modelo in modelos}
    for treino, teste in gerar_splits(len(anos), R_VALIDACAO, SEED_VALIDACAO):
        media, desvio = ajustar_zscore(anos[treino])
        xtr = aplicar_zscore(anos[treino], media, desvio).reshape(-1, 1)
        xte = aplicar_zscore(anos[teste], media, desvio).reshape(-1, 1)
        ytr, yte = y[treino], y[teste]
        beta = mqo_polinomial(xtr, ytr, q)
        yhat = predizer(matriz_polinomial(xte, q), beta)
        resultados_mse[modelos[0]].append(mse(yte, yhat))
        resultados_r2[modelos[0]].append(r2(yte, yhat))
        beta = mqo_tradicional(xtr, ytr)
        yhat = predizer(xte, beta)
        resultados_mse[modelos[1]].append(mse(yte, yhat))
        resultados_r2[modelos[1]].append(r2(yte, yhat))
        for posicao, lbd in enumerate(LAMBDAS[1:], start=2):
            beta = mqo_regularizado(xtr, ytr, lbd)
            yhat = predizer(xte, beta)
            resultados_mse[modelos[posicao]].append(mse(yte, yhat))
            resultados_r2[modelos[posicao]].append(r2(yte, yhat))
    return modelos, resultados_mse, resultados_r2


def resumir(valores):
    vetor = np.asarray(valores)
    return vetor.mean(), vetor.std(), vetor.max(), vetor.min()


def salvar_csv(caminho, modelos, resultados):
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["modelo", "media", "desvio_padrao", "maior_valor", "menor_valor"])
        for modelo in modelos:
            escritor.writerow([modelo, *[f"{v:.12e}" for v in resumir(resultados[modelo])]])


def imprimir_tabela(modelos, resultados, titulo, notacao):
    print("\n" + "=" * 94 + f"\n{titulo}\n" + "=" * 94)
    print(f"{'Modelo':<28}{'Média':>16}{'Desvio-padrão':>18}{'Maior valor':>16}{'Menor valor':>16}")
    print("-" * 94)
    for modelo in modelos:
        texto = "".join(format(v, notacao).rjust(16) for v in resumir(resultados[modelo]))
        print(f"{modelo:<28}{texto}")


def gerar_figuras(anos, pib, y, q, r2_treino, r2_teste, modelos, resultados_mse, resultados_r2):
    plt.figure(figsize=(7, 4.5))
    plt.scatter(anos, pib / 1e12, s=35, c="#1f77b4", edgecolors="k", linewidths=0.4)
    plt.xlabel("Ano"); plt.ylabel("PIB (trilhões de USD)")
    plt.title("Gráfico de espalhamento — PIB da China (1960–2014)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / "fig1_espalhamento.png", dpi=150); plt.close()

    # Ajustes descritivos com a base completa; não são usados nas avaliações de teste.
    media, desvio = ajustar_zscore(anos)
    x = aplicar_zscore(anos, media, desvio).reshape(-1, 1)
    grade = np.linspace(anos.min(), anos.max(), 400)
    grade_x = aplicar_zscore(grade, media, desvio).reshape(-1, 1)
    plt.figure(figsize=(7.5, 5))
    plt.scatter(anos, pib / 1e12, s=30, c="k", label="dados observados", zorder=3)
    plt.plot(grade, predizer(grade_x, mqo_tradicional(x, y)).ravel() / 1e12, lw=2, label="MQO tradicional")
    for lbd in LAMBDAS[1:]:
        beta = mqo_regularizado(x, y, lbd)
        plt.plot(grade, predizer(grade_x, beta).ravel() / 1e12, lw=1, ls="--", label=f"Tikhonov $\\lambda$={lbd:g}")
    beta_poly = mqo_polinomial(x, y, q)
    plt.plot(grade, predizer(matriz_polinomial(grade_x, q), beta_poly).ravel() / 1e12,
             lw=2.5, c="crimson", label=f"Polinomial q={q}")
    plt.xlabel("Ano"); plt.ylabel("PIB (trilhões de USD)"); plt.title("Modelos ajustados à base completa")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / "fig2_modelos_ajustados.png", dpi=150); plt.close()

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.3))
    for eixo in eixos:
        eixo.plot(range(1, Q_MAX + 1), r2_treino, "o-", label="$R^2$ treino")
        eixo.plot(range(1, Q_MAX + 1), r2_teste, "s-", label="$R^2$ validação")
        eixo.axvline(q, color="r", ls="--", label=f"q* = {q}")
        eixo.set_xlabel("ordem q do polinômio"); eixo.set_ylabel(f"$R^2$ médio ({R_PODA} reamostragens)")
        eixo.grid(alpha=0.3); eixo.legend(fontsize=8, loc="lower right")
    eixos[0].set_title("Visão geral")
    eixos[1].set_title("Ampliação da região de saturação"); eixos[1].set_ylim(0.955, 1.001); eixos[1].set_xlim(3.5, Q_MAX + 0.3)
    fig.suptitle("Seleção da ordem do polinômio por poda"); plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / "fig3_poda_q.png", dpi=150); plt.close()

    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    eixos[0].boxplot([np.asarray(resultados_mse[m]) / 1e24 for m in modelos], tick_labels=range(1, 7))
    eixos[0].set_title("MSE por rodada ($\\times 10^{24}$)"); eixos[0].set_yscale("log")
    eixos[1].boxplot([resultados_r2[m] for m in modelos], tick_labels=range(1, 7)); eixos[1].set_title("$R^2$ por rodada")
    for eixo in eixos: eixo.set_xlabel("modelo"); eixo.grid(alpha=0.3)
    fig.suptitle("Random Subsampling (R=500) | 1: Polinomial; 2: MQO; 3–6: Tikhonov")
    plt.tight_layout(); plt.savefig(PASTA_FIGURAS / "fig4_boxplots_regressao.png", dpi=150); plt.close()


def salvar_saida(q, q_bruto, r2_treino, r2_teste, modelos, resultados_mse, resultados_r2):
    with Path("saida_regressao.txt").open("w", encoding="utf-8") as arquivo:
        arquivo.write("AUDITORIA: shape=(55, 2); anos=1960-2014; NaN=0; Inf=0\n")
        arquivo.write(f"SELEÇÃO_Q: R={R_PODA}; tolerância={TOL_R2}; q_melhor={q_bruto}; q_escolhido={q}\n")
        arquivo.write("q;R2_treino_medio;R2_validacao_medio\n")
        for ordem in range(1, Q_MAX + 1):
            arquivo.write(f"{ordem};{r2_treino[ordem-1]:.8f};{r2_teste[ordem-1]:.8f}\n")
        arquivo.write("\nVALIDAÇÃO_FINAL: R=500; treino=80%; teste=20%; seed=4242\n")
        arquivo.write("modelo;MSE_media;MSE_desvio;MSE_max;MSE_min;R2_media;R2_desvio;R2_max;R2_min\n")
        for modelo in modelos:
            arquivo.write(modelo + ";" + ";".join(f"{v:.12e}" for v in (*resumir(resultados_mse[modelo]), *resumir(resultados_r2[modelo]))) + "\n")


def main():
    PASTA_FIGURAS.mkdir(exist_ok=True)
    anos, pib = carregar_dados(ARQUIVO_DADOS)
    y = pib.reshape(-1, 1)
    print("=" * 70 + "\nTAREFA DE REGRESSÃO — PIB da China\n" + "=" * 70)
    print(f"Auditoria OK: shape=(55, 2), período={int(anos.min())}–{int(anos.max())}, NaN=0, Inf=0")
    print(f"Dimensões organizadas: X={(len(anos), 1)}; y={y.shape}")
    q, q_bruto, r2_treino, r2_teste = selecionar_q(anos, y)
    print("\n--- Seleção de q por poda com R² ---")
    for ordem in range(1, Q_MAX + 1):
        print(f"q={ordem:2d} | R² treino={r2_treino[ordem-1]: .6f} | R² validação={r2_teste[ordem-1]: .6f}")
    print(f"Maior R² médio: q={q_bruto}; menor ordem a até {TOL_R2} do máximo: q*={q}")
    inicio = time.perf_counter()
    modelos, resultados_mse, resultados_r2 = validar_modelos(anos, y, q)
    print(f"\nValidação final concluída: {R_VALIDACAO} rodadas em {time.perf_counter()-inicio:.2f} s")
    imprimir_tabela(modelos, resultados_mse, "MÉTRICA: MSE no conjunto de teste", ".4e")
    imprimir_tabela(modelos, resultados_r2, "MÉTRICA: R² no conjunto de teste", ".6f")
    salvar_csv(Path("resultados_mse.csv"), modelos, resultados_mse)
    salvar_csv(Path("resultados_r2.csv"), modelos, resultados_r2)
    salvar_saida(q, q_bruto, r2_treino, r2_teste, modelos, resultados_mse, resultados_r2)
    gerar_figuras(anos, pib, y, q, r2_treino, r2_teste, modelos, resultados_mse, resultados_r2)
    print("\nArquivos da regressão regenerados com sucesso.")


if __name__ == "__main__":
    main()
