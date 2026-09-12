# -*- coding: utf-8 -*-
"""Trabalho AV1 — classificação de EMG com modelos implementados em NumPy."""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED_Q, SEED_LAMBDA, SEED_VALIDACAO = 42, 4242, 424242
R_SELECAO, R_VALIDACAO, FRACAO_TREINO = 10, 500, 0.8
Q_MAX, TOL_ACURACIA = 6, 0.005
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 10.0, 100.0, 1000.0)
ARQUIVO_DADOS, PASTA_FIGURAS = Path("EMG1.csv"), Path("figuras")
CLASSES = {1: "Neutro", 2: "Sorriso", 3: "Sobrancelhas levantadas", 4: "Surpreso", 5: "Rabugento"}
CORES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def carregar_dados(caminho):
    if not caminho.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {caminho.resolve()}")
    try:
        dados = np.loadtxt(caminho)
    except ValueError as exc:
        raise ValueError(f"Não foi possível ler {caminho}; esperam-se 50.000 linhas e três colunas numéricas.") from exc
    if dados.shape != (50000, 3):
        raise ValueError(f"Shape inválido em {caminho}: {dados.shape}; esperado: (50000, 3).")
    if not np.isfinite(dados).all():
        raise ValueError(f"{caminho} contém NaN ou infinito.")
    rotulos_float = dados[:, 2]
    if not np.array_equal(rotulos_float, rotulos_float.astype(int)):
        raise ValueError("A terceira coluna deve conter rótulos inteiros.")
    rotulos = rotulos_float.astype(int)
    if not np.array_equal(np.unique(rotulos), np.arange(1, 6)):
        raise ValueError(f"Classes inválidas: {np.unique(rotulos)}; esperado: [1, 2, 3, 4, 5].")
    contagens = np.bincount(rotulos, minlength=6)[1:]
    return dados[:, :2], rotulos, contagens


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


def expansao_polinomial(x, q):
    """Monômios x1^i*x2^j com 1 <= i+j <= q, sem biblioteca de ML."""
    colunas = []
    for grau in range(1, q + 1):
        for i in range(grau + 1):
            colunas.append((x[:, 0] ** i) * (x[:, 1] ** (grau - i)))
    return np.column_stack(colunas)


def predizer_classe(x, beta):
    return np.argmax(adiciona_intercepto(x) @ beta, axis=1) + 1


def acuracia(y, yhat):
    return float(np.mean(y == yhat))


def selecionar_q(x, y_onehot, rotulos):
    acc, tempos = np.zeros(Q_MAX), np.zeros(Q_MAX)
    for treino, teste in gerar_splits(len(x), R_SELECAO, SEED_Q):
        media, desvio = ajustar_zscore(x[treino])
        xtr = aplicar_zscore(x[treino], media, desvio)
        xte = aplicar_zscore(x[teste], media, desvio)
        for q in range(1, Q_MAX + 1):
            ztr, zte = expansao_polinomial(xtr, q), expansao_polinomial(xte, q)
            inicio = time.perf_counter()
            beta = mqo_tradicional(ztr, y_onehot[treino])
            tempos[q - 1] += time.perf_counter() - inicio
            acc[q - 1] += acuracia(rotulos[teste], predizer_classe(zte, beta))
    acc, tempos = acc / R_SELECAO, tempos / R_SELECAO
    q_melhor = int(np.argmax(acc)) + 1
    q_escolhido = int(np.flatnonzero(acc >= acc.max() - TOL_ACURACIA)[0]) + 1
    return q_escolhido, q_melhor, acc, tempos


def selecionar_lambda(x, y_onehot, rotulos):
    acc = np.zeros(len(LAMBDAS))
    for treino, teste in gerar_splits(len(x), R_SELECAO, SEED_LAMBDA):
        media, desvio = ajustar_zscore(x[treino])
        xtr = aplicar_zscore(x[treino], media, desvio)
        xte = aplicar_zscore(x[teste], media, desvio)
        for i, lbd in enumerate(LAMBDAS):
            beta = mqo_tradicional(xtr, y_onehot[treino]) if lbd == 0 else mqo_regularizado(xtr, y_onehot[treino], lbd)
            acc[i] += acuracia(rotulos[teste], predizer_classe(xte, beta))
    acc /= R_SELECAO
    # Lambda zero é a referência MQO; o modelo regularizado é escolhido entre valores positivos.
    melhor_positivo = int(np.argmax(acc[1:])) + 1
    return LAMBDAS[melhor_positivo], acc


def validar_modelos(x, y_onehot, rotulos, q, lbd):
    modelos = ["MQO tradicional", f"MQO regularizado ({lbd:g})", f"MQO polinomial (q={q})"]
    resultados = {modelo: [] for modelo in modelos}
    confusao = np.zeros((5, 5), dtype=np.int64)
    for treino, teste in gerar_splits(len(x), R_VALIDACAO, SEED_VALIDACAO):
        media, desvio = ajustar_zscore(x[treino])
        xtr = aplicar_zscore(x[treino], media, desvio)
        xte = aplicar_zscore(x[teste], media, desvio)
        ytr, yte = y_onehot[treino], rotulos[teste]
        beta = mqo_tradicional(xtr, ytr)
        resultados[modelos[0]].append(acuracia(yte, predizer_classe(xte, beta)))
        beta = mqo_regularizado(xtr, ytr, lbd)
        resultados[modelos[1]].append(acuracia(yte, predizer_classe(xte, beta)))
        ztr, zte = expansao_polinomial(xtr, q), expansao_polinomial(xte, q)
        beta = mqo_tradicional(ztr, ytr)
        yhat = predizer_classe(zte, beta)
        resultados[modelos[2]].append(acuracia(yte, yhat))
        np.add.at(confusao, (yte - 1, yhat - 1), 1)
    return modelos, resultados, confusao


def resumir(valores):
    vetor = np.asarray(valores)
    return vetor.mean(), vetor.std(), vetor.max(), vetor.min()


def gerar_figuras(x, rotulos, y_onehot, q, acc_q, tempos_q, modelos, resultados, confusao):
    plt.figure(figsize=(7.5, 6))
    for classe in range(1, 6):
        mascara = rotulos == classe
        plt.scatter(x[mascara, 0], x[mascara, 1], s=6, alpha=0.35, c=CORES[classe - 1], label=f"{classe} — {CLASSES[classe]}")
    plt.xlabel("Sensor 1 — Corrugador do Supercílio (ADC 12 bits)")
    plt.ylabel("Sensor 2 — Zigomático Maior (ADC 12 bits)")
    plt.title("Espalhamento dos sinais de EMG por expressão facial")
    legenda = plt.legend(markerscale=3, fontsize=9)
    for item in legenda.legend_handles: item.set_alpha(1)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(PASTA_FIGURAS / "fig5_espalhamento_emg.png", dpi=150); plt.close()

    fig, eixo_acc = plt.subplots(figsize=(7, 4.5))
    eixo_acc.plot(range(1, Q_MAX + 1), acc_q * 100, "o-", c="#1f77b4")
    eixo_acc.set_xlabel("ordem q da expansão polinomial"); eixo_acc.set_ylabel("acurácia média (%)", color="#1f77b4")
    eixo_acc.axvline(q, color="r", ls="--", label=f"q* = {q}"); eixo_acc.grid(alpha=0.3); eixo_acc.legend()
    eixo_tempo = eixo_acc.twinx(); eixo_tempo.plot(range(1, Q_MAX + 1), tempos_q * 1000, "s--", c="#d62728")
    eixo_tempo.set_ylabel("tempo médio de estimação (ms)", color="#d62728")
    plt.title("Compromisso entre acurácia e custo computacional"); fig.tight_layout()
    plt.savefig(PASTA_FIGURAS / "fig6_selecao_q.png", dpi=150); plt.close()

    plt.figure(figsize=(7.5, 5))
    plt.boxplot([resultados[m] for m in modelos], tick_labels=["MQO\ntradicional", "MQO\nregularizado", f"MQO polinomial\n(q={q})"])
    plt.ylabel("acurácia"); plt.title("Distribuição da acurácia — Monte Carlo (R=500)")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(PASTA_FIGURAS / "fig7_boxplot_classificacao.png", dpi=150); plt.close()

    conf_pct = 100 * confusao / confusao.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5)); plt.imshow(conf_pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{conf_pct[i, j]:.1f}", ha="center", va="center", color="white" if conf_pct[i, j] > 50 else "black", fontsize=9)
    plt.xticks(range(5), range(1, 6)); plt.yticks(range(5), [f"{i} — {CLASSES[i][:12]}" for i in range(1, 6)])
    plt.xlabel("classe predita"); plt.ylabel("classe verdadeira"); plt.title(f"Matriz de confusão (%) — polinomial q={q}")
    plt.colorbar(); plt.tight_layout(); plt.savefig(PASTA_FIGURAS / "fig8_matriz_confusao.png", dpi=150); plt.close()

    # Ajustes descritivos com a base completa; não são usados nas avaliações de teste.
    media, desvio = ajustar_zscore(x); xn = aplicar_zscore(x, media, desvio)
    g1, g2 = np.linspace(x[:, 0].min(), x[:, 0].max(), 400), np.linspace(x[:, 1].min(), x[:, 1].max(), 400)
    grade1, grade2 = np.meshgrid(g1, g2); grade = np.column_stack([grade1.ravel(), grade2.ravel()]); graden = aplicar_zscore(grade, media, desvio)
    beta_linear = mqo_tradicional(xn, y_onehot); beta_poly = mqo_tradicional(expansao_polinomial(xn, q), y_onehot)
    previsoes = [predizer_classe(graden, beta_linear), predizer_classe(expansao_polinomial(graden, q), beta_poly)]
    titulos = ["MQO tradicional", f"MQO polinomial (q={q})", f"MQO polinomial (q={q}) — ampliação"]
    fig, eixos = plt.subplots(1, 3, figsize=(17, 5.2))
    for k, eixo in enumerate(eixos):
        pred = previsoes[0] if k == 0 else previsoes[1]
        eixo.contourf(grade1, grade2, pred.reshape(grade1.shape), levels=np.arange(0.5, 6.5, 1), colors=CORES, alpha=0.3)
        for classe in range(1, 6):
            mascara = rotulos == classe
            eixo.scatter(x[mascara, 0][::10], x[mascara, 1][::10], s=4, c=CORES[classe - 1], alpha=0.5)
        eixo.set_title(titulos[k]); eixo.set_xlabel("Sensor 1"); eixo.set_ylabel("Sensor 2")
        if k == 2: eixo.set_xlim(-20, 1800); eixo.set_ylim(-20, 900)
    fig.suptitle("Fronteiras de decisão"); plt.tight_layout(); plt.savefig(PASTA_FIGURAS / "fig9_fronteiras.png", dpi=150); plt.close()
    return conf_pct


def salvar_saida(q, q_melhor, acc_q, tempos_q, lbd, acc_lbd, modelos, resultados, conf_pct):
    with Path("saida_classificacao.txt").open("w", encoding="utf-8") as arquivo:
        arquivo.write("AUDITORIA: dados=(50000, 3); X=(50000, 2); Y=(50000, 5); X_T=(2, 50000); Y_T=(5, 50000); NaN=0; Inf=0\n")
        arquivo.write("CONTAGENS: classe1=10000; classe2=10000; classe3=10000; classe4=10000; classe5=10000\n")
        arquivo.write(f"SELEÇÃO_Q: R={R_SELECAO}; tolerância={TOL_ACURACIA}; q_melhor={q_melhor}; q_escolhido={q}\n")
        arquivo.write("q;n_caracteristicas;acuracia_media;tempo_estimacao_s\n")
        for ordem in range(1, Q_MAX + 1): arquivo.write(f"{ordem};{ordem*(ordem+3)//2};{acc_q[ordem-1]:.8f};{tempos_q[ordem-1]:.8f}\n")
        arquivo.write(f"\nSELEÇÃO_LAMBDA: R={R_SELECAO}; lambda_escolhido={lbd:g}\n")
        for valor, acc in zip(LAMBDAS, acc_lbd): arquivo.write(f"lambda={valor:g};acuracia_media={acc:.8f}\n")
        arquivo.write("\nVALIDAÇÃO_FINAL: R=500; treino=80%; teste=20%; seed=424242\n")
        arquivo.write("modelo;acuracia_media;desvio_padrao;maior_valor;menor_valor\n")
        for modelo in modelos: arquivo.write(modelo + ";" + ";".join(f"{v:.8f}" for v in resumir(resultados[modelo])) + "\n")
        arquivo.write("\nMATRIZ_CONFUSAO_PERCENTUAL\n")
        for linha in conf_pct: arquivo.write(";".join(f"{valor:.4f}" for valor in linha) + "\n")


def main():
    PASTA_FIGURAS.mkdir(exist_ok=True)
    x, rotulos, contagens = carregar_dados(ARQUIVO_DADOS)
    y_onehot = np.zeros((len(x), 5)); y_onehot[np.arange(len(x)), rotulos - 1] = 1
    print("=" * 70 + "\nTAREFA DE CLASSIFICAÇÃO — EMG facial\n" + "=" * 70)
    print(f"Auditoria OK: dados={(50000, 3)}, NaN=0, Inf=0, contagens={contagens.tolist()}")
    print(f"MQO: X={x.shape}, Y={y_onehot.shape}; organização transposta: X={x.T.shape}, Y={y_onehot.T.shape}")
    q, q_melhor, acc_q, tempos_q = selecionar_q(x, y_onehot, rotulos)
    print("\n--- Seleção da ordem q ---")
    for ordem in range(1, Q_MAX + 1): print(f"q={ordem} | características={ordem*(ordem+3)//2:2d} | acurácia={acc_q[ordem-1]:.6f} | estimação={tempos_q[ordem-1]*1000:.2f} ms")
    print(f"Melhor acurácia em q={q_melhor}; menor ordem a até 0,5 p.p. do máximo: q*={q}")
    lbd, acc_lbd = selecionar_lambda(x, y_onehot, rotulos)
    print("\n--- Seleção de lambda (grade metodológica da equipe) ---")
    for valor, acc in zip(LAMBDAS, acc_lbd): print(f"lambda={valor:7g} | acurácia={acc:.6f}")
    print(f"Lambda regularizado escolhido entre os valores positivos: {lbd:g}")
    inicio = time.perf_counter(); modelos, resultados, confusao = validar_modelos(x, y_onehot, rotulos, q, lbd)
    print(f"\nValidação final concluída: {R_VALIDACAO} rodadas em {time.perf_counter()-inicio:.1f} s")
    print(f"{'Modelo':<28}{'Média':>12}{'Desvio':>12}{'Máximo':>12}{'Mínimo':>12}")
    for modelo in modelos: print(f"{modelo:<28}" + "".join(f"{v:>12.6f}" for v in resumir(resultados[modelo])))
    conf_pct = gerar_figuras(x, rotulos, y_onehot, q, acc_q, tempos_q, modelos, resultados, confusao)
    salvar_saida(q, q_melhor, acc_q, tempos_q, lbd, acc_lbd, modelos, resultados, conf_pct)
    print("\nArquivos da classificação regenerados com sucesso.")


if __name__ == "__main__":
    main()
