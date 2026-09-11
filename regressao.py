# -*- coding: utf-8 -*-
"""
Inteligencia Artificial Computacional - Trabalho AV1
TAREFA DE REGRESSAO - china_gdp.csv

Modelos implementados do zero (apenas numpy para algebra linear):
    - MQO tradicional (minimos quadrados ordinarios)
    - MQO regularizado (Tikhonov / Ridge) com lambda = {0, 0.25, 0.5, 0.75, 1}
    - Regressao Polinomial via MQO (ordem q definida por poda com metrica R2)

Validacao: Random Subsampling Validation, R = 500 rodadas, 80% treino / 20% teste.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
FIG = "figuras/"

# =====================================================================
# 1. LEITURA E VISUALIZACAO INICIAL DOS DADOS
# =====================================================================
dados = np.loadtxt("china_gdp.csv", delimiter=",", skiprows=1)
anos = dados[:, 0]
pib = dados[:, 1]
N = dados.shape[0]

print("=" * 70)
print("TAREFA DE REGRESSAO - PIB da China")
print("=" * 70)
print(f"Numero de amostras N = {N}  |  periodo: {int(anos.min())}-{int(anos.max())}")
print(f"PIB minimo  = {pib.min():.4e} USD   ({int(anos[np.argmin(pib)])})")
print(f"PIB maximo  = {pib.max():.4e} USD   ({int(anos[np.argmax(pib)])})")

plt.figure(figsize=(7, 4.5))
plt.scatter(anos, pib / 1e12, s=35, c="#1f77b4", edgecolors="k", linewidths=0.4)
plt.xlabel("Ano")
plt.ylabel("PIB (trilhões de USD)")
plt.title("Gráfico de espalhamento – PIB da China (1960–2014)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG + "fig1_espalhamento.png", dpi=150)
plt.close()

# =====================================================================
# 2. ORGANIZACAO DOS DADOS: X (N x p) e y (N x 1)
# =====================================================================
# p = 1 variavel regressora (ano). Padronizacao do regressor (z-score) para
# condicionamento numerico: ano^8 com ano ~ 2000 ultrapassa 1e26 e destroi a
# solucao do sistema normal. A padronizacao e' uma transformacao afim, logo
# nao altera o espaco de funcoes gerado pelo polinomio.
mu_x, sd_x = anos.mean(), anos.std()
X = ((anos - mu_x) / sd_x).reshape(N, 1)   # R^{N x p}, p = 1
y = pib.reshape(N, 1)                      # R^{N x 1}
print(f"\nDimensoes organizadas -> X: {X.shape}   y: {y.shape}")


# =====================================================================
# 3. IMPLEMENTACAO DOS MODELOS (do zero)
# =====================================================================
def adiciona_intercepto(X):
    """Concatena a coluna de 1's (termo de intercepto/bias)."""
    return np.hstack((np.ones((X.shape[0], 1)), X))


def mqo_tradicional(X, y):
    """beta = (X^T X)^-1 X^T y, com X ja contendo a coluna de 1's."""
    Xb = adiciona_intercepto(X)
    return np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y


def mqo_regularizado(X, y, lbd):
    """Tikhonov: beta = (X^T X + lambda*I)^-1 X^T y (intercepto nao penalizado)."""
    Xb = adiciona_intercepto(X)
    I = np.identity(Xb.shape[1])
    I[0, 0] = 0.0                      # nao penaliza o intercepto
    return np.linalg.pinv(Xb.T @ Xb + lbd * I) @ Xb.T @ y


def matriz_polinomial(x, q):
    """Constroi [x, x^2, ..., x^q] (o intercepto e' acrescentado depois)."""
    return np.hstack([x ** k for k in range(1, q + 1)])


def mqo_polinomial(X, y, q, lbd=0.0):
    """Regressao polinomial de ordem q estimada via MQO."""
    Z = matriz_polinomial(X, q)
    return mqo_regularizado(Z, y, lbd) if lbd > 0 else mqo_tradicional(Z, y)


def predizer(X, beta):
    return adiciona_intercepto(X) @ beta


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


def r2(y, yhat):
    sq_res = np.sum((y - yhat) ** 2)
    sq_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - sq_res / sq_tot)


# =====================================================================
# 4. DEFINICAO DA ORDEM q DO POLINOMIO (ESTRATEGIA DE PODA VIA R2)
# =====================================================================
Q_MAX = 12
R_PODA = 100          # rodadas de reamostragem usadas so' para escolher q
TOL = 0.005           # tolerancia de perda de R2 aceita na poda

r2_tr = np.zeros(Q_MAX)
r2_te = np.zeros(Q_MAX)

for r in range(R_PODA):
    idx = rng.permutation(N)
    corte = int(0.8 * N)
    tr, te = idx[:corte], idx[corte:]
    for q in range(1, Q_MAX + 1):
        b = mqo_polinomial(X[tr], y[tr], q)
        r2_tr[q - 1] += r2(y[tr], predizer(matriz_polinomial(X[tr], q), b))
        r2_te[q - 1] += r2(y[te], predizer(matriz_polinomial(X[te], q), b))

r2_tr /= R_PODA
r2_te /= R_PODA

print("\n--- Escolha da ordem q (poda com metrica R2) ---")
print(f"{'q':>3} {'R2 treino':>12} {'R2 teste':>12}")
for q in range(1, Q_MAX + 1):
    print(f"{q:>3} {r2_tr[q-1]:>12.5f} {r2_te[q-1]:>12.5f}")

q_otimo_bruto = int(np.argmax(r2_te)) + 1
# Poda: parte do melhor q e reduz a ordem enquanto a perda de R2 for irrelevante
q_star = q_otimo_bruto
for q in range(1, q_otimo_bruto + 1):
    if r2_te[q - 1] >= r2_te[q_otimo_bruto - 1] - TOL:
        q_star = q
        break
print(f"\nMaior R2 de teste em q = {q_otimo_bruto} (R2 = {r2_te[q_otimo_bruto-1]:.5f})")
print(f"Ordem escolhida apos a poda (tolerancia {TOL}): q* = {q_star}"
      f" (R2 = {r2_te[q_star-1]:.5f})")

fig, axp = plt.subplots(1, 2, figsize=(11, 4.3))
for k, a in enumerate(axp):
    a.plot(range(1, Q_MAX + 1), r2_tr, "o-", label="$R^2$ treino")
    a.plot(range(1, Q_MAX + 1), r2_te, "s-", label="$R^2$ teste")
    a.axvline(q_star, color="r", ls="--", label=f"q* = {q_star} (poda)")
    a.set_xlabel("ordem q do polinômio")
    a.set_ylabel("$R^2$ médio (100 reamostragens)")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="lower right")
axp[0].set_title("Visão geral")
axp[1].set_title("Ampliação da região de saturação")
axp[1].set_ylim(0.955, 1.001)
axp[1].set_xlim(3.5, Q_MAX + 0.3)
fig.suptitle("Seleção da ordem do polinômio por poda")
plt.tight_layout()
plt.savefig(FIG + "fig3_poda_q.png", dpi=150)
plt.close()

# =====================================================================
# 5. ESTIMATIVAS DE beta COM TODOS OS DADOS (6 estimativas)
# =====================================================================
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
print("\n--- Vetores beta estimados com a base completa ---")
beta_trad = mqo_tradicional(X, y)
print(f"MQO tradicional        : b0={beta_trad[0,0]:.6e}  b1={beta_trad[1,0]:.6e}")
for lbd in LAMBDAS[1:]:
    b = mqo_regularizado(X, y, lbd)
    print(f"MQO Tikhonov (l={lbd:<4}) : b0={b[0,0]:.6e}  b1={b[1,0]:.6e}")
beta_poly = mqo_polinomial(X, y, q_star)
print(f"Polinomial (q={q_star})        : " +
      "  ".join(f"b{k}={beta_poly[k,0]:.3e}" for k in range(beta_poly.shape[0])))

# Curvas ajustadas
grid = np.linspace(anos.min(), anos.max(), 400)
grid_s = ((grid - mu_x) / sd_x).reshape(-1, 1)
plt.figure(figsize=(7.5, 5))
plt.scatter(anos, pib / 1e12, s=30, c="k", label="dados observados", zorder=3)
plt.plot(grid, predizer(grid_s, beta_trad).ravel() / 1e12, lw=2,
         label="MQO tradicional")
for lbd in LAMBDAS[1:]:
    b = mqo_regularizado(X, y, lbd)
    plt.plot(grid, predizer(grid_s, b).ravel() / 1e12, lw=1, ls="--",
             label=f"Tikhonov $\\lambda$={lbd}")
plt.plot(grid, predizer(matriz_polinomial(grid_s, q_star), beta_poly).ravel() / 1e12,
         lw=2.5, c="crimson", label=f"Polinomial q={q_star}")
plt.xlabel("Ano")
plt.ylabel("PIB (trilhões de USD)")
plt.title("Modelos ajustados à base completa")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG + "fig2_modelos_ajustados.png", dpi=150)
plt.close()

# Sensibilidade do lambda (justifica a discussao do relatorio)
print("\n--- Sensibilidade ao lambda (norma do desvio em relacao ao MQO) ---")
for lbd in LAMBDAS[1:]:
    b = mqo_regularizado(X, y, lbd)
    dif = np.linalg.norm(b - beta_trad) / np.linalg.norm(beta_trad)
    print(f"lambda = {lbd:<5} -> desvio relativo de beta = {dif:.3e}")

# =====================================================================
# 6. RANDOM SUBSAMPLING VALIDATION (R = 500, 80/20)
# =====================================================================
R = 500
modelos = [f"Polinomial (q={q_star})", "MQO tradicional"] + \
          [f"MQO regularizado ({l})" for l in LAMBDAS[1:]]
MSE = {m: [] for m in modelos}
R2 = {m: [] for m in modelos}

t0 = time.time()
for r in range(R):
    idx = rng.permutation(N)
    corte = int(0.8 * N)
    tr, te = idx[:corte], idx[corte:]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    # Polinomial
    b = mqo_polinomial(Xtr, ytr, q_star)
    yh = predizer(matriz_polinomial(Xte, q_star), b)
    MSE[modelos[0]].append(mse(yte, yh)); R2[modelos[0]].append(r2(yte, yh))

    # MQO tradicional
    b = mqo_tradicional(Xtr, ytr)
    yh = predizer(Xte, b)
    MSE[modelos[1]].append(mse(yte, yh)); R2[modelos[1]].append(r2(yte, yh))

    # MQO regularizado
    for j, lbd in enumerate(LAMBDAS[1:]):
        b = mqo_regularizado(Xtr, ytr, lbd)
        yh = predizer(Xte, b)
        nome = modelos[2 + j]
        MSE[nome].append(mse(yte, yh)); R2[nome].append(r2(yte, yh))

print(f"\nValidacao concluida: R = {R} rodadas em {time.time()-t0:.2f} s")

# =====================================================================
# 7. TABELAS DE RESULTADOS
# =====================================================================
def tabela(dic, titulo, casas="e"):
    print("\n" + "=" * 92)
    print(titulo)
    print("=" * 92)
    cab = f"{'Modelo':<26}{'Media':>16}{'Desvio-Padrao':>16}{'Maior Valor':>16}{'Menor Valor':>16}"
    print(cab)
    print("-" * 92)
    linhas = []
    for m in modelos:
        v = np.array(dic[m])
        vals = [v.mean(), v.std(), v.max(), v.min()]
        if casas == "e":
            txt = "".join(f"{x:>16.4e}" for x in vals)
        else:
            txt = "".join(f"{x:>16.4f}" for x in vals)
        print(f"{m:<26}" + txt)
        linhas.append([m] + vals)
    return linhas


lin_mse = tabela(MSE, "METRICA: MSE (erro quadratico medio no conjunto de teste)")
lin_r2 = tabela(R2, "METRICA: R2 (coeficiente de determinacao no conjunto de teste)", casas="f")

np.savetxt("resultados_mse.csv",
           np.array([[l[1], l[2], l[3], l[4]] for l in lin_mse]),
           delimiter=",", header="media,desvio,maior,menor", comments="")
np.savetxt("resultados_r2.csv",
           np.array([[l[1], l[2], l[3], l[4]] for l in lin_r2]),
           delimiter=",", header="media,desvio,maior,menor", comments="")

# Boxplots
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].boxplot([np.array(MSE[m]) / 1e24 for m in modelos], tick_labels=[str(i) for i in range(1,7)])
ax[0].set_title("MSE por rodada ($\\times 10^{24}$)")
ax[0].set_xlabel("modelo")
ax[0].set_yscale("log")
ax[1].boxplot([R2[m] for m in modelos], tick_labels=[str(i) for i in range(1,7)])
ax[1].set_title("$R^2$ por rodada")
ax[1].set_xlabel("modelo")
for a in ax:
    a.grid(alpha=0.3)
fig.suptitle("Random Subsampling Validation (R=500) | 1:Polinomial 2:MQO "
             "3-6:Tikhonov (0.25, 0.5, 0.75, 1)")
plt.tight_layout()
plt.savefig(FIG + "fig4_boxplots_regressao.png", dpi=150)
plt.close()

with open("saida_regressao.txt", "w") as f:
    f.write(f"q* = {q_star}\n")
    for m in modelos:
        v, w = np.array(MSE[m]), np.array(R2[m])
        f.write(f"{m};{v.mean():.6e};{v.std():.6e};{v.max():.6e};{v.min():.6e};"
                f"{w.mean():.6f};{w.std():.6f};{w.max():.6f};{w.min():.6f}\n")
print("\nArquivos gerados em figuras/ e saida_regressao.txt")
