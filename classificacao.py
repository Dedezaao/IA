# -*- coding: utf-8 -*-
"""
Inteligencia Artificial Computacional - Trabalho AV1
TAREFA DE CLASSIFICACAO - sinais de EMG facial (EMG1.csv)

Sensor 1: Corrugador do Supercilio | Sensor 2: Zigomatico Maior
Classes: 1-Neutro 2-Sorriso 3-Sobrancelhas levantadas 4-Surpreso 5-Rabugento

Modelos implementados do zero (apenas numpy):
    - MQO tradicional (regressao sobre codificacao one-hot + argmax)
    - MQO regularizado (Tikhonov)
    - MQO polinomial (expansao polinomial de ordem q + MQO)

Validacao: Monte Carlo (amostragem aleatoria), R = 500, 80% treino / 20% teste.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
FIG = "figuras/"
CLASSES = {1: "Neutro", 2: "Sorriso", 3: "Sobrancelhas levantadas",
           4: "Surpreso", 5: "Rabugento"}

# =====================================================================
# 1. ORGANIZACAO DOS DADOS
# =====================================================================
dados = np.loadtxt("EMG1.csv")          # arquivo com N linhas e 3 colunas
X = dados[:, 0:2]                       # R^{N x p}
rotulos = dados[:, 2].astype(int)
N, p = X.shape
C = len(CLASSES)

# Y one-hot em R^{N x C} (modelos via MQO)
Y = np.zeros((N, C))
Y[np.arange(N), rotulos - 1] = 1

# Organizacao alternativa exigida pelos modelos gaussianos bayesianos
X_bayes = X.T                           # R^{p x N}
Y_bayes = Y.T                           # R^{C x N}

print("=" * 70)
print("TAREFA DE CLASSIFICACAO - EMG facial")
print("=" * 70)
print(f"X (MQO)   : {X.shape}      Y (MQO)   : {Y.shape}")
print(f"X (bayes) : {X_bayes.shape}   Y (bayes) : {Y_bayes.shape}")
print(f"N = {N}, p = {p}, C = {C}")
for c in range(1, C + 1):
    m = rotulos == c
    print(f"  classe {c} ({CLASSES[c]:<24}): {m.sum():>6} amostras | "
          f"sensor1 media={X[m,0].mean():7.1f} sd={X[m,0].std():6.1f} | "
          f"sensor2 media={X[m,1].mean():7.1f} sd={X[m,1].std():6.1f}")

# =====================================================================
# 2. VISUALIZACAO INICIAL (espalhamento por categoria)
# =====================================================================
cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
plt.figure(figsize=(7.5, 6))
for c in range(1, C + 1):
    m = rotulos == c
    plt.scatter(X[m, 0], X[m, 1], s=6, alpha=0.35, c=cores[c - 1],
                label=f"{c} - {CLASSES[c]}")
plt.xlabel("Sensor 1 – Corrugador do Supercílio (ADC 12 bits)")
plt.ylabel("Sensor 2 – Zigomático Maior (ADC 12 bits)")
plt.title("Espalhamento dos sinais de EMG por expressão facial")
leg = plt.legend(markerscale=3, fontsize=9)
for lh in leg.legend_handles:
    lh.set_alpha(1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG + "fig5_espalhamento_emg.png", dpi=150)
plt.close()


# =====================================================================
# 3. MODELOS
# =====================================================================
def adiciona_intercepto(A):
    return np.hstack((np.ones((A.shape[0], 1)), A))


def mqo_tradicional(Xtr, Ytr):
    Xb = adiciona_intercepto(Xtr)
    return np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ Ytr


def mqo_regularizado(Xtr, Ytr, lbd):
    Xb = adiciona_intercepto(Xtr)
    I = np.identity(Xb.shape[1])
    I[0, 0] = 0.0                       # intercepto nao penalizado
    return np.linalg.pinv(Xb.T @ Xb + lbd * I) @ Xb.T @ Ytr


def expansao_polinomial(A, q):
    """Monomios x1^i * x2^j com 1 <= i+j <= q."""
    cols = []
    for grau in range(1, q + 1):
        for i in range(grau + 1):
            cols.append((A[:, 0] ** i) * (A[:, 1] ** (grau - i)))
    return np.column_stack(cols)


def predizer_classe(Xte, B):
    """Regra de decisao: argmax da saida do discriminante linear."""
    return np.argmax(adiciona_intercepto(Xte) @ B, axis=1) + 1


def acuracia(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# Padronizacao (z-score) das caracteristicas: necessaria para condicionar a
# expansao polinomial (4095^6 ~ 4.7e21 inviabiliza o sistema normal).
mu, sd = X.mean(axis=0), X.std(axis=0)
Xn = (X - mu) / sd

# =====================================================================
# 4. ESCOLHA DO HIPERPARAMETRO q (acuracia x tempo de estimacao)
# =====================================================================
Q_MAX = 6
R_SEL = 10
acc_q = np.zeros(Q_MAX)
tempo_q = np.zeros(Q_MAX)
n_feat = np.zeros(Q_MAX, dtype=int)

Z_cache = {q: expansao_polinomial(Xn, q) for q in range(1, Q_MAX + 1)}
for q in range(1, Q_MAX + 1):
    n_feat[q - 1] = Z_cache[q].shape[1]

for r in range(R_SEL):
    idx = rng.permutation(N)
    corte = int(0.8 * N)
    tr, te = idx[:corte], idx[corte:]
    for q in range(1, Q_MAX + 1):
        Z = Z_cache[q]
        t0 = time.perf_counter()
        B = mqo_tradicional(Z[tr], Y[tr])
        tempo_q[q - 1] += time.perf_counter() - t0
        acc_q[q - 1] += acuracia(rotulos[te], predizer_classe(Z[te], B))

acc_q /= R_SEL
tempo_q /= R_SEL

print("\n--- Selecao da ordem q do classificador polinomial ---")
print(f"{'q':>3}{'n. caracteristicas':>20}{'acuracia media':>17}{'tempo medio (ms)':>19}")
for q in range(1, Q_MAX + 1):
    print(f"{q:>3}{n_feat[q-1]:>20}{acc_q[q-1]:>17.4f}{tempo_q[q-1]*1000:>19.2f}")

# Compromisso: menor q cuja acuracia esta a menos de 0.5 ponto percentual do maximo
lim = acc_q.max() - 0.005
q_star = int(np.argmax(acc_q >= lim)) + 1
print(f"\nMelhor acuracia: q = {int(np.argmax(acc_q))+1} ({acc_q.max():.4f})")
print(f"Ordem escolhida pelo compromisso acuracia/tempo: q* = {q_star} "
      f"(acuracia {acc_q[q_star-1]:.4f}, {tempo_q[q_star-1]*1000:.2f} ms)")

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(range(1, Q_MAX + 1), acc_q * 100, "o-", c="#1f77b4", label="acurácia")
ax1.set_xlabel("ordem q da expansão polinomial")
ax1.set_ylabel("acurácia média (%)", color="#1f77b4")
ax1.axvline(q_star, color="r", ls="--", label=f"q* = {q_star}")
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(range(1, Q_MAX + 1), tempo_q * 1000, "s--", c="#d62728")
ax2.set_ylabel("tempo médio de estimação (ms)", color="#d62728")
plt.title("Compromisso entre acurácia e custo computacional")
fig.tight_layout()
plt.savefig(FIG + "fig6_selecao_q.png", dpi=150)
plt.close()

# =====================================================================
# 4b. ESCOLHA DO HIPERPARAMETRO lambda do MQO regularizado
# =====================================================================
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 10.0, 100.0, 1000.0]
print("\n--- Sensibilidade do MQO regularizado ao lambda ---")
acc_lbd = []
for lbd in LAMBDAS:
    a = 0.0
    for r in range(R_SEL):
        idx = rng.permutation(N)
        corte = int(0.8 * N)
        tr, te = idx[:corte], idx[corte:]
        B = mqo_regularizado(Xn[tr], Y[tr], lbd)
        a += acuracia(rotulos[te], predizer_classe(Xn[te], B))
    acc_lbd.append(a / R_SEL)
    print(f"lambda = {lbd:>8} -> acuracia media = {acc_lbd[-1]:.5f}")
lbd_star = LAMBDAS[int(np.argmax(acc_lbd))]
print(f"Lambda escolhido: {lbd_star}")

# =====================================================================
# 5. VALIDACAO POR MONTE CARLO (R = 500, 80/20)
# =====================================================================
R = 500
modelos = ["MQO tradicional", f"MQO regularizado ({lbd_star})",
           f"MQO polinomial (q={q_star})"]
ACC = {m: [] for m in modelos}
Zq = Z_cache[q_star]
conf = np.zeros((C, C))

t0 = time.time()
for r in range(R):
    idx = rng.permutation(N)
    corte = int(0.8 * N)
    tr, te = idx[:corte], idx[corte:]
    yte = rotulos[te]

    B = mqo_tradicional(Xn[tr], Y[tr])
    ACC[modelos[0]].append(acuracia(yte, predizer_classe(Xn[te], B)))

    B = mqo_regularizado(Xn[tr], Y[tr], lbd_star)
    ACC[modelos[1]].append(acuracia(yte, predizer_classe(Xn[te], B)))

    B = mqo_tradicional(Zq[tr], Y[tr])
    yhat = predizer_classe(Zq[te], B)
    ACC[modelos[2]].append(acuracia(yte, yhat))
    if r < 50:                           # matriz de confusao acumulada
        for v, w in zip(yte, yhat):
            conf[v - 1, w - 1] += 1

print(f"\nValidacao concluida: R = {R} rodadas em {time.time()-t0:.1f} s")

# =====================================================================
# 6. TABELA DE RESULTADOS
# =====================================================================
print("\n" + "=" * 88)
print("METRICA: ACURACIA (taxa de acerto no conjunto de teste)")
print("=" * 88)
print(f"{'Modelos':<30}{'Media':>14}{'Desvio-Padrao':>16}{'Maior Valor':>14}{'Menor Valor':>14}")
print("-" * 88)
linhas = []
for m in modelos:
    v = np.array(ACC[m])
    print(f"{m:<30}{v.mean():>14.4f}{v.std():>16.4f}{v.max():>14.4f}{v.min():>14.4f}")
    linhas.append([m, v.mean(), v.std(), v.max(), v.min()])

# Matriz de confusao (percentual por linha) do melhor modelo
conf_pct = 100 * conf / conf.sum(axis=1, keepdims=True)
print("\nMatriz de confusao do modelo polinomial (% por classe verdadeira):")
print("            " + "".join(f"{c:>8}" for c in range(1, C + 1)))
for i in range(C):
    print(f"real {i+1:<7}" + "".join(f"{conf_pct[i,j]:>8.1f}" for j in range(C)))

plt.figure(figsize=(6, 5))
plt.imshow(conf_pct, cmap="Blues", vmin=0, vmax=100)
for i in range(C):
    for j in range(C):
        plt.text(j, i, f"{conf_pct[i,j]:.1f}", ha="center", va="center",
                 color="white" if conf_pct[i, j] > 50 else "black", fontsize=9)
plt.xticks(range(C), [str(i + 1) for i in range(C)])
plt.yticks(range(C), [f"{i+1} - {CLASSES[i+1][:12]}" for i in range(C)])
plt.xlabel("classe predita")
plt.ylabel("classe verdadeira")
plt.title(f"Matriz de confusão (%) – MQO polinomial q={q_star}")
plt.colorbar()
plt.tight_layout()
plt.savefig(FIG + "fig8_matriz_confusao.png", dpi=150)
plt.close()

plt.figure(figsize=(7.5, 5))
plt.boxplot([ACC[m] for m in modelos],
            tick_labels=["MQO\ntradicional", f"MQO regul.\n($\\lambda$={lbd_star})",
                         f"MQO polinomial\n(q={q_star})"])
plt.ylabel("acurácia")
plt.title(f"Distribuição da acurácia – Monte Carlo (R={R})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG + "fig7_boxplot_classificacao.png", dpi=150)
plt.close()

# Fronteiras de decisao
g1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 400)
g2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 400)
G1, G2 = np.meshgrid(g1, g2)
grade = np.column_stack([G1.ravel(), G2.ravel()])
graden = (grade - mu) / sd
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
for k, (titulo, Zf) in enumerate([
        ("MQO tradicional", None), (f"MQO polinomial (q={q_star})", q_star),
        (f"MQO polinomial (q={q_star}) – ampliação", q_star)]):
    if Zf is None:
        B = mqo_tradicional(Xn, Y)
        pred = predizer_classe(graden, B)
    else:
        B = mqo_tradicional(expansao_polinomial(Xn, Zf), Y)
        pred = predizer_classe(expansao_polinomial(graden, Zf), B)
    ax[k].contourf(G1, G2, pred.reshape(G1.shape), levels=np.arange(0.5, C + 1),
                   colors=cores, alpha=0.3)
    for c in range(1, C + 1):
        m = rotulos == c
        ax[k].scatter(X[m, 0][::10], X[m, 1][::10], s=4, c=cores[c - 1], alpha=0.5)
    ax[k].set_title(titulo)
    ax[k].set_xlabel("Sensor 1")
    ax[k].set_ylabel("Sensor 2")
    if k == 2:
        ax[k].set_xlim(-20, 1800)
        ax[k].set_ylim(-20, 900)
plt.suptitle("Fronteiras de decisão")
plt.tight_layout()
plt.savefig(FIG + "fig9_fronteiras.png", dpi=150)
plt.close()

with open("saida_classificacao.txt", "w") as f:
    f.write(f"q*={q_star};lambda*={lbd_star}\n")
    for l in linhas:
        f.write(f"{l[0]};{l[1]:.6f};{l[2]:.6f};{l[3]:.6f};{l[4]:.6f}\n")
    f.write("acc_por_q;" + ";".join(f"{a:.5f}" for a in acc_q) + "\n")
    f.write("tempo_por_q_ms;" + ";".join(f"{t*1000:.3f}" for t in tempo_q) + "\n")
    f.write("nfeat_por_q;" + ";".join(str(n) for n in n_feat) + "\n")
    f.write("acc_por_lambda;" + ";".join(f"{a:.5f}" for a in acc_lbd) + "\n")
    np.savetxt(f, conf_pct, fmt="%.2f", delimiter=";", header="matriz_confusao_pct")
print("\nArquivos gerados em figuras/ e saida_classificacao.txt")
