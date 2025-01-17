import numpy as np
import matplotlib.pyplot as plt
import odlab

np.random.seed(92384)

dims = 2
steps = 1000
max_k = 400
uncorr_chain = np.random.randn(dims, steps)
corr_chain = np.zeros_like(uncorr_chain)


def step(x):
    prop = np.random.randn(2)*5e-2
    y = prop + x
    dr = np.linalg.norm(y) - np.linalg.norm(x)
    if dr > 0:
        y *= np.exp(-dr)
    return y


for ind in range(1, steps):
    corr_chain[:, ind] = step(corr_chain[:, ind - 1])


def plot_chain_autocorr(axes, chain, **kwargs):
    MC_gamma = odlab.statistics.autocovariance(chain, min_k=0, max_k=max_k)
    Kv = np.arange(0, max_k)
    flat_ax = axes.flatten()
    for vari in range(dims):
        ax = flat_ax[vari]
        ax.plot(Kv, MC_gamma[vari, :] / MC_gamma[vari, 0], **kwargs)
        ax.set(
            xlabel="$k$",
            ylabel="$\\hat{\\gamma}_k/\\hat{\\gamma}_0$",
            title=f'Autocorrelation for "{vari}"',
        )
    return fig, axes


fig, axes = plt.subplots(3, 2, figsize=(15, 15))

axes[0, 0].scatter(uncorr_chain[0, :], uncorr_chain[1, :], s=1)
axes[0, 0].set_title("Uncorrelated chain")

axes[0, 1].scatter(corr_chain[0, :], corr_chain[1, :], s=1)
axes[0, 1].set_title("Correlated chain")

plot_chain_autocorr(axes[1:, 0], uncorr_chain, c="b", label="Uncorrelated chain")
plot_chain_autocorr(axes[1:, 1], corr_chain, c="g", label="Correlated chain")

fig.suptitle("Markov Chain autocorrelation functions")

plt.show()
