import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

results = np.load("results.npz")
markers = {
    "ours": "o",
    "pcdet": "s",
    "lilan": "p",
    "official": "1",
    "rrpn": "Py"
}

forward_mean = defaultdict(list)
forward_std = defaultdict(list)
backward_mean = defaultdict(list)
backward_std = defaultdict(list)
label_source = {}


for k in results.keys():
    source, rtype, direction, n = k.split('.')
    label = f"{source} ({rtype})"
    label_source[label] = source
    if direction == "forward":
        forward_mean[label].append((n, np.mean(results[k])))
        forward_std[label].append((n, np.std(results[k])))
    elif direction == "backward":
        backward_mean[label].append((n, np.mean(results[k])))
        backward_std[label].append((n, np.std(results[k])))


fig, ax = plt.subplots()
for label in forward_mean:
    n, means = zip(*forward_mean[label])
    n_, stds = zip(*forward_std[label])
    assert np.all(n == n_)
    n, means, stds = np.array(n).astype(float), np.array(means), np.array(stds)

    color = np.random.rand(3,)
    marker = markers[label_source[label]]
    ax.plot(n, means, marker, c=color, label=label)
    ax.errorbar(n, means, yerr=stds, c=color)
    ax.fill_between(n, means-stds, means+stds, alpha=0.2, color=color)

ax.set_xscale("log")
ax.set_yscale("log")
plt.legend()
fig.savefig("forward.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
for label in backward_mean:
    n, means = zip(*backward_mean[label])
    n_, stds = zip(*backward_std[label])
    assert np.all(n == n_)
    n, means, stds = np.array(n).astype(float), np.array(means), np.array(stds)

    color = np.random.rand(3,)
    marker = markers[label_source[label]]
    ax.plot(n, means, marker, c=color, label=label)
    ax.errorbar(n, means, yerr=stds, c=color)
    ax.fill_between(n, means-stds, means+stds, alpha=0.2, color=color)

ax.set_xscale("log")
ax.set_yscale("log")
plt.legend()
fig.savefig("backward.pdf", bbox_inches='tight')

# plt.show()
