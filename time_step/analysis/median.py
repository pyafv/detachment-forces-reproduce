import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


dts = [0.1, 0.05, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]

# Define the colormap and colors
start, end = 0.1, 0.92
cmap = plt.cm.viridis  # or 'plasma'
colors = cmap(np.linspace(start, end, len(dts)))

medians = []
ups = []
lows = []

for dt in dts:

    data = np.load("../data/dt%g.npz" % dt)
    
    rupture_times = data['rupture_times']
    rupture_sizes = data['rupture_sizes']

    print(f"max_rupture_time: {rupture_times.max()}")

    # remove assays already rupture before starting time
    mask = (rupture_sizes > 0) & (rupture_times < dt)

    rupture_times = rupture_times[~mask]
    rupture_sizes = rupture_sizes[~mask]

    nExps = rupture_sizes.size  # number of assays
    rupture_times = rupture_times[rupture_sizes > 0]
    rupture_sizes = rupture_sizes[rupture_sizes > 0]
    N = len(rupture_times)  # number of assays that have ruptures
    print(f"{N=}/{nExps=}")

    if N == 0:
        rupture_times = np.ones(1)

    T = np.ones(nExps) * rupture_times.max()
    T[:N] = rupture_times

    E = np.zeros(nExps)
    E[:N] = np.ones(N)

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, label=r"$\Delta t=%g$" % dt, timeline=np.arange(0, 1000+dt, dt))

    idx = np.where(np.array(dts) == dt)[0][0]

    median_ = kmf.median_survival_time_
    median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
    low = median_confidence_interval_[r"$\Delta t=%g$_lower_0.95" % dt][0.5]
    up = median_confidence_interval_[r"$\Delta t=%g$_upper_0.95" % dt][0.5]

    medians.append(median_)
    ups.append(up)
    lows.append(low)


fig, ax = plt.subplots(figsize=(3.2, 3))
ax.errorbar(dts[:], medians[:], yerr=[(np.array(medians) - np.array(lows))[:], (np.array(ups) - np.array(medians))[:]],
            fmt='o--', label="Original", color="C0", markersize=5, markerfacecolor='None', markeredgewidth=1.5, capsize=4, clip_on=False, zorder=1)

ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"Median survival time $t_{1/2}$")
ax.set_xscale("log")
ax.set_yscale("log")


#------------------------------------------------------------

dts = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

medians = []
ups = []
lows = []
for dt in dts:

    data = np.load("../data/trunc_dt%g.npz" % dt)
    
    rupture_times = data['rupture_times']
    rupture_sizes = data['rupture_sizes']

    print(f"max_rupture_time: {rupture_times.max()}")

    # remove assays already rupture before starting time
    mask = (rupture_sizes > 0) & (rupture_times < dt)

    rupture_times = rupture_times[~mask]
    rupture_sizes = rupture_sizes[~mask]

    nExps = rupture_sizes.size  # number of assays
    rupture_times = rupture_times[rupture_sizes > 0]
    rupture_sizes = rupture_sizes[rupture_sizes > 0]
    N = len(rupture_times)  # number of assays that have ruptures
    print(f"{N=}/{nExps=}")

    if N == 0:
        rupture_times = np.ones(1)

    T = np.ones(nExps) * rupture_times.max()
    T[:N] = rupture_times

    E = np.zeros(nExps)
    E[:N] = np.ones(N)

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, label=r"$\Delta t=%g$" % dt, timeline=np.arange(0, 1000+dt, dt))

    idx = np.where(np.array(dts) == dt)[0][0]

    median_ = kmf.median_survival_time_
    median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
    low = median_confidence_interval_[r"$\Delta t=%g$_lower_0.95" % dt][0.5]
    up = median_confidence_interval_[r"$\Delta t=%g$_upper_0.95" % dt][0.5]

    medians.append(median_)
    ups.append(up)
    lows.append(low)


ax.errorbar(dts[:], medians[:], yerr=[(np.array(medians) - np.array(lows))[:], (np.array(ups) - np.array(medians))[:]],
            fmt='^--', label="Truncated", color="C3", markersize=5, markerfacecolor='None', markeredgewidth=1.5, capsize=4, clip_on=False, zorder=2)

ax.set_xlim(1e-3, 0.1)
ax.set_ylim(bottom=1)

ax.legend(frameon=False, loc='upper right', fontsize=10)
plt.savefig("median.png", dpi=150, bbox_inches='tight')
