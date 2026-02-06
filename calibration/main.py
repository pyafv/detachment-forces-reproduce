import numpy as np
import matplotlib.pyplot as plt
import pyafv as afv
import pyafv.calibrate as cal


KA = 1.0
KP = 1.0
A0 = np.pi
P0 = 4.8
Lambda = 0.2
 
phys = afv.PhysicalParams(KA=KA, KP=KP, A0=A0, P0=P0, Lambda=Lambda)
sim = cal.DeformablePolygonSimulator(phys)

# Initial steady-state shape
fig, ax = plt.subplots()
sim.plot_2d(ax)
ax.set_title('Steady state')
plt.show()


# ------------  Apply F = -3 -------------
ext_force = -3.0
sim.simulate(ext_force, dt=1e-3, nsteps=50_000)

fig, ax = plt.subplots()
ax.set_title(f"After apply F={ext_force}, {sim.detached=}")
sim.plot_2d(ax)
# Plot the external forces as arrows
ax.quiver(
    sim.pts1[1:-1, 0], sim.pts1[1:-1, 1], -0.075 * ext_force, 0,
    angles="xy", scale_units="xy", scale=1, width=0.003, color='C7', zorder=2
)
ax.quiver(
    sim.pts2[1:-1, 0], sim.pts2[1:-1, 1], 0.075 * ext_force, 0,
    angles="xy", scale_units="xy", scale=1, width=0.003, color='C7', zorder=2
)
plt.show()
plt.close(fig)


# ------------  Apply F = 3 -------------
ext_force = 3.0
sim.simulate(ext_force, dt=1e-3, nsteps=50_000)

fig, ax = plt.subplots()
ax.set_title(f"After apply F={ext_force}, {sim.detached=}")
sim.plot_2d(ax)
# Plot the external forces as arrows
ax.quiver(
    sim.pts1[1:-1, 0], sim.pts1[1:-1, 1], -0.075 * ext_force, 0,
    angles="xy", scale_units="xy", scale=1, width=0.003, color='C7', zorder=2
)
ax.quiver(
    sim.pts2[1:-1, 0], sim.pts2[1:-1, 1], 0.075 * ext_force, 0,
    angles="xy", scale_units="xy", scale=1, width=0.003, color='C7', zorder=2
)
plt.show()
plt.close(fig)
