import numpy as np
import pyafv as afv

import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


P0s = np.linspace(4, 6, 11)
P0s = np.concatenate((P0s, [6.1, 6.2]))
Lambdas = np.linspace(0.0, 0.5, 11)

detachment_forces = np.load('detachment_forces_DP.npy')

# interpolator: detachment_forces should have shape (len(P0s), len(Lambdas))
interp = RegularGridInterpolator((P0s, Lambdas), detachment_forces)

# line: Lambda = lambda^(n) + 2 K_P P0
P0_line = np.linspace(min(P0s), 4.3, 100)
Lambda_line = 2 * P0_line - 7.9

# keep only the part inside the plotting/domain range
mask = (Lambda_line >= min(Lambdas)) & (Lambda_line <= max(Lambdas))
P0_line_in = P0_line[mask]
Lambda_line_in = Lambda_line[mask]

# interpolate force values on the line
points = np.column_stack((P0_line_in, Lambda_line_in))
forces_line = interp(points)

# plotting
fig, ax = plt.subplots(figsize=(2.8, 2.8))
ax.plot(P0_line_in, forces_line, lw=3, color='C2', label="DP")


#==================================================
# FV: ell = 1.0
#==================================================
delta = 0.45

detachment_forces = []
for idx in range(len(P0_line_in)):
    P0 = P0_line_in[idx]
    Lambda = Lambda_line_in[idx]

    KP = 1.0
    A0 = np.pi
    phys = afv.PhysicalParams(r=1.0, KP=KP, A0=A0, P0=P0, Lambda=Lambda)
    l0 = phys.r
    dc = np.sqrt(4 * (l0**2) - delta**2)

    l = l0
    distance = dc
    epsilon = l - (distance/2.)

    theta = 2 * np.pi - 2 * np.arctan2(np.sqrt(l**2 - (l - epsilon)**2), l - epsilon)
    A = (l - epsilon) * np.sqrt(l**2 - (l - epsilon)**2) + 0.5 * (l**2 * theta)
    P = 2 * np.sqrt(l**2 - (l - epsilon)**2) + l * theta

    f = 4. * np.sqrt((2-epsilon) * epsilon) * (A - A0 + KP * ((P - P0)/(2 - epsilon)) + (Lambda/2) * (1./((2-epsilon)*epsilon)))

    detachment_forces.append(f)

detachment_forces = np.array(detachment_forces)

ax.plot(P0_line_in, detachment_forces, lw=3, color='C3', label=r"FV: $\ell=1$")

#==================================================
# FV: ell = ell_0
#==================================================

detachment_forces = []
for idx in range(len(P0_line_in)):
    P0 = P0_line_in[idx]
    Lambda = Lambda_line_in[idx]

    KP = 1.0
    A0 = np.pi
    phys = afv.PhysicalParams(r=1.0, KP=KP, A0=A0, P0=P0, Lambda=Lambda)
    l0, d0 = phys.get_steady_state()
    dc = np.sqrt(4 * (l0**2) - delta**2)

    l = l0
    distance = dc
    epsilon = l - (distance/2.)

    theta = 2 * np.pi - 2 * np.arctan2(np.sqrt(l**2 - (l - epsilon)**2), l - epsilon)
    A = (l - epsilon) * np.sqrt(l**2 - (l - epsilon)**2) + 0.5 * (l**2 * theta)
    P = 2 * np.sqrt(l**2 - (l - epsilon)**2) + l * theta

    f = 4. * np.sqrt((2-epsilon) * epsilon) * (A - A0 + KP * ((P - P0)/(2 - epsilon)) + (Lambda/2) * (1./((2-epsilon)*epsilon)))
    
    detachment_forces.append(f)

detachment_forces = np.array(detachment_forces)

ax.plot(P0_line_in, detachment_forces, lw=3, color='C4', label=r"FV: $\ell=\ell_0$")


ax.set_xlabel(r'$P_0$')
ax.set_ylabel(r'$f_\mathrm{detach}$')

ax.set_xlim(min(P0_line_in), max(P0_line_in))
ax.set_ylim(bottom=0)
ax.set_xticks([4.05, 4.15], minor=True)

ax.set_title(r'$\Lambda=\lambda^{(n)}+2K_P P_0$')

plt.legend(frameon=False)
plt.savefig('vary_lambda_c.png', dpi=150, bbox_inches='tight')
