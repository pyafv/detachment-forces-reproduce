import numpy as np
from mpi4py import MPI
import pyafv as afv
import pyafv.calibrate as cal


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


P0s = np.linspace(4, 6, 11)
P0s = np.concatenate((P0s, [6.1, 6.2]))
Lambdas = np.linspace(0.0, 0.5, 11)  # Tension parameter values

if size != len(P0s):
    if rank == 0:
        print(f"Error: number of MPI processes ({size}) must equal number of P0 values ({len(P0s)}).")
    exit(1)


# ---------- parameters ----------
KA = 1.0
KP = 1.0
A0 = np.pi
P0 = P0s[rank]

detachment_forces =[]
for idx, Lambda in enumerate(Lambdas):
    if rank == 0:
        print(f"\nProcessing {idx+1}/{len(Lambdas)}...")

    phys = afv.PhysicalParams(KA=KA, KP=KP, A0=A0, P0=P0, Lambda=Lambda)
    f_detach, phys_cal = cal.auto_calibrate(phys, show=(rank == 0))
    # l0, delta = phys_cal.l0, phys_cal.delta

    detachment_forces.append(f_detach)

detachment_forces = np.array(detachment_forces)

# Gather results from all processes
all_detachment_forces = comm.gather(detachment_forces, root=0)
if rank == 0:
    all_detachment_forces = np.array(all_detachment_forces)
    # shape: (len(P0s), len(Lambdas))

    np.save("detachment_forces_DP.npy", all_detachment_forces)
    print("\nDone!")
