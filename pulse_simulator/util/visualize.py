import numpy as np
import matplotlib.pyplot as plt
from .operator import operator, state

def visualize_1q_population(system, simulator):
    plt.figure(figsize=(20,4))
    for i in [0,1]:
        unitary = np.array(simulator.unitary)
        pi0 = np.abs(unitary[:,i,0])**2
        pi1 = np.abs(unitary[:,i,1])**2
        pi2 = np.abs(unitary[:,i,2])**2

        plt.subplot(2,1,i+1)
        plt.title(f"Start from {i} state")
        plt.axhline(0, color="black", linestyle="--")
        plt.axhline(1, color="black", linestyle="--")
        plt.plot(simulator.time, pi0)
        plt.plot(simulator.time, pi1)
        plt.plot(simulator.time, pi2)
        plt.ylabel(f"Population")
        plt.xlim(0, simulator.time[-1])
        if i != 1:
            plt.tick_params(labelbottom=False)
        else:
            plt.xlabel("Duration (ns)")
    plt.tight_layout()
    plt.show()

def visualize_hamiltonian_tomography(system, simulator, control):
    SC0 = state(system, {control:"S0"})
    SC1 = state(system, {control:"S1"})
    for idx, qubit in system.qubits.items():
        L = operator(system, {idx:"Ql"})
        X = operator(system, {idx:"Qx"})
        Y = operator(system, {idx:"Qy"})
        Z = operator(system, {idx:"Qz"})

        p0 = []
        p1 = []
        for u in simulator.unitary:
            F0 = u@SC0@u.T.conj()
            F1 = u@SC1@u.T.conj()
            p0.append([np.trace(L@F0), np.trace(X@F0), np.trace(Y@F0), np.trace(Z@F0)])
            p1.append([np.trace(L@F1), np.trace(X@F1), np.trace(Y@F1), np.trace(Z@F1)])
        p0 = np.array(p0).T.real
        p1 = np.array(p1).T.real

        print(f"Qubit ({idx})")
        plt.figure(figsize=(8,6))
        for i, label in enumerate(["Leak","X","Y","Z"]):
            plt.subplot(4,1,i+1)
            if label == "Z":
                plt.plot(simulator.time, p0[i]/2)
                plt.plot(simulator.time, p1[i]/2)
            else:
                plt.plot(simulator.time, p0[i]/2)
                plt.plot(simulator.time, p1[i]/2)
            plt.ylabel(label)
            if i != 3:
                plt.tick_params(labelbottom=False)
            else:
                plt.xlabel("Duration (ns)")
            if label == "Leak":
                plt.yscale("log")
                plt.ylim(1e-5, 1)
            elif label == "Z":
                plt.axhline(0, color="black", linestyle="--")
                plt.axhline(1, color="black", linestyle="--")
            else:
                plt.axhline(-1, color="black", linestyle="--")
                plt.axhline(1, color="black", linestyle="--")
        plt.tight_layout()
        plt.show()