import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .operator import operator, state

def analyze_cr_vector(time, data):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(data).T
    phase = np.unwrap(np.angle(proj[0] + 1j*proj[1]))
    norm = np.mean(np.gradient(phase)/np.gradient(time)/(2*np.pi))
    axis = np.cross(pca.components_[0], pca.components_[1])
    vector = norm * axis
    return vector

def analyze_cr_pauli(time, data):
    v00 = analyze_cr_vector(time, data[0].T)
    v10 = analyze_cr_vector(time, data[1].T)
    vi  = 0.5*(v00 + v10)
    vz  = 0.5*(v00 - v10)
    pauli = np.pi*np.hstack([vi,vz])
    return pauli

def visualize_hamiltonian_tomography_z(system, time, unitary, control, analyze=False, trig_pos=None):
    SC0 = state(system, {control:"S0"})
    SC1 = state(system, {control:"S1"})
    for idx, qubit in system.qubits.items():
        L = operator(system, {idx:"Ql"})
        X = operator(system, {idx:"Qx"})
        Y = operator(system, {idx:"Qy"})
        Z = operator(system, {idx:"Qz"})

        p0 = []
        p1 = []
        for u in unitary:
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
            if trig_pos is not None:
                for pos in trig_pos:
                    plt.axvline(pos, color="red", linestyle="--")
            if label == "Z":
                plt.plot(time, p0[i]/2)
                plt.plot(time, p1[i]/2)
            else:
                plt.plot(time, p0[i])
                plt.plot(time, p1[i])
            plt.ylabel(label)
            if i != 3:
                plt.tick_params(labelbottom=False)
            else:
                plt.xlabel("Duration (ns)")
            if label == "Leak":
                plt.yscale("log")
                plt.ylim(1e-5, 1)
            elif label == "Z":
                plt.axhline(0.0, color="black", linestyle="--")
                plt.axhline(0.5, color="black", linestyle="--")
                plt.axhline(1.0, color="black", linestyle="--")
            else:
                plt.axhline(-1, color="black", linestyle="--")
                plt.axhline(0, color="black", linestyle="--")
                plt.axhline(+1, color="black", linestyle="--")
            plt.xlim(time[0], time[-1])
        plt.tight_layout()
        plt.show()
        
        if analyze:
            if idx != control:
                data = [p0[1:], p1[1:]]
                pauli = analyze_cr_pauli(time, data)

                plt.figure(figsize=(8,2))
                plt.axhline(0, color="black", linestyle="--")
                plt.bar(range(6), pauli)
                plt.xticks(range(6), ["IX","IY","IZ","ZX","ZY","ZZ"])
                plt.xlabel("Pauli")
                plt.ylabel("Coefficients (MHz)")
                plt.show()
        
def visualize_hamiltonian_tomography_x(system, time, unitary, control, analyze=False, trig_pos=None):
    state_p = dict(zip(list(system.qubits.keys()), ["Sp"]*len(system.qubits)))
    state_m = copy.copy(state_p)
    state_m[control] = "Sm"
    
    SCp = state(system, state_p)
    SCm = state(system, state_m)
    for idx, qubit in system.qubits.items():
        L = operator(system, {idx:"Ql"})
        X = operator(system, {idx:"Qx"})
        Y = operator(system, {idx:"Qy"})
        Z = operator(system, {idx:"Qz"})

        pp = []
        pm = []
        for u in unitary:
            Fp = u@SCp@u.T.conj()
            Fm = u@SCm@u.T.conj()
            pp.append([np.trace(L@Fp), np.trace(X@Fp), np.trace(Y@Fp), np.trace(Z@Fp)])
            pm.append([np.trace(L@Fm), np.trace(X@Fm), np.trace(Y@Fm), np.trace(Z@Fm)])
        pp = np.array(pp).T.real
        pm = np.array(pm).T.real

        print(f"Qubit ({idx})")
        plt.figure(figsize=(8,6))
        for i, label in enumerate(["Leak","X","Y","Z"]):
            plt.subplot(4,1,i+1)
            if trig_pos is not None:
                for pos in trig_pos:
                    plt.axvline(pos, color="red", linestyle="--")
            if label == "Z":
                plt.plot(time, pp[i]/2)
                plt.plot(time, pm[i]/2)
            else:
                plt.plot(time, pp[i])
                plt.plot(time, pm[i])
            plt.ylabel(label)
            if i != 3:
                plt.tick_params(labelbottom=False)
            else:
                plt.xlabel("Duration (ns)")
            if label == "Leak":
                plt.yscale("log")
                plt.ylim(1e-5, 1)
            elif label == "Z":
                plt.axhline(0.0, color="black", linestyle="--")
                plt.axhline(0.5, color="black", linestyle="--")
                plt.axhline(1.0, color="black", linestyle="--")
            else:
                plt.axhline(-1, color="black", linestyle="--")
                plt.axhline(0, color="black", linestyle="--")
                plt.axhline(+1, color="black", linestyle="--")
            plt.xlim(time[0], time[-1])
        plt.tight_layout()
        plt.show()
        
        if analyze:
            if idx != control:
                data = [pp[1:], pm[1:]]
                pauli = analyze_cr_pauli(time, data)

                plt.figure(figsize=(8,2))
                plt.axhline(0, color="black", linestyle="--")
                plt.bar(range(6), pauli)
                plt.xticks(range(6), ["XI","YI","ZI","XX","YX","ZX"])
                plt.xlabel("Pauli")
                plt.ylabel("Coefficients")
                plt.show()
        
#         p_ph = np.unwrap(np.angle(pp[1] + 1j*pp[2]))
#         m_ph = np.unwrap(np.angle(pm[1] + 1j*pm[2]))
        
#         plt.figure(figsize=(8,3))
#         plt.plot(time, p_ph)
#         plt.plot(time, m_ph)
#         plt.xlabel("Duration (ns)")
#         plt.ylabel("Phase (rad)")
#         plt.show()