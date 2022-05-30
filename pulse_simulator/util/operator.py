from .tensor_product import TensorProduct

def state(system, str_dict):
    """generate state with the shape of the multi-qubit system
    Args:
        system (System) : multi-qubit system
        str_dict (dict) : {0:"S0", 1:"S1", 4:"Sp" ...} (missing index is transpiled as "S0")
    Returns:
        output (np.array) : density matrix of the target state
    """
    dims = [q.dim for q in system.qubits.values()]
    tp = TensorProduct(*dims)
    for idx, qubit in system.qubits.items():
        if idx in str_dict.keys():
            tp.prod(getattr(qubit, str_dict[idx]), idx)
        else:
            if "default" in str_dict.keys():
                tp.prod(getattr(qubit, str_dict["default"]), idx)
            else:
                tp.prod(getattr(qubit, "S0"), idx)
    output = tp.get_operator()
    return output

def operator(system, str_dict):
    """generate operator with the shape of the multi-qubit system
    Args:
        system (System) : multi-qubit system
        str_dict (dict) : {0:"X", 1:"sZ", 4:"A" ...} (missing index is transpiled as "I")
    Returns:
        output (np.array) : unitary matrix of the target operator
    """
    dims = [q.dim for q in system.qubits.values()]
    tp = TensorProduct(*dims)
    for idx, qubit in system.qubits.items():
        if idx in str_dict.keys():
            tp.prod(getattr(qubit, str_dict[idx]), idx)
        else:
            if "default" in str_dict.keys():
                tp.prod(getattr(qubit, str_dict["default"]), idx)
            else:
                pass
    output = tp.get_operator()
    return output