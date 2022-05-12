from .tensor_product import TensorProduct

def state(system, str_dict):
    """generate state with the shape of the multi-qubit system
    Args:
        system (System) : multi-qubit system
        str_dict (dict) : {0:"S0", 1:"S1", 4:"Sp" ...} (missing index is transpiled as "S0")
    """
    dims = [q.dim for q in system.qubits.values()]
    tp = TensorProduct(*dims)
    for idx, qubit in system.qubits.items():
        if idx in str_dict.keys():
            tp.prod(getattr(qubit, str_dict[idx]), idx)
        else:
            tp.prod(getattr(qubit, "S0"), idx)
    return tp.get_operator()

def operator(system, str_dict):
    """generate operator with the shape of the multi-qubit system
    Args:
        system (System) : multi-qubit system
        str_dict (dict) : {0:"X", 1:"sZ", 4:"A" ...} (missing index is transpiled as "I")
    """
    dims = [q.dim for q in system.qubits.values()]
    tp = TensorProduct(*dims)
    for idx, string in str_dict.items():
        tp.prod(getattr(system.qubits[idx], string), idx)
    return tp.get_operator()