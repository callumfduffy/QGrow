from __future__ import annotations

from collections.abc import Callable
from typing import List, Union

import numpy as np
import qadence as q
import torch
import torch.nn as nn
from qadence.parameters import Parameter
from qadence.types import BasisSet, ReuploadScaling, TParameter

RotationTypes = type[Union[q.RX, q.RY, q.RZ, q.PHASE]]

#seed = 42
#np.random.seed(seed)
#torch.manual_seed(seed)


def feature_map_new(
    n_qubits: int,
    support: tuple[int, ...] | None = None,
    param: Parameter | str = "phi",
    op: RotationTypes = q.RX,
    fm_type: BasisSet | Callable | str = BasisSet.FOURIER,
    reupload_scaling: ReuploadScaling
    | Callable
    | str = ReuploadScaling.CONSTANT,
    feature_range: tuple[float, float] | None = None,
    target_range: tuple[float, float] | None = None,
    multiplier: Parameter | TParameter | None = None,
    param_prefix: str | None = None,
    param_val: float | None = None,
) -> q.blocks.KronBlock:
    """Construct a feature map of a given type.

    Arguments:
        n_qubits: Number of qubits the feature map covers. Results in `support=range(n_qubits)`.
        support: Puts one feature-encoding rotation gate on every qubit in `support`. n_qubits in
            this case specifies the total overall qubits of the circuit, which may be wider than the
            support itself, but not narrower.
        param: Parameter of the feature map; you can pass a string or Parameter;
            it will be set as non-trainable (FeatureParameter) regardless.
        op: Rotation operation of the feature map; choose from RX, RY, RZ or PHASE.
        fm_type: Basis set for data encoding; choose from `BasisSet.FOURIER` for Fourier
            encoding, or `BasisSet.CHEBYSHEV` for Chebyshev polynomials of the first kind.
        reupload_scaling: how the feature map scales the data that is re-uploaded for each qubit.
            choose from `ReuploadScaling` enumeration or provide your own function with a single
            int as input and int or float as output.
        feature_range: range of data that the input data provided comes from. Used to map input data
            to the correct domain of the feature-encoding function.
        target_range: range of data the data encoder assumes as the natural range. For example,
            in Chebyshev polynomials it is (-1, 1), while for Fourier it may be chosen as (0, 2*PI).
            Used to map data to the correct domain of the feature-encoding function.
        multiplier: overall multiplier; this is useful for reuploading the feature map serially with
            different scalings; can be a number or parameter/expression.
        param_prefix: string prefix to create trainable parameters multiplying the feature parameter
            inside the feature-encoding function. Note that currently this does not take into
            account the domain of the feature-encoding function.

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import feature_map, BasisSet, ReuploadScaling

    fm = feature_map(3, fm_type=BasisSet.FOURIER)
    print(f"{fm = }")

    fm = feature_map(3, fm_type=BasisSet.CHEBYSHEV)
    print(f"{fm = }")

    fm = feature_map(3, fm_type=BasisSet.FOURIER, reupload_scaling = ReuploadScaling.TOWER)
    print(f"{fm = }")
    ```
    """

    # Process input
    if support is None:
        support = tuple(range(n_qubits))
    elif len(support) != n_qubits:
        raise ValueError("Wrong qubit support supplied")

    rotations = [q.RX, q.RY, q.RZ, q.PHASE]
    if op not in rotations:
        raise ValueError(
            f"Operation {op} not supported. "
            f"Please provide one from {[rot.__name__ for rot in rotations]}."
        )

    scaled_fparam = q.constructors.feature_maps.fm_parameter_scaling(
        fm_type, param, feature_range=feature_range, target_range=target_range
    )

    transform_func = q.constructors.feature_maps.fm_parameter_func(fm_type)

    basis_tag = (
        fm_type.value if isinstance(fm_type, BasisSet) else str(fm_type)
    )
    rs_func, rs_tag = q.constructors.feature_maps.fm_reupload_scaling_fn(
        reupload_scaling
    )

    # Set overall multiplier
    multiplier = 1 if multiplier is None else Parameter(multiplier)

    # Build feature map
    op_list = []
    fparam = scaled_fparam
    for i, qubit in enumerate(support):
        if param_prefix is not None:
            train_param = Parameter(
                name=param_prefix, trainable=True, value=param_val
            )
            fparam = train_param * scaled_fparam
        op_list.append(
            op(qubit, multiplier * rs_func(i) * transform_func(fparam))
        )
    fm = q.kron(*op_list)

    fm.tag = rs_tag + " " + basis_tag + " FM"
    return fm


def trainable_freq_circ(
    n_qubits: int,
    depth: int,
    inputs: List[str],
    init_func: Callable,
    fm_init: Callable = None,
    ansatz_gates: List[object] = [q.RY],
    fm_gates: List[object] = [q.RX],
):
    operations = []
    for d in range(depth):
        ansatz_layers = []
        # add in trainable rotation gates
        for gate_idx, a_gate in enumerate(ansatz_gates):
            rot_1 = []
            rot_2 = []
            for i in range(n_qubits):
                if type(init_func) is tuple:
                    v1 = torch.tensor(init_func[0](size=1))
                    v2 = -1.0 * v1
                else:
                    v1 = torch.tensor(init_func(size=1))
                    v2 = torch.tensor(init_func(size=1))

                param_1 = Parameter(
                    name=f"theta_{d}{i}{gate_idx}{0}", trainable=True, value=v1
                )
                param_2 = Parameter(
                    name=f"theta_{d}{i}{gate_idx}{1}", trainable=True, value=v2
                )

                g1 = a_gate(target=i, parameter=param_1)
                g2 = a_gate(target=i, parameter=param_2)
                rot_1.append(g1)
                rot_2.append(g2)
            ansatz_layers.append(q.kron(g for g in rot_1))
            ansatz_layers.append(q.kron(g for g in rot_2))

        for i in range(n_qubits - 1):
            ansatz_layers.append(q.CNOT(i, i + 1))

        for fm_idx, fm_gate in enumerate(fm_gates):
            fm1_list = []
            fm2_list = []
            for i in range(n_qubits):
                if fm_init == "equal_spacing":
                    fm1_val = 1.0
                    fm2_val = 1.0

                elif type(fm_init) is tuple:
                    fm1_val = torch.tensor(fm_init[0](size=1))
                    fm2_val = -1.0 * fm1_val

                else:
                    fm1_val = torch.tensor(fm_init(size=1))
                    fm2_val = torch.tensor(fm_init(size=1))

                fm1 = feature_map_new(
                    1,
                    support=[i],
                    param=inputs[i],
                    op=fm_gate,
                    param_prefix=f"phi_{d}{i}{fm_idx}{0}",
                    param_val=fm1_val,
                )
                fm2 = feature_map_new(
                    1,
                    support=[i],
                    param=inputs[i],
                    op=fm_gate,
                    param_prefix=f"phi_{d}{i}{fm_idx}{1}",
                    param_val=fm2_val,
                )
                fm1_list.append(fm1)
                fm2_list.append(fm2)

            ansatz_layers.append(q.kron(fm for fm in fm1_list))
            ansatz_layers.append(q.kron(fm for fm in fm2_list))

        operations.append(q.chain(l for l in ansatz_layers))

    # add a final ansatz layer at the end
    ansatz_layers = []
    for gate_idx, a_gate in enumerate(ansatz_gates):
        rot_1 = []
        rot_2 = []
        for i in range(n_qubits):
            if type(init_func) is tuple:
                v1 = torch.tensor(init_func[0](size=1))
                v2 = -1.0 * v1
            else:
                v1 = torch.tensor(init_func(size=1))
                v2 = init_func(size=1)
            param_1 = Parameter(
                name=f"theta_{d}{i}{gate_idx}{0}", trainable=True, value=v1
            )
            param_2 = Parameter(
                name=f"theta_{d}{i}{gate_idx}{1}", trainable=True, value=v2
            )

            g1 = a_gate(target=i, parameter=param_1)
            g2 = a_gate(target=i, parameter=param_2)
            rot_1.append(g1)
            rot_2.append(g2)
        ansatz_layers.append(q.kron(g for g in rot_1))
        ansatz_layers.append(q.kron(g for g in rot_2))

    for i in range(n_qubits - 1):
        ansatz_layers.append(q.CNOT(i, i + 1))
    operations.append(q.chain(b for b in ansatz_layers))

    obs = q.Z(0)
    circ = q.QuantumCircuit(n_qubits, q.chain(o for o in operations))
    model = q.QNN(circuit=circ, observable=obs, inputs=inputs)
    return model


def growth_fm_circ(
    n_qubits: int,
    depth: int,
    fm_depth: int,
    inputs: List[str],
    init_func: Callable,
    fm_init: Callable = None,
    growth_type: str = "serial",
    ansatz_gates: List[object] = [q.RY],
    fm_gates: List[object] = [q.RX],
):
    if growth_type == "serial":
        fm_idx = list(range(fm_depth))
    elif growth_type == "interleave":
        inteval = int(np.floor(depth / 2))
        fm_idx = [inteval]
        for i in range(int((fm_depth - 1) / 2)):
            fm_idx.append(int(fm_idx[-1] + 1))
            fm_idx.insert(0, int(fm_idx[0] - 1))

    operations = []
    for d in range(depth):
        ansatz_layers = []
        for gate_idx, a_gate in enumerate(ansatz_gates):
            rot_1 = []
            rot_2 = []
            for i in range(n_qubits):
                # identity init
                v1 = torch.tensor(init_func(size=1))
                v2 = -1.0 * v1
                param_1 = Parameter(
                    name=f"theta_{d}{i}{gate_idx}{0}", trainable=True, value=v1
                )
                param_2 = Parameter(
                    name=f"theta_{d}{i}{gate_idx}{1}", trainable=True, value=v2
                )

                g1 = a_gate(target=i, parameter=param_1)
                g2 = a_gate(target=i, parameter=param_2)
                rot_1.append(g1)
                rot_2.append(g2)
            ansatz_layers.append(q.kron(g for g in rot_1))
            ansatz_layers.append(q.kron(g for g in rot_2))

        for i in range(n_qubits - 1):
            ansatz_layers.append(q.CNOT(i, i + 1))

        if d in fm_idx:
            for fm_id, fm_gate in enumerate(fm_gates):
                fm1_list = []
                fm2_list = []
                for i in range(n_qubits):
                    fm1_val = torch.tensor(fm_init(size=1))
                    fm2_val = -1.0 * fm1_val
                    fm1 = feature_map_new(
                        1,
                        support=[i],
                        param=inputs[i],
                        op=fm_gate,
                        param_prefix=f"phi_{d}{i}{fm_id}{0}",
                        param_val=fm1_val,
                    )
                    fm2 = feature_map_new(
                        1,
                        support=[i],
                        param=inputs[i],
                        op=fm_gate,
                        param_prefix=f"phi_{d}{i}{fm_id}{1}",
                        param_val=fm2_val,
                    )
                    fm1_list.append(fm1)
                    fm2_list.append(fm2)
                ansatz_layers.append(q.kron(fm for fm in fm1_list))
                ansatz_layers.append(q.kron(fm for fm in fm2_list))

        operations.append(q.chain(l for l in ansatz_layers))

    ansatz_layers = []
    for gate_idx, a_gate in enumerate(ansatz_gates):
        rot_1 = []
        rot_2 = []
        for i in range(n_qubits):
            v1 = init_func(size=1)
            v2 = -1 * v1
            param_1 = Parameter(
                name=f"theta_{d}{i}{gate_idx}{0}", trainable=True, value=v1
            )
            param_2 = Parameter(
                name=f"theta_{d}{i}{gate_idx}{1}", trainable=True, value=v2
            )

            g1 = a_gate(target=i, parameter=param_1)
            g2 = a_gate(target=i, parameter=param_2)
            rot_1.append(g1)
            rot_2.append(g2)
        ansatz_layers.append(q.kron(g for g in rot_1))
        ansatz_layers.append(q.kron(g for g in rot_2))

    for i in range(n_qubits - 1):
        ansatz_layers.append(q.CNOT(i, i + 1))
    operations.append(q.chain(b for b in ansatz_layers))

    obs = q.Z(0)
    circ = q.QuantumCircuit(n_qubits, q.chain(o for o in operations))
    model = q.QNN(circuit=circ, observable=obs, inputs=inputs)
    return model


def growth_circ(
    n_qubits: int,
    depth: int,
    inputs: List[str],
    init_func: Callable,
    fm_init: Callable = None,
    ansatz_gates: List[object] = [q.RY],
    fm_gates: List[object] = [q.RX],
):
    operations = []
    for d in range(depth):
        ansatz_layers = []
        for gate_idx, a_gate in enumerate(ansatz_gates):
            rot_1 = []
            rot_2 = []
            for i in range(n_qubits):
                v1 = torch.tensor(init_func(size=1))
                v2 = -1.0 * v1
                param_1 = Parameter(
                    name=f"theta_{d}{i}{gate_idx}{0}", trainable=True, value=v1
                )
                param_2 = Parameter(
                    name=f"theta_{d}{i}{gate_idx}{1}", trainable=True, value=v2
                )

                g1 = a_gate(target=i, parameter=param_1)
                g2 = a_gate(target=i, parameter=param_2)
                rot_1.append(g1)
                rot_2.append(g2)
            ansatz_layers.append(q.kron(g for g in rot_1))
            ansatz_layers.append(q.kron(g for g in rot_2))

        for i in range(n_qubits - 1):
            ansatz_layers.append(q.CNOT(i, i + 1))

        for fm_idx, fm_gate in enumerate(fm_gates):
            fm1_list = []
            fm2_list = []
            for i in range(n_qubits):
                fm1_val = torch.tensor(fm_init(size=1))
                fm2_val = -1.0 * fm1_val

                fm1 = feature_map_new(
                    1,
                    support=[i],
                    param=inputs[i],
                    op=fm_gate,
                    param_prefix=f"phi_{d}{i}{fm_idx}{0}",
                    param_val=fm1_val,
                )
                fm2 = feature_map_new(
                    1,
                    support=[i],
                    param=inputs[i],
                    op=fm_gate,
                    param_prefix=f"phi_{d}{i}{fm_idx}{1}",
                    param_val=fm2_val,
                )
                fm1_list.append(fm1)
                fm2_list.append(fm2)

            ansatz_layers.append(q.kron(fm for fm in fm1_list))
            ansatz_layers.append(q.kron(fm for fm in fm2_list))

        operations.append(q.chain(l for l in ansatz_layers))

    ansatz_layers = []
    for gate_idx, a_gate in enumerate(ansatz_gates):
        rot_1 = []
        rot_2 = []
        for i in range(n_qubits):
            v1 = init_func(size=1)
            v2 = -1 * v1
            param_1 = Parameter(
                name=f"theta_{d}{i}{gate_idx}{0}", trainable=True, value=v1
            )
            param_2 = Parameter(
                name=f"theta_{d}{i}{gate_idx}{1}", trainable=True, value=v2
            )

            g1 = a_gate(target=i, parameter=param_1)
            g2 = a_gate(target=i, parameter=param_2)
            rot_1.append(g1)
            rot_2.append(g2)
        ansatz_layers.append(q.kron(g for g in rot_1))
        ansatz_layers.append(q.kron(g for g in rot_2))

    for i in range(n_qubits - 1):
        ansatz_layers.append(q.CNOT(i, i + 1))
    operations.append(q.chain(b for b in ansatz_layers))

    obs = q.Z(0)
    circ = q.QuantumCircuit(n_qubits, q.chain(o for o in operations))
    model = q.QNN(circuit=circ, observable=obs, inputs=inputs)
    return model
