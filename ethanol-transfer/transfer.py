import os
import numpy as np
import h5py
import json
import argparse
import pandas as pd
from collections import OrderedDict

# Atomic number to symbol map
_ATOMIC_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
    29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr',
    39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
    49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce',
    59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er',
    69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra',
    89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf',
    99: 'Es', 100: 'Fm'
}

def num_to_symbol(n):
    try:
        n_int = int(n)
    except Exception:
        return str(n)
    return _ATOMIC_SYMBOLS.get(n_int, str(n))

def guess_keys(npz):
    files = list(npz.files)
    # common names
    pos_keys = ['positions', 'coords', 'R', 'r', 'pos']
    force_keys = ['forces', 'F', 'force', 'forces_array']
    energy_keys = ['energies', 'E', 'y', 'energy', 'energies_array']
    elements_keys = ['elements', 'species', 'Z', 'atomic_numbers', 'atom_types']

    found = {}
    for k in pos_keys:
        if k in files:
            found['positions'] = k
            break
    for k in force_keys:
        if k in files:
            found['forces'] = k
            break
    for k in energy_keys:
        if k in files:
            found['energies'] = k
            break
    for k in elements_keys:
        if k in files:
            found['elements'] = k
            break
    found['all_keys'] = files
    return found

def ensure_3d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    # try to squeeze extra dims
    arr = arr.reshape((-1, arr.shape[-1]))
    return arr

def create_info_json(output_dir, idx, num_atoms, elements, rcut=5.0):
    json_path = os.path.join(output_dir, f"md17_{idx}", "info.json")
    unique_elements = list(OrderedDict.fromkeys(elements))
    elements_orbital_map = {el: [] for el in unique_elements}
    element_cutoff_radius = {el: rcut for el in unique_elements}
    info_data = {
        "atoms_quantity": num_atoms,
        "spinful": False,
        "elements_orbital_map": elements_orbital_map,
        "elements_force_rcut_map": element_cutoff_radius,
        "max_num_neighbors": 500,
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2, ensure_ascii=False)
    return json_path

def save_poscar(output_dir, idx, elements, positions, cell):
    poscar_path = os.path.join(output_dir, f"md17_{idx}", "POSCAR")
    os.makedirs(os.path.dirname(poscar_path), exist_ok=True)
    unique_elements = []
    counts = []
    for e in elements:
        if e not in unique_elements:
            unique_elements.append(e)
    for e in unique_elements:
        counts.append(elements.count(e))
    with open(poscar_path, 'w', encoding='utf-8') as f:
        f.write("MD17 STRUCTURE\n")
        f.write("  1000.0000000\n")
        for v in cell:
            f.write(f"{v[0]:15.10f} {v[1]:15.10f} {v[2]:15.10f}\n")
        f.write(" ".join(unique_elements) + "\n")
        f.write(" ".join(map(str, counts)) + "\n")
        f.write("Cartesian\n")
        for pos in positions:
            f.write(f"{pos[0]:15.10f} {pos[1]:15.10f} {pos[2]:15.10f}\n")
    return poscar_path

def save_force_h5(output_dir, idx, natoms, cell, forces, energy=None):
    h5_path = os.path.join(output_dir, f"md17_{idx}", "force.h5")
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset("force", data=forces)
        f.create_dataset("cell", data=cell)
        if energy is not None:
            f.create_dataset("energy", data=energy)
        f.create_dataset("stress", data=np.zeros((6,)))
        f.attrs["formula"] = np.array("X%d" % natoms, dtype='S')
        f.attrs["natoms"] = natoms
    return h5_path

def convert_units_if_needed(energies, forces):
    """
    Convert energies and forces from kcal/mol (and kcal/mol/Å) to eV (and eV/Å).
    Note: the user confirmed input units are kcal/mol (energy) and kcal/mol/Å (force),
    so this applies a fixed conversion factor 0.0433641.
    Returns (energies_eV, forces_eV, note)
    """
    energies = np.asarray(energies, dtype=float)
    forces = np.asarray(forces, dtype=float)
    factor = 0.0433641
    energies = energies * factor
    forces = forces * factor
    unit_note = 'converted from kcal/mol (and kcal/mol/Angstrom) to eV (and eV/Angstrom) using factor 0.0433641'
    return energies, forces, unit_note

def parse_npz_file(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    keys = guess_keys(data)
    print('npz keys found:', keys['all_keys'])
    # positions
    if 'positions' in keys:
        pos_key = keys['positions']
        positions_all = data[pos_key]
    else:
        raise KeyError("Could not find 'positions' field in npz; please check available keys")
    # forces
    if 'forces' in keys:
        force_key = keys['forces']
        forces_all = data[force_key]
    else:
        raise KeyError("Could not find 'forces' field in npz; please check available keys")
    # energies (optional)
    energies_all = None
    if 'energies' in keys:
        energies_all = data[keys['energies']]
    # elements (optional)
    elements_all = None
    if 'nuclear_charges' in data.files:
        elements_all = data['nuclear_charges']

    # Normalize shapes: structures as first dimension
    pos_arr = np.asarray(positions_all)
    force_arr = np.asarray(forces_all)

    # Ensure first dim is structures
    if pos_arr.ndim == 2 and force_arr.ndim == 2:
        # single structure
        pos_arr = pos_arr[np.newaxis, ...]
        force_arr = force_arr[np.newaxis, ...]
    elif pos_arr.ndim == 3 and force_arr.ndim == 3:
        pass
    else:
        # try to reshape if shapes align
        try:
            n_structs = min(pos_arr.shape[0], force_arr.shape[0])
            pos_arr = pos_arr[:n_structs]
            force_arr = force_arr[:n_structs]
        except Exception:
            raise ValueError('Could not interpret positions/forces shapes: {} / {}'.format(pos_arr.shape, force_arr.shape))

    # energies
    if energies_all is not None:
        e_arr = np.asarray(energies_all)
        if e_arr.ndim == 0:
            e_arr = np.repeat(e_arr, pos_arr.shape[0])
        elif e_arr.ndim == 1 and e_arr.shape[0] >= pos_arr.shape[0]:
            e_arr = e_arr[:pos_arr.shape[0]]
        else:
            # try to broadcast
            e_arr = np.asarray(e_arr)
    else:
        e_arr = np.full((pos_arr.shape[0],), np.nan)

    # elements
    if elements_all is None:
        # try to infer from single structure by element symbols arrays or atomic numbers
        elements_list = None
    else:
        elements_list = list(elements_all)
        # if elements_all is (n_structs, natoms) or (natoms,) handle
        if isinstance(elements_all, np.ndarray):
            if elements_all.ndim == 2:
                # pick first structure's element list if it's same for all
                elements_list = list(elements_all[0])
            elif elements_all.ndim == 1 and len(elements_all) == pos_arr.shape[1]:
                elements_list = list(elements_all)
            else:
                # fallback
                elements_list = [str(x) for x in elements_all]
    return pos_arr, force_arr, e_arr, elements_list

def find_npz_file():
    """
    Automatically find the .npz file based on directory structure.
    Assumes current directory is Name-transfer and npz file is in Name/ subfolder.
    """
    current_dir = os.path.basename(os.path.abspath('.'))
    if not current_dir.endswith('-transfer'):
        raise ValueError(f"Current directory name '{current_dir}' does not end with '-transfer'")
    
    # Extract molecule name by removing '-transfer' suffix
    molecule_name = current_dir.replace('-transfer', '')
    npz_dir = os.path.join('.', molecule_name)
    
    if not os.path.exists(npz_dir):
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")
    
    # Look for .npz files in the directory
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")
    elif len(npz_files) > 1:
        print(f"Warning: Multiple .npz files found in {npz_dir}, using: {npz_files[0]}")
    
    npz_path = os.path.join(npz_dir, npz_files[0])
    print(f"Auto-detected NPZ file: {npz_path}")
    return npz_path

def load_training_indices(csv_path, max_structs=None):
    """
    Load training indices from CSV file and ensure they are within valid range.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Index file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, header=None)
    indices = df.iloc[:, 0].values  # Assume indices are in first column
    
    # Validate index range
    if max_structs is not None:
        indices = indices[indices < max_structs]
        if len(indices) == 0:
            raise ValueError("All indices are out of valid range!")
        print(f"Loaded {len(indices)} valid indices (total structures: {max_structs}).")
    
    return np.sort(indices)  # Return sorted indices

def main():
    parser = argparse.ArgumentParser(description='Convert MD17-like .npz using specified indices')
    parser.add_argument('--output-dir', '-o', default=os.path.join(os.path.dirname(__file__), 'dft'))
    parser.add_argument('--index-csv', default="./index_train_01.csv", help='Training indices CSV file path')
    parser.add_argument('--rcut', type=float, default=5.0)
    parser.add_argument('--cell-scale', type=float, default=1.0)
    args = parser.parse_args()

    # Automatically find the NPZ file
    npz_path = find_npz_file()
    output_dir = args.output_dir
    index_csv_path = args.index_csv
    
    os.makedirs(output_dir, exist_ok=True)

    # Parse NPZ file to get data
    pos_arr, force_arr, e_arr, elements_list = parse_npz_file(npz_path)
    n_structs = pos_arr.shape[0]

    # Load training indices
    selected_indices = load_training_indices(index_csv_path, max_structs=n_structs)
    n = len(selected_indices)

    # Unit conversion
    energies_converted, forces_converted, note = convert_units_if_needed(e_arr, force_arr)
    print('Unit conversion:', note)
    print(f'Using index file: {index_csv_path}')
    print(f'Processing {n} structures (total available: {n_structs})')

    # Element processing logic
    if elements_list is None:
        natoms = pos_arr.shape[1]
        elements = ['X'] * natoms
    else:
        if all([isinstance(x, (np.integer, int, float)) or (isinstance(x, bytes) and x.isdigit()) for x in elements_list]):
            elements = [num_to_symbol(x) for x in elements_list]
        else:
            elements = [str(x).decode() if isinstance(x, bytes) else str(x) for x in elements_list]

    cell = np.eye(3) * args.cell_scale

    # Process each structure by index
    for i, original_idx in enumerate(selected_indices):
        positions = ensure_3d(pos_arr[original_idx])
        forces = ensure_3d(forces_converted[original_idx])
        energy = float(energies_converted[original_idx]) if not np.isnan(energies_converted[original_idx]) else None
        natoms = positions.shape[0]

        # Element assignment logic
        if elements_list is not None and isinstance(elements_list, (list, np.ndarray)) and len(elements_list) == n_structs:
            el = elements_list[original_idx]
            if isinstance(el, (list, np.ndarray)):
                elements_this = [x.decode() if isinstance(x, bytes) else str(x) for x in el]
            else:
                elements_this = [str(x) for x in elements_list]
        else:
            elements_this = elements if len(elements) == natoms else ['X'] * natoms

        # Save files
        save_poscar(output_dir, i, elements_this, positions, cell)
        save_force_h5(output_dir, i, natoms, cell, forces, energy)
        create_info_json(output_dir, i, natoms, elements_this, rcut=args.rcut)
        
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"Processed structure {i+1}/{n} (original index: {original_idx})")

    print(f"Processing completed: {n} structures converted, output directory: {output_dir}")

if __name__ == '__main__':
    main()