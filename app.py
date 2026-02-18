import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import shutil
import requests

from scipy.spatial import distance

from Bio.PDB import PDBParser, Select, PDBIO

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from meeko import MoleculePreparation
from vina import Vina

import py3Dmol
from stmol import showmol

from streamlit_ketcher import st_ketcher


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    layout="wide",
    page_title="NucLigs Pro Ultra",
    page_icon="🧬"
)

TEMP_DIR = "temp_workspace"
os.makedirs(TEMP_DIR, exist_ok=True)


# =========================================================
# CONFIDENCE SCORE MODEL
# =========================================================

def calculate_confidence(affinity, le, hbonds, contacts):

    score = 0

    # affinity contribution
    score += min(max((-affinity) * 5, 0), 40)

    # ligand efficiency
    score += min(le * 50, 25)

    # hydrogen bonds
    score += min(hbonds * 5, 20)

    # contacts
    score += min(contacts * 2, 15)

    return round(score, 1)


# =========================================================
# INTERACTION ANALYSIS
# =========================================================

def analyze_interactions(protein_model, ligand_pdbqt):

    rec_coords = []
    rec_labels = []

    for chain in protein_model:
        for res in chain:

            if res.get_resname() == "HOH":
                continue

            for atom in res:
                rec_coords.append(atom.coord)
                rec_labels.append(
                    f"{res.get_resname()}{res.get_id()[1]}"
                )

    rec_coords = np.array(rec_coords)

    lig_coords = []

    for line in ligand_pdbqt.split("\n"):

        if line.startswith(("ATOM", "HETATM")):

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            lig_coords.append([x, y, z])

    lig_coords = np.array(lig_coords)

    if len(lig_coords) == 0:

        return 0, 0, []

    dist = distance.cdist(lig_coords, rec_coords)

    contacts = np.where(dist < 4.0)

    hbonds = np.sum(dist < 3.5)

    hydrophobic = np.sum((dist > 3.5) & (dist < 5.0))

    residues = list(set(rec_labels[i] for i in contacts[1]))

    return hbonds, hydrophobic, residues


# =========================================================
# PROPERTY CALCULATOR
# =========================================================

def calculate_properties(mol, affinity):

    heavy_atoms = mol.GetNumHeavyAtoms()

    le = (-affinity / heavy_atoms) if heavy_atoms > 0 else 0

    mw = Descriptors.MolWt(mol)

    logp = Descriptors.MolLogP(mol)

    return round(le, 3), round(mw, 1), round(logp, 2)


# =========================================================
# STRUCTURE DRAWER / INPUT
# =========================================================

with st.sidebar:

    st.header("Library Input")

    mode = st.radio(
        "Input method",
        ["Upload File", "Paste SMILES", "Draw Molecule"]
    )

    df = None

    if mode == "Upload File":

        file = st.file_uploader(
            "Upload Excel/CSV/TXT",
            type=["xlsx", "csv", "txt"]
        )

        if file:

            if file.name.endswith(".xlsx"):

                df = pd.read_excel(file)

            else:

                df = pd.read_csv(file, sep=None, engine="python")

            df.columns = [c.lower() for c in df.columns]

            if "name" not in df.columns:

                df["name"] = [
                    f"Mol_{i}" for i in range(len(df))
                ]

    elif mode == "Paste SMILES":

        text = st.text_area("Paste SMILES")

        if text:

            smiles = text.splitlines()

            df = pd.DataFrame({

                "name":
                [f"Mol_{i}" for i in range(len(smiles))],

                "smiles": smiles

            })

    elif mode == "Draw Molecule":

        smiles = st_ketcher()

        if smiles:

            df = pd.DataFrame({

                "name": ["Drawn"],
                "smiles": [smiles]

            })


if df is None:

    df = pd.DataFrame({

        "name": ["Example"],
        "smiles": ["CCO"]

    })


st.success(f"{len(df)} molecules ready")


# =========================================================
# LOAD RECEPTOR
# =========================================================

st.header("Target")

pdb_id = st.text_input("PDB ID", "1UA2")

pdb_path = os.path.join(TEMP_DIR, f"{pdb_id}.pdb")

if st.button("Load Target"):

    r = requests.get(
        f"https://files.rcsb.org/download/{pdb_id}.pdb"
    )

    open(pdb_path, "w").write(r.text)

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure("rec", pdb_path)

    coords = np.array(
        [atom.coord for atom in structure.get_atoms()]
    )

    center = coords.mean(axis=0)

    box = coords.max(axis=0) - coords.min(axis=0) + 10

    st.session_state.structure = structure

    st.session_state.center = center

    st.session_state.box = box

    st.success("Target ready")


# =========================================================
# RUN SCREENING
# =========================================================

if st.button("Run Screening"):

    structure = st.session_state.structure

    center = st.session_state.center

    box = st.session_state.box

    receptor_pdbqt = os.path.join(
        TEMP_DIR,
        "receptor.pdbqt"
    )

    subprocess.run([
        "obabel",
        pdb_path,
        "-O",
        receptor_pdbqt,
        "-xr",
        "--partialcharge",
        "gasteiger"
    ])

    results = []

    progress = st.progress(0)

    for i, row in df.iterrows():

        mol = Chem.MolFromSmiles(row.smiles)

        mol = Chem.AddHs(mol)

        AllChem.EmbedMolecule(mol)

        prep = MoleculePreparation()

        prep.prepare(mol)

        lig = prep.write_pdbqt_string()

        v = Vina()

        v.set_receptor(receptor_pdbqt)

        v.set_ligand_from_string(lig)

        v.compute_vina_maps(
            center=[float(x) for x in center],
            box_size=[float(x) for x in box]
        )

        v.dock(exhaustiveness=8, n_poses=1)

        affinity = v.score()[0]

        pose = v.poses(n_poses=1)

        hbonds, hydrophobic, residues = analyze_interactions(
            structure[0],
            pose
        )

        le, mw, logp = calculate_properties(mol, affinity)

        confidence = calculate_confidence(
            affinity,
            le,
            hbonds,
            len(residues)
        )

        results.append({

            "Ligand": row.name,

            "Affinity":
            round(affinity, 2),

            "LE":
            le,

            "Confidence":
            confidence,

            "HBonds":
            hbonds,

            "Hydrophobic":
            hydrophobic,

            "Contacts":
            len(residues),

            "MW":
            mw,

            "LogP":
            logp

        })

        progress.progress((i + 1) / len(df))


# =========================================================
# PROFESSIONAL RESULT TABLE
# =========================================================

    res = pd.DataFrame(results)

    res = res.sort_values("Confidence", ascending=False)

    res.insert(0, "Rank", range(1, len(res) + 1))


    st.subheader("Screening Results")

    st.dataframe(

        res,

        column_config={

            "Rank":
            st.column_config.NumberColumn(),

            "Confidence":
            st.column_config.ProgressColumn(
                min_value=0,
                max_value=100
            ),

            "Affinity":
            st.column_config.NumberColumn(
                format="%.2f kcal/mol"
            ),

            "LE":
            st.column_config.NumberColumn(
                format="%.3f"
            ),

            "HBonds":
            st.column_config.ProgressColumn(
                min_value=0,
                max_value=res["HBonds"].max()
            ),

            "Contacts":
            st.column_config.ProgressColumn(
                min_value=0,
                max_value=res["Contacts"].max()
            )

        },

        use_container_width=True

    )


# =========================================================
# DOWNLOAD
# =========================================================

    st.download_button(

        "Download Results",

        res.to_csv(index=False),

        "screening_results.csv"
    )
