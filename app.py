import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import joblib

from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

import py3Dmol
from stmol import showmol


# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(layout="wide", page_title="NucLigs Pro Ultra — Physics Engine")

TEMP = "dock_temp"
os.makedirs(TEMP, exist_ok=True)


# ============================================
# PHYSICAL CONSTANTS
# ============================================

EPSILON = 0.2        # vdW depth
SIGMA = 3.5          # vdW radius
COULOMB_CONST = 332  # electrostatic constant


# ============================================
# RECEPTOR LOADER
# ============================================

def load_receptor(pdb_file):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure("rec", pdb_file)

    coords = []
    labels = []

    for model in structure:
        for chain in model:
            for residue in chain:

                if residue.get_resname() == "HOH":
                    continue

                label = f"{residue.get_resname()}_{chain.id}_{residue.get_id()[1]}"

                for atom in residue:

                    coords.append(atom.coord)
                    labels.append(label)

    return structure, np.array(coords), labels


# ============================================
# LIGAND GENERATOR
# ============================================

def generate_ligand(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    AllChem.MMFFOptimizeMolecule(mol)

    coords = mol.GetConformer().GetPositions()

    return mol, np.array(coords)


# ============================================
# PHYSICS ENERGY TERMS
# ============================================

def lennard_jones(rec, lig):

    dist = cdist(lig, rec)

    dist[dist < 0.1] = 0.1

    term = (SIGMA/dist)**12 - (SIGMA/dist)**6

    energy = np.sum(4 * EPSILON * term)

    return energy


def electrostatic(rec, lig):

    dist = cdist(lig, rec)

    dist[dist < 0.1] = 0.1

    energy = np.sum(COULOMB_CONST / dist)

    return energy


def hydrogen_energy(rec, lig):

    dist = cdist(lig, rec)

    hbonds = np.sum(dist < 3.5)

    return -1.5 * hbonds, hbonds


def desolvation_energy(rec, lig):

    dist = cdist(lig, rec)

    buried = np.sum(dist < 5.0)

    return -0.2 * buried


def total_energy(rec, lig):

    evdw = lennard_jones(rec, lig)

    eelec = electrostatic(rec, lig)

    ehb, hbcount = hydrogen_energy(rec, lig)

    edes = desolvation_energy(rec, lig)

    total = evdw + eelec + ehb + edes

    return total, hbcount


# ============================================
# POSE SAMPLING
# ============================================

def sample_poses(lig_coords, center, nposes=50):

    poses = []

    lig_center = lig_coords.mean(axis=0)

    lig_coords = lig_coords - lig_center

    for i in range(nposes):

        theta = np.random.uniform(0, 2*np.pi)

        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,0,1]
        ])

        pose = lig_coords @ rot.T

        pose = pose + center

        poses.append(pose)

    return poses


# ============================================
# INTERACTION ANALYZER
# ============================================

def count_contacts(rec_coords, rec_labels, lig_coords, cutoff=4.0):

    dist = cdist(lig_coords, rec_coords)

    idx = np.where(dist < cutoff)[1]

    residues = list(set(rec_labels[i] for i in idx))

    return residues, len(residues)


# ============================================
# PHYSICOCHEMICAL PROPERTIES
# ============================================

def ligand_props(mol, energy):

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    heavy = mol.GetNumHeavyAtoms()

    LE = (-energy/heavy) if heavy > 0 else 0

    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    violations = 0

    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1

    return mw, logp, tpsa, LE, violations


# ============================================
# MACHINE LEARNING RESCORING
# ============================================

def load_ml():

    if os.path.exists("rescoring_model.pkl"):
        return joblib.load("rescoring_model.pkl")

    return None


ML_MODEL = load_ml()


def ml_score(features):

    if ML_MODEL is None:
        return sum(features)

    return ML_MODEL.predict([features])[0]


# ============================================
# CONFIDENCE SCORE
# ============================================

def confidence(energy, LE, contacts, hb, violations):

    score = 0

    score += min((-energy)*0.1, 40)

    score += min(LE*100, 25)

    score += min(contacts*2, 15)

    score += min(hb*2, 10)

    if violations <= 1:
        score += 10

    return round(score,1)


# ============================================
# STREAMLIT UI
# ============================================

st.title("NucLigs Pro Ultra — Physics-Based Virtual Screening")

st.write("Fully physics- and biochemistry-based screening engine (No docking software used)")


# LOAD LIGANDS

file = st.file_uploader("Upload ligand Excel", type=["xlsx"])

if file:

    lig_df = pd.read_excel(file)

else:

    lig_df = pd.DataFrame({

        "name":["LigA","LigB"],
        "smiles":["CCO","CCN"]

    })


# LOAD RECEPTOR

pdbid = st.text_input("Enter PDB ID","1UA2")

pdb_path = os.path.join(TEMP,"rec.pdb")

if st.button("Load Target"):

    url = f"https://files.rcsb.org/download/{pdbid}.pdb"

    r = requests.get(url)

    open(pdb_path,"w").write(r.text)

    st.success("Target loaded")


# VISUALIZATION

if os.path.exists(pdb_path):

    view = py3Dmol.view(width=800,height=400)

    view.addModel(open(pdb_path).read(),"pdb")

    view.setStyle({"cartoon":{"color":"white"}})

    view.zoomTo()

    showmol(view,height=400,width=800)


# RUN SCREENING

if st.button("Run Physics Screening"):

    structure, rec_coords, rec_labels = load_receptor(pdb_path)

    center = rec_coords.mean(axis=0)

    results = []

    for i,row in lig_df.iterrows():

        mol, lig_coords = generate_ligand(row.smiles)

        if mol is None:
            continue

        poses = sample_poses(lig_coords, center)

        best_energy = 1e9
        best_pose = None
        best_hb = 0

        for pose in poses:

            energy, hb = total_energy(rec_coords, pose)

            if energy < best_energy:

                best_energy = energy
                best_pose = pose
                best_hb = hb

        residues, contacts = count_contacts(rec_coords, rec_labels, best_pose)

        mw, logp, tpsa, LE, violations = ligand_props(mol, best_energy)

        ml = ml_score([best_energy, LE, contacts, best_hb])

        conf = confidence(best_energy, LE, contacts, best_hb, violations)

        results.append({

            "Name":row.name,
            "Binding Energy":best_energy,
            "Ligand Efficiency":LE,
            "Contacts":contacts,
            "HBonds":best_hb,
            "ML Score":ml,
            "Confidence":conf,
            "MW":mw,
            "LogP":logp

        })


    df = pd.DataFrame(results)

    df = df.sort_values("Confidence", ascending=False)

    st.dataframe(df)

    st.download_button(

        "Download Results",

        df.to_csv(index=False),

        "physics_screening_results.csv"
    )
