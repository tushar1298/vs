import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import shutil
import requests
import joblib

from Bio.PDB import PDBParser, Select, PDBIO
from scipy.spatial.distance import cdist

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

from meeko import MoleculePreparation
from vina import Vina

import py3Dmol
from stmol import showmol


# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(layout="wide", page_title="NucLigs Pro Ultra")

TEMP = "dock_temp"
os.makedirs(TEMP, exist_ok=True)


# ============================================
# CHECK GPU SUPPORT
# ============================================

def check_gpu():

    return shutil.which("autodock_gpu") is not None


GPU_AVAILABLE = check_gpu()


# ============================================
# AUTOMATIC POCKET DETECTION
# ============================================

def detect_pocket(structure):

    coords = []

    for atom in structure.get_atoms():

        if atom.get_parent().get_resname() != "HOH":

            coords.append(atom.coord)

    coords = np.array(coords)

    center = coords.mean(axis=0)

    size = coords.max(axis=0) - coords.min(axis=0)

    box = size + 10

    return center, box


# ============================================
# INTERACTION ANALYZER
# ============================================

class InteractionAnalyzer:

    def __init__(self, structure, ligand_pdbqt):

        self.rec_atoms = []
        self.rec_labels = []

        for model in structure:
            for chain in model:
                for residue in chain:

                    if residue.get_resname() == "HOH":
                        continue

                    label = f"{residue.get_resname()}_{chain.id}_{residue.get_id()[1]}"

                    for atom in residue:

                        self.rec_atoms.append(atom.coord)
                        self.rec_labels.append(label)

        self.rec_atoms = np.array(self.rec_atoms)

        self.lig_atoms = []

        for line in ligand_pdbqt.splitlines():

            if line.startswith("ATOM") or line.startswith("HETATM"):

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                self.lig_atoms.append([x,y,z])

        self.lig_atoms = np.array(self.lig_atoms)


    def contacts(self, cutoff=4.0):

        if len(self.lig_atoms)==0:
            return [],0

        dist = cdist(self.lig_atoms, self.rec_atoms)

        idx = np.where(dist < cutoff)

        residues = list(set(self.rec_labels[i] for i in idx[1]))

        return residues, len(residues)


# ============================================
# HYDROGEN BOND DETECTOR
# ============================================

class HydrogenBondDetector:

    def __init__(self, lig, rec):

        self.lig = lig
        self.rec = rec


    def detect(self, cutoff=3.5):

        dist = cdist(self.lig, self.rec)

        hb = np.where(dist < cutoff)

        return len(hb[0])


# ============================================
# PHYSICOCHEMICAL PROPERTIES
# ============================================

def ligand_props(mol, energy):

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    heavy = mol.GetNumHeavyAtoms()

    LE = (-energy/heavy) if heavy>0 else 0

    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    violations = 0

    if mw>500: violations+=1
    if logp>5: violations+=1
    if hbd>5: violations+=1
    if hba>10: violations+=1

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

    score += min((-energy)*4, 40)
    score += min(LE*100, 25)
    score += min(contacts*2, 15)
    score += min(hb*2, 10)

    if violations<=1:
        score += 10

    return round(score,1)


# ============================================
# RECEPTOR PREPARATION
# ============================================

def prepare_receptor(pdb):

    pdbqt = os.path.join(TEMP,"rec.pdbqt")

    subprocess.run([
        "obabel",
        pdb,
        "-O",
        pdbqt,
        "-xr",
        "--partialcharge",
        "gasteiger"
    ])

    return pdbqt


# ============================================
# DOCKING ENGINE
# ============================================

def dock_vina(rec_pdbqt, lig_pdbqt, center, box):

    v = Vina()

    v.set_receptor(rec_pdbqt)

    v.set_ligand_from_string(lig_pdbqt)

    v.compute_vina_maps(center=center, box_size=box)

    v.dock(exhaustiveness=16, n_poses=5)

    poses = v.poses(n_poses=5)

    energies = v.energies(n_poses=5)

    return poses[0], energies[0][0]


# ============================================
# STREAMLIT UI
# ============================================

st.title("NucLigs Pro Ultra — Scientific Docking Platform")

st.write("Publication-grade RNA/protein virtual screening")


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

if st.button("Load Target"):

    url = f"https://files.rcsb.org/download/{pdbid}.pdb"

    pdb_path = os.path.join(TEMP,"rec.pdb")

    r = requests.get(url)

    open(pdb_path,"w").write(r.text)

    st.success("Target loaded")


# VISUALIZATION

if os.path.exists(os.path.join(TEMP,"rec.pdb")):

    view = py3Dmol.view(width=800,height=400)

    view.addModel(open(os.path.join(TEMP,"rec.pdb")).read(),"pdb")

    view.setStyle({"cartoon":{"color":"white"}})

    view.zoomTo()

    showmol(view,height=400,width=800)


# RUN SCREENING

if st.button("Run Screening"):

    parser = PDBParser()

    structure = parser.get_structure("rec", os.path.join(TEMP,"rec.pdb"))

    center, box = detect_pocket(structure)

    rec_pdbqt = prepare_receptor(os.path.join(TEMP,"rec.pdb"))

    results = []

    for i,row in lig_df.iterrows():

        mol = Chem.MolFromSmiles(row.smiles)

        mol = Chem.AddHs(mol)

        AllChem.EmbedMolecule(mol)

        prep = MoleculePreparation()

        prep.prepare(mol)

        lig_pdbqt = prep.write_pdbqt_string()

        pose, energy = dock_vina(rec_pdbqt, lig_pdbqt, center, box)

        analyzer = InteractionAnalyzer(structure, pose)

        residues, contacts = analyzer.contacts()

        hb_detector = HydrogenBondDetector(
            analyzer.lig_atoms,
            analyzer.rec_atoms
        )

        hb = hb_detector.detect()

        mw, logp, tpsa, LE, violations = ligand_props(mol, energy)

        ml = ml_score([energy, LE, contacts, hb])

        conf = confidence(energy, LE, contacts, hb, violations)

        results.append({

            "Name":row.name,
            "Affinity":energy,
            "LE":LE,
            "Contacts":contacts,
            "HBonds":hb,
            "ML Score":ml,
            "Confidence":conf,
            "MW":mw,
            "LogP":logp

        })


    df = pd.DataFrame(results)

    df = df.sort_values("Confidence",ascending=False)

    st.dataframe(df)

    st.download_button(

        "Download Results",

        df.to_csv(index=False),

        "screening_results.csv"
    )
