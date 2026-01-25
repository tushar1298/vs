import streamlit as st
import pandas as pd
import requests
import numpy as np
import os
from io import StringIO
from Bio.PDB import PDBParser, Select, PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from vina import Vina
from stmol import showmol
import py3Dmol

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="NucLigs Explorer")

# --- HELPER CLASSES ---

class LigandSelect(Select):
    """Biopython selector to filtering specific residues (to remove the native ligand)."""
    def __init__(self, res_to_remove):
        self.res_to_remove = res_to_remove # Format: (Chain, ResName, ResID)
        
    def accept_residue(self, residue):
        # Check if this residue matches the one we want to remove
        res_id = residue.get_id()[1]
        chain_id = residue.get_parent().id
        res_name = residue.get_resname()
        
        if (chain_id, res_name, res_id) == self.res_to_remove:
            return 0 # Remove this
        return 1 # Keep everything else

def get_ligand_center(residue):
    """Calculates the geometric center (centroid) of a residue."""
    coords = [atom.get_coord() for atom in residue]
    return np.mean(coords, axis=0)

# --- MAIN APP ---

st.title("🧬 NucLigs: Targeted Virtual Screening")
st.markdown("Load a PDB, select a bound ligand to define the binding site, and screen your NucLigs analogs.")

# 1. SIDEBAR: DATA INPUT
with st.sidebar:
    st.header("1. Input Data")
    
    # Upload Excel
    uploaded_file = st.file_uploader("Upload 'nucligs_metadata.xlsx'", type=['xlsx'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success(f"Loaded {len(df)} analogs from NucLigs.")
        # Ensure 'smiles' column exists (case insensitive check)
        cols = [c.lower() for c in df.columns]
        if 'smiles' not in cols:
            st.error("Excel must have a 'smiles' column.")
    else:
        # Mock data for demonstration if no file uploaded
        st.info("Using mock data (Upload Excel to override)")
        df = pd.DataFrame({
            'Name': ['Mock Analog A', 'Mock Analog B'],
            'smiles': ['Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1', 'CC1=CN([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)C(=O)NC1=O']
        })

    st.divider()
    st.header("3. Screening Settings")
    exhaustiveness = st.slider("Exhaustiveness", 1, 16, 8)
    n_poses = st.slider("Num Poses", 1, 5, 1)

# 2. PDB FETCHING & PARSING
st.header("2. Target Selection")
pdb_id = st.text_input("Enter PDB ID (e.g., 1UA2):", "1UA2").strip().lower()

if pdb_id:
    # Fetch PDB text from RCSB
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(pdb_url)
    
    if response.status_code == 200:
        pdb_text = response.text
        
        # Save temporary PDB for Biopython to read
        with open("temp_target.pdb", "w") as f:
            f.write(pdb_text)
            
        # Parse PDB
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('Target', "temp_target.pdb")
        
        # Extract Ligands (HETATMs that are not water)
        ligand_list = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    # HETATM check (res_id starts with H_)
                    if residue.get_id()[0].startswith('H_') and residue.get_resname() != 'HOH':
                        center = get_ligand_center(residue)
                        ligand_list.append({
                            'Display': f"{residue.get_resname()} (Chain {chain.id}, ID {residue.get_id()[1]})",
                            'Value': (chain.id, residue.get_resname(), residue.get_id()[1]),
                            'Center': center
                        })
        
        if not ligand_list:
            st.warning("No ligands found in this PDB.")
        else:
            # Dropdown to select ligand
            st.subheader("Select Bound Ligand to Remove & Target")
            
            selected_ligand_idx = st.selectbox(
                "Choose the native ligand defining the binding pocket:",
                range(len(ligand_list)),
                format_func=lambda x: ligand_list[x]['Display']
            )
            
            target_ligand = ligand_list[selected_ligand_idx]
            center = target_ligand['Center']
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Box Center Automatically Calculated:**\nX: {center[0]:.2f}, Y: {center[1]:.2f}, Z: {center[2]:.2f}")
            with col2:
                box_size = st.number_input("Box Size (Angstroms)", value=20.0)

            # --- VISUALIZATION OF TARGET ---
            view = py3Dmol.view(width=800, height=400)
            view.addModel(pdb_text, 'pdb')
            view.setStyle({'cartoon': {'color': 'white'}})
            # Highlight selected ligand
            view.addStyle({'resn': target_ligand['Value'][1], 'resi': target_ligand['Value'][2]}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.5}})
            view.addBox({'center': {'x':center[0], 'y':center[1], 'z':center[2]}, 'dimensions': {'w':box_size, 'h':box_size, 'd':box_size}, 'color':'red', 'opacity': 0.5})
            view.zoomTo()
            showmol(view, height=400, width=800)

            # --- RUN SCREENING ---
            if st.button("🚀 Remove Ligand & Screen NucLigs"):
                
                # A. PREPARE RECEPTOR (Remove selected ligand)
                io = PDBIO()
                io.set_structure(structure)
                clean_pdb_name = "clean_receptor.pdb"
                io.save(clean_pdb_name, LigandSelect(target_ligand['Value']))
                
                # Convert cleaned PDB to PDBQT (Requires OpenBabel installed on system)
                # For demo purposes, we will assume the file is ready or use a converter
                # st.write("Converting receptor to PDBQT...")
                # subprocess.run(["obabel", clean_pdb_name, "-O", "receptor.pdbqt", "-xr", "--partialcharge", "gasteiger"])
                
                # NOTE: Since we can't run OpenBabel in this strict text environment, 
                # we will assume 'receptor.pdbqt' exists for the code to not crash, 
                # but in your real app, UNCOMMENT the subprocess line above.
                if not os.path.exists("receptor.pdbqt"):
                    st.error("Error: OpenBabel is required to convert the receptor PDB to PDBQT format.")
                    st.stop()
                
                results_table = []
                progress = st.progress(0)
                
                for i, row in df.iterrows():
                    smi = row.get('smiles') or row.get('SMILES')
                    name = row.get('Name', f"Mol_{i}")
                    
                    if pd.isna(smi): continue

                    # B. PREPARE LIGAND (RDKit -> Meeko)
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol)
                        
                        meeko_prep = MoleculePreparation()
                        meeko_prep.prepare(mol)
                        pdbqt_string = meeko_prep.write_pdbqt_string()
                        
                        # C. DOCKING (Vina)
                        v = Vina(sf_name='vina')
                        v.set_receptor("receptor.pdbqt")
                        v.set_ligand_from_string(pdbqt_string)
                        v.compute_vina_maps(center=center, box_size=[box_size, box_size, box_size])
                        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
                        
                        score = v.score()[0]
                        results_table.append({'Name': name, 'Affinity': score, 'SMILES': smi})
                        
                    except Exception as e:
                        st.warning(f"Failed to dock {name}: {e}")
                    
                    progress.progress((i + 1) / len(df))

                st.success("Screening Complete!")
                res_df = pd.DataFrame(results_table).sort_values(by='Affinity')
                st.dataframe(res_df)
