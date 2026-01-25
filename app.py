import streamlit as st
import pandas as pd
import requests
import numpy as np
import os
import subprocess
import shutil
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

def check_openbabel():
    """Checks if OpenBabel is installed and accessible."""
    return shutil.which("obabel") is not None

# --- MAIN APP ---
st.title("🧬 NucLigs: Virtual Screening Tool")
st.markdown("""
**Workflow:**
1. Upload your Nucleotide Analog Library (Excel).
2. Input a PDB ID (Target).
3. Select the bound ligand to define the binding pocket.
4. Run Docking.
""")

# Check for OpenBabel
if not check_openbabel():
    st.error("⚠️ System Error: 'openbabel' is not installed. If on Streamlit Cloud, ensure 'packages.txt' contains 'openbabel'.")

# 1. SIDEBAR: DATA INPUT
with st.sidebar:
    st.header("1. Input Data")
    
    # Upload Excel
    uploaded_file = st.file_uploader("Upload 'nucligs_metadata.xlsx'", type=['xlsx'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success(f"Loaded {len(df)} analogs.")
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        if 'smiles' not in df.columns:
            st.error("Excel must have a 'smiles' column.")
            st.stop()
    else:
        st.info("Using Demo Data (Upload Excel to override)")
        # Demo: Gemcitabine and Remdesivir analogs
        df = pd.DataFrame({
            'name': ['Gemcitabine Analog', 'Remdesivir Metabolite'],
            'smiles': [
                'NC1=NC(=O)N(C=C1)C2C(C(C(O2)CO)O)(F)F', 
                'Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1'
            ]
        })

    st.divider()
    st.header("3. Screening Settings")
    exhaustiveness = st.slider("Exhaustiveness (Accuracy)", 1, 16, 4) # Lower default for cloud speed
    n_poses = st.slider("Num Poses", 1, 5, 1)

# 2. PDB FETCHING & PARSING
st.header("2. Target Preparation")
pdb_id = st.text_input("Enter PDB ID (e.g., 1UA2 for HIV-RT):", "1UA2").strip().lower()

if pdb_id:
    # Use a temporary directory for processing
    temp_dir = "temp_docking"
    os.makedirs(temp_dir, exist_ok=True)
    
    pdb_path = os.path.join(temp_dir, "target.pdb")
    
    # Fetch PDB
    if not os.path.exists(pdb_path):
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(pdb_url)
        if response.status_code == 200:
            with open(pdb_path, "w") as f:
                f.write(response.text)
        else:
            st.error("Invalid PDB ID")
            st.stop()
            
    # Parse PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('Target', pdb_path)
    
    # Extract Ligands
    ligand_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0].startswith('H_') and residue.get_resname() != 'HOH':
                    center = get_ligand_center(residue)
                    ligand_list.append({
                        'Display': f"{residue.get_resname()} (Chain {chain.id}, ID {residue.get_id()[1]})",
                        'Value': (chain.id, residue.get_resname(), residue.get_id()[1]),
                        'Center': center
                    })
    
    if not ligand_list:
        st.warning("No ligands found to define binding site.")
    else:
        st.subheader("Select Binding Pocket")
        selected_ligand_idx = st.selectbox(
            "Choose the bound ligand to remove & target:",
            range(len(ligand_list)),
            format_func=lambda x: ligand_list[x]['Display']
        )
        
        target_ligand = ligand_list[selected_ligand_idx]
        center = target_ligand['Center']
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Grid Center:**\nX: {center[0]:.2f}, Y: {center[1]:.2f}, Z: {center[2]:.2f}")
        with col2:
            box_size = st.number_input("Box Size (Å)", value=20.0)

        # Draw 3D Box
        view = py3Dmol.view(width=800, height=400)
        view.addModel(open(pdb_path).read(), 'pdb')
        view.setStyle({'cartoon': {'color': 'white'}})
        view.addStyle({'resn': target_ligand['Value'][1], 'resi': target_ligand['Value'][2]}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.5}})
        view.addBox({'center': {'x':center[0], 'y':center[1], 'z':center[2]}, 'dimensions': {'w':box_size, 'h':box_size, 'd':box_size}, 'color':'red', 'opacity': 0.5})
        view.zoomTo()
        showmol(view, height=400, width=800)

        # --- EXECUTION ---
        if st.button("🚀 Run Virtual Screening"):
            results_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # A. Remove Ligand & Save Clean PDB
            clean_pdb_path = os.path.join(temp_dir, "receptor_clean.pdb")
            receptor_pdbqt_path = os.path.join(temp_dir, "receptor.pdbqt")
            
            io = PDBIO()
            io.set_structure(structure)
            io.save(clean_pdb_path, LigandSelect(target_ligand['Value']))
            
            # B. Convert PDB -> PDBQT using OpenBabel
            status_text.text("Preparing Receptor (OpenBabel)...")
            cmd = [
                "obabel", clean_pdb_path, 
                "-O", receptor_pdbqt_path, 
                "-xr", "--partialcharge", "gasteiger"
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                st.error("Failed to convert receptor to PDBQT.")
                st.stop()

            results_table = []
            
            # C. Loop through NucLigs
            total_mols = len(df)
            for i, row in df.iterrows():
                name = row.get('name', f"Mol_{i}")
                smi = row.get('smiles')
                
                status_text.text(f"Docking: {name}")
                
                try:
                    # RDKit Setup
                    mol = Chem.MolFromSmiles(smi)
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol)
                    
                    # Meeko PDBQT prep
                    meeko_prep = MoleculePreparation()
                    meeko_prep.prepare(mol)
                    ligand_pdbqt = meeko_prep.write_pdbqt_string()
                    
                    # Vina Docking
                    v = Vina(sf_name='vina')
                    v.set_receptor(receptor_pdbqt_path)
                    v.set_ligand_from_string(ligand_pdbqt)
                    v.compute_vina_maps(center=center, box_size=[box_size, box_size, box_size])
                    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
                    
                    score = v.score()[0]
                    results_table.append({'Name': name, 'Affinity (kcal/mol)': score, 'SMILES': smi})
                    
                except Exception as e:
                    print(f"Error docking {name}: {e}")
                
                progress_bar.progress((i + 1) / total_mols)
            
            status_text.text("Done!")
            res_df = pd.DataFrame(results_table).sort_values(by='Affinity (kcal/mol)')
            
            st.subheader("Results")
            st.dataframe(res_df)
            
            # Download CSV
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "docking_results.csv", "text/csv")
