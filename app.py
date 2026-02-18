import streamlit as st
import pandas as pd
import requests
import numpy as np
import os
import subprocess
import shutil
from Bio.PDB import PDBParser, Select, PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation
from vina import Vina
from stmol import showmol
import py3Dmol
from scipy.spatial import distance

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="NucLigs Pro: Bio-Physics Screener")

# --- ADVANCED PHYSICS & BIOLOGY CLASSES ---

class InteractionAnalyzer:
    """
    Physics-based analysis of the binding pocket.
    Calculates distances between Ligand heavy atoms and Receptor residues.
    """
    def __init__(self, receptor_structure, ligand_pdbqt_string):
        self.receptor_atoms = []
        self.receptor_residues = []
        
        # 1. Parse Receptor Atoms (Heavy atoms only for speed)
        for model in receptor_structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() != "HOH":
                        for atom in residue:
                            self.receptor_atoms.append(atom.get_coord())
                            self.receptor_residues.append(f"{residue.get_resname()}{residue.get_id()[1]}")
        self.receptor_coords = np.array(self.receptor_atoms)

        # 2. Parse Ligand Atoms from Vina Output String
        self.lig_coords = []
        for line in ligand_pdbqt_string.split('\n'):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # PDBQT format: X is 30-38, Y is 38-46, Z is 46-54
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    self.lig_coords.append([x, y, z])
                except ValueError:
                    continue
        self.lig_coords = np.array(self.lig_coords)

    def get_interacting_residues(self, cutoff=3.5):
        """Returns unique list of residues within 'cutoff' Angstroms of ligand."""
        if len(self.lig_coords) == 0: return "Parsing Error"
        
        # Physics: Calculate Euclidean distance matrix
        dists = distance.cdist(self.lig_coords, self.receptor_coords, 'euclidean')
        
        # Find receptor atoms within cutoff
        interacting_indices = np.where(dists < cutoff)[1]
        unique_residues = sorted(list(set([self.receptor_residues[i] for i in interacting_indices])))
        
        return ", ".join(unique_residues)

def calculate_physchem_props(mol, score):
    """
    Calculates Ligand Efficiency (LE) and Lipinski Violations.
    Physics-Based Metric: LE = -Score / Heavy_Atoms
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    
    # Physics-Based Metrics
    le = (-score / heavy_atoms) if heavy_atoms > 0 else 0
    
    # Biological: Lipinski Rule of 5 Check
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    pass_lipinski = "✅" if violations <= 1 else "⚠️"
    
    return {
        "MW": round(mw, 1),
        "LogP": round(logp, 2),
        "LE": round(le, 2),
        "Lipinski": pass_lipinski
    }

# --- STANDARD CLASSES ---
class LigandSelect(Select):
    def __init__(self, res_to_remove):
        self.res_to_remove = res_to_remove
    def accept_residue(self, residue):
        res_id = residue.get_id()[1]
        chain_id = residue.get_parent().id
        res_name = residue.get_resname()
        if (chain_id, res_name, res_id) == self.res_to_remove: return 0
        return 1

def get_ligand_center(residue):
    coords = [atom.get_coord() for atom in residue]
    return np.mean(coords, axis=0)

def check_openbabel():
    return shutil.which("obabel") is not None

# --- MAIN APP ---
st.title("🧬 NucLigs Pro: Bio-Physics Screening")
st.markdown("Advanced screening with **Interaction Fingerprinting** and **Ligand Efficiency** analysis.")

if not check_openbabel():
    st.error("⚠️ System Error: OpenBabel missing. Add 'openbabel', 'libboost-all-dev', 'swig', 'build-essential' to packages.txt")

# 1. SIDEBAR
with st.sidebar:
    st.header("1. NucLigs Library")
    uploaded_file = st.file_uploader("Upload Metadata (Excel)", type=['xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df.columns = [c.lower() for c in df.columns]
        if 'smiles' not in df.columns:
            st.error("Excel must have a 'smiles' column.")
            st.stop()
    else:
        st.info("Using Demo Data")
        df = pd.DataFrame({
            'name': ['Gemcitabine Analog', 'Remdesivir Metabolite', 'High MW Control'],
            'smiles': [
                'NC1=NC(=O)N(C=C1)C2C(C(C(O2)CO)O)(F)F', 
                'Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1',
                'CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NC2=CC=C(C=C2)C3=CC=CC=C3' # Random larger mol
            ]
        })
    
    st.divider()
    st.header("3. Physics Settings")
    exhaustiveness = st.slider("Docking Precision (Exhaustiveness)", 1, 32, 8, help="Higher = more accurate but slower.")
    contact_cutoff = st.slider("Interaction Cutoff (Å)", 2.5, 5.0, 4.0, help="Distance to define a residue contact.")

# 2. TARGET PREP
st.header("2. Target & Pocket Definition")
pdb_id = st.text_input("PDB ID:", "1UA2").strip().lower()

if pdb_id:
    temp_dir = "temp_docking"
    os.makedirs(temp_dir, exist_ok=True)
    pdb_path = os.path.join(temp_dir, "target.pdb")
    
    # Fetch PDB
    if not os.path.exists(pdb_path):
        response = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
        if response.status_code == 200:
            with open(pdb_path, "w") as f:
                f.write(response.text)
        else:
            st.error("Invalid PDB ID")
            st.stop()
            
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('Target', pdb_path)
    
    ligand_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0].startswith('H_') and residue.get_resname() != 'HOH':
                    ligand_list.append({
                        'Display': f"{residue.get_resname()} (Chain {chain.id} {residue.get_id()[1]})",
                        'Value': (chain.id, residue.get_resname(), residue.get_id()[1]),
                        'Center': get_ligand_center(residue)
                    })
    
    if not ligand_list:
        st.warning("No bound ligands found in PDB to define pocket.")
    else:
        selected_ligand_idx = st.selectbox("Select Bound Ligand (Pocket Center):", range(len(ligand_list)), format_func=lambda x: ligand_list[x]['Display'])
        target_ligand = ligand_list[selected_ligand_idx]
        center = target_ligand['Center']
        
        # VISUALIZATION
        view = py3Dmol.view(width=800, height=400)
        view.addModel(open(pdb_path).read(), 'pdb')
        view.setStyle({'cartoon': {'color': 'white'}})
        view.addStyle({'resn': target_ligand['Value'][1], 'resi': target_ligand['Value'][2]}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.5}})
        
        # Safe float conversion for JSON
        safe_center = {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])}
        view.addBox({'center': safe_center, 'dimensions': {'w':20, 'h':20, 'd':20}, 'color':'red', 'opacity': 0.5})
        view.zoomTo()
        showmol(view, height=400, width=800)

        if st.button("🚀 Run Bio-Physics Screening"):
            status = st.empty()
            progress = st.progress(0)
            
            # Prep Receptor
            clean_path = os.path.join(temp_dir, "receptor_clean.pdb")
            pdbqt_path = os.path.join(temp_dir, "receptor.pdbqt")
            io = PDBIO()
            io.set_structure(structure)
            io.save(clean_path, LigandSelect(target_ligand['Value']))
            
            try:
                subprocess.run(["obabel", clean_path, "-O", pdbqt_path, "-xr", "--partialcharge", "gasteiger"], check=True)
            except Exception as e:
                st.error("Receptor preparation failed. Is OpenBabel installed?")
                st.stop()
            
            results = []
            
            for i, row in df.iterrows():
                name = row.get('name', f"Mol_{i}")
                smi = row.get('smiles')
                
                status.text(f"Analyzing: {name}")
                
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        mol = Chem.AddHs(mol)
                        # Increased maxAttempts for complex nucleosides
                        if AllChem.EmbedMolecule(mol, maxAttempts=5000) == -1:
                             # Fallback: Compute 2D coords if 3D fails (rare but possible)
                             AllChem.Compute2DCoords(mol)
                        
                        # Meeko Prep
                        meeko_prep = MoleculePreparation()
                        meeko_prep.prepare(mol)
                        lig_pdbqt = meeko_prep.write_pdbqt_string()
                        
                        # Docking
                        v = Vina(sf_name='vina')
                        v.set_receptor(pdbqt_path)
                        v.set_ligand_from_string(lig_pdbqt)
                        v.compute_vina_maps(center=center, box_size=[20, 20, 20])
                        v.dock(exhaustiveness=exhaustiveness, n_poses=1)
                        
                        # 1. Physics: Raw Energy
                        score = v.score()[0]
                        docked_pdbqt = v.poses(n_poses=1)
                        
                        # 2. Physics: Ligand Efficiency & Lipinski
                        props = calculate_physchem_props(mol, score)
                        
                        # 3. Biology: Interaction Fingerprint
                        analyzer = InteractionAnalyzer(structure, docked_pdbqt)
                        contacts = analyzer.get_interacting_residues(cutoff=contact_cutoff)
                        
                        results.append({
                            'Name': name,
                            'Affinity (kcal/mol)': score,
                            'Ligand Efficiency': props['LE'], # Higher is better (>0.3)
                            'Lipinski Pass': props['Lipinski'],
                            'Interacting Residues': contacts,
                            'MW': props['MW'],
                            'LogP': props['LogP']
                        })
                    else:
                        st.warning(f"Skipping invalid SMILES: {name}")    
                except Exception as e:
                    print(f"Docking failed for {name}: {e}")
                
                progress.progress((i+1)/len(df))
            
            status.success("Analysis Complete!")
            if results:
                res_df = pd.DataFrame(results).sort_values(by='Affinity (kcal/mol)')
                
                st.subheader("High-Confidence Results")
                st.markdown("""
                **Interpretation Guide:**
                1. **Affinity:** Binding strength (more negative is better).
                2. **Ligand Efficiency (LE):** Quality of binding. **Target > 0.3**.
                3. **Lipinski Pass:** Checks oral drug-likeness.
                4. **Interactions:** Ensure these match the active site residues (e.g., catalytic triad).
                """)
                
                # Stylized dataframe
                st.dataframe(
                    res_df.style.background_gradient(subset=['Ligand Efficiency'], cmap='Greens')
                          .format({"Affinity (kcal/mol)": "{:.2f}", "Ligand Efficiency": "{:.2f}"})
                )
                
                # Download CSV
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Report", csv, "docking_report.csv", "text/csv")
