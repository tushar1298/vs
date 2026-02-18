import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import shutil
import requests
from io import BytesIO
from scipy.spatial import distance

# Bioinformatics & Chemistry
from Bio.PDB import PDBParser, Select, PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation
from vina import Vina

# Visualization
import py3Dmol
from stmol import showmol

# ============================================
# 1. CONFIGURATION & CONSTANTS
# ============================================
st.set_page_config(
    layout="wide", 
    page_title="NucLigs Pro: Bio-Physics Engine",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50; 
        color: white; 
        font-weight: bold;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    h1 { color: #2e7d32; }
    h2 { color: #1b5e20; border-bottom: 2px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

TEMP_DIR = "temp_workspace"
os.makedirs(TEMP_DIR, exist_ok=True)

# ============================================
# 2. HELPER CLASSES & FUNCTIONS
# ============================================

class PocketSelector(Select):
    """Selects everything EXCEPT specific residues (used to remove native ligands)."""
    def __init__(self, res_to_exclude):
        self.res_to_exclude = res_to_exclude # (Chain, ResName, ResID)
        
    def accept_residue(self, residue):
        unique_id = (residue.get_parent().id, residue.get_resname(), residue.get_id()[1])
        return 0 if unique_id == self.res_to_exclude else 1

def check_system_dependencies():
    """Ensures OpenBabel is installed for Receptor preparation."""
    if shutil.which("obabel") is None:
        st.error("❌ Critical Error: **OpenBabel** is not installed on this system.")
        st.info("If running on Streamlit Cloud, add `openbabel` to your `packages.txt`.")
        st.stop()

def get_geometric_center(residue):
    """Calculates the center of mass for a residue."""
    coords = [atom.get_coord() for atom in residue]
    return np.mean(coords, axis=0)

def analyze_interactions(receptor_model, ligand_pdbqt_string, cutoff=4.0):
    """
    Physics-based contact analysis.
    Returns: List of residues interacting with the ligand.
    """
    # 1. Extract Receptor Coordinates
    rec_coords = []
    rec_labels = []
    for chain in receptor_model:
        for res in chain:
            if res.get_resname() != "HOH":
                for atom in res:
                    rec_coords.append(atom.get_coord())
                    rec_labels.append(f"{res.get_resname()}{res.get_id()[1]}")
    rec_coords = np.array(rec_coords)

    # 2. Extract Ligand Coordinates from PDBQT
    lig_coords = []
    for line in ligand_pdbqt_string.split('\n'):
        if line.startswith(("ATOM", "HETATM")):
            try:
                # PDBQT columns for X, Y, Z
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                lig_coords.append([x, y, z])
            except ValueError:
                continue
    lig_coords = np.array(lig_coords)

    if len(lig_coords) == 0: return "N/A"

    # 3. Distance Matrix Calculation
    dists = distance.cdist(lig_coords, rec_coords, 'euclidean')
    
    # 4. Filter by Cutoff (Van der Waals contact)
    interacting_indices = np.where(dists < cutoff)[1]
    unique_contacts = sorted(list(set([rec_labels[i] for i in interacting_indices])))
    
    return ", ".join(unique_contacts)

def calculate_properties(mol, affinity):
    """Calculates Ligand Efficiency (LE) and Drug-Likeness."""
    heavy_atoms = mol.GetNumHeavyAtoms()
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    
    # Physics Metric: Ligand Efficiency (LE)
    # Target: > 0.3 kcal/mol per heavy atom
    le = (-affinity / heavy_atoms) if heavy_atoms > 0 else 0
    
    # Biological Metric: Lipinski Violations
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if Descriptors.NumHDonors(mol) > 5: violations += 1
    if Descriptors.NumHAcceptors(mol) > 10: violations += 1
    
    status = "✅ Pass" if violations <= 1 else f"⚠️ {violations} Violations"
    
    return round(le, 2), status, round(mw, 1), round(logp, 2)

# ============================================
# 3. MAIN APPLICATION UI
# ============================================

st.title("🧬 NucLigs Pro: Bio-Physics Screening Engine")
st.markdown("A high-precision tool for virtual screening of nucleotide analogs using **AutoDock Vina** and **Interaction Fingerprinting**.")

check_system_dependencies()

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("1. Library Input")
    uploaded_file = st.file_uploader("Upload Metadata (Excel)", type=['xlsx'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]
        if 'smiles' not in df.columns:
            st.error("Excel must contain a 'smiles' column.")
            st.stop()
        st.success(f"Loaded {len(df)} molecules.")
    else:
        st.info("Using Demo Library")
        df = pd.DataFrame({
            'name': ['Gemcitabine Analog', 'Remdesivir Metabolite', 'ATP Analog'],
            'smiles': [
                'NC1=NC(=O)N(C=C1)C2C(C(C(O2)CO)O)(F)F', 
                'Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1',
                'NC1=C2N=CN(C2=NC=N1)C3OC(COP(=O)(O)O)C(O)C3O'
            ]
        })

    st.divider()
    st.header("3. Simulation Settings")
    exhaustiveness = st.slider("Search Exhaustiveness", 1, 32, 8, help="Higher = More accurate but slower.")
    box_padding = st.slider("Box Padding (Å)", 10, 30, 20, help="Size of the search space around the active site.")
    contact_dist = st.slider("Contact Cutoff (Å)", 2.5, 5.0, 4.0, help="Max distance to define a residue interaction.")

# --- MAIN: RECEPTOR PREP ---
st.header("2. Target Preparation")

col1, col2 = st.columns([1, 2])

with col1:
    pdb_input = st.text_input("PDB ID:", "1UA2", help="Enter 4-letter PDB code").strip().lower()

if pdb_input:
    # Use dynamic filename based on input ID to prevent caching old files
    pdb_filename = f"{pdb_input}.pdb"
    pdb_path = os.path.join(TEMP_DIR, pdb_filename)
    
    # Fetch PDB
    if not os.path.exists(pdb_path):
        with st.spinner(f"Downloading {pdb_input.upper()} from RCSB..."):
            try:
                # Add headers to mimic a browser
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(f"https://files.rcsb.org/download/{pdb_input}.pdb", headers=headers)
                
                if r.status_code != 200:
                    st.error(f"❌ Error: Could not find PDB ID '{pdb_input.upper()}' on RCSB.")
                    st.stop()
                
                # Check for valid PDB content
                if "HEADER" not in r.text[:100] and "ATOM" not in r.text[:1000]:
                     st.error("❌ Invalid file format received from RCSB.")
                     st.stop()

                with open(pdb_path, "w") as f:
                    f.write(r.text)
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.stop()
    
    # Parse Structure
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_input, pdb_path)
    except Exception:
        st.error("❌ Failed to parse PDB file. It might be corrupt.")
        if os.path.exists(pdb_path): os.remove(pdb_path)
        st.stop()
    
    # Identify Potential Pockets (Bound Ligands)
    ligands = []
    for model in structure:
        for chain in model:
            for res in chain:
                # H_ prefix denotes Heteroatom (ligand/water)
                # Exclude common non-ligands like HOH (water), PO4, SO4, etc.
                if res.get_id()[0].startswith('H_') and res.get_resname() not in ['HOH', 'DOD', 'PO4', 'SO4']:
                    ligands.append({
                        'display': f"{res.get_resname()} (Chain {chain.id} {res.get_id()[1]})",
                        'id': (chain.id, res.get_resname(), res.get_id()[1]),
                        'center': get_geometric_center(res)
                    })
    
    if not ligands:
        st.warning(f"No bound ligands found in {pdb_input.upper()}.")
        st.info("The tool currently requires a bound ligand to define the docking center automatically.")
        st.stop()

    with col2:
        selected_lig_idx = st.selectbox(
            "Select Active Site (Defined by Bound Ligand):", 
            range(len(ligands)), 
            format_func=lambda i: ligands[i]['display']
        )
        target_site = ligands[selected_lig_idx]
        center = target_site['center']

    # --- VISUALIZATION ---
    with st.expander("Active Site Visualization", expanded=True):
        view = py3Dmol.view(width=800, height=400)
        view.addModel(open(pdb_path).read(), 'pdb')
        view.setStyle({'cartoon': {'color': 'white'}})
        
        # Highlight Active Site Ligand
        view.addStyle(
            {'resn': target_site['id'][1], 'resi': target_site['id'][2], 'chain': target_site['id'][0]}, 
            {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.5}}
        )
        
        # Show Search Box (Safe Float Conversion for JSON)
        safe_center = {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])}
        view.addBox({
            'center': safe_center, 
            'dimensions': {'w': box_padding, 'h': box_padding, 'd': box_padding}, 
            'color': 'red', 'opacity': 0.4
        })
        view.zoomTo()
        showmol(view, height=400, width=800)

    # --- SIMULATION EXECUTION ---
    if st.button("🚀 Run Physics Screening Engine"):
        
        results_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # A. PREPARE RECEPTOR (Clean & Convert)
        status_text.text("⚙️ Preparing Receptor Topology...")
        clean_pdb = os.path.join(TEMP_DIR, "clean_receptor.pdb")
        receptor_pdbqt = os.path.join(TEMP_DIR, "receptor.pdbqt")
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(clean_pdb, PocketSelector(target_site['id']))
        
        try:
            subprocess.run([
                "obabel", clean_pdb, "-O", receptor_pdbqt, "-xr", "--partialcharge", "gasteiger"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            st.error("OpenBabel conversion failed.")
            st.stop()

        # B. DOCKING LOOP
        results = []
        total = len(df)
        
        for i, row in df.iterrows():
            name = row.get('name', f"Mol_{i}")
            smi = row.get('smiles')
            
            status_text.text(f"⚗️ Simulating: {name}")
            
            try:
                # 1. 3D Embedding (RDKit)
                mol = Chem.MolFromSmiles(smi)
                if not mol: raise ValueError("Invalid SMILES")
                mol = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol, maxAttempts=5000) == -1:
                    AllChem.Compute2DCoords(mol) # Fallback
                
                # 2. Ligand Prep (Meeko)
                prep = MoleculePreparation()
                prep.prepare(mol)
                ligand_pdbqt = prep.write_pdbqt_string()
                
                # 3. Docking (Vina)
                v = Vina(sf_name='vina')
                v.set_receptor(receptor_pdbqt)
                v.set_ligand_from_string(ligand_pdbqt)
                v.compute_vina_maps(center=center, box_size=[box_padding]*3)
                v.dock(exhaustiveness=exhaustiveness, n_poses=1)
                
                # 4. Analysis
                affinity = v.score()[0]
                le, lipinski, mw, logp = calculate_properties(mol, affinity)
                interactions = analyze_interactions(structure[0], v.poses(n_poses=1), cutoff=contact_dist)
                
                results.append({
                    "Name": name,
                    "Affinity (kcal/mol)": affinity,
                    "Ligand Efficiency": le,
                    "Drug-Likeness": lipinski,
                    "Interacting Residues": interactions,
                    "MW": mw,
                    "LogP": logp
                })
                
            except Exception as e:
                print(f"Failed {name}: {e}")
            
            progress_bar.progress((i + 1) / total)

        status_text.text("✅ Simulation Complete!")
        
        # C. REPORTING
        if results:
            res_df = pd.DataFrame(results).sort_values(by="Affinity (kcal/mol)")
            
            with results_container:
                st.subheader("🏆 Screening Results")
                
                # Metrics for best hit
                best_mol = res_df.iloc[0]
                m1, m2, m3 = st.columns(3)
                m1.metric("Top Affinity", f"{best_mol['Affinity (kcal/mol)']:.2f} kcal/mol")
                m2.metric("Best Efficiency", f"{res_df['Ligand Efficiency'].max():.2f}")
                m3.metric("Molecules Screened", total)
                
                # Styled Dataframe
                st.dataframe(
                    res_df.style.background_gradient(subset=['Affinity (kcal/mol)'], cmap='viridis_r')
                          .background_gradient(subset=['Ligand Efficiency'], cmap='Greens')
                          .format({"Affinity (kcal/mol)": "{:.2f}", "Ligand Efficiency": "{:.2f}"})
                )
                
                # Download
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Physics Report",
                    data=csv,
                    file_name="nucligs_screening_results.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No valid results generated. Check your input SMILES.")
