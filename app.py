import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import shutil
import requests
from io import BytesIO, StringIO
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
from streamlit_ketcher import st_ketcher

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

class LigandRemover(Select):
    """Specific Removal: Keeps everything EXCEPT the selected residue."""
    def __init__(self, res_to_exclude):
        self.res_to_exclude = res_to_exclude 
    def accept_residue(self, residue):
        unique_id = (residue.get_parent().id, residue.get_resname(), residue.get_id()[1])
        return 0 if unique_id == self.res_to_exclude else 1

class WaterRemover(Select):
    """Blind Cleaning: Removes HOH (Water) but keeps everything else."""
    def accept_residue(self, residue):
        return 0 if residue.get_resname() == "HOH" else 1

def get_geometric_center(residue):
    """Calculates the center of mass for a specific residue."""
    coords = [atom.get_coord() for atom in residue]
    return np.mean(coords, axis=0)

def get_whole_protein_bbox(structure):
    """
    Calculates the center and size of the ENTIRE protein for Blind Docking.
    Returns: center (x,y,z), dimensions (w,h,d)
    """
    coords = []
    for model in structure:
        for chain in model:
            for res in chain:
                # Use standard amino acids to define the protein box
                if res.get_id()[0] == ' ': 
                    for atom in res:
                        coords.append(atom.get_coord())
    
    if not coords: return np.array([0,0,0]), np.array([20,20,20])
    
    coords = np.array(coords)
    min_coord = np.min(coords, axis=0)
    max_coord = np.max(coords, axis=0)
    
    center = (max_coord + min_coord) / 2
    size = (max_coord - min_coord) + 10 # Add 10 Angstrom padding
    
    return center, size

def analyze_interactions(receptor_model, ligand_pdbqt_string, cutoff=4.0):
    """Physics-based contact analysis."""
    rec_coords = []
    rec_labels = []
    for chain in receptor_model:
        for res in chain:
            if res.get_resname() != "HOH":
                for atom in res:
                    rec_coords.append(atom.get_coord())
                    rec_labels.append(f"{res.get_resname()}{res.get_id()[1]}")
    rec_coords = np.array(rec_coords)

    lig_coords = []
    for line in ligand_pdbqt_string.split('\n'):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                lig_coords.append([x, y, z])
            except ValueError:
                continue
    lig_coords = np.array(lig_coords)

    if len(lig_coords) == 0: return "N/A"

    dists = distance.cdist(lig_coords, rec_coords, 'euclidean')
    interacting_indices = np.where(dists < cutoff)[1]
    unique_contacts = sorted(list(set([rec_labels[i] for i in interacting_indices])))
    
    return ", ".join(unique_contacts)

def calculate_properties(mol, affinity):
    """Calculates Ligand Efficiency (LE) and Drug-Likeness."""
    heavy_atoms = mol.GetNumHeavyAtoms()
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    le = (-affinity / heavy_atoms) if heavy_atoms > 0 else 0
    
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
st.markdown("A high-precision tool for virtual screening of nucleotide analogs using **AutoDock Vina**.")

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("1. Library Input")
    
    # INPUT METHOD SELECTOR
    input_method = st.radio(
        "Choose Input Method:", 
        ["📂 Upload File", "📝 Paste List", "✏️ Draw Structure"]
    )
    
    df = None # Initialize dataframe

    # --- OPTION A: UPLOAD FILE ---
    if input_method == "📂 Upload File":
        uploaded_file = st.file_uploader("Upload (Excel, CSV, TXT)", type=['xlsx', 'csv', 'txt'])
        if uploaded_file:
            try:
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if file_ext == '.xlsx':
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file, sep=None, engine='python')
                
                # Normalize columns
                df.columns = [str(c).lower().strip() for c in df.columns]
                if 'smiles' not in df.columns:
                    if len(df.columns) == 1:
                        df.columns = ['smiles']
                    else:
                        st.error("❌ File must contain a 'smiles' column.")
                        df = None
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # --- OPTION B: PASTE TEXT ---
    elif input_method == "📝 Paste List":
        st.caption("Paste SMILES (one per line). Optional: 'Name, SMILES'")
        paste_data = st.text_area("Input Data", height=200, 
                                placeholder="Gemcitabine, NC1=NC(=O)N(C=C1)\nCCO")
        
        if paste_data:
            data_list = []
            for line in paste_data.split('\n'):
                line = line.strip()
                if not line: continue
                
                parts = line.split(',')
                if len(parts) >= 2:
                    # Assume format: Name, SMILES
                    data_list.append({'name': parts[0].strip(), 'smiles': parts[1].strip()})
                else:
                    # Assume format: SMILES only
                    data_list.append({'name': f"Mol_{len(data_list)+1}", 'smiles': parts[0].strip()})
            
            if data_list:
                df = pd.DataFrame(data_list)

    # --- OPTION C: DRAW STRUCTURE (KETCHER) ---
    elif input_method == "✏️ Draw Structure":
        st.caption("Draw your molecule below:")
        
        # Ketcher Editor
        drawn_smiles = st_ketcher("c1ccccc1")
        
        col_name, col_add = st.columns([2, 1])
        with col_name:
            mol_name = st.text_input("Molecule Name", "Drawn_Ligand_01")
        
        if drawn_smiles:
            st.success(f"SMILES: `{drawn_smiles}`")
            # Create a 1-row DataFrame
            df = pd.DataFrame([{'name': mol_name, 'smiles': drawn_smiles}])

    # --- INPUT SUMMARY ---
    if df is not None:
        st.success(f"✅ Loaded {len(df)} molecules")
        with st.expander("Preview Library"):
            st.dataframe(df.head())
    else:
        # Fallback to Demo Data if nothing is provided
        st.info("Using Demo Library (Default)")
        df = pd.DataFrame({
            'name': ['Gemcitabine Analog', 'Remdesivir Metabolite'],
            'smiles': [
                'NC1=NC(=O)N(C=C1)C2C(C(C(O2)CO)O)(F)F', 
                'Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1'
            ]
        })

    st.divider()
    
    st.header("3. Docking Strategy")
    docking_mode = st.radio(
        "Screening Mode", 
        ["Active Site (Ligand-Guided)", "Blind Docking (Whole Surface)"],
        help="Active Site: Targets a known pocket.\nBlind Docking: Searches the entire protein surface."
    )
    
    exhaustiveness = st.slider("Search Exhaustiveness", 1, 32, 8, help="Higher = More accurate but slower.")
    
    if docking_mode == "Active Site (Ligand-Guided)":
        box_padding = st.slider("Box Size (Å)", 10, 30, 20)
    else:
        st.info("ℹ️ Box size is automatically calculated to cover the whole protein.")

# --- MAIN: RECEPTOR PREP ---
st.header("2. Target Preparation")

col1, col2 = st.columns([1, 2])

with col1:
    pdb_input = st.text_input("PDB ID:", "1UA2", help="Enter 4-letter PDB code").strip().lower()

if pdb_input:
    pdb_filename = f"{pdb_input}.pdb"
    pdb_path = os.path.join(TEMP_DIR, pdb_filename)
    
    if not os.path.exists(pdb_path):
        with st.spinner(f"Downloading {pdb_input.upper()}..."):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(f"https://files.rcsb.org/download/{pdb_input}.pdb", headers=headers)
                if r.status_code != 200 or "HEADER" not in r.text[:100]:
                     st.error("❌ Invalid PDB ID or File.")
                     st.stop()
                with open(pdb_path, "w") as f:
                    f.write(r.text)
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.stop()
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_input, pdb_path)
    except Exception:
        st.error("❌ Failed to parse PDB file.")
        if os.path.exists(pdb_path): os.remove(pdb_path)
        st.stop()

    # --- MODE LOGIC ---
    target_site_id = None # Used only for exclusion in Guided Mode
    
    if docking_mode == "Active Site (Ligand-Guided)":
        # Identify Bound Ligands
        ligands = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.get_id()[0].startswith('H_') and res.get_resname() not in ['HOH', 'DOD', 'PO4', 'SO4']:
                        ligands.append({
                            'display': f"{res.get_resname()} (Chain {chain.id} {res.get_id()[1]})",
                            'id': (chain.id, res.get_resname(), res.get_id()[1]),
                            'center': get_geometric_center(res)
                        })
        
        if not ligands:
            st.warning(f"No bound ligands found. Switching to Blind Docking might be better.")
            st.stop()

        with col2:
            selected_lig_idx = st.selectbox(
                "Select Pocket (Defined by Bound Ligand):", 
                range(len(ligands)), 
                format_func=lambda i: ligands[i]['display']
            )
            target_site = ligands[selected_lig_idx]
            target_site_id = target_site['id']
            
            # Set Vina Parameters
            center = target_site['center']
            box_dim = [box_padding, box_padding, box_padding] # User defined

    else: # BLIND DOCKING
        with col2:
            st.success("🌍 Blind Docking Mode Active")
            st.markdown("The grid box will encompass the entire protein surface.")
            
            # Calculate Whole Protein Box
            prot_center, prot_size = get_whole_protein_bbox(structure)
            center = prot_center
            box_dim = prot_size
            
            st.caption(f"Grid Center: {center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}")
            st.caption(f"Grid Size: {box_dim[0]:.1f} x {box_dim[1]:.1f} x {box_dim[2]:.1f} Å")

    # --- VISUALIZATION ---
    with st.expander("Grid Visualization", expanded=True):
        view = py3Dmol.view(width=800, height=400)
        view.addModel(open(pdb_path).read(), 'pdb')
        view.setStyle({'cartoon': {'color': 'white'}})
        
        if docking_mode == "Active Site (Ligand-Guided)":
             view.addStyle(
                {'resn': target_site['id'][1], 'resi': target_site['id'][2], 'chain': target_site['id'][0]}, 
                {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.5}}
            )
        
        # Show Grid Box
        safe_center = {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])}
        view.addBox({
            'center': safe_center, 
            'dimensions': {'w': float(box_dim[0]), 'h': float(box_dim[1]), 'd': float(box_dim[2])}, 
            'color': 'blue' if docking_mode == "Blind Docking (Whole Surface)" else 'red', 
            'opacity': 0.4
        })
        view.zoomTo()
        showmol(view, height=400, width=800)

    # --- SIMULATION EXECUTION ---
    if st.button("🚀 Run Screening"):
        
        results_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_log = st.expander("⚠️ Error Log (Check if results are empty)", expanded=False)
        
        # A. PREPARE RECEPTOR
        status_text.text("⚙️ Preparing Receptor Topology...")
        clean_pdb = os.path.join(TEMP_DIR, "clean_receptor.pdb")
        receptor_pdbqt = os.path.join(TEMP_DIR, "receptor.pdbqt")
        
        io = PDBIO()
        io.set_structure(structure)
        
        # Select cleaning strategy based on mode
        if docking_mode == "Active Site (Ligand-Guided)":
            io.save(clean_pdb, LigandRemover(target_site_id))
        else:
            io.save(clean_pdb, WaterRemover())
        
        # SYSTEM CHECK: OPENBABEL
        if shutil.which("obabel") is None:
            st.error("❌ Critical Error: **OpenBabel** is not installed on this system.")
            st.info("If running on Streamlit Cloud, add `openbabel` to your `packages.txt`.")
            st.stop()
            
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
        success_count = 0
        
        for i, row in df.iterrows():
            name = row.get('name', f"Mol_{i}")
            smi = row.get('smiles')
            status_text.text(f"⚗️ Simulating: {name}")
            
            try:
                # 1. Embed
                mol = Chem.MolFromSmiles(smi)
                if not mol: raise ValueError("Invalid SMILES")
                mol = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol, maxAttempts=5000) == -1:
                    AllChem.Compute2DCoords(mol)
                
                # 2. Prep
                prep = MoleculePreparation()
                prep.prepare(mol)
                ligand_pdbqt = prep.write_pdbqt_string()
                
                # 3. Dock
                v = Vina(sf_name='vina')
                v.set_receptor(receptor_pdbqt)
                v.set_ligand_from_string(ligand_pdbqt)
                
                # --- CRITICAL FIX: Ensure these are lists of floats, not NumPy arrays ---
                safe_center = [float(c) for c in center]
                safe_box_dim = [float(d) for d in box_dim]
                
                v.compute_vina_maps(center=safe_center, box_size=safe_box_dim)
                v.dock(exhaustiveness=exhaustiveness, n_poses=1)
                
                # 4. Analyze
                if len(v.score()) > 0:
                    affinity = v.score()[0]
                    le, lipinski, mw, logp = calculate_properties(mol, affinity)
                    interactions = analyze_interactions(structure[0], v.poses(n_poses=1))
                    
                    results.append({
                        "Name": name,
                        "Affinity (kcal/mol)": affinity,
                        "Ligand Efficiency": le,
                        "Drug-Likeness": lipinski,
                        "Interacting Residues": interactions,
                        "MW": mw,
                        "LogP": logp
                    })
                    success_count += 1
                
            except Exception as e:
                # Log errors to the expander so we can see what went wrong
                error_log.write(f"❌ **{name}** Failed: `{str(e)}`")
            
            progress_bar.progress((i + 1) / total)

        status_text.text(f"✅ Simulation Complete! ({success_count}/{total} successful)")
        
        if results:
            res_df = pd.DataFrame(results).sort_values(by="Affinity (kcal/mol)")
            with results_container:
                st.subheader("🏆 Screening Results")
                
                best_mol = res_df.iloc[0]
                m1, m2, m3 = st.columns(3)
                m1.metric("Top Affinity", f"{best_mol['Affinity (kcal/mol)']:.2f} kcal/mol")
                m2.metric("Best Efficiency", f"{res_df['Ligand Efficiency'].max():.2f}")
                m3.metric("Molecules Screened", f"{success_count}/{total}")
                
                st.dataframe(
                    res_df.style.background_gradient(subset=['Affinity (kcal/mol)'], cmap='viridis_r')
                          .background_gradient(subset=['Ligand Efficiency'], cmap='Greens')
                          .format({"Affinity (kcal/mol)": "{:.2f}", "Ligand Efficiency": "{:.2f}"})
                )
                
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "screening_results.csv", "text/csv")
        else:
            st.error("🚫 No results generated. Please check the 'Error Log' above.")
