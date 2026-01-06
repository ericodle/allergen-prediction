#!/usr/bin/env python3
"""
Convert SDF files to molecular descriptors using RDKit
This script processes SDF files from ChAIPred directory and generates descriptor CSVs
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')

def convert_sdf_to_descriptors(sdf_file, output_csv, data_dir="/home/eo/allergen-prediction/ChAIPred"):
    """
    Convert SDF file to molecular descriptors CSV using PaDEL
    
    Args:
        sdf_file: Name of the SDF file (e.g., 'Pos_train.sdf')
        output_csv: Name of the output CSV file (e.g., 'Pos_train_descriptors.csv')
        data_dir: Directory containing the SDF files
    """
    sdf_path = os.path.join(data_dir, sdf_file)
    output_path = os.path.join("/home/eo/allergen-prediction/results", output_csv)
    
    if not os.path.exists(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")
    
    print(f"Processing {sdf_file}...")
    
    # Create output directory if it doesn't exist
    os.makedirs("/home/eo/allergen-prediction/results", exist_ok=True)
    
    # Use RDKit to extract molecular descriptors
    # Read SDF file
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
    molecules = []
    names = []
    
    print(f"Reading molecules from {sdf_file}...")
    for mol in supplier:
        if mol is not None:
            molecules.append(mol)
            # Try to get name from molecule properties
            try:
                name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"mol_{len(molecules)}"
            except:
                name = f"mol_{len(molecules)}"
            names.append(name)
    
    print(f"Found {len(molecules)} molecules in {sdf_file}")
    
    if len(molecules) == 0:
        raise ValueError(f"No valid molecules found in {sdf_file}")
    
    # Calculate descriptors using RDKit
    # Get all descriptor functions
    descriptor_funcs = [x[1] for x in Descriptors._descList]
    descriptor_names = [x[0] for x in Descriptors._descList]
    
    print(f"Calculating {len(descriptor_names)} descriptors for each molecule...")
    
    # Calculate descriptors for each molecule
    descriptor_data = []
    failed_count = 0
    for i, mol in enumerate(molecules):
        try:
            desc_values = [func(mol) for func in descriptor_funcs]
            descriptor_data.append(desc_values)
        except Exception as e:
            # If calculation fails, use NaN
            descriptor_data.append([np.nan] * len(descriptor_names))
            failed_count += 1
            if failed_count <= 5:  # Print first few errors
                print(f"  Warning: Failed to calculate descriptors for molecule {i+1}: {e}")
    
    if failed_count > 0:
        print(f"  Total molecules with failed descriptor calculation: {failed_count}")
    
    # Create DataFrame
    df = pd.DataFrame(descriptor_data, columns=descriptor_names)
    df.insert(0, 'Name', names)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully created {output_csv} with {len(df)} molecules and {len(descriptor_names)} descriptors")

def main():
    """Convert all SDF files in ChAIPred to descriptor CSVs"""
    print("="*80)
    print("Converting SDF files to molecular descriptors using RDKit")
    print("="*80)
    
    # Define the SDF files and their corresponding output names
    files_to_process = [
        ('Pos_train.sdf', 'Pos_train_descriptors.csv'),
        ('Neg_train.sdf', 'Neg_train_descriptors.csv'),
        ('Pos_test.sdf', 'Pos_test_descriptors.csv'),
        ('Neg_test.sdf', 'Neg_test_descriptors.csv')
    ]
    
    for sdf_file, output_csv in files_to_process:
        try:
            convert_sdf_to_descriptors(sdf_file, output_csv)
        except Exception as e:
            print(f"Failed to process {sdf_file}: {e}")
            continue
    
    print("\n" + "="*80)
    print("Conversion complete!")
    print("="*80)

if __name__ == "__main__":
    main()

