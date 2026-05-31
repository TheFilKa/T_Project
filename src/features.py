from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def add_smiles_features(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Optional features. Safe no-op if RDKit is not installed or smiles_col is missing."""
    if not smiles_col or smiles_col not in df.columns:
        return df

    out = df.copy()
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
    except Exception:
        out["smiles_len"] = out[smiles_col].astype(str).str.len()
        out["canonical_smiles"] = out[smiles_col].map(canonicalize_smiles)
        return out

    mols = out[smiles_col].map(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
    out["is_valid_smiles"] = mols.map(lambda m: int(m is not None))
    out["canonical_smiles"] = mols.map(lambda m: Chem.MolToSmiles(m, canonical=True) if m else None)
    out["smiles_len"] = out[smiles_col].astype(str).str.len()
    out["rdkit_molwt"] = mols.map(lambda m: Descriptors.MolWt(m) if m else np.nan)
    out["rdkit_logp"] = mols.map(lambda m: Crippen.MolLogP(m) if m else np.nan)
    out["rdkit_tpsa"] = mols.map(lambda m: rdMolDescriptors.CalcTPSA(m) if m else np.nan)
    out["rdkit_h_donors"] = mols.map(lambda m: Lipinski.NumHDonors(m) if m else np.nan)
    out["rdkit_h_acceptors"] = mols.map(lambda m: Lipinski.NumHAcceptors(m) if m else np.nan)
    out["rdkit_rot_bonds"] = mols.map(lambda m: Lipinski.NumRotatableBonds(m) if m else np.nan)
    out["rdkit_rings"] = mols.map(lambda m: rdMolDescriptors.CalcNumRings(m) if m else np.nan)
    out["rdkit_aromatic_rings"] = mols.map(lambda m: rdMolDescriptors.CalcNumAromaticRings(m) if m else np.nan)
    return out


def make_morgan_fingerprints(df: pd.DataFrame, smiles_col: str, radius: int, n_bits: int) -> pd.DataFrame:
    if not smiles_col or smiles_col not in df.columns:
        return pd.DataFrame(index=df.index)

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception as exc:
        raise ImportError("RDKit is required for Morgan fingerprints") from exc

    fps = np.zeros((len(df), n_bits), dtype=np.uint8)
    for i, smi in enumerate(df[smiles_col].fillna("")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, fps[i])
    return pd.DataFrame(fps, columns=[f"morgan_{i}" for i in range(n_bits)], index=df.index)


def build_feature_frame(X: pd.DataFrame, config: dict, fit_test_safe: bool = True) -> pd.DataFrame:
    """Create only features available without target information."""
    smiles_col = config["data"].get("smiles_col")
    features_cfg = config["features"]
    out = X.copy()

    if smiles_col and (features_cfg.get("use_smiles_basic") or features_cfg.get("use_rdkit_descriptors")):
        out = add_smiles_features(out, smiles_col)

    if smiles_col and features_cfg.get("use_morgan_fp"):
        fp = make_morgan_fingerprints(
            out,
            smiles_col=smiles_col,
            radius=features_cfg.get("morgan_radius", 2),
            n_bits=features_cfg.get("morgan_n_bits", 2048),
        )
        out = pd.concat([out.drop(columns=[smiles_col], errors="ignore"), fp], axis=1)

    # drop raw string columns unless they are explicitly handled elsewhere
    object_cols = out.select_dtypes(include=["object"]).columns
    return out.drop(columns=object_cols, errors="ignore")


def make_preprocessor(X: pd.DataFrame, config: dict) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=config["features"].get("missing_strategy", "median"))),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    steps = [("columns", ColumnTransformer(transformers, remainder="drop"))]
    if config["features"].get("remove_constant_features", True):
        steps.append(("variance", VarianceThreshold(threshold=0.0)))
    return Pipeline(steps)
