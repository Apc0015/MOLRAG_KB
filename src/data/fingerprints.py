"""
Molecular fingerprint generation using RDKit

Based on blueprint specifications:
- Morgan Fingerprint (ECFP4)
- Radius = 2
- 2048-bit vectors
- Tanimoto/Dice similarity
"""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MolecularFingerprints:
    """
    Generate and compare molecular fingerprints

    Supports:
    - Morgan (ECFP) fingerprints
    - RDKit fingerprints
    - MACCS keys
    """

    def __init__(
        self,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
        use_features: bool = False,
        use_chirality: bool = True
    ):
        """
        Initialize fingerprint generator

        Args:
            fingerprint_type: Type of fingerprint (morgan, rdkit, maccs)
            radius: Radius for Morgan fingerprints (2 = ECFP4)
            n_bits: Number of bits in fingerprint
            use_features: Use feature-based Morgan fingerprints
            use_chirality: Include chirality information
        """
        self.fingerprint_type = fingerprint_type.lower()
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        self.use_chirality = use_chirality

        logger.info(
            f"Initialized {fingerprint_type} fingerprints: "
            f"radius={radius}, n_bits={n_bits}, chirality={use_chirality}"
        )

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES to RDKit molecule

        Args:
            smiles: SMILES string

        Returns:
            RDKit molecule object or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
            return mol
        except Exception as e:
            logger.error(f"Failed to parse SMILES '{smiles}': {e}")
            return None

    def generate_fingerprint(
        self,
        smiles: str
    ) -> Optional[DataStructs.ExplicitBitVect]:
        """
        Generate molecular fingerprint from SMILES

        Args:
            smiles: SMILES string

        Returns:
            RDKit fingerprint bit vector or None if generation fails
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None

        try:
            if self.fingerprint_type == "morgan":
                # Morgan (ECFP) fingerprints
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.radius,
                    nBits=self.n_bits,
                    useFeatures=self.use_features,
                    useChirality=self.use_chirality
                )
            elif self.fingerprint_type == "rdkit":
                # RDKit fingerprints
                fp = RDKFingerprint(mol, fpSize=self.n_bits)
            elif self.fingerprint_type == "maccs":
                # MACCS keys (166 bits)
                fp = MACCSkeys.GenMACCSKeys(mol)
            else:
                logger.error(f"Unknown fingerprint type: {self.fingerprint_type}")
                return None

            return fp

        except Exception as e:
            logger.error(f"Failed to generate fingerprint for '{smiles}': {e}")
            return None

    def fingerprint_to_array(
        self,
        fingerprint: DataStructs.ExplicitBitVect
    ) -> np.ndarray:
        """
        Convert RDKit fingerprint to numpy array

        Args:
            fingerprint: RDKit fingerprint bit vector

        Returns:
            Numpy array of fingerprint bits
        """
        arr = np.zeros((len(fingerprint),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return arr

    def fingerprint_to_list(
        self,
        fingerprint: DataStructs.ExplicitBitVect
    ) -> List[int]:
        """
        Convert RDKit fingerprint to Python list

        Args:
            fingerprint: RDKit fingerprint bit vector

        Returns:
            List of fingerprint bits (for JSON serialization)
        """
        return self.fingerprint_to_array(fingerprint).tolist()

    def smiles_to_array(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert SMILES directly to fingerprint array

        Args:
            smiles: SMILES string

        Returns:
            Numpy array of fingerprint or None if generation fails
        """
        fp = self.generate_fingerprint(smiles)
        if fp is None:
            return None
        return self.fingerprint_to_array(fp)

    def smiles_to_list(self, smiles: str) -> Optional[List[int]]:
        """
        Convert SMILES directly to fingerprint list

        Args:
            smiles: SMILES string

        Returns:
            List of fingerprint bits or None if generation fails
        """
        arr = self.smiles_to_array(smiles)
        if arr is None:
            return None
        return arr.tolist()

    def calculate_similarity(
        self,
        fp1: DataStructs.ExplicitBitVect,
        fp2: DataStructs.ExplicitBitVect,
        metric: str = "tanimoto"
    ) -> float:
        """
        Calculate similarity between two fingerprints

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            metric: Similarity metric (tanimoto, dice, cosine)

        Returns:
            Similarity score (0-1)
        """
        metric = metric.lower()

        try:
            if metric == "tanimoto":
                return DataStructs.TanimotoSimilarity(fp1, fp2)
            elif metric == "dice":
                return DataStructs.DiceSimilarity(fp1, fp2)
            elif metric == "cosine":
                return DataStructs.CosineSimilarity(fp1, fp2)
            else:
                logger.warning(f"Unknown metric '{metric}', using Tanimoto")
                return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    def compare_smiles(
        self,
        smiles1: str,
        smiles2: str,
        metric: str = "tanimoto"
    ) -> Optional[float]:
        """
        Calculate similarity between two molecules (SMILES)

        Args:
            smiles1: First molecule SMILES
            smiles2: Second molecule SMILES
            metric: Similarity metric

        Returns:
            Similarity score or None if fingerprint generation fails
        """
        fp1 = self.generate_fingerprint(smiles1)
        fp2 = self.generate_fingerprint(smiles2)

        if fp1 is None or fp2 is None:
            return None

        return self.calculate_similarity(fp1, fp2, metric)

    def find_most_similar(
        self,
        query_smiles: str,
        candidate_smiles: List[str],
        top_k: int = 10,
        metric: str = "tanimoto"
    ) -> List[Tuple[str, float]]:
        """
        Find most similar molecules from candidates

        Args:
            query_smiles: Query molecule SMILES
            candidate_smiles: List of candidate SMILES
            top_k: Number of top results to return
            metric: Similarity metric

        Returns:
            List of (smiles, similarity_score) tuples, sorted by descending similarity
        """
        query_fp = self.generate_fingerprint(query_smiles)
        if query_fp is None:
            logger.error(f"Failed to generate fingerprint for query: {query_smiles}")
            return []

        similarities = []

        for candidate in candidate_smiles:
            candidate_fp = self.generate_fingerprint(candidate)
            if candidate_fp is None:
                continue

            similarity = self.calculate_similarity(query_fp, candidate_fp, metric)
            similarities.append((candidate, similarity))

        # Sort by descending similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def batch_generate(
        self,
        smiles_list: List[str]
    ) -> List[Optional[np.ndarray]]:
        """
        Generate fingerprints for multiple SMILES in batch

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of fingerprint arrays (None for failed generations)
        """
        fingerprints = []

        for smiles in smiles_list:
            fp_array = self.smiles_to_array(smiles)
            fingerprints.append(fp_array)

        logger.info(
            f"Generated {sum(fp is not None for fp in fingerprints)}/{len(smiles_list)} fingerprints"
        )

        return fingerprints
