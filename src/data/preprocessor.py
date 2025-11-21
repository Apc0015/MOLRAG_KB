"""
SMILES preprocessing and validation

Based on blueprint specifications for molecular preprocessing
"""

from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, MolStandardize

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SMILESPreprocessor:
    """
    Preprocess and validate SMILES strings

    Features:
    - Canonicalization
    - Salt removal
    - Charge neutralization
    - Validation
    """

    def __init__(
        self,
        canonicalize: bool = True,
        remove_salts: bool = True,
        neutralize_charges: bool = False
    ):
        """
        Initialize SMILES preprocessor

        Args:
            canonicalize: Convert to canonical SMILES
            remove_salts: Remove salt fragments
            neutralize_charges: Neutralize charges
        """
        self.canonicalize = canonicalize
        self.remove_salts = remove_salts
        self.neutralize_charges = neutralize_charges

        # Initialize standardizer components
        if remove_salts:
            self.salt_remover = MolStandardize.fragment.LargestFragmentChooser()

        if neutralize_charges:
            self.uncharger = MolStandardize.charge.Uncharger()

        logger.info(
            f"Initialized SMILES preprocessor: "
            f"canonicalize={canonicalize}, remove_salts={remove_salts}, "
            f"neutralize={neutralize_charges}"
        )

    def is_valid_smiles(self, smiles: str) -> bool:
        """
        Check if SMILES is valid

        Args:
            smiles: SMILES string

        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Convert to canonical SMILES

        Args:
            smiles: Input SMILES

        Returns:
            Canonical SMILES or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logger.warning(f"Failed to canonicalize '{smiles}': {e}")
            return None

    def remove_salt_fragments(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Remove salt fragments, keeping largest fragment

        Args:
            mol: RDKit molecule

        Returns:
            Molecule with salts removed
        """
        try:
            return self.salt_remover.choose(mol)
        except Exception as e:
            logger.warning(f"Failed to remove salts: {e}")
            return mol

    def neutralize(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Neutralize charges on molecule

        Args:
            mol: RDKit molecule

        Returns:
            Neutralized molecule
        """
        try:
            return self.uncharger.uncharge(mol)
        except Exception as e:
            logger.warning(f"Failed to neutralize charges: {e}")
            return mol

    def preprocess(self, smiles: str) -> Optional[str]:
        """
        Full preprocessing pipeline

        Args:
            smiles: Input SMILES string

        Returns:
            Preprocessed SMILES or None if invalid
        """
        # Validate input
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None

        # Remove salts
        if self.remove_salts:
            mol = self.remove_salt_fragments(mol)

        # Neutralize charges
        if self.neutralize_charges:
            mol = self.neutralize(mol)

        # Convert back to SMILES
        try:
            processed_smiles = Chem.MolToSmiles(mol, canonical=self.canonicalize)
            return processed_smiles
        except Exception as e:
            logger.error(f"Failed to convert molecule to SMILES: {e}")
            return None

    def get_molecular_properties(self, smiles: str) -> dict:
        """
        Calculate basic molecular properties

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of molecular properties
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        try:
            properties = {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "num_h_acceptors": Descriptors.NumHAcceptors(mol),
                "num_h_donors": Descriptors.NumHDonors(mol),
                "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
                "tpsa": Descriptors.TPSA(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms()
            }
            return properties
        except Exception as e:
            logger.error(f"Failed to calculate properties for '{smiles}': {e}")
            return {}

    def passes_lipinski_rule_of_five(self, smiles: str) -> bool:
        """
        Check if molecule passes Lipinski's Rule of Five

        Criteria:
        - Molecular weight ≤ 500 Da
        - LogP ≤ 5
        - H-bond donors ≤ 5
        - H-bond acceptors ≤ 10

        Args:
            smiles: SMILES string

        Returns:
            True if passes all criteria
        """
        props = self.get_molecular_properties(smiles)
        if not props:
            return False

        return (
            props.get("molecular_weight", float("inf")) <= 500
            and props.get("logp", float("inf")) <= 5
            and props.get("num_h_donors", float("inf")) <= 5
            and props.get("num_h_acceptors", float("inf")) <= 10
        )
