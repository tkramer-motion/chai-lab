from enum import StrEnum
from typing import Optional

from pydantic import Field, BaseModel, model_validator


class AlphaFoldBaseModel(BaseModel):
    email_addresses: str = Field(default="", description="Email addresses comma separated to send results to")


class RankingData(BaseModel):
    confidence_score: float
    ptm: float
    iptm: float
    ligand_iptm: Optional[float] = None
    protein_iptm: Optional[float] = None
    complex_plddt: float
    complex_iplddt: Optional[float] = None
    complex_pde: Optional[float] = None
    complex_ipde: Optional[float] = None
    total_clashes: Optional[int] = None


class AlphaFold3EntityType(StrEnum):
    protein = "protein"
    ligand = "ligand"
    dna = "dna"
    rna = "rna"


class AlphaFold3Entity(BaseModel):
    entity_type: AlphaFold3EntityType = Field(AlphaFold3EntityType.protein)
    id: str = Field(..., description="CHAIN_ID")
    sequence: Optional[str] = Field(None, description="only for protein, dna, rna")
    smiles: Optional[str] = Field(None, description="only for ligand")

    @model_validator(mode="after")
    def check_valid(self):
        if not self.id:
            raise ValueError("An ID must be specified")

        num_not_null = 0
        if self.sequence:
            num_not_null += 1
        if self.smiles:
            num_not_null += 1

        if num_not_null != 1:
            raise ValueError("Only 1 of sequence, smiles, ccd can be specified")

        if self.entity_type == AlphaFold3EntityType.protein and not self.sequence:
            raise ValueError("When entity_type is protein sequence must be specified")

        if self.entity_type == AlphaFold3EntityType.rna and not self.sequence:
            raise ValueError("When entity_type is rna sequence must be specified")

        if self.entity_type == AlphaFold3EntityType.dna and not self.sequence:
            raise ValueError("When entity_type is dna sequence must be specified")

        return self


class AlphaFold3Sequences(BaseModel):
    name: str = Field(...)
    sequences: list[AlphaFold3Entity] = Field(...)


class AlphaFold3BaseModel(AlphaFoldBaseModel):
    reference_pdb: Optional[str] = Field(None, description="S3 url of reference PDB to use")


class AlphaFold3Prediction(BaseModel):
    cif_path: str
    ranking_data: RankingData


class AlphaFold3Result(BaseModel):
    predictions: list[AlphaFold3Prediction]
    json_path: str
    sequences: AlphaFold3Sequences
