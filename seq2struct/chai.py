from enum import StrEnum
from tempfile import NamedTemporaryFile
from typing import Optional

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pydantic import Field, BaseModel

from seq2struct import AlphaFold3Sequences, AlphaFold3Entity, AlphaFold3EntityType


class ChaiConstraintType(StrEnum):
    contact = "contact"
    pocket = "pocket"


class ChaiConstraint(BaseModel):
    chainA: str = Field(...)
    res_idxA: Optional[str] = Field(None)
    chainB: str = Field(...)
    res_idxB: Optional[str] = Field(None)
    connection_type: ChaiConstraintType = Field(ChaiConstraintType.pocket)
    max_distance_angstrom: float = Field(...)


class ChaiMSAEntity(AlphaFold3Entity):
    msas: Optional[list[str]] = Field(None)


class ChaiSequences(AlphaFold3Sequences):
    sequences: list[ChaiMSAEntity] = Field(...)
    constraints: list[ChaiConstraint] = Field(default_factory=list, description="Constraints to use")


class ChaiMSAModel(BaseModel):
    num_trunk_recycles: int = Field(3, description="The number of recycling steps to use for prediction.", gt=0)
    num_diffn_timesteps: int = Field(200, description="The number of sampling steps to use for prediction.", gt=0)
    sequences: ChaiSequences = Field(..., description="Entities to predict")
    use_esm_embeddings: bool = Field(True, description="Whether to use ESM embeddings")


def create_input_files(model: ChaiMSAModel) -> tuple[str, Optional[str]]:
    with NamedTemporaryFile("w", suffix=".fasta", delete=False) as fasta_file:
        entries = []

        for sequence in model.sequences.sequences:
            if sequence.entity_type == AlphaFold3EntityType.ligand:
                seq = Seq(sequence.smiles)
            else:
                seq = Seq(sequence.sequence)
            entries.append(SeqRecord(seq, id="|".join([sequence.entity_type.value, f"name={sequence.id}"])))
        SeqIO.write(entries, fasta_file, "fasta")
        fasta_file.flush()

    if model.sequences.constraints:
        with NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as csv_file:
            csv_file.write("restraint_id,chainA,res_idxA,chainB,res_idxB,connection_type,max_distance_angstrom,min_distance_angstrom,confidence,comment\n")
            for i, constraint in enumerate(model.sequences.constraints):
                csv_file.write(f"{f'cons{i}'},{constraint.chainA},{constraint.res_idxA if constraint.res_idxA else ''},{constraint.chainB},{constraint.res_idxB if constraint.res_idxB else ''},{constraint.connection_type.value},{constraint.max_distance_angstrom},0.0,1.0,comment\n")
            csv_file.flush()
    else:
        csv_file = None

    return fasta_file.name, csv_file.name if csv_file else None
