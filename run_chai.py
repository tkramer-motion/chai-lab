import dataclasses
import json
import logging
import os.path
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence, Iterable

import smart_open
from chai_lab.chai1 import run_inference, StructureCandidates
from chai_lab.data.parsing.msas.aligned_pqt import expected_basename, merge_multi_a3m_to_aligned_dataframe
from chai_lab.data.parsing.msas.data_source import MSADataSource

from seq2struct import AlphaFold3EntityType
from seq2struct.chai import ChaiMSAModel, create_input_files
from seq2struct.serialization import TensorEncoder


def _convert_sto_seq_to_a3m(
        query_non_gaps: Sequence[bool], sto_seq: str) -> Iterable[str]:
    for is_query_res_non_gap, sequence_res in zip(query_non_gaps, sto_seq):
        if is_query_res_non_gap:
            yield sequence_res
        elif sequence_res != '-':
            yield sequence_res.lower()


def convert_stockholm_to_a3m(stockholm_format: str,
                             max_sequences: Optional[int] = None,
                             remove_first_row_gaps: bool = True) -> str:
    """Converts MSA in Stockholm format to the A3M format."""
    descriptions = {}
    sequences = {}
    reached_max_sequences = False

    for line in stockholm_format.splitlines():
        reached_max_sequences = max_sequences and len(sequences) >= max_sequences
        if line.strip() and not line.startswith(('#', '//')):
            # Ignore blank lines, markup and end symbols - remainder are alignment
            # sequence parts.
            seqname, aligned_seq = line.split(maxsplit=1)
            if seqname not in sequences:
                if reached_max_sequences:
                    continue
                sequences[seqname] = ''
            sequences[seqname] += aligned_seq

    for line in stockholm_format.splitlines():
        if line[:4] == '#=GS':
            # Description row - example format is:
            # #=GS UniRef90_Q9H5Z4/4-78            DE [subseq from] cDNA: FLJ22755 ...
            columns = line.split(maxsplit=3)
            seqname, feature = columns[1:3]
            value = columns[3] if len(columns) == 4 else ''
            if feature != 'DE':
                continue
            if reached_max_sequences and seqname not in sequences:
                continue
            descriptions[seqname] = value
            if len(descriptions) == len(sequences):
                break

    # Convert sto format to a3m line by line
    a3m_sequences = {}
    if remove_first_row_gaps:
        # query_sequence is assumed to be the first sequence
        query_sequence = next(iter(sequences.values()))
        query_non_gaps = [res != '-' for res in query_sequence]
    for seqname, sto_sequence in sequences.items():
        # Dots are optional in a3m format and are commonly removed.
        out_sequence = sto_sequence.replace('.', '')
        if remove_first_row_gaps:
            out_sequence = ''.join(
                _convert_sto_seq_to_a3m(query_non_gaps, out_sequence))
        a3m_sequences[seqname] = out_sequence

    fasta_chunks = (f">{k} {descriptions.get(k, '')}\n{a3m_sequences[k]}"
                    for k in a3m_sequences)
    return '\n'.join(fasta_chunks) + '\n'


def merge_files_in_directory(directory: str) -> Path:
    dir_path = Path(directory)
    assert dir_path.is_dir()

    mapped_a3m_files = {}
    for file in dir_path.glob("*"):
        dbname = file.stem.replace("_hits", "").replace("hits_", "")
        try:
            msa_src = MSADataSource(dbname)
            logging.info(f"Found {msa_src} MSAs in {file}")
        except ValueError:
            if "bfd" in dbname:
                msa_src = MSADataSource.BFD_UNICLUST
            elif "pdb" in dbname:
                msa_src = MSADataSource.PDB70
            else:
                msa_src = MSADataSource.UNIREF90
                logging.warning(
                    f"Could not determine source for {file=}; default to {msa_src}"
                )
        mapped_a3m_files[file] = msa_src
    df = merge_multi_a3m_to_aligned_dataframe(
        mapped_a3m_files, insert_keys_for_sources="uniprot"
    )
    query_seq: str = df.iloc[0]["sequence"]
    df.to_parquet(dir_path / expected_basename(query_seq))
    return dir_path / expected_basename(query_seq)


def predict(fasta_path: str, output_dir: str, msa_directory: Optional[str], num_trunk_recycles: int, num_diffn_timesteps: int, constraint_path: Optional[str], use_esm_embeddings: bool) -> StructureCandidates:
    return run_inference(
        fasta_file=Path(fasta_path),
        output_dir=Path(output_dir),
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        device="cpu",
        use_esm_embeddings=use_esm_embeddings,
        msa_directory=Path(msa_directory) if msa_directory else None,
        use_msa_server=False,
        constraint_path=constraint_path
    )


if __name__ == '__main__':
    with smart_open.open(sys.argv[1], mode="r") as f:
        model = ChaiMSAModel.model_validate_json(f.read())
    base_s3_url = sys.argv[2]

    fasta_file, constraints_fn = create_input_files(model)

    with TemporaryDirectory() as output_dir:
        with TemporaryDirectory() as msa_dir:
            msa_count = 0
            for entity in model.sequences.sequences:
                if entity.entity_type == AlphaFold3EntityType.protein:
                    with TemporaryDirectory() as temp_msa_dir:
                        if entity.msas:
                            for a3m_fn in entity.msas:
                                if Path(a3m_fn).name != "pdb_hits.sto":
                                    with smart_open.open(a3m_fn, "rb") as src:
                                        if a3m_fn.endswith(".sto"):
                                            a3m = convert_stockholm_to_a3m(src.read().decode())
                                            with smart_open.open(os.path.join(temp_msa_dir, Path(a3m_fn).name), "w") as dst:
                                                dst.write(a3m)
                                        else:
                                            with smart_open.open(os.path.join(temp_msa_dir, Path(a3m_fn).name), "wb") as dst:
                                                shutil.copyfileobj(src, dst)

                            pqt = merge_files_in_directory(temp_msa_dir)
                            shutil.copyfile(pqt, os.path.join(msa_dir, pqt.name))
                            msa_count += 1

            result = predict(fasta_file, output_dir, msa_dir if msa_count else None, num_trunk_recycles=model.num_trunk_recycles, num_diffn_timesteps=model.num_diffn_timesteps, constraint_path=constraints_fn, use_esm_embeddings=model.use_esm_embeddings)

        for i, cif in enumerate(result.cif_paths):
            with smart_open.open(cif, "rb") as src:
                with smart_open.smart_open(os.path.join(base_s3_url, cif.name), "wb") as dst:
                    shutil.copyfileobj(src, dst)

            result.cif_paths[i] = os.path.join(base_s3_url, cif.name)

        with smart_open.open(os.path.join(base_s3_url, "results.json"), "w") as f:
            d = dataclasses.asdict(result)
            json.dump({key: d[key] for key in ["cif_paths", "ranking_data"]}, f, cls=TensorEncoder)
