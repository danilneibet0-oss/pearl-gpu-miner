"""
pearl_proof.py — упаковка найденного решения в PlainProof (Merkle-доказательства строк A и столбцов B)
и локальная проверка перед отправкой. Требует собранный py-pearl-mining (модуль pearl_mining).

Формат и проверка — официальные (pearl_mining), здесь только склейка:
  tree(A)   = MerkleTree(pad_to_chunk(A row-major), key=job_key)
  tree(B^T) = MerkleTree(pad_to_chunk(B^T row-major), key=job_key)
  proof(A)  = tree(A).get_multileaf_proof(leaf_indices_from_rows(a_rows, (m, k)))
  proof(B)  = то же для строк B^T (столбцов B)
"""

from __future__ import annotations

import numpy as np

import pearl_mining as pm
import pearlhash_ref as ref


def to_pm_header(header: ref.IncompleteBlockHeader) -> "pm.IncompleteBlockHeader":
    return pm.IncompleteBlockHeader.from_bytes(header.to_bytes())


def header_from_job(incomplete_header_bytes: bytes) -> ref.IncompleteBlockHeader:
    """76 байт из задания gateway -> заголовок (версия, prev, merkle_root, время, nbits)."""
    return ref.IncompleteBlockHeader.from_bytes(incomplete_header_bytes)


def _matrix_proof(rows_data: np.ndarray, indices: list[int], key: bytes) -> "pm.MatrixMerkleProof":
    n_rows, k = rows_data.shape
    padded = ref.pad_to_chunk(np.ascontiguousarray(rows_data, dtype=np.int8).tobytes())
    tree = pm.MerkleTree(padded, key)
    leaves = pm.MerkleTree.compute_leaf_indices_from_rows(list(indices), (n_rows, k))
    return pm.MatrixMerkleProof(tree.get_multileaf_proof(leaves), list(indices))


def build_plain_proof(sol: ref.Solution, header: ref.IncompleteBlockHeader,
                      cfg: ref.MiningConfiguration) -> "pm.PlainProof":
    """sol.a: m×k, sol.b: k×n (int8, без шума)."""
    m, k = sol.a.shape
    n = sol.b.shape[1]
    key = ref.job_key(header, cfg)
    bt = np.ascontiguousarray(sol.b.T)
    return pm.PlainProof(
        m, n, k, cfg.rank,
        _matrix_proof(sol.a, sol.a_rows, key),
        _matrix_proof(bt, sol.b_cols, key),
        None,
    )


def verify_before_submit(cert_version: int, header: ref.IncompleteBlockHeader,
                         proof: "pm.PlainProof", nbits_override: int | None = None) -> tuple[bool, str]:
    """Официальная проверка PlainProof (verify_plain_proof_for_cert_version)."""
    return pm.verify_plain_proof_for_cert_version(cert_version, to_pm_header(header), proof, nbits_override)


def proof_to_base64(proof: "pm.PlainProof") -> str:
    return proof.to_base64()
