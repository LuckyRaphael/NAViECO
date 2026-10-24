"""Offline human gene harmonization using the geneSync 3.2.1 database."""

from collections import defaultdict
from pathlib import Path
import re

import pandas as pd

GENE_DB = Path(__file__).resolve().parent / "data" / "human_gene_db.tsv.gz"


def map_genes(query_genes, reference_genes, database=GENE_DB):
    """Map Query genes to Reference genes, preserving an auditable result."""
    query = pd.Index(query_genes).astype(str)
    reference = pd.Index(reference_genes).astype(str)

    if not query.is_unique or not reference.is_unique:
        raise ValueError("Query and Reference gene names must be unique")

    target = set(reference)

    def normalize(name):
        if re.fullmatch(r"ENSG\d+(?:\.\d+)?", name, re.IGNORECASE):
            return name.split(".")[0].upper()
        return name if name.isdecimal() else name.upper()

    wanted = {normalize(g) for g in query if g not in target}
    hits = defaultdict(lambda: defaultdict(set))

    if wanted:
        db = pd.read_csv(database, sep="\t", dtype=str, keep_default_na=False)

        for i, row in enumerate(
            db[["gene_id", "symbol", "symbol_authority",
                "synonyms", "ensembl_id"]].itertuples(index=False, name=None)
        ):
            gid, symbol, authority, synonyms, ensembl = row
            canonical = authority or symbol
            if not canonical:
                continue

            entry = (gid or f"row_{i}", canonical, bool(authority))

            def add(name, kind):
                if name and normalize(name) in wanted:
                    hits[kind][normalize(name)].add(entry)

            add(authority, "exact_authority")
            add(symbol, "exact_symbol")
            add(gid, "id_match")

            for name in ensembl.split("|"):
                add(name, "ensembl_match")

            for name in synonyms.split("|"):
                add(name, "synonym")

    rows = []

    for gene in query:
        record = {
            "input": gene,
            "reference_gene": None,
            "match_type": "not_found",
            "status": "unmatched",
            "candidate_count": 0,
        }

        if gene in target:
            record.update(
                reference_gene=gene,
                match_type="exact_reference",
                status="matched",
                candidate_count=1,
            )
        else:
            key = normalize(gene)

            if re.fullmatch(r"ENSG\d+(?:\.\d+)?", gene, re.IGNORECASE):
                kinds = ("ensembl_match",)
            elif gene.isdecimal():
                kinds = ("id_match",)
            else:
                kinds = ("exact_authority", "exact_symbol", "synonym")

            for kind in kinds:
                candidates = hits[kind].get(key, set())
                if not candidates:
                    continue

                unique = {(gid, symbol) for gid, symbol, _ in candidates}
                record["match_type"] = kind
                record["candidate_count"] = len(unique)

                chosen = None

                if len(unique) == 1:
                    chosen = next(iter(unique))

                elif kind == "exact_symbol":
                    authoritative = {
                        (gid, symbol)
                        for gid, symbol, has_authority in candidates
                        if has_authority
                    }
                    if len(authoritative) == 1:
                        chosen = next(iter(authoritative))

                if chosen is None:
                    record["status"] = "ambiguous"
                elif chosen[1] in target:
                    record.update(
                        reference_gene=chosen[1],
                        status="matched",
                    )
                else:
                    record["status"] = "not_in_reference"

                break

        rows.append(record)

    # Prefer exact Reference names. Never automatically sum duplicated features.
    ranks = {
        "exact_reference": 0,
        "exact_authority": 1,
        "ensembl_match": 1,
        "id_match": 1,
        "exact_symbol": 2,
        "synonym": 3,
    }

    grouped = defaultdict(list)
    for i, record in enumerate(rows):
        if record["status"] == "matched":
            grouped[record["reference_gene"]].append(i)

    for indices in grouped.values():
        if len(indices) == 1:
            continue

        best_rank = min(ranks[rows[i]["match_type"]] for i in indices)
        best = [i for i in indices if ranks[rows[i]["match_type"]] == best_rank]
        winner = best[0] if len(best) == 1 else None

        for i in indices:
            if i != winner:
                rows[i]["reference_gene"] = None
                rows[i]["status"] = "collision"

    return pd.DataFrame(rows)


__all__ = ["map_genes"]
