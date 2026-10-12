"""Build a signed exclusion ledger from verified earlier image pools."""
from __future__ import annotations

import iris_lora_colab as core


def build_exclusion_ledger(manifests):
    if not manifests:
        raise ValueError("Previous training manifests are required.")
    identifiers, image_hashes, fingerprints = set(), set(), []
    for manifest in manifests:
        expected = core.stable_hash({"recipe": manifest["recipe"], "records": manifest["records"]})
        if expected != manifest["fingerprint"]:
            raise ValueError("A previous dataset manifest failed its fingerprint check.")
        fingerprints.append(expected)
        for row in manifest["records"]:
            identifiers.add(str(row["image_id"]))
            image_hashes.add(row["image_sha256"])
    return dict(schema=1, prior_dataset_fingerprints=sorted(fingerprints),
                image_ids=sorted(identifiers), image_sha256=sorted(image_hashes))


def exclusion_recipe(ledger):
    return dict(sha256=core.stable_hash(ledger), excluded_id_count=len(ledger["image_ids"]),
                excluded_image_hash_count=len(ledger["image_sha256"]),
                prior_dataset_fingerprints=ledger["prior_dataset_fingerprints"])
