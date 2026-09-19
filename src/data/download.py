"""Download da base da Olist pelo kagglehub com manifesto de integridade.

Uso:
    python -m src.data.download            # baixa e copia o que ainda nao existe
    python -m src.data.download --force    # copia de novo, mesmo que exista

Regras:
- a autenticacao no Kaggle fica com o kagglehub (token em ~/.kaggle ou variavel de
  ambiente); nenhuma credencial passa por este modulo;
- os CSVs sao copiados do cache do kagglehub para data/raw/olist/;
- o SHA-256 e o tamanho de cada CSV sao gravados em data/manifest.json. Na primeira
  vez o manifesto e criado; nas seguintes, um hash diferente do registrado indica
  que a copia local nao e a mesma usada no projeto, e o aviso aparece no log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DATASET = "olistbr/brazilian-ecommerce"
RAW_DIR = Path("data/raw/olist")
MANIFEST = Path("data/manifest.json")

CSV_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict, path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_against_manifest(raw_dir: Path = RAW_DIR, manifest_path: Path = MANIFEST) -> list[str]:
    """Lista os arquivos ausentes ou com hash diferente do manifesto."""
    files = load_manifest(manifest_path).get("files", {})
    problems = []
    for name, meta in files.items():
        local = raw_dir / name
        if not local.exists():
            problems.append(f"{name}: ausente")
        elif sha256_of(local) != meta["sha256"]:
            problems.append(f"{name}: sha256 diferente do manifesto")
    return problems


def download(force: bool = False) -> dict:
    import kagglehub

    source = Path(kagglehub.dataset_download(DATASET))
    version = source.name if source.name.isdigit() else None
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    old = load_manifest().get("files", {})
    files = {}
    for name in CSV_FILES:
        src, dst = source / name, RAW_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"{name} nao veio no pacote do Kaggle ({source})")
        if force or not dst.exists():
            shutil.copy2(src, dst)
        digest = sha256_of(dst)
        if name in old and old[name]["sha256"] != digest:
            print(f"aviso: {name} difere do manifesto registrado")
        files[name] = {"sha256": digest, "bytes": dst.stat().st_size}
        print(f"{name:<42} {files[name]['bytes']:>12,} bytes")

    manifest = {
        "dataset": DATASET,
        "url": f"https://www.kaggle.com/datasets/{DATASET}",
        "licenca": "CC BY-NC-SA 4.0",
        "versao_kaggle": version,
        "files": files,
    }
    save_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="copia de novo mesmo que exista")
    args = parser.parse_args()
    download(force=args.force)
    problems = check_against_manifest()
    print("manifesto conferido" if not problems else "\n".join(problems))


if __name__ == "__main__":
    main()
