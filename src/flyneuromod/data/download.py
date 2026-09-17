"""Fetching the connectome files this library builds on.

No data is committed to the repository. These files keep their own licences and
citation requirements (see ``docs/references.md``); they are downloaded from the
projects that publish them.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/raw")
TIMEOUT_SECONDS = 120

SHIU_MODEL_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
FLYWIRE_ANNOTATIONS_COMMIT = "8587524c1748ce5ef2080822a2fc890fc03bf597"
"""Pinned commits: the tests assert exact neuron counts, so the data must not move."""


@dataclass(frozen=True, slots=True)
class DataFile:
    """One downloadable input file.

    Attributes
    ----------
    name:
        File name on disk.
    url:
        Where to fetch it from, pinned to a commit.
    sha256:
        Expected checksum; a download that does not match is rejected.
    approximate_size_mb:
        Rough size, so the user knows what a download will cost.
    source:
        Who publishes it and under what terms.
    """

    name: str
    url: str
    sha256: str
    approximate_size_mb: int
    source: str


FILES = (
    DataFile(
        name="Completeness_783.csv",
        url=(
            "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/"
            f"{SHIU_MODEL_COMMIT}/Completeness_783.csv"
        ),
        sha256="bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
        approximate_size_mb=3,
        source="Shiu et al. 2024 brain model repository (MIT), FlyWire v783 neuron list",
    ),
    DataFile(
        name="Connectivity_783.parquet",
        url=(
            "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/"
            f"{SHIU_MODEL_COMMIT}/Connectivity_783.parquet"
        ),
        sha256="efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
        approximate_size_mb=96,
        source="Shiu et al. 2024 brain model repository (MIT), FlyWire v783 connectivity",
    ),
    DataFile(
        name="flywire_annotations_v783.tsv",
        url=(
            "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
            f"{FLYWIRE_ANNOTATIONS_COMMIT}/supplemental_files/"
            "Supplemental_file1_neuron_annotations.tsv"
        ),
        sha256="9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
        approximate_size_mb=30,
        source="Schlegel et al. 2024 cell-type annotations",
    ),
)


def total_size_mb() -> int:
    """Total download size in megabytes."""
    return sum(f.approximate_size_mb for f in FILES)


def sha256_of(path: Path) -> str:
    """Checksum of a file, so a download can be verified or recorded."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    directory: str | Path = DEFAULT_DATA_DIR, force: bool = False
) -> dict[str, Path]:
    """Download the connectome files that are not present yet.

    Parameters
    ----------
    directory:
        Where to put them.
    force:
        Re-download files that already exist.

    Returns
    -------
    dict
        Mapping from file name to its path on disk.

    Raises
    ------
    RuntimeError
        If a download fails, naming the file and the URL.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for data_file in FILES:
        target = directory / data_file.name
        paths[data_file.name] = target
        if target.is_file() and not force:
            if sha256_of(target) != data_file.sha256:
                logger.warning(
                    "%s is present but does not match the pinned checksum; "
                    "results may differ from the documented ones (use --force to re-download)",
                    data_file.name,
                )
            else:
                logger.info("%s already present and verified", data_file.name)
            continue

        logger.info(
            "downloading %s (~%d MB) from %s",
            data_file.name,
            data_file.approximate_size_mb,
            data_file.source,
        )
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(data_file.url, timeout=TIMEOUT_SECONDS) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
        except Exception as error:  # noqa: BLE001 - re-raised with context below
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"could not download {data_file.name} from {data_file.url}: {error}"
            ) from error

        checksum = sha256_of(temporary)
        if checksum != data_file.sha256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"{data_file.name} has checksum {checksum}, expected {data_file.sha256}; "
                "the upstream file changed or the download was corrupted"
            )
        temporary.replace(target)

    return paths
