"""Strictly parse repository YAML and the Citation File Format document."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def main() -> None:
    paths = sorted((ROOT / ".github").rglob("*.yml"))
    paths += sorted((ROOT / ".github").rglob("*.yaml"))
    paths += sorted((ROOT / "code" / "baselines").glob("*/configs/*.yaml"))
    paths += sorted((ROOT / "code" / "baselines").glob("*/reproducibility.yaml"))
    paths.append(ROOT / "link-exceptions.yaml")
    paths.append(ROOT / "CITATION.cff")
    if not paths:
        raise RuntimeError("No YAML/CFF files were discovered")
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            yaml.load(stream, Loader=UniqueKeyLoader)
    print(f"Parsed {len(paths)} YAML/CFF files")


if __name__ == "__main__":
    main()
