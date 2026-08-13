"""Build verbatim benchmark inputs from fixed ranges of public RFC sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RANGES = {
    "rfc_identity_01": [(3986, 32, 38), (3986, 588, 597), (3986, 1113, 1120),
                        (9110, 384, 390), (9110, 692, 704), (9110, 851, 858)],
    "rfc_identity_02": [(8446, 288, 303), (8446, 357, 364), (8446, 410, 418),
                        (8446, 432, 438), (8446, 452, 460)],
    "rfc_identity_03": [(7540, 400, 411), (7540, 3670, 3678), (7540, 3718, 3725),
                        (7540, 3774, 3782), (9000, 1756, 1764)],
    "rfc_identity_04": [(9000, 2499, 2508), (9000, 3120, 3128), (9000, 3514, 3523),
                        (9000, 3537, 3546), (6749, 2708, 2717), (6749, 3258, 3268)],
    "rfc_identity_05": [(9293, 15, 22), (9293, 242, 251), (9293, 328, 336),
                        (9420, 20, 35), (9420, 244, 252), (9420, 1136, 1148)],
    "rfc_identity_06": [(9420, 416, 429), (9420, 430, 438), (9420, 287, 296),
                        (6749, 2424, 2432), (9112, 257, 264)],
}


def build(source_dir: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for document_id, ranges in RANGES.items():
        blocks = []
        refs = []
        for rfc, start, end in ranges:
            path = source_dir / f"rfc{rfc}.txt"
            raw = path.read_bytes()
            lines = raw.decode("utf-8-sig").splitlines()
            blocks.append("\n".join(lines[start - 1:end]).strip())
            refs.append({"rfc": rfc, "url": f"https://www.rfc-editor.org/rfc/rfc{rfc}.txt",
                         "line_start": start, "line_end": end,
                         "source_sha256": hashlib.sha256(raw).hexdigest()})
        text = "\n\n".join(blocks) + "\n"
        target = output_dir / f"{document_id}.txt"
        target.write_text(text, encoding="utf-8")
        manifest.append({"document_id": document_id, "path": f"benchmark_sources/{target.name}",
                         "sha256": hashlib.sha256(text.encode()).hexdigest(), "verbatim_ranges": refs})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source_dir, args.output_dir)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
