"""根据已复核的采样拼图冻结负例标签。"""

import cache_env
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "out/clarity_hardneg_20261019"
    items = []
    hand = root / "data/external/freihand_research"
    public = json.loads((hand / "download_manifest.json").read_text())
    for row in public["items"]:
        items.append(
            dict(
                path=row["path"],
                sha256=row["sha256"],
                boxes=[],
                source="FreiHAND_research",
                kind="hand_negative",
                group=row["group"],
                split=row["split"],
                label_status="no_visible_fallen_person",
                label_source="Assistant reviewed all 96 author training images in four contact sheets: only hands/forearms with held objects, no visible fallen body",
                terms=public["terms"],
            )
        )
    for folder, group, provenance in (
        (
            "raw_unreviewed",
            "current_dorm_chair_standing",
            "Assistant reviewed all 3 raw frames: chair/bed, hanging clothes and upright partial person; no visible fallen body",
        ),
        (
            "raw_partials",
            "current_hands_black_chest",
            "Assistant reviewed all 12 raw frames: hands/forearms, black-shirt chest and upright partial body. User clarified black garment is own chest, not a clothing picture. No visible fallen body",
        ),
    ):
        directory = evidence / folder
        raw = json.loads((directory / "capture.json").read_text())
        for row in raw["samples"]:
            path = (directory / row["file"]).with_suffix(".png")
            items.append(
                dict(
                    path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    raw_sha256=row["sha256"],
                    boxes=[],
                    source="user_camera",
                    kind="current_negative",
                    group=group,
                    split="train",
                    label_status="no_visible_fallen_person",
                    label_source=provenance,
                    input_source="ov5640",
                    test_context="live_scene",
                )
            )
    # 相机新采样不作为独立测试，后续需换取景和物理场景。
    scope = (
        "New camera captures are training scenes and same-scene regressions. FreiHAND selection uses disjoint "
        "index blocks, without a subject-independence claim. Original FallDataset/CAUCA validation remains fixed. "
        "Previously viewed test subjects are regression only. Phone-screen replay is separate from field acceptance."
    )
    output = evidence / "reviewed_negatives.json"
    output.write_text(
        json.dumps(dict(scope=scope, items=items, public_source=public), indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                images=len(items),
                camera=sum(v["source"] == "user_camera" for v in items),
            )
        )
    )


if __name__ == "__main__":
    main()
