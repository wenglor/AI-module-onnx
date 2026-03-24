import json
from pathlib import Path
from typing import Any, Dict, List


def merge_coco(json_paths: List[str], output_path: str) -> None:
    """
    Merge multiple COCO JSON files into a single COCO JSON file.

    Example:
    merge_coco(
        [
            "../data/1/annotations/instances_default.json",
            "../data/2/annotations/instances_default.json",
            "../data/3/annotations/instances_default.json",
        ],
        "../data/coco-annotations/annotations/instances_default.json",
    )
    """
    if len(json_paths) < 1:
        raise ValueError("At least one COCO JSON file must be provided.")

    # Load all COCO JSON files
    cocos: List[Dict[str, Any]] = []
    for path in json_paths:
        with open(path, "r") as f:
            cocos.append(json.load(f))

    first_coco = cocos[0]

    # --- Category validation and setup ---
    if "categories" not in first_coco or not first_coco["categories"]:
        raise ValueError(
            "The first COCO file must contain a non-empty 'categories' list."
        )

    master_category_list = first_coco["categories"]
    master_names = {cat["name"] for cat in master_category_list}
    master_name_to_id = {cat["name"]: cat["id"] for cat in master_category_list}
    master_name_to_super = {
        cat["name"]: cat.get("supercategory", "") for cat in master_category_list
    }

    # Validate that all files have exactly the same category names and consistent supercategories
    for idx, coco in enumerate(cocos[1:], start=2):
        if "categories" not in coco or not coco["categories"]:
            raise ValueError(f"COCO file {idx} does not contain a 'categories' list.")

        this_names = {cat["name"] for cat in coco["categories"]}
        if this_names != master_names:
            raise ValueError(
                f"Category names in file {idx} do not match the first file. "
                "All files must have exactly the same category names."
            )

        for cat in coco["categories"]:
            name = cat["name"]
            if cat.get("supercategory", "") != master_name_to_super[name]:
                raise ValueError(
                    f"Supercategory mismatch for category '{name}' in file {idx}."
                )

    # Build merged structure (use first file's info/licenses/categories)
    merged: Dict[str, Any] = {
        "info": first_coco.get("info", {}),
        "licenses": first_coco.get("licenses", []),
        "categories": master_category_list,
    }

    merged["images"] = []
    merged["annotations"] = []

    # Track images by file_name to detect duplicates
    # Maps file_name -> {"image": image_dict, "image_id": int, "file_index": int}
    images_by_filename: Dict[str, Dict[str, Any]] = {}

    # Track all annotations, keyed by image_id
    annotations_by_image_id: Dict[int, List[Dict[str, Any]]] = {}

    # Counter for new image IDs
    next_image_id = 1
    next_ann_id = 1

    # --- Process all files ---
    for file_idx, coco in enumerate(cocos):
        # Category mapping: old category_id -> master (first file's) category_id
        if file_idx == 0:
            # First file's categories are the master
            cat_map = {cat["id"]: cat["id"] for cat in coco["categories"]}
        else:
            cat_map = {
                cat["id"]: master_name_to_id[cat["name"]] for cat in coco["categories"]
            }

        # Image reindexing with duplicate detection
        image_map: Dict[int, int] = {}
        for img in coco.get("images", []):
            old_id = img["id"]
            file_name = img.get("file_name", "")

            if file_name in images_by_filename:
                # Duplicate found - print warning
                prev_file_idx = images_by_filename[file_name]["file_index"]
                print(
                    f"WARNING: Duplicate image '{file_name}' found in file {file_idx + 1} "
                    f"(previously in file {prev_file_idx + 1}). Using the later occurrence."
                )

                # Get the existing image_id that we'll reuse
                new_id = images_by_filename[file_name]["image_id"]
                image_map[old_id] = new_id

                # Replace the image data with the new one
                new_img = img.copy()
                new_img["id"] = new_id
                images_by_filename[file_name]["image"] = new_img
                images_by_filename[file_name]["file_index"] = file_idx

                # Remove old annotations for this image
                if new_id in annotations_by_image_id:
                    annotations_by_image_id[new_id] = []
            else:
                # New image
                new_id = next_image_id
                next_image_id += 1
                image_map[old_id] = new_id

                new_img = img.copy()
                new_img["id"] = new_id
                images_by_filename[file_name] = {
                    "image": new_img,
                    "image_id": new_id,
                    "file_index": file_idx,
                }

        # Annotation reindexing and remapping
        for ann in coco.get("annotations", []):
            old_image_id = ann["image_id"]
            if old_image_id not in image_map:
                raise ValueError(
                    f"Annotation references unknown image_id {old_image_id}."
                )

            new_image_id = image_map[old_image_id]
            old_cat_id = ann["category_id"]
            if old_cat_id not in cat_map:
                raise ValueError(
                    f"Annotation references unknown category_id {old_cat_id}."
                )

            new_cat_id = cat_map[old_cat_id]

            new_ann = ann.copy()
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = new_image_id
            new_ann["category_id"] = new_cat_id

            # Store annotation by image_id
            if new_image_id not in annotations_by_image_id:
                annotations_by_image_id[new_image_id] = []
            annotations_by_image_id[new_image_id].append(new_ann)

    # Build final merged lists from the tracked data
    for img_data in images_by_filename.values():
        merged["images"].append(img_data["image"])

    for annotations in annotations_by_image_id.values():
        merged["annotations"].extend(annotations)

    # Write merged JSON
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=4)

    print(f"Merged {len(cocos)} COCO JSON file(s) saved to {output_path}")
    print(f"  Total images: {len(merged['images'])}")
    print(f"  Total annotations: {len(merged['annotations'])}")
    print(f"  Categories (using IDs from first file): {len(merged['categories'])}")


def remap_coco_ids(coco_path, id_map):
    p = Path(coco_path)
    data = json.loads(p.read_text())
    # remap categories
    cat_by_old = {c["id"]: c for c in data["categories"]}
    for old, new in id_map.items():
        cat_by_old[old]["id"] = new
    # remap annotations
    for ann in data["annotations"]:
        if ann["category_id"] in id_map:
            ann["category_id"] = id_map[ann["category_id"]]
    p.write_text(json.dumps(data))
    print(f"Remapped {coco_path}")


def transform_coco_to_rfdetr_format(folder_path, max_category_id):

    id_map = {i + 1: i for i in range(max_category_id)}

    remap_coco_ids(folder_path / "annotations/instances_default.json", id_map)
    (folder_path / "annotations/instances_default.json").rename(
        folder_path / "_annotations.coco.json"
    )
    for item in (folder_path / "images/default").iterdir():
        item.rename(folder_path / item.name)

    (folder_path / "annotations").rmdir()
    (folder_path / "images/default").rmdir()
    (folder_path / "images").rmdir()
