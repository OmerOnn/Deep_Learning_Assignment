import os

def parse_pairs_file(pairs_path, images_root):
    pairs = []

    with open(pairs_path, 'r') as f:
        lines = f.readlines()[1:]  # שים לב – השורה הראשונה זה count

    for line in lines:
        parts = line.strip().split()

        if len(parts) == 3:
            # Same person
            name = parts[0]
            img1 = int(parts[1])
            img2 = int(parts[2])

            path1 = os.path.join(images_root, name, f"{name}_{img1:04d}.jpg")
            path2 = os.path.join(images_root, name, f"{name}_{img2:04d}.jpg")

            label = 1

        elif len(parts) == 4:
            # Different persons
            name1, img1, name2, img2 = parts

            path1 = os.path.join(images_root, name1, f"{name1}_{int(img1):04d}.jpg")
            path2 = os.path.join(images_root, name2, f"{name2}_{int(img2):04d}.jpg")

            label = 0

        pairs.append((path1, path2, label))

    return pairs