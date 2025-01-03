import argparse
import numpy as np
import cv2


def create_stereogram(depth_map_path, output_path, pattern_width=100):
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    height, width = depth_map.shape
    pattern = np.random.randint(0, 256, (height, pattern_width), dtype=np.uint8)
    stereogram = np.tile(pattern, (1, width // pattern_width + 1))[:, :width]
    depth_map = depth_map.astype(np.int32)

    for y in range(height):
        for x in range(width):
            offset = depth_map[y, x] // 3
            if x - pattern_width - offset >= 0:
                stereogram[y, x] = stereogram[y, x - pattern_width - offset]

    cv2.imwrite(output_path, stereogram)


def main():
    parser = argparse.ArgumentParser(description="Create a stereogram from a depth map.")
    parser.add_argument("depth_map", help="Path to the input depth map image.")
    parser.add_argument("output", help="Path to save the output stereogram image.")
    args = parser.parse_args()

    create_stereogram(args.depth_map, args.output)


if __name__ == "__main__":
    main()
