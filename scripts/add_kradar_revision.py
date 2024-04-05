import argparse
import os
import shutil


def main(src: str, dst: str, revision: str = 'v2'):
    # Copy label data for all sequences
    for seq in os.listdir(src):
        # Create label folder
        os.makedirs(os.path.join(dst, seq, f"info_label_{revision}"), exist_ok=True)

        # Copy all labels
        for filename in os.listdir(os.path.join(src, seq)):
            shutil.copy2(os.path.join(src, seq, filename),
                         os.path.join(dst, seq, f"info_label_{revision}", filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='/data/kradar/KRadar_refined_label_by_UWIPL',
                        help="Path to the raw dataset folder.")
    parser.add_argument('--dst', type=str, default='/data/kradar/raw',
                        help="Path to save the processed dataset.")
    args = parser.parse_args()

    main(src=args.src, dst=args.dst)
