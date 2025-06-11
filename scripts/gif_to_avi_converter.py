import os
import cv2
import imageio
import argparse

def gif_to_avi(input_path):
    gif = imageio.mimread(input_path)
    if not gif:
        return
    height, width, _ = gif[0].shape
    avi_path = os.path.splitext(input_path)[0] + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10  # or calculate based on gif metadata
    out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
    for frame in gif:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert GIF to AVI')
    parser.add_argument('--input_gif', type=str, help='Path to input .gif file')
    args = parser.parse_args()
    gif_to_avi(args.input_gif)


# python scripts/gif_to_avi_converter.py --input_gif 1-3-1.gif