#!/usr/bin/env python3

import os
import sys
import argparse
import math
import cv2
import pyzed.sl as sl

def zed2_to_kitti(input_file, output_dir=None):
    """
    Converts a .zed2 (or .svo) file to a KITTI-like stereo dataset:
      output_dir/
      ├── calib.txt
      ├── times.txt
      ├── image_0
      │   └── 000000.png
      │   └── 000001.png
      │   ...
      └── image_1
          └── 000000.png
          └── 000001.png
          └── ...
    """

    # If output_dir is not specified, use the same directory as input_file
    if output_dir is None:
        output_dir = os.path.splitext(input_file)[0] + "_kitti_dataset"

    # Create necessary subdirectories
    image_0_dir = os.path.join(output_dir, "image_0")
    image_1_dir = os.path.join(output_dir, "image_1")
    os.makedirs(image_0_dir, exist_ok=True)
    os.makedirs(image_1_dir, exist_ok=True)

    times_file = open(os.path.join(output_dir, "times.txt"), 'w')

    # Initialize ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    print("Opening Input File: ", input_file)
    init_params.set_from_svo_file(input_file)  # or .zed2 file
    init_params.coordinate_units = sl.UNIT.METER
    init_params.svo_real_time_mode = False  # allows reading at whatever speed we want

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED file: {status}")
        sys.exit(1)

    runtime_params = sl.RuntimeParameters()
    # Retrieve calibration parameters
    cam_info = zed.get_camera_information()
    left_cam_params = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    right_cam_params = zed.get_camera_information().camera_configuration.calibration_parameters.right_cam

    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    baseline = calibration_params.stereo_transform.get_translation().get()[0] #in meters

    # TIP: If you want rectified images, retrieve with sl.VIEW.LEFT/RIGHT
    # If you prefer raw/unrectified, use sl.VIEW.LEFT_UNRECTIFIED/RIGHT_UNRECTIFIED
    view_left = sl.VIEW.LEFT
    view_right = sl.VIEW.RIGHT

    left_image = sl.Mat()
    right_image = sl.Mat()

    frame_count = 0

    # Loop through frames
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve rectified left/right images
            zed.retrieve_image(left_image, view_left)
            zed.retrieve_image(right_image, view_right)

            # Retrieve timestamp (in nanoseconds)
            ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

            # Convert sl.Mat to NumPy array (BGR format) for saving
            left_np = left_image.get_data()[:, :, :3]  # discard alpha if 4 channels
            right_np = right_image.get_data()[:, :, :3]

            # Build filenames (KITTI expects zero-padded frame indices)
            filename_left = f"{frame_count:06d}.png"
            filename_right = f"{frame_count:06d}.png"

            # Write images
            cv2.imwrite(os.path.join(image_0_dir, filename_left), left_np)
            cv2.imwrite(os.path.join(image_1_dir, filename_right), right_np)

            # Write timestamp to times.txt as seconds in float
            # Convert nanoseconds to seconds
            times_file.write(f"{ts * 1e-9:.9f}\n")

            frame_count += 1
        else:
            # No more frames or an error occurred
            break

    print(f"Exported {frame_count} stereo frames to {output_dir}")
    times_file.close()

    # Close ZED camera
    zed.close()

    # Create KITTI-style calib.txt
    # The standard KITTI format typically has lines like:
    #
    #  P0: fx 0  cx  0
    #       0  fy cy  0
    #       0  0  1   0
    #
    #  P1: fx 0  cx  tx
    #       0  fy cy  0
    #       0  0  1   0
    #
    # where tx = -fx * baseline (if rectified).
    #
    # For the ZED, baseline is in meters. Also check sign convention.
    # Typically, for a standard rectified pair, Tx = -fx * baseline.
    #
    # We'll assume the left camera is P0 and the right camera is P1.

    fx_left = left_cam_params.fx
    fy_left = left_cam_params.fy
    cx_left = left_cam_params.cx
    cy_left = left_cam_params.cy

    fx_right = right_cam_params.fx
    fy_right = right_cam_params.fy
    cx_right = right_cam_params.cx
    cy_right = right_cam_params.cy

    # For rectified pairs from the ZED, the same fx, fy, cx, cy is typically used,
    # but let's read them separately. The baseline can be used to compute Tx.
    # Note: In some calibrations, T[0] might already be negative if the cameras are side by side.
    # KITTI typically has P1 with tx = -fx * baseline.
    #
    # We can interpret baseline as positive, so we set tx = -fx_left * baseline.

    tx = -fx_left * baseline

    calib_str = []
    # P0 line
    P0 = f"P0: {fx_left:.9f} 0.0 {cx_left:.9f} 0.0 0.0 {fy_left:.9f} {cy_left:.9f} 0.0 0.0 0.0 1.0 0.0"
    calib_str.append(P0)
    # P1 line
    P1 = f"P1: {fx_right:.9f} 0.0 {cx_right:.9f} {tx:.9f} 0.0 {fy_right:.9f} {cy_right:.9f} 0.0 0.0 0.0 1.0 0.0"
    calib_str.append(P1)

    with open(os.path.join(output_dir, "calib.txt"), 'w') as f:
        for line in calib_str:
            f.write(line + "\n")

    print(f"Saved calibration file to {os.path.join(output_dir, 'calib.txt')}")


def main():
    parser = argparse.ArgumentParser(description="Convert .zed2/.svo to KITTI-like dataset")
    parser.add_argument("input_file", help="Path to the input .zed2 or .svo file")
    parser.add_argument("--output_dir", "-o", help="Path to the output directory (default: same as input file)")

    args = parser.parse_args()

    zed2_to_kitti(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
