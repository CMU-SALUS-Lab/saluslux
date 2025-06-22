import math
import numpy as np
from scipy.interpolate import interp1d, griddata
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm  # For better illuminance visualization
import re
import pandas as pd
from datetime import datetime
from scipy.fft import fft2, fftshift
from scipy.interpolate import RegularGridInterpolator

class LightSource:
    def __init__(self, pole_position, arm_length, mounting_angle, orientation, photometric_data):
        """
        Initialize a light source separate from the pole.

        :param pole_position: (x_p, y_p, h_p) location of the pole.
        :param arm_length: Distance from pole to light head.
        :param mounting_angle: Angle of the arm (0° = horizontal, 90° = vertical).
        :param orientation: Rotation angle
        :param photometric_data: IES light distribution data.
        """
        self.x_p, self.y_p, self.h_p = pole_position
        self.arm_length = arm_length
        self.mounting_angle = mounting_angle
        self.orientation = orientation
        self.photometric_data = photometric_data

        # Compute the actual light source position
        self.x = self.x_p + arm_length * math.cos(math.radians(orientation))
        self.y = self.y_p + arm_length * math.sin(math.radians(orientation))
        self.h = self.h_p + arm_length * math.sin(math.radians(mounting_angle))

def fit_interpolated_candela(candela_matrix, vertical_angles, horizontal_angles, extrapolate=False):
    """
    Returns a callable interpolator function that approximates candela values
    using bilinear interpolation on the original IES angular grid.
    Allows toggling extrapolation behavior.
    """
    interpolator = RegularGridInterpolator(
        (horizontal_angles, vertical_angles),
        candela_matrix,
        bounds_error=not extrapolate,
        fill_value=None if extrapolate else 0.0
    )

    def candela_interp(h_deg, v_deg):
        h_deg = np.atleast_1d(h_deg)
        v_deg = np.atleast_1d(v_deg)
        query_points = np.stack(np.meshgrid(h_deg, v_deg, indexing='ij'), axis=-1).reshape(-1, 2)
        interpolated = interpolator(query_points)
        return interpolated.reshape(h_deg.shape + v_deg.shape).squeeze()

    return candela_interp, {
        'method': 'interpolation',
        'angle_grids': (horizontal_angles, vertical_angles),
        'extrapolate': extrapolate
    }


def parse_ies(file_path, use_ml=False, light_to_sky=False, debug=False, extrapolate=False):
    """Improved IES file parser adhering to LM-63 standard."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    metadata = {}
    candela_values = []
    vertical_angles = []
    horizontal_angles = []

    filename = os.path.basename(file_path).lower()
    parsing_candela = False # Flag to indicate candela data parsing
    vertical_angle_count, horizontal_angle_count = None, None
    found_IES_type = False
    for line in lines:
        if line.startswith("[") and "]" in line:
            end_bracket = line.index("]")
            key = line[1:end_bracket].strip()
            value = line[end_bracket+1:].strip()
            metadata[key] = value
            if key == "IES_type":
                found_IES_type = True

    if found_IES_type == False:
        if use_ml:
            metadata['IES_type'] = "ML_predicted_type"  # You could integrate real model here
        else:
            # Heuristic rules based on filename
            if re.search(r'(g1-5|5wq|5mq|t5|type5|v)', filename):
                metadata['IES_type'] = 'Type V'
            elif re.search(r'(g1-4|t4|type4|iv)', filename):
                metadata['IES_type'] = 'Type IV'
            elif re.search(r'(g1-3|t3|type3|iii)', filename):
                metadata['IES_type'] = 'Type III'
            elif re.search(r'(g1-2|t2r|t2|type2|ii)', filename):
                metadata['IES_type'] = 'Type II'
            else:
                metadata['IES_type'] = 'Unknown'

    # for line in lines:
    for i, line in enumerate(lines):
        line = line.strip()
        # Skip empty lines
        if not line:
            continue

        # Parse metadata
        if line.startswith("["):
            parts = line.strip('[]').split(' ', 1)
            if len(parts) == 2:
                key, value = parts
                metadata[key] = value
            continue

        # Parse tilt information and candela header
        if line.startswith("TILT"):
            metadata["tilt"] = line.split("=")[1].strip()
            continue

        # Parse the key photometric parameters
        photometric_params_index = None
        if vertical_angle_count is None and len(line.split()) >= 10:
            parts = line.split()
            photometric_params_index = int(i) # Store the index for later use
            try:
                metadata["number_of_lamps"] = int(parts[0])
                metadata["lumens_per_lamp"] = float(parts[1])
                metadata["candela_multiplier"] = float(parts[2])
                vertical_angle_count = int(parts[3])
                horizontal_angle_count = int(parts[4])
                metadata["photometric_type"] = int(parts[5])
                metadata["units_type"] = int(parts[6])
                metadata["luminous_dimensions"] = tuple(map(float, parts[7:10]))

                # Try to detect wattage from next line
                next_line = lines[photometric_params_index + 1].strip() #Ex. 1.000 1 150.3
                parts = list(map(float, next_line.split()))
                if len(parts) >= 3:
                    metadata["ballast_factor"] = parts[0]
                    metadata["file_generation_type"] = parts[1]
                    metadata["wattage"] = parts[2]
                elif len(parts) == 2:
                    metadata["ballast_factor"] = parts[0]
                    metadata["file_generation_type"] = parts[1]
                    metadata["wattage"] = 0
                elif len(parts) == 1:
                    metadata["ballast_factor"] = parts[0]
                    metadata["file_generation_type"] = 0
                    metadata["wattage"] = 0
                else:
                    metadata["ballast_factor"] = 1.0  # default per LM-63
                    metadata["file_generation_type"] = 1.00001  # undefined
                    metadata["wattage"] = 0
                lines[i + 1] = ''  # This prevents it from being re-processed in the next loop
                print(f"Photometric Parameters: {metadata}") if debug else None
                continue
            except ValueError:
                raise ValueError(f"Error parsing photometric parameters: {line}")

        # Parse vertical angles
        if vertical_angle_count is not None and len(vertical_angles) < vertical_angle_count:
            vertical_angles.extend(map(float, line.split()))
            if len(vertical_angles) > vertical_angle_count:
                # Trim excess values
                vertical_angles = vertical_angles[:vertical_angle_count]
            # print(f"Vertical Angles ({len(vertical_angles)}/{vertical_angle_count}): {vertical_angles}")
            continue

        # Parse horizontal angles
        if horizontal_angle_count is not None and len(horizontal_angles) < horizontal_angle_count:
            horizontal_angles.extend(map(float, line.split()))
            if len(horizontal_angles) > horizontal_angle_count:
                # Trim excess values
                horizontal_angles = horizontal_angles[:horizontal_angle_count]
            # print(f"Horizontal Angles ({len(horizontal_angles)}/{horizontal_angle_count}): {horizontal_angles}")
            continue

        # Parse candela values
        if len(vertical_angles) == vertical_angle_count and len(horizontal_angles) == horizontal_angle_count:
            if not parsing_candela:
                parsing_candela = True
                # print("Switching to parsing candela values...")
            candela_values.extend(map(float, line.split()))
            # print(f"Current Candela Values Count: {len(candela_values)}")

    # Validate parsed data
    if vertical_angle_count is None or horizontal_angle_count is None:
        raise ValueError("Missing vertical or horizontal angle count in the IES file.")

    expected_candela_count = vertical_angle_count * horizontal_angle_count
    if len(candela_values) != expected_candela_count:
        # print(f"Expected Candela Count: {expected_candela_count}, Parsed: {len(candela_values)}")
        raise ValueError(f"Mismatch in candela data. Expected {expected_candela_count}, got {len(candela_values)}.")

    candela_matrix = np.array(candela_values).reshape((horizontal_angle_count, vertical_angle_count))
    # in case metadata["candela_multiplier"] is not 1, multiply the candela matrix
    if metadata.get("candela_multiplier", 1) != 1:
        candela_matrix *= metadata["candela_multiplier"]
        # print(f"Applying candela multiplier: {metadata['candela_multiplier']}")

    vertical_angles = np.array(vertical_angles)
    horizontal_angles = np.array(horizontal_angles)

    if horizontal_angles[-1] <= 90:
        print("There is only 1/4 of the horizontal angles") if debug else None
        horizontal_angles = np.concatenate([horizontal_angles, 180 - horizontal_angles[::-1], 180 + horizontal_angles, 360 - horizontal_angles[::-1]])
        candela_matrix = np.vstack([candela_matrix, candela_matrix[::-1], candela_matrix, candela_matrix[::-1]])
        # Remove duplicates
        unique_ha, idx = np.unique(horizontal_angles, return_index=True)
        horizontal_angles = unique_ha
        candela_matrix = candela_matrix[idx, :]
    elif horizontal_angles[-1] <= 180:
        print("There is only 1/2 of the horizontal angles") if debug else None
        ha_mirror = 360 - horizontal_angles[::-1]
        if horizontal_angles[-1] == ha_mirror[0]:
            ha_mirror = ha_mirror[1:]  # remove duplicate 180° if mirrored
        horizontal_angles = np.concatenate([horizontal_angles, ha_mirror])
        candela_matrix = np.vstack([candela_matrix, candela_matrix[::-1]])

        # Sort and deduplicate horizontal angles
        sort_idx = np.argsort(horizontal_angles)
        horizontal_angles = horizontal_angles[sort_idx]
        candela_matrix = candela_matrix[sort_idx, :]

    if vertical_angles[-1] <= 90:
        print("There is only 1/2 of the vertical angles") if debug else None
        va_mirror = 180 - vertical_angles[::-1]
        if vertical_angles[-1] == va_mirror[0]:
            va_mirror = va_mirror[1:] # remove duplicate 90° (or the end) if mirrored
        vertical_angles = np.concatenate([vertical_angles, va_mirror])
        if light_to_sky:
            candela_matrix = np.hstack([candela_matrix, candela_matrix[:, ::-1]])
        else:
            zero_cols = np.zeros((candela_matrix.shape[0], len(va_mirror)))
            candela_matrix = np.hstack([candela_matrix, zero_cols])
    print(f"Horizontal Angles: {horizontal_angles}") if debug else None
    print(f"Vertical Angles: {vertical_angles}") if debug else None
    assert candela_matrix.shape == (len(horizontal_angles), len(vertical_angles))
    assert len(np.unique(horizontal_angles)) == len(horizontal_angles), "Duplicate values in horizontal_angles"
    assert len(np.unique(vertical_angles)) == len(vertical_angles), "Duplicate values in vertical_angles"

    candela_fn, candela_model_info = fit_interpolated_candela(
        candela_matrix,
        vertical_angles,
        horizontal_angles,
        extrapolate=extrapolate
    )
    
    metadata['source_model'] = metadata.get('LUMCAT', os.path.basename(file_path))
    # print(f"Source Model: {metadata['source_model']}")
    metadata['source_brand'] = metadata.get('MANUFAC', 'Unknown')
    # print(f"Source Brand: {metadata['source_brand']}")
    metadata['lumen'] = metadata.get('lumens_per_lamp', 0) * metadata.get('number_of_lamps', 1)
    return {
        'metadata': metadata,
        'vertical_angles': vertical_angles,
        'horizontal_angles': horizontal_angles,
        'candela_values': candela_matrix,
        'lumens': metadata['lumen'],
        'candela_function': candela_fn,
        'candela_model_info': candela_model_info
    } 

def plot_photometry_ies(datafromies, show=True, save=False, saveto = None, debug=False, plot_fit = False):
    V = datafromies['vertical_angles']
    H = datafromies['horizontal_angles']
    Z = datafromies['candela_values']
    F = datafromies['candela_function']

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})

    for az in [0, 180]:
        theta = np.radians(V)
        r_raw = Z[H.tolist().index(az)] if az in H else np.zeros_like(V)
        r_fit = F(az, V)
        axs[0].plot(theta, r_raw, label=f"Raw {az}°")
        axs[0].plot(theta, r_fit, '--', label=f"Fit {az}°") if plot_fit else None

    axs[0].set_title("Vertical Candela Profiles")
    axs[0].set_theta_zero_location("S")  # Downward direction is center
    axs[0].set_theta_direction(-1)
    axs[0].legend()

    for vcut in [0, 30, 45, 60, 75]:
        h_theta = np.radians(H)
        raw = Z[:, V.tolist().index(vcut)] if vcut in V else np.zeros_like(H)
        fit = F(H, vcut)
        axs[1].plot(h_theta, raw, label=f"Raw {vcut}°")
        axs[1].plot(h_theta, fit, '--', label=f"Fit {vcut}°") if plot_fit else None

    axs[1].set_title("Horizontal Candela Cuts")
    axs[1].legend()
    plt.suptitle(f"Candela Fit", fontsize=14) if plot_fit else plt.suptitle(f"Candela Data", fontsize=14)

    if debug:
        print(f"Horizontal angles: {H}")
        print(f"Vertical angles: {V}")

    if save and saveto:
        plt.savefig(saveto)
    if show:
        plt.show()
    plt.close(fig)