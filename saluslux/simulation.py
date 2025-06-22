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

def generate_unified_grid(grid_size, bounds_list, height=0):
    """
    Generates a unified grid covering all calculation surfaces.
    """
    all_points = []

    for bounds in bounds_list:
        x_min, x_max, y_min, y_max = bounds
        rows, cols = grid_size
        x_vals = np.linspace(x_min, x_max, cols)
        y_vals = np.linspace(y_min, y_max, rows)

        for x in x_vals:
            for y in y_vals:
                all_points.append((x, y, height))
    return all_points

def compute_illuminance(grid_points, light_sources, normal_vector, height=1.5):
    """
    Computes illuminance for a given surface normal at multiple grid points.

    Parameters:
        grid_points: List of (x, y, z) tuples representing the calculation surface.
        light_sources: List of LightSource objects.
        normal_vector: (Nx, Ny, Nz) unit vector defining the surface direction.

    Returns:
        List of (x, y, illuminance) tuples.
    """
    results = []
    N = np.array(normal_vector) / np.linalg.norm(normal_vector)

    for x, y, z in grid_points:
        z = height  # Apply height offset
        total_illuminance = 0

        for light in light_sources:
            # Vector from light to calculation point
            dx = x - light.x
            dy = y - light.y
            dz = z - light.h
            d = np.linalg.norm([dx, dy, dz])

            if d == 0:
                continue # Avoid division by zero
            L = np.array([dx, dy, dz])
            L_unit = L / d  # Normalize light direction vector

            phi_math = np.degrees(np.arctan2(dy, dx)) % 360
            phi_relative = (phi_math - light.orientation) % 360

            # Vertical angle calculation
            cos_theta = np.dot(N, -L_unit)
            cos_theta =  np.clip(cos_theta, -1, 1)  # Ensure cos_theta is within valid range
            # should be between -1 and 1, but due to bit precision errors, it can be slightly outside this range
            #theta = np.degrees(np.arccos(dz/d))  # This is wrong becuase it makes dark holes
            theta = np.degrees(np.arccos(cos_theta))  # Convert to degrees
            # if the light is behind the surface, skip it, no light falling on the surface
            if cos_theta < 0:
                continue
            # # Photometric data lookup -------------------------------------------
            # vertical_angles = light.photometric_data['vertical_angles']
            # horizontal_angles = light.photometric_data['horizontal_angles']
            # candela_matrix = light.photometric_data['candela_values']

            # # Find nearest horizontal angle index
            # # Find closest horizontal angle (simplified example) you need bilinear interpolation between nearest angles
            # h_idx = np.abs(horizontal_angles - phi_relative).argmin()

            # # Interpolate vertical profile
            # intensity_interp = interp1d(vertical_angles, candela_matrix[h_idx],
            #                           kind='linear', fill_value="extrapolate", bounds_error=False)
            # luminous_intensity = intensity_interp(theta)

            luminous_intensity = light.photometric_data['candela_function'](phi_relative, theta)


            # Illuminance calculation -------------------------------------------
            E = (luminous_intensity / d**2) * cos_theta
            total_illuminance += E

        results.append((x, y, total_illuminance))
    return results

def compute_illuminance_sc(grid_points, light_sources, normal_vector=(0, 1, 0), height=1.5, directions=None):
    """
    Computes semi-cylindrical illuminance by averaging vertical illuminance
    from directions around a forward-facing vector N (normal).

    Parameters:
        grid_points: List of (x, y, z)
        light_sources: List of LightSource
        normal_vector: Direction the cylinder is "facing" (e.g., pedestrian forward)
        height: Height at which the cylinder is centered
        directions: List of angles in degrees (relative to forward)

    Returns:
        List of (x, y, E_sc)
    """
    if directions is None:
        directions = [-90, -60, -45, -30, 0, 30, 45, 60, 90]

    # Normalize the base forward vector
    forward = np.array(normal_vector[:2])
    forward = forward / np.linalg.norm(forward)

    # Precompute rotated directions (unit vectors)
    dir_vectors = []
    for angle in directions:
        angle_rad = math.radians(angle)
        rot = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad),  math.cos(angle_rad)]
        ])
        rotated = rot @ forward
        dir_vectors.append((rotated[0], rotated[1], 0))

    # Accumulate all results
    illum_sum = np.zeros(len(grid_points))

    for dir_vec in dir_vectors:
        illum = compute_illuminance(grid_points, light_sources, normal_vector=dir_vec, height=height)
        illum_sum += np.array([e[2] for e in illum])

    illum_avg = illum_sum / len(directions)
    results = [(pt[0], pt[1], e) for pt, e in zip(grid_points, illum_avg)]
    return results


def compute_luminance(illuminance_results, reflection_factor=0.20):
    """
    Computes luminance from illuminance.
    Returns same (x, y, L) structure.
    """
    # https://www.youtube.com/watch?v=HPNW0we-ft0 Reflectance Models | Radiometry and Reflectance
    # https://www.handprint.com/HP/WCL/color3.html#luminousintensity
    # A hemisphere has 2π steradians
    # potentially complex and depends on (1) the illuminance onto the surface,
    # (2) the surface reflectance or albedo of the surface (the proportion of light falling on the surface that is reflected from it),
    # (3) the angle of incidence of the light,
    # (4) the angle of view to the surface, and
    # (5) how much the surface diffuses or randomly scatters the light. These issues are explored on a later page, but a few points should be mentioned here.
    # There are several alternative luminance measures that attempt to equate the luminance of surface with the illuminance incident on it.
    # The most common are the metric apostilb and millilambert (preferable to the inconveniently small lambert) and the USA foot lambert:
    # These are all related to the general formula for the luminance of a surface that is viewed perpendicular to the surface and illuminated from all sides by a perfectly diffuse light:
    return [(x, y, (E * reflection_factor) / math.pi) for (x, y, E) in illuminance_results]

def compute_glare_rating(grid_points, observer_eye_height, observer_view_vec, light_sources, rho=0.2,proximity=1.8):
    """
    Compute CIE Glare Rating (GR) at each point on a grid.

    Parameters:
        grid_points: list of (x, y) tuples where GR should be computed
        observer_eye_height: height (z) of the observer's eye, typically 1.5 m
        observer_view_vec: 3D unit vector for observer's line of sight
        light_sources: list of LightSource objects
        E_HOR_avg: average horizontal illuminance in the environment (when not specified, defaults to 0)
        rho: surface reflectance (default = 0.2)

    Returns:
        List of GR values, one per grid point (as in (x, y, GR))
    """
    view_vec = np.array(observer_view_vec)
    view_vec = view_vec / np.linalg.norm(view_vec)

    L = compute_luminance(compute_illuminance(grid_points, light_sources, normal_vector = (0, 0, 1)), reflection_factor=rho)
    L_arr = np.array(L)  # shape (N, 3)
    results = []

    for (x, y, z) in grid_points:
        lv_sum = 0
        for light in light_sources:
            dx = light.x - x
            dy = light.y - y
            dz = light.h - observer_eye_height
            vec_to_light = np.array([dx, dy, dz])
            distance = np.linalg.norm(vec_to_light)
            if distance == 0:
                continue

            vec_to_light_unit = vec_to_light / distance

            cos_q = np.dot(view_vec, vec_to_light_unit)
            # cos_q should be between -1 and 1, but due to bit precision errors, it can be slightly outside this range
            cos_q = np.clip(cos_q, -1, 1)
            # if the light is behind the field of view, skip it
            if cos_q < 0:
                continue
            q_i = np.degrees(np.arccos(cos_q)) # qi = the angle between the observer's line of sight and the direction of light.
            if q_i <= 1.5 or q_i >= 60: # skip if the angle is too small or too large
                # as explained in the CIE 112:1994 https://pdfcoffee.com/qdownload/cie-112-1994-2-pdf-free.html
                continue

            phi_math = np.degrees(np.arctan2(dy, dx)) % 360
            phi_relative = (phi_math - light.orientation) % 360
            # vertical_angles = light.photometric_data['vertical_angles']
            # horizontal_angles = light.photometric_data['horizontal_angles']
            # candela_matrix = light.photometric_data['candela_values']

            # h_idx = np.abs(horizontal_angles - phi_relative).argmin()

            # # Interpolate vertical profile
            # intensity_interp = interp1d(vertical_angles, candela_matrix[h_idx], kind='linear',
            #                             fill_value="extrapolate", bounds_error=False)
            # I_theta = intensity_interp(theta)

            # luminous intensity
            theta = q_i # it is the same as q_i in this case
            cos_theta = cos_q
            I_theta = light.photometric_data['candela_function'](phi_relative, theta)

            # Illuminance calculation
            E_eye = (I_theta / (distance ** 2)) * cos_theta
            lv_sum += E_eye / (q_i ** 2)
        # L_VL: Veiling luminance on the eye produced by the luminaires
        L_VL = 10 * lv_sum
        # L_VE: Veiling luminance on the observer's eye produced by the environment.
        # now we need to calculate L_av, choose the average of the near points from xy (assumeing 1.8 m)
        # L_av = np.mean([l[2] for l in L if (math.sqrt((l[0] - x)**2) < 1.8) and (math.sqrt((l[1] - y)**2) < 1.8)]) # This line is slow
        # Instead, do this with NumPy
        
        dx = L_arr[:, 0] - x
        dy = L_arr[:, 1] - y
        dist_sq = dx**2 + dy**2
        nearby = L_arr[dist_sq < (proximity ** 2)]
        L_av = np.mean(nearby[:, 2]) if len(nearby) > 0 else 0
        L_VE = (0.035 * L_av)
        # print(f"L_VL: {L_VL}, L_VE: {L_VE}")
        if L_VL == 0: # log (0) is undefined, and it means no glare definitely
            GR = 0
        else:
            GR = 24 + 27 * np.log10(L_VL/ (L_VE**0.9))
        # print(f"GR: {GR}")
        results.append((x, y, GR))
    return results

def compute_glare_rating_any(grid_points, observer_eye_height, light_sources, rho=0.2, proximity=1.8):
    """similar to compute_glare_rating but for any observer_view_vec (pick a direction pointing to most glaring light)"""
    results = [] # for storing glare results
    for point in grid_points:
        # Change in plan - Find the GR that is the most glaring light source
        x, y, z = point
        max_glare = 0
        for light in light_sources:
            dx = light.x - x
            dy = light.y - y
            dz = light.h - observer_eye_height
            observer_view_vec = (dx, dy, dz)
            gr_i = compute_glare_rating([point], observer_eye_height, observer_view_vec, light_sources, rho,proximity)[0][2]
            max_glare = max(max_glare, gr_i)

        # # Use the closest light source to determine the observer's view vector
        # if closest_light is not None:
        #     observer_view_vec = (closest_light.x - x, closest_light.y - y, closest_light.h - observer_eye_height)
        #     # Normalize the view vector
        #     norm = np.linalg.norm(observer_view_vec)
        #     if norm > 0:
        #         # it should be = (closest_light.x - x, closest_light.y - y, closest_light.h - observer_eye_height)
        #         observer_view_vec = observer_view_vec / norm
        #     # Call the compute_glare_rating function with the normalized view vector BUT JUST FOR THIS POINT
        #     glare_results = compute_glare_rating([point], observer_eye_height, observer_view_vec, light_sources, rho)
        #     glare_results = glare_results[0][2]  # Extract the glare rating for this point

        results.append((x, y, max_glare))
    return results

def extracts_min(results, surface_bounds):
    """
    Extracts the minimum [anything] for a given surface.
    """
    filtered_points = [p for p in results if
                       surface_bounds[0] <= p[0] <= surface_bounds[1] and
                       surface_bounds[2] <= p[1] <= surface_bounds[3]]

    if filtered_points:
        min_v = min([p[2] for p in filtered_points])
        return min_v
    return 0

def extracts_max(results, surface_bounds):
    """
    Extracts the maximun [anything] for a given surface.
    """
    filtered_points = [p for p in results if
                       surface_bounds[0] <= p[0] <= surface_bounds[1] and
                       surface_bounds[2] <= p[1] <= surface_bounds[3]]

    if filtered_points:
        max_v = max([p[2] for p in filtered_points])
        return max_v
    return 0

def extracts_avg(results, surface_bounds):
    """
    Extracts the average [anything] for a given surface.
    """
    filtered_points = [p for p in results if
                       surface_bounds[0] <= p[0] <= surface_bounds[1] and
                       surface_bounds[2] <= p[1] <= surface_bounds[3]]

    if filtered_points:
        val = np.mean([p[2] for p in filtered_points])
        return val
    return 0

def intersect_simulation(scenario, calculation_surfaces_withname, light_sources,resolution =20, showplot=0, saveplot=0):
    """
    Runs a single lighting simulation based on the given scenario parameters.
    """
    sim_name = scenario["scenario_name"].unique()[0]
    config = scenario["config"].unique()[0]
    offset_to_z_ratio = scenario["offset_to_z_ratio"].unique()[0]
    # strip calculation_surfaces withname to just the values like (x1, x2, y1, y2)
    calculation_surfaces = list(calculation_surfaces_withname.values())

    # Generate grid points covering all surfaces
    unified_grid = generate_unified_grid((resolution, resolution), calculation_surfaces)

    # Compute illuminance (H)
    print("...calculating horizontal illuminance")
    illuminance_results_H = compute_illuminance(unified_grid, light_sources, normal_vector = (0, 0, 1))

    # Compute illuminance (V)
    print("...calculating vertical illuminance")
    illuminance_results_V_N = compute_illuminance(unified_grid, light_sources, normal_vector = (0, 1, 0))
    illuminance_results_V_E = compute_illuminance(unified_grid, light_sources, normal_vector = (1, 0, 0))
    illuminance_results_V_S = compute_illuminance(unified_grid, light_sources, normal_vector = (0, -1, 0))
    illuminance_results_V_W = compute_illuminance(unified_grid, light_sources, normal_vector = (-1, 0, 0))
    illuminance_results_V_NW = compute_illuminance(unified_grid, light_sources, normal_vector = (-1, 1, 0))
    illuminance_results_V_NE = compute_illuminance(unified_grid, light_sources, normal_vector = (1, 1, 0))
    illuminance_results_V_SW = compute_illuminance(unified_grid, light_sources, normal_vector = (-1, -1, 0))
    illuminance_results_V_SE = compute_illuminance(unified_grid, light_sources, normal_vector = (1, -1, 0))

    # Computes semi-cylindrical illuminance
    print("...calculating semi-cylindrical illuminance")
    dir_octa = [-90, -45, 0, 45, 90]
    Esc_results_N = compute_illuminance_sc(unified_grid, light_sources, normal_vector=(0, 1, 0), height=1.5, directions = dir_octa)
    Esc_results_E = compute_illuminance_sc(unified_grid, light_sources, normal_vector=(1, 0, 0), height=1.5, directions = dir_octa)
    Esc_results_S = compute_illuminance_sc(unified_grid, light_sources, normal_vector=(0, -1, 0), height=1.5, directions = dir_octa)
    Esc_results_W = compute_illuminance_sc(unified_grid, light_sources, normal_vector=(-1, 0, 0), height=1.5, directions = dir_octa)
    Esc_results_NW = compute_illuminance_sc(unified_grid, light_sources, normal_vector=(-1, 1, 0), height=1.5, directions = dir_octa)
    Esc_results_NE = compute_illuminance_sc(unified_grid, light_sources, normal_vector=(1, 1, 0), height=1.5, directions = dir_octa)
    Esc_results_SW = compute_illuminance_sc(unified_grid, light_sources, normal_vector=(-1, -1, 0), height=1.5, directions = dir_octa)
    Esc_results_SE = compute_illuminance_sc(unified_grid, light_sources, normal_vector=(1, -1, 0), height=1.5, directions = dir_octa)

    # Compute luminance
    print("...calculating luminance")
    luminance_result = compute_luminance(illuminance_results_H, reflection_factor=0.20)

    # Compute glare rating
    print("...calculating glare rating")
    glare_anywhere = compute_glare_rating_any(unified_grid, observer_eye_height=1.5, light_sources=light_sources)
    glare_results_N = compute_glare_rating(unified_grid, observer_eye_height=1.5, observer_view_vec=(0, 1, 0), light_sources=light_sources)
    glare_results_E = compute_glare_rating(unified_grid, observer_eye_height=1.5, observer_view_vec=(1, 0, 0), light_sources=light_sources)
    glare_results_S = compute_glare_rating(unified_grid, observer_eye_height=1.5, observer_view_vec=(0, -1, 0), light_sources=light_sources)
    glare_results_W = compute_glare_rating(unified_grid, observer_eye_height=1.5, observer_view_vec=(-1, 0, 0), light_sources=light_sources)
    glare_results_NW = compute_glare_rating(unified_grid, observer_eye_height=1.5, observer_view_vec=(-1, 1, 0), light_sources=light_sources)
    glare_results_NE = compute_glare_rating(unified_grid, observer_eye_height=1.5, observer_view_vec=(1, 1, 0), light_sources=light_sources)
    glare_results_SW = compute_glare_rating(unified_grid, observer_eye_height=1.5, observer_view_vec=(-1, -1, 0), light_sources=light_sources)
    glare_results_SE = compute_glare_rating(unified_grid, observer_eye_height=1.5, observer_view_vec=(1, -1, 0), light_sources=light_sources)

    # Extraction: use this namming convention for all results
    # "NW_m	NE_m	SW_m	SE_m	NW 	N_i	N_o	NE 	E_i	E_o	SE	S_i	S_o	SW	W_i	W_o"
    # 16 small surfaces on 4 way intersection as the name suggests
    # and for each surface, we have min max average, so 16*3 = 48 at most for each parameter
    # Extract H Illuminance results
    E_h_NW_m_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["NW_m"])
    E_h_NE_m_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["NE_m"])
    E_h_SW_m_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["SW_m"])
    E_h_SE_m_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["SE_m"])
    E_h_NW_min = extracts_max(illuminance_results_H, calculation_surfaces_withname["NW"])
    E_h_N_i_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["N_i"])
    E_h_N_o_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["N_o"])
    E_h_NE_min = extracts_max(illuminance_results_H, calculation_surfaces_withname["NE"])
    E_h_E_i_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["E_i"])
    E_h_E_o_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["E_o"])
    E_h_SE_min = extracts_max(illuminance_results_H, calculation_surfaces_withname["SE"])
    E_h_S_i_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["S_i"])
    E_h_S_o_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["S_o"])
    E_h_SW_min = extracts_max(illuminance_results_H, calculation_surfaces_withname["SW"])
    E_h_W_i_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["W_i"])
    E_h_W_o_min = extracts_min(illuminance_results_H, calculation_surfaces_withname["W_o"])

    E_h_NW_m_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["NW_m"])
    E_h_NE_m_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["NE_m"])
    E_h_SW_m_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["SW_m"])
    E_h_SE_m_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["SE_m"])
    E_h_NW_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["NW"])
    E_h_N_i_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["N_i"])
    E_h_N_o_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["N_o"])
    E_h_NE_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["NE"])
    E_h_E_i_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["E_i"])
    E_h_E_o_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["E_o"])
    E_h_SE_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["SE"])
    E_h_S_i_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["S_i"])
    E_h_S_o_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["S_o"])
    E_h_SW_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["SW"])
    E_h_W_i_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["W_i"])
    E_h_W_o_avg = extracts_avg(illuminance_results_H, calculation_surfaces_withname["W_o"])

    E_h_NW_m_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["NW_m"])
    E_h_NE_m_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["NE_m"])
    E_h_SW_m_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["SW_m"])
    E_h_SE_m_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["SE_m"])
    E_h_NW_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["NW"])
    E_h_N_i_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["N_i"])
    E_h_N_o_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["N_o"])
    E_h_NE_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["NE"])
    E_h_E_i_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["E_i"])
    E_h_E_o_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["E_o"])
    E_h_SE_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["SE"])
    E_h_S_i_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["S_i"])
    E_h_S_o_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["S_o"])
    E_h_SW_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["SW"])
    E_h_W_i_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["W_i"])
    E_h_W_o_max = extracts_max(illuminance_results_H, calculation_surfaces_withname["W_o"])

    # Extract V Illuminance results
    # for E_v, each surface has unique directions (as I plan in the slide), be careful
    E_v_NW_m_min = None # in the middle of the interestion(No pedestrian)
    E_v_NE_m_min = None # in the middle of the interestion(No pedestrian)
    E_v_SW_m_min = None # in the middle of the interestion(No pedestrian)
    E_v_SE_m_min = None # in the middle of the interestion(No pedestrian)
    E_v_NW_min = extracts_max(illuminance_results_V_NE, calculation_surfaces_withname["NW"])
    E_v_N_i_min = extracts_min(illuminance_results_V_N, calculation_surfaces_withname["N_i"])
    E_v_N_o_min = extracts_min(illuminance_results_V_S, calculation_surfaces_withname["N_o"])
    E_v_NE_min = extracts_max(illuminance_results_V_SE, calculation_surfaces_withname["NE"])
    E_v_E_i_min = extracts_min(illuminance_results_V_E, calculation_surfaces_withname["E_i"])
    E_v_E_o_min = extracts_min(illuminance_results_V_W, calculation_surfaces_withname["E_o"])
    E_v_SE_min = extracts_max(illuminance_results_V_SW, calculation_surfaces_withname["SE"])
    E_v_S_i_min = extracts_min(illuminance_results_V_S, calculation_surfaces_withname["S_i"])
    E_v_S_o_min = extracts_min(illuminance_results_V_N, calculation_surfaces_withname["S_o"])
    E_v_SW_min = extracts_max(illuminance_results_V_NW, calculation_surfaces_withname["SW"])
    E_v_W_i_min = extracts_min(illuminance_results_V_W, calculation_surfaces_withname["W_i"])
    E_v_W_o_min = extracts_min(illuminance_results_V_E, calculation_surfaces_withname["W_o"])

    E_v_NW_m_avg = None # in the middle of the interestion(No pedestrian)
    E_v_NE_m_avg = None # in the middle of the interestion(No pedestrian)
    E_v_SW_m_avg = None # in the middle of the interestion(No pedestrian)
    E_v_SE_m_avg = None # in the middle of the interestion(No pedestrian)
    E_v_NW_avg = extracts_avg(illuminance_results_V_NE, calculation_surfaces_withname["NW"])
    E_v_N_i_avg = extracts_avg(illuminance_results_V_N, calculation_surfaces_withname["N_i"])
    E_v_N_o_avg = extracts_avg(illuminance_results_V_S, calculation_surfaces_withname["N_o"])
    E_v_NE_avg = extracts_avg(illuminance_results_V_SE, calculation_surfaces_withname["NE"])
    E_v_E_i_avg = extracts_avg(illuminance_results_V_E, calculation_surfaces_withname["E_i"])
    E_v_E_o_avg = extracts_avg(illuminance_results_V_W, calculation_surfaces_withname["E_o"])
    E_v_SE_avg = extracts_avg(illuminance_results_V_SW, calculation_surfaces_withname["SE"])
    E_v_S_i_avg = extracts_avg(illuminance_results_V_S, calculation_surfaces_withname["S_i"])
    E_v_S_o_avg = extracts_avg(illuminance_results_V_N, calculation_surfaces_withname["S_o"])
    E_v_SW_avg = extracts_avg(illuminance_results_V_NW, calculation_surfaces_withname["SW"])
    E_v_W_i_avg = extracts_avg(illuminance_results_V_W, calculation_surfaces_withname["W_i"])
    E_v_W_o_avg = extracts_avg(illuminance_results_V_E, calculation_surfaces_withname["W_o"])

    E_v_NW_m_max = None # in the middle of the interestion(No pedestrian)
    E_v_NE_m_max = None # in the middle of the interestion(No pedestrian)
    E_v_SW_m_max = None # in the middle of the interestion(No pedestrian)
    E_v_SE_m_max = None # in the middle of the interestion(No pedestrian)
    E_v_NW_max = extracts_max(illuminance_results_V_NE, calculation_surfaces_withname["NW"])
    E_v_N_i_max = extracts_max(illuminance_results_V_N, calculation_surfaces_withname["N_i"])
    E_v_N_o_max = extracts_max(illuminance_results_V_S, calculation_surfaces_withname["N_o"])
    E_v_NE_max = extracts_max(illuminance_results_V_SE, calculation_surfaces_withname["NE"])
    E_v_E_i_max = extracts_max(illuminance_results_V_E, calculation_surfaces_withname["E_i"])
    E_v_E_o_max = extracts_max(illuminance_results_V_W, calculation_surfaces_withname["E_o"])
    E_v_SE_max = extracts_max(illuminance_results_V_SW, calculation_surfaces_withname["SE"])
    E_v_S_i_max = extracts_max(illuminance_results_V_S, calculation_surfaces_withname["S_i"])
    E_v_S_o_max = extracts_max(illuminance_results_V_N, calculation_surfaces_withname["S_o"])
    E_v_SW_max = extracts_max(illuminance_results_V_NW, calculation_surfaces_withname["SW"])
    E_v_W_i_max = extracts_max(illuminance_results_V_W, calculation_surfaces_withname["W_i"])
    E_v_W_o_max = extracts_max(illuminance_results_V_E, calculation_surfaces_withname["W_o"])

    # Extract semi-cylindrical
    # This is similar to the V illuminance
    E_sc_NW_m_min = None # in the middle of the interestion(No pedestrian)
    E_sc_NE_m_min = None # in the middle of the interestion(No pedestrian)
    E_sc_SW_m_min = None # in the middle of the interestion(No pedestrian)
    E_sc_SE_m_min = None # in the middle of the interestion(No pedestrian)
    E_sc_NW_min = extracts_max(Esc_results_NE, calculation_surfaces_withname["NW"])
    E_sc_N_i_min = extracts_min(Esc_results_N, calculation_surfaces_withname["N_i"])
    E_sc_N_o_min = extracts_min(Esc_results_S, calculation_surfaces_withname["N_o"])
    E_sc_NE_min = extracts_max(Esc_results_SE, calculation_surfaces_withname["NE"])
    E_sc_E_i_min = extracts_min(Esc_results_E, calculation_surfaces_withname["E_i"])
    E_sc_E_o_min = extracts_min(Esc_results_W, calculation_surfaces_withname["E_o"])
    E_sc_SE_min = extracts_max(Esc_results_SW, calculation_surfaces_withname["SE"])
    E_sc_S_i_min = extracts_min(Esc_results_S, calculation_surfaces_withname["S_i"])
    E_sc_S_o_min = extracts_min(Esc_results_N, calculation_surfaces_withname["S_o"])
    E_sc_SW_min = extracts_max(Esc_results_NW, calculation_surfaces_withname["SW"])
    E_sc_W_i_min = extracts_min(Esc_results_W, calculation_surfaces_withname["W_i"])
    E_sc_W_o_min = extracts_min(Esc_results_E, calculation_surfaces_withname["W_o"])

    E_sc_NW_m_avg = None # in the middle of the interestion(No pedestrian)
    E_sc_NE_m_avg = None # in the middle of the interestion(No pedestrian)
    E_sc_SW_m_avg = None # in the middle of the interestion(No pedestrian)
    E_sc_SE_m_avg = None # in the middle of the interestion(No pedestrian)
    E_sc_NW_avg = extracts_avg(Esc_results_NE, calculation_surfaces_withname["NW"])
    E_sc_N_i_avg = extracts_avg(Esc_results_N, calculation_surfaces_withname["N_i"])
    E_sc_N_o_avg = extracts_avg(Esc_results_S, calculation_surfaces_withname["N_o"])
    E_sc_NE_avg = extracts_avg(Esc_results_SE, calculation_surfaces_withname["NE"])
    E_sc_E_i_avg = extracts_avg(Esc_results_E, calculation_surfaces_withname["E_i"])
    E_sc_E_o_avg = extracts_avg(Esc_results_W, calculation_surfaces_withname["E_o"])
    E_sc_SE_avg = extracts_avg(Esc_results_SW, calculation_surfaces_withname["SE"])
    E_sc_S_i_avg = extracts_avg(Esc_results_S, calculation_surfaces_withname["S_i"])
    E_sc_S_o_avg = extracts_avg(Esc_results_N, calculation_surfaces_withname["S_o"])
    E_sc_SW_avg = extracts_avg(Esc_results_NW, calculation_surfaces_withname["SW"])
    E_sc_W_i_avg = extracts_avg(Esc_results_W, calculation_surfaces_withname["W_i"])
    E_sc_W_o_avg = extracts_avg(Esc_results_E, calculation_surfaces_withname["W_o"])

    E_sc_NW_m_max = None # in the middle of the interestion(No pedestrian)
    E_sc_NE_m_max = None # in the middle of the interestion(No pedestrian)
    E_sc_SW_m_max = None # in the middle of the interestion(No pedestrian)
    E_sc_SE_m_max = None # in the middle of the interestion(No pedestrian)
    E_sc_NW_max = extracts_max(Esc_results_NE, calculation_surfaces_withname["NW"])
    E_sc_N_i_max = extracts_max(Esc_results_N, calculation_surfaces_withname["N_i"])
    E_sc_N_o_max = extracts_max(Esc_results_S, calculation_surfaces_withname["N_o"])
    E_sc_NE_max = extracts_max(Esc_results_SE, calculation_surfaces_withname["NE"])
    E_sc_E_i_max = extracts_max(Esc_results_E, calculation_surfaces_withname["E_i"])
    E_sc_E_o_max = extracts_max(Esc_results_W, calculation_surfaces_withname["E_o"])
    E_sc_SE_max = extracts_max(Esc_results_SW, calculation_surfaces_withname["SE"])
    E_sc_S_i_max = extracts_max(Esc_results_S, calculation_surfaces_withname["S_i"])
    E_sc_S_o_max = extracts_max(Esc_results_N, calculation_surfaces_withname["S_o"])
    E_sc_SW_max = extracts_max(Esc_results_NW, calculation_surfaces_withname["SW"])
    E_sc_W_i_max = extracts_max(Esc_results_W, calculation_surfaces_withname["W_i"])
    E_sc_W_o_max = extracts_max(Esc_results_E, calculation_surfaces_withname["W_o"])

    # Extract luminance
    L_NW_m_min = extracts_min(luminance_result, calculation_surfaces_withname["NW_m"])
    L_NE_m_min = extracts_min(luminance_result, calculation_surfaces_withname["NE_m"])
    L_SW_m_min = extracts_min(luminance_result, calculation_surfaces_withname["SW_m"])
    L_SE_m_min = extracts_min(luminance_result, calculation_surfaces_withname["SE_m"])
    L_NW_min = extracts_max(luminance_result, calculation_surfaces_withname["NW"])
    L_N_i_min = extracts_min(luminance_result, calculation_surfaces_withname["N_i"])
    L_N_o_min = extracts_min(luminance_result, calculation_surfaces_withname["N_o"])
    L_NE_min = extracts_max(luminance_result, calculation_surfaces_withname["NE"])
    L_E_i_min = extracts_min(luminance_result, calculation_surfaces_withname["E_i"])
    L_E_o_min = extracts_min(luminance_result, calculation_surfaces_withname["E_o"])
    L_SE_min = extracts_max(luminance_result, calculation_surfaces_withname["SE"])
    L_S_i_min = extracts_min(luminance_result, calculation_surfaces_withname["S_i"])
    L_S_o_min = extracts_min(luminance_result, calculation_surfaces_withname["S_o"])
    L_SW_min = extracts_max(luminance_result, calculation_surfaces_withname["SW"])
    L_W_i_min = extracts_min(luminance_result, calculation_surfaces_withname["W_i"])
    L_W_o_min = extracts_min(luminance_result, calculation_surfaces_withname["W_o"])
    L_NW_m_avg = extracts_avg(luminance_result, calculation_surfaces_withname["NW_m"])
    L_NE_m_avg = extracts_avg(luminance_result, calculation_surfaces_withname["NE_m"])
    L_SW_m_avg = extracts_avg(luminance_result, calculation_surfaces_withname["SW_m"])
    L_SE_m_avg = extracts_avg(luminance_result, calculation_surfaces_withname["SE_m"])
    L_NW_avg = extracts_avg(luminance_result, calculation_surfaces_withname["NW"])
    L_N_i_avg = extracts_avg(luminance_result, calculation_surfaces_withname["N_i"])
    L_N_o_avg = extracts_avg(luminance_result, calculation_surfaces_withname["N_o"])
    L_NE_avg = extracts_avg(luminance_result, calculation_surfaces_withname["NE"])
    L_E_i_avg = extracts_avg(luminance_result, calculation_surfaces_withname["E_i"])
    L_E_o_avg = extracts_avg(luminance_result, calculation_surfaces_withname["E_o"])
    L_SE_avg = extracts_avg(luminance_result, calculation_surfaces_withname["SE"])
    L_S_i_avg = extracts_avg(luminance_result, calculation_surfaces_withname["S_i"])
    L_S_o_avg = extracts_avg(luminance_result, calculation_surfaces_withname["S_o"])
    L_SW_avg = extracts_avg(luminance_result, calculation_surfaces_withname["SW"])
    L_W_i_avg = extracts_avg(luminance_result, calculation_surfaces_withname["W_i"])
    L_W_o_avg = extracts_avg(luminance_result, calculation_surfaces_withname["W_o"])
    L_NW_m_max = extracts_max(luminance_result, calculation_surfaces_withname["NW_m"])
    L_NE_m_max = extracts_max(luminance_result, calculation_surfaces_withname["NE_m"])
    L_SW_m_max = extracts_max(luminance_result, calculation_surfaces_withname["SW_m"])
    L_SE_m_max = extracts_max(luminance_result, calculation_surfaces_withname["SE_m"])
    L_NW_max = extracts_max(luminance_result, calculation_surfaces_withname["NW"])
    L_N_i_max = extracts_max(luminance_result, calculation_surfaces_withname["N_i"])
    L_N_o_max = extracts_max(luminance_result, calculation_surfaces_withname["N_o"])
    L_NE_max = extracts_max(luminance_result, calculation_surfaces_withname["NE"])
    L_E_i_max = extracts_max(luminance_result, calculation_surfaces_withname["E_i"])
    L_E_o_max = extracts_max(luminance_result, calculation_surfaces_withname["E_o"])
    L_SE_max = extracts_max(luminance_result, calculation_surfaces_withname["SE"])
    L_S_i_max = extracts_max(luminance_result, calculation_surfaces_withname["S_i"])
    L_S_o_max = extracts_max(luminance_result, calculation_surfaces_withname["S_o"])
    L_SW_max = extracts_max(luminance_result, calculation_surfaces_withname["SW"])
    L_W_i_max = extracts_max(luminance_result, calculation_surfaces_withname["W_i"])
    L_W_o_max = extracts_max(luminance_result, calculation_surfaces_withname["W_o"])

    # Extract glare rating: For this we have 2 cases, for drivers'eye and pedestrians' eye
    # for drivers' eye, we dont need at the conrner, so no NW, NE, SW, SE
    GR_NW_m_min = extracts_min(glare_results_SW, calculation_surfaces_withname["NW_m"])
    GR_NE_m_min = extracts_min(glare_results_NW, calculation_surfaces_withname["NE_m"])
    GR_SW_m_min = extracts_min(glare_results_SE, calculation_surfaces_withname["SW_m"])
    GR_SE_m_min = extracts_min(glare_results_NE, calculation_surfaces_withname["SE_m"])
    GR_NW_min = None
    GR_N_i_min = extracts_min(glare_results_S, calculation_surfaces_withname["N_i"])
    GR_N_o_min = extracts_min(glare_results_N, calculation_surfaces_withname["N_o"])
    GR_NE_min = None
    GR_E_i_min = extracts_min(glare_results_W, calculation_surfaces_withname["E_i"])
    GR_E_o_min = extracts_min(glare_results_E, calculation_surfaces_withname["E_o"])
    GR_SE_min = None
    GR_S_i_min = extracts_min(glare_results_N, calculation_surfaces_withname["S_i"])
    GR_S_o_min = extracts_min(glare_results_S, calculation_surfaces_withname["S_o"])
    GR_SW_min = None
    GR_W_i_min = extracts_min(glare_results_E, calculation_surfaces_withname["W_i"])
    GR_W_o_min = extracts_min(glare_results_W, calculation_surfaces_withname["W_o"])

    GR_NW_m_avg = extracts_avg(glare_results_SW, calculation_surfaces_withname["NW_m"])
    GR_NE_m_avg = extracts_avg(glare_results_NW, calculation_surfaces_withname["NE_m"])
    GR_SW_m_avg = extracts_avg(glare_results_SE, calculation_surfaces_withname["SW_m"])
    GR_SE_m_avg = extracts_avg(glare_results_NE, calculation_surfaces_withname["SE_m"])
    GR_NW_avg = None
    GR_N_i_avg = extracts_avg(glare_results_S, calculation_surfaces_withname["N_i"])
    GR_N_o_avg = extracts_avg(glare_results_N, calculation_surfaces_withname["N_o"])
    GR_NE_avg = None
    GR_E_i_avg = extracts_avg(glare_results_W, calculation_surfaces_withname["E_i"])
    GR_E_o_avg = extracts_avg(glare_results_E, calculation_surfaces_withname["E_o"])
    GR_SE_avg = None
    GR_S_i_avg = extracts_avg(glare_results_N, calculation_surfaces_withname["S_i"])
    GR_S_o_avg = extracts_avg(glare_results_S, calculation_surfaces_withname["S_o"])
    GR_SW_avg = None
    GR_W_i_avg = extracts_avg(glare_results_E, calculation_surfaces_withname["W_i"])
    GR_W_o_avg = extracts_avg(glare_results_W, calculation_surfaces_withname["W_o"])

    GR_NW_m_max = extracts_max(glare_results_SW, calculation_surfaces_withname["NW_m"])
    GR_NE_m_max = extracts_max(glare_results_NW, calculation_surfaces_withname["NE_m"])
    GR_SW_m_max = extracts_max(glare_results_SE, calculation_surfaces_withname["SW_m"])
    GR_SE_m_max = extracts_max(glare_results_NE, calculation_surfaces_withname["SE_m"])
    GR_NW_max = None
    GR_N_i_max = extracts_max(glare_results_S, calculation_surfaces_withname["N_i"])
    GR_N_o_max = extracts_max(glare_results_N, calculation_surfaces_withname["N_o"])
    GR_NE_max = None
    GR_E_i_max = extracts_max(glare_results_W, calculation_surfaces_withname["E_i"])
    GR_E_o_max = extracts_max(glare_results_E, calculation_surfaces_withname["E_o"])
    GR_SE_max = None
    GR_S_i_max = extracts_max(glare_results_N, calculation_surfaces_withname["S_i"])
    GR_S_o_max = extracts_max(glare_results_S, calculation_surfaces_withname["S_o"])
    GR_SW_max = None
    GR_W_i_max = extracts_max(glare_results_E, calculation_surfaces_withname["W_i"])
    GR_W_o_max = extracts_max(glare_results_W, calculation_surfaces_withname["W_o"])

    # Extract glare rating: For pedestrians' eye, we ignore the midle of the intersection NW_m, NE_m, SW_m, SE_m
    GR_ped_NW_m_min = None
    GR_ped_NE_m_min = None
    GR_ped_SW_m_min = None
    GR_ped_SE_m_min = None
    GR_ped_NW_min = extracts_min(glare_results_SE, calculation_surfaces_withname["NW"])
    GR_ped_N_i_min = extracts_min(glare_results_E, calculation_surfaces_withname["N_i"])
    GR_ped_N_o_min = extracts_min(glare_results_W, calculation_surfaces_withname["N_o"])
    GR_ped_NE_min = extracts_min(glare_results_SW, calculation_surfaces_withname["NE"])
    GR_ped_E_i_min = extracts_min(glare_results_S, calculation_surfaces_withname["E_i"])
    GR_ped_E_o_min = extracts_min(glare_results_N, calculation_surfaces_withname["E_o"])
    GR_ped_SE_min = extracts_min(glare_results_NW, calculation_surfaces_withname["SE"])
    GR_ped_S_i_min = extracts_min(glare_results_W, calculation_surfaces_withname["S_i"])
    GR_ped_S_o_min = extracts_min(glare_results_E, calculation_surfaces_withname["S_o"])
    GR_ped_SW_min = extracts_min(glare_results_NE, calculation_surfaces_withname["SW"])
    GR_ped_W_i_min = extracts_min(glare_results_N, calculation_surfaces_withname["W_i"])
    GR_ped_W_o_min = extracts_min(glare_results_S, calculation_surfaces_withname["W_o"])
    GR_ped_NW_m_avg = None
    GR_ped_NE_m_avg = None
    GR_ped_SW_m_avg = None
    GR_ped_SE_m_avg = None
    GR_ped_NW_avg = extracts_avg(glare_results_SE, calculation_surfaces_withname["NW"])
    GR_ped_N_i_avg = extracts_avg(glare_results_E, calculation_surfaces_withname["N_i"])
    GR_ped_N_o_avg = extracts_avg(glare_results_W, calculation_surfaces_withname["N_o"])
    GR_ped_NE_avg = extracts_avg(glare_results_SW, calculation_surfaces_withname["NE"])
    GR_ped_E_i_avg = extracts_avg(glare_results_S, calculation_surfaces_withname["E_i"])
    GR_ped_E_o_avg = extracts_avg(glare_results_N, calculation_surfaces_withname["E_o"])
    GR_ped_SE_avg = extracts_avg(glare_results_NW, calculation_surfaces_withname["SE"])
    GR_ped_S_i_avg = extracts_avg(glare_results_W, calculation_surfaces_withname["S_i"])
    GR_ped_S_o_avg = extracts_avg(glare_results_E, calculation_surfaces_withname["S_o"])
    GR_ped_SW_avg = extracts_avg(glare_results_NE, calculation_surfaces_withname["SW"])
    GR_ped_W_i_avg = extracts_avg(glare_results_N, calculation_surfaces_withname["W_i"])
    GR_ped_W_o_avg = extracts_avg(glare_results_S, calculation_surfaces_withname["W_o"])
    GR_ped_NW_m_max = None
    GR_ped_NE_m_max = None
    GR_ped_SW_m_max = None
    GR_ped_SE_m_max = None
    GR_ped_NW_max = extracts_max(glare_results_SE, calculation_surfaces_withname["NW"])
    GR_ped_N_i_max = extracts_max(glare_results_E, calculation_surfaces_withname["N_i"])
    GR_ped_N_o_max = extracts_max(glare_results_W, calculation_surfaces_withname["N_o"])
    GR_ped_NE_max = extracts_max(glare_results_SW, calculation_surfaces_withname["NE"])
    GR_ped_E_i_max = extracts_max(glare_results_S, calculation_surfaces_withname["E_i"])
    GR_ped_E_o_max = extracts_max(glare_results_N, calculation_surfaces_withname["E_o"])
    GR_ped_SE_max = extracts_max(glare_results_NW, calculation_surfaces_withname["SE"])
    GR_ped_S_i_max = extracts_max(glare_results_W, calculation_surfaces_withname["S_i"])
    GR_ped_S_o_max = extracts_max(glare_results_E, calculation_surfaces_withname["S_o"])
    GR_ped_SW_max = extracts_max(glare_results_NE, calculation_surfaces_withname["SW"])
    GR_ped_W_i_max = extracts_max(glare_results_N, calculation_surfaces_withname["W_i"])
    GR_ped_W_o_max = extracts_max(glare_results_S, calculation_surfaces_withname["W_o"])

    # Use the first light source as reference for logging in that scenario
    first_source = light_sources[0]

    # Prepare log entry
    meta = first_source.photometric_data["metadata"]
    log_entry = [
        sim_name,
        meta.get("source_brand", "Unknown"),
        meta.get("source_model", "Unknown"),
        meta.get("IES_type", "Unknown"),
        meta.get("wattage", 0),
        meta.get("lumen", 0),
        first_source.h_p,  # pole_height
        config, offset_to_z_ratio,
        E_h_NW_m_min, E_h_NE_m_min, E_h_SW_m_min, E_h_SE_m_min,
        E_h_NW_min, E_h_N_i_min, E_h_N_o_min, E_h_NE_min, E_h_E_i_min, E_h_E_o_min,
        E_h_SE_min, E_h_S_i_min, E_h_S_o_min, E_h_SW_min, E_h_W_i_min, E_h_W_o_min,
        E_h_NW_m_avg, E_h_NE_m_avg, E_h_SW_m_avg, E_h_SE_m_avg,
        E_h_NW_avg, E_h_N_i_avg, E_h_N_o_avg, E_h_NE_avg, E_h_E_i_avg, E_h_E_o_avg,
        E_h_SE_avg, E_h_S_i_avg, E_h_S_o_avg, E_h_SW_avg, E_h_W_i_avg, E_h_W_o_avg,
        E_h_NW_m_max, E_h_NE_m_max, E_h_SW_m_max, E_h_SE_m_max,
        E_h_NW_max, E_h_N_i_max, E_h_N_o_max, E_h_NE_max, E_h_E_i_max, E_h_E_o_max,
        E_h_SE_max, E_h_S_i_max, E_h_S_o_max, E_h_SW_max, E_h_W_i_max, E_h_W_o_max,
        E_v_NW_m_min, E_v_NE_m_min, E_v_SW_m_min, E_v_SE_m_min,
        E_v_NW_min, E_v_N_i_min, E_v_N_o_min, E_v_NE_min, E_v_E_i_min, E_v_E_o_min,
        E_v_SE_min, E_v_S_i_min, E_v_S_o_min, E_v_SW_min, E_v_W_i_min, E_v_W_o_min,
        E_v_NW_m_avg, E_v_NE_m_avg, E_v_SW_m_avg, E_v_SE_m_avg,
        E_v_NW_avg, E_v_N_i_avg, E_v_N_o_avg, E_v_NE_avg, E_v_E_i_avg, E_v_E_o_avg,
        E_v_SE_avg, E_v_S_i_avg, E_v_S_o_avg, E_v_SW_avg, E_v_W_i_avg, E_v_W_o_avg,
        E_v_NW_m_max, E_v_NE_m_max, E_v_SW_m_max, E_v_SE_m_max,
        E_v_NW_max, E_v_N_i_max, E_v_N_o_max, E_v_NE_max, E_v_E_i_max, E_v_E_o_max,
        E_v_SE_max, E_v_S_i_max, E_v_S_o_max, E_v_SW_max, E_v_W_i_max, E_v_W_o_max,
        E_sc_NW_m_min, E_sc_NE_m_min, E_sc_SW_m_min, E_sc_SE_m_min,
        E_sc_NW_min, E_sc_N_i_min, E_sc_N_o_min, E_sc_NE_min, E_sc_E_i_min, E_sc_E_o_min,
        E_sc_SE_min, E_sc_S_i_min, E_sc_S_o_min, E_sc_SW_min, E_sc_W_i_min, E_sc_W_o_min,
        E_sc_NW_m_avg, E_sc_NE_m_avg, E_sc_SW_m_avg, E_sc_SE_m_avg,
        E_sc_NW_avg, E_sc_N_i_avg, E_sc_N_o_avg, E_sc_NE_avg, E_sc_E_i_avg, E_sc_E_o_avg,
        E_sc_SE_avg, E_sc_S_i_avg, E_sc_S_o_avg, E_sc_SW_avg, E_sc_W_i_avg, E_sc_W_o_avg,
        E_sc_NW_m_max, E_sc_NE_m_max, E_sc_SW_m_max, E_sc_SE_m_max,
        E_sc_NW_max, E_sc_N_i_max, E_sc_N_o_max, E_sc_NE_max, E_sc_E_i_max, E_sc_E_o_max,
        E_sc_SE_max, E_sc_S_i_max, E_sc_S_o_max, E_sc_SW_max, E_sc_W_i_max, E_sc_W_o_max,
        L_NW_m_min, L_NE_m_min, L_SW_m_min, L_SE_m_min,
        L_NW_min, L_N_i_min, L_N_o_min, L_NE_min, L_E_i_min, L_E_o_min,
        L_SE_min, L_S_i_min, L_S_o_min, L_SW_min, L_W_i_min, L_W_o_min,
        L_NW_m_avg, L_NE_m_avg, L_SW_m_avg, L_SE_m_avg,
        L_NW_avg, L_N_i_avg, L_N_o_avg, L_NE_avg, L_E_i_avg, L_E_o_avg,
        L_SE_avg, L_S_i_avg, L_S_o_avg, L_SW_avg, L_W_i_avg, L_W_o_avg,
        L_NW_m_max, L_NE_m_max, L_SW_m_max, L_SE_m_max,
        L_NW_max, L_N_i_max, L_N_o_max, L_NE_max, L_E_i_max, L_E_o_max,
        L_SE_max, L_S_i_max, L_S_o_max, L_SW_max, L_W_i_max, L_W_o_max,
        GR_NW_m_min, GR_NE_m_min, GR_SW_m_min, GR_SE_m_min,
        GR_NW_min, GR_N_i_min, GR_N_o_min, GR_NE_min, GR_E_i_min, GR_E_o_min,
        GR_SE_min, GR_S_i_min, GR_S_o_min, GR_SW_min, GR_W_i_min, GR_W_o_min,
        GR_NW_m_avg, GR_NE_m_avg, GR_SW_m_avg, GR_SE_m_avg,
        GR_NW_avg, GR_N_i_avg, GR_N_o_avg, GR_NE_avg, GR_E_i_avg, GR_E_o_avg,
        GR_SE_avg, GR_S_i_avg, GR_S_o_avg, GR_SW_avg, GR_W_i_avg, GR_W_o_avg,
        GR_NW_m_max, GR_NE_m_max, GR_SW_m_max, GR_SE_m_max,
        GR_NW_max, GR_N_i_max, GR_N_o_max, GR_NE_max, GR_E_i_max, GR_E_o_max,
        GR_SE_max, GR_S_i_max, GR_S_o_max, GR_SW_max, GR_W_i_max, GR_W_o_max,
        GR_ped_NW_m_min, GR_ped_NE_m_min, GR_ped_SW_m_min, GR_ped_SE_m_min,
        GR_ped_NW_min, GR_ped_N_i_min, GR_ped_N_o_min, GR_ped_NE_min, GR_ped_E_i_min, GR_ped_E_o_min,
        GR_ped_SE_min, GR_ped_S_i_min, GR_ped_S_o_min, GR_ped_SW_min, GR_ped_W_i_min, GR_ped_W_o_min,
        GR_ped_NW_m_avg, GR_ped_NE_m_avg, GR_ped_SW_m_avg, GR_ped_SE_m_avg,
        GR_ped_NW_avg, GR_ped_N_i_avg, GR_ped_N_o_avg, GR_ped_NE_avg, GR_ped_E_i_avg, GR_ped_E_o_avg,
        GR_ped_SE_avg, GR_ped_S_i_avg, GR_ped_S_o_avg, GR_ped_SW_avg, GR_ped_W_i_avg, GR_ped_W_o_avg,
        GR_ped_NW_m_max, GR_ped_NE_m_max, GR_ped_SW_m_max, GR_ped_SE_m_max,
        GR_ped_NW_max, GR_ped_N_i_max, GR_ped_N_o_max, GR_ped_NE_max, GR_ped_E_i_max, GR_ped_E_o_max,
        GR_ped_SE_max, GR_ped_S_i_max, GR_ped_S_o_max, GR_ped_SW_max, GR_ped_W_i_max, GR_ped_W_o_max,
    ]
    log_header = [
        "scenario_name", "source_brand", "source_model", "IES_type", "wattage", "lumen", "pole_height", "config",
        "offset_to_z_ratio",
        "E_h_NW_m_min", "E_h_NE_m_min", "E_h_SW_m_min", "E_h_SE_m_min",
        "E_h_NW_min", "E_h_N_i_min", "E_h_N_o_min", "E_h_NE_min", "E_h_E_i_min", "E_h_E_o_min",
        "E_h_SE_min", "E_h_S_i_min", "E_h_S_o_min", "E_h_SW_min", "E_h_W_i_min", "E_h_W_o_min",
        "E_h_NW_m_avg", "E_h_NE_m_avg", "E_h_SW_m_avg", "E_h_SE_m_avg",
        "E_h_NW_avg", "E_h_N_i_avg", "E_h_N_o_avg", "E_h_NE_avg", "E_h_E_i_avg", "E_h_E_o_avg",
        "E_h_SE_avg", "E_h_S_i_avg", "E_h_S_o_avg", "E_h_SW_avg", "E_h_W_i_avg", "E_h_W_o_avg",
        "E_h_NW_m_max", "E_h_NE_m_max", "E_h_SW_m_max", "E_h_SE_m_max",
        "E_h_NW_max", "E_h_N_i_max", "E_h_N_o_max", "E_h_NE_max", "E_h_E_i_max", "E_h_E_o_max",
        "E_h_SE_max", "E_h_S_i_max", "E_h_S_o_max", "E_h_SW_max", "E_h_W_i_max", "E_h_W_o_max",
        "E_v_NW_m_min", "E_v_NE_m_min", "E_v_SW_m_min", "E_v_SE_m_min",
        "E_v_NW_min", "E_v_N_i_min", "E_v_N_o_min", "E_v_NE_min", "E_v_E_i_min", "E_v_E_o_min",
        "E_v_SE_min", "E_v_S_i_min", "E_v_S_o_min", "E_v_SW_min", "E_v_W_i_min", "E_v_W_o_min",
        "E_v_NW_m_avg", "E_v_NE_m_avg", "E_v_SW_m_avg", "E_v_SE_m_avg",
        "E_v_NW_avg", "E_v_N_i_avg", "E_v_N_o_avg", "E_v_NE_avg", "E_v_E_i_avg", "E_v_E_o_avg",
        "E_v_SE_avg", "E_v_S_i_avg", "E_v_S_o_avg", "E_v_SW_avg", "E_v_W_i_avg", "E_v_W_o_avg",
        "E_v_NW_m_max", "E_v_NE_m_max", "E_v_SW_m_max", "E_v_SE_m_max",
        "E_v_NW_max", "E_v_N_i_max", "E_v_N_o_max", "E_v_NE_max", "E_v_E_i_max", "E_v_E_o_max",
        "E_v_SE_max", "E_v_S_i_max", "E_v_S_o_max", "E_v_SW_max", "E_v_W_i_max", "E_v_W_o_max",
        "E_sc_NW_m_min", "E_sc_NE_m_min", "E_sc_SW_m_min", "E_sc_SE_m_min",
        "E_sc_NW_min", "E_sc_N_i_min", "E_sc_N_o_min", "E_sc_NE_min", "E_sc_E_i_min", "E_sc_E_o_min",
        "E_sc_SE_min", "E_sc_S_i_min", "E_sc_S_o_min", "E_sc_SW_min", "E_sc_W_i_min", "E_sc_W_o_min",
        "E_sc_NW_m_avg", "E_sc_NE_m_avg", "E_sc_SW_m_avg", "E_sc_SE_m_avg",
        "E_sc_NW_avg", "E_sc_N_i_avg", "E_sc_N_o_avg", "E_sc_NE_avg", "E_sc_E_i_avg", "E_sc_E_o_avg",
        "E_sc_SE_avg", "E_sc_S_i_avg", "E_sc_S_o_avg", "E_sc_SW_avg", "E_sc_W_i_avg", "E_sc_W_o_avg",
        "E_sc_NW_m_max", "E_sc_NE_m_max", "E_sc_SW_m_max", "E_sc_SE_m_max",
        "E_sc_NW_max", "E_sc_N_i_max", "E_sc_N_o_max", "E_sc_NE_max", "E_sc_E_i_max", "E_sc_E_o_max",
        "E_sc_SE_max", "E_sc_S_i_max", "E_sc_S_o_max", "E_sc_SW_max", "E_sc_W_i_max", "E_sc_W_o_max",
        "L_NW_m_min", "L_NE_m_min", "L_SW_m_min", "L_SE_m_min",
        "L_NW_min", "L_N_i_min", "L_N_o_min", "L_NE_min", "L_E_i_min", "L_E_o_min",
        "L_SE_min", "L_S_i_min", "L_S_o_min", "L_SW_min", "L_W_i_min", "L_W_o_min",
        "L_NW_m_avg", "L_NE_m_avg", "L_SW_m_avg", "L_SE_m_avg",
        "L_NW_avg", "L_N_i_avg", "L_N_o_avg", "L_NE_avg", "L_E_i_avg", "L_E_o_avg",
        "L_SE_avg", "L_S_i_avg", "L_S_o_avg", "L_SW_avg", "L_W_i_avg", "L_W_o_avg",
        "L_NW_m_max", "L_NE_m_max", "L_SW_m_max", "L_SE_m_max",
        "L_NW_max", "L_N_i_max", "L_N_o_max", "L_NE_max", "L_E_i_max", "L_E_o_max",
        "L_SE_max", "L_S_i_max", "L_S_o_max", "L_SW_max", "L_W_i_max", "L_W_o_max",
        "GR_NW_m_min", "GR_NE_m_min", "GR_SW_m_min", "GR_SE_m_min",
        "GR_NW_min", "GR_N_i_min", "GR_N_o_min", "GR_NE_min", "GR_E_i_min", "GR_E_o_min",
        "GR_SE_min", "GR_S_i_min", "GR_S_o_min", "GR_SW_min", "GR_W_i_min", "GR_W_o_min",
        "GR_NW_m_avg", "GR_NE_m_avg", "GR_SW_m_avg", "GR_SE_m_avg",
        "GR_NW_avg", "GR_N_i_avg", "GR_N_o_avg", "GR_NE_avg", "GR_E_i_avg", "GR_E_o_avg",
        "GR_SE_avg", "GR_S_i_avg", "GR_S_o_avg", "GR_SW_avg", "GR_W_i_avg", "GR_W_o_avg",
        "GR_NW_m_max", "GR_NE_m_max", "GR_SW_m_max", "GR_SE_m_max",
        "GR_NW_max", "GR_N_i_max", "GR_N_o_max", "GR_NE_max", "GR_E_i_max", "GR_E_o_max",
        "GR_SE_max", "GR_S_i_max", "GR_S_o_max", "GR_SW_max", "GR_W_i_max", "GR_W_o_max",
        "GR_ped_NW_m_min", "GR_ped_NE_m_min", "GR_ped_SW_m_min", "GR_ped_SE_m_min",
        "GR_ped_NW_min", "GR_ped_N_i_min", "GR_ped_N_o_min", "GR_ped_NE_min", "GR_ped_E_i_min", "GR_ped_E_o_min",
        "GR_ped_SE_min", "GR_ped_S_i_min", "GR_ped_S_o_min", "GR_ped_SW_min", "GR_ped_W_i_min", "GR_ped_W_o_min",
        "GR_ped_NW_m_avg", "GR_ped_NE_m_avg", "GR_ped_SW_m_avg", "GR_ped_SE_m_avg",
        "GR_ped_NW_avg", "GR_ped_N_i_avg", "GR_ped_N_o_avg", "GR_ped_NE_avg", "GR_ped_E_i_avg", "GR_ped_E_o_avg",
        "GR_ped_SE_avg", "GR_ped_S_i_avg", "GR_ped_S_o_avg", "GR_ped_SW_avg", "GR_ped_W_i_avg", "GR_ped_W_o_avg",
        "GR_ped_NW_m_max", "GR_ped_NE_m_max", "GR_ped_SW_m_max", "GR_ped_SE_m_max",
        "GR_ped_NW_max", "GR_ped_N_i_max", "GR_ped_N_o_max", "GR_ped_NE_max", "GR_ped_E_i_max", "GR_ped_E_o_max",
        "GR_ped_SE_max", "GR_ped_S_i_max", "GR_ped_S_o_max", "GR_ped_SW_max", "GR_ped_W_i_max", "GR_ped_W_o_max",
    ]
    # Check if log file exists, create headers if not
    # Log the results
    # Create directory folder if it doesn't exist
    # if not os.path.exists("datalog"):
    #     os.makedirs("datalog")
    # The main log file
    log_file="datalog/experiment_log.csv"
    file_exists = os.path.isfile(log_file)
    log_data = [log_entry]
    # In addition to the main log file, create a separate CSV for each run
    df_log = pd.DataFrame(log_data, columns=log_header)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"simulation_log_{timestamp}.csv" #- Unique filename for each run
    log_path = f"datalog/{log_filename}"
    df_log.to_csv(log_path, index=False)
    print(f"Experiment '{sim_name}' logged successfully!")
    # Append to the main log file
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file does not exist
        if not file_exists:
            writer.writerow(log_header)
        # Append log entry
        writer.writerow(log_entry)

    if showplot or saveplot:
        intersect_visualize_results(
            scenario_name = sim_name,
            calculation_surfaces_withname = calculation_surfaces_withname,
            light_sources=light_sources,
            illuminance_results_H = illuminance_results_H,
            illuminance_results_V_N=illuminance_results_V_N,
            illuminance_results_V_E=illuminance_results_V_E,
            illuminance_results_V_S=illuminance_results_V_S,
            illuminance_results_V_W=illuminance_results_V_W,
            illuminance_results_V_NW=illuminance_results_V_NW,
            illuminance_results_V_NE=illuminance_results_V_NE,
            illuminance_results_V_SW=illuminance_results_V_SW,
            illuminance_results_V_SE=illuminance_results_V_SE,
            Esc_results_N=Esc_results_N,
            Esc_results_E=Esc_results_E,
            Esc_results_S=Esc_results_S,
            Esc_results_W=Esc_results_W,
            Esc_results_NW=Esc_results_NW,
            Esc_results_NE=Esc_results_NE,
            Esc_results_SW=Esc_results_SW,
            Esc_results_SE=Esc_results_SE,
            luminance_result=luminance_result,
            glare_anywhere = glare_anywhere,
            glare_results_N = glare_results_N, glare_results_E = glare_results_E,
            glare_results_S = glare_results_S, glare_results_W = glare_results_W,
            glare_results_NW = glare_results_NW, glare_results_NE = glare_results_NE,
            glare_results_SW = glare_results_SW, glare_results_SE = glare_results_SE,
            showplot=showplot,
            saveplot=saveplot
        )

def intersect_visualize_results(scenario_name, calculation_surfaces_withname, light_sources,
                    illuminance_results_H,
                    illuminance_results_V_N, illuminance_results_V_E,
                    illuminance_results_V_S, illuminance_results_V_W,
                    illuminance_results_V_NW, illuminance_results_V_NE,
                    illuminance_results_V_SW, illuminance_results_V_SE,
                    Esc_results_N, Esc_results_E, Esc_results_S, Esc_results_W,
                    Esc_results_NW, Esc_results_NE, Esc_results_SW, Esc_results_SE,
                    luminance_result,
                    glare_anywhere,
                    glare_results_N, glare_results_E, glare_results_S, glare_results_W,
                    glare_results_NW, glare_results_NE, glare_results_SW, glare_results_SE,
                    showplot = 1 , saveplot = 1):
    """
    Create 2×3 grid visualization with:
    - Combined crosswalk visualizations in single subplots
    - Calculation surfaces visible in all plots
    """
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12)) # fig should be H*2-4
    # Ideally should put the file name or Brand & Wattage too, not just scenario_name
    fig.suptitle(f"Lighting Simulation Results: {scenario_name}", fontsize=16)

    # Extract coordinates and values
    x = [p[0] for p in illuminance_results_H]
    y = [p[1] for p in illuminance_results_H]
    e_h = [p[2] for p in illuminance_results_H]
    # lu = [p[2] for p in luminance_result]
    glare_anywhere_value = [p[2] for p in glare_anywhere]

    # from calculation_surfaces_withname extract the bounds
    # calculation_surfaces_withname is a dictionary with keys as names and values as tuples (x1, x2, y1, y2)
    calculation_surfaces = list(calculation_surfaces_withname.values())

    # Plot 1: Horizontal Illuminance (top-left)
    plot_surfaces_and_luminaires(axes[0, 0], calculation_surfaces, light_sources)
    plot_heatmap_with_surfaces(axes[0, 0], x, y, e_h, calculation_surfaces,
                             "Horizontal Illuminance (lux)", log_scale=True)

    # Plot 2: Combined Vertical Illuminance (top-middle)
    plot_surfaces_and_luminaires(axes[0, 1], calculation_surfaces, light_sources)
    # plot_combined_crosswalks(axes[0, 1],
    #                        [illuminance_results_V_N, illuminance_results_V_E,
    #                         illuminance_results_V_S, illuminance_results_V_W],
    #                        calculation_surfaces[1:],  # Just crosswalks
    #                        "Vertical Illuminance by Crosswalk (lux)")
    # Update - each surface has its own direction results. This is a bit tricky.
    # Think about at which direction the light hit pedestrian body and then to car, think about it for each surface
    # Use the name of the surface to get the right illuminance results
    # "NW_m	NE_m	SW_m	SE_m	NW 	N_i	N_o	NE 	E_i	E_o	SE	S_i	S_o	SW	W_i	W_o" but we only need the first 4,
    #  meaning 12 surfaces
    # NW get illuminance_results_V_NE, N_i get illuminance_results_V_N, N_o get illuminance_results_V_S, NE get illuminance_results_V_SE
    plot_combined_crosswalks(axes[0, 1],
                             # 12 illuminance results, recall the order in which we logged vertical illuminance
                             [illuminance_results_V_NE, illuminance_results_V_N, illuminance_results_V_S, illuminance_results_V_SE,
                              illuminance_results_V_E, illuminance_results_V_W, illuminance_results_V_SW, illuminance_results_V_S,
                              illuminance_results_V_N, illuminance_results_V_NW, illuminance_results_V_W, illuminance_results_V_E
                             ],
                             [calculation_surfaces_withname["NW"], calculation_surfaces_withname["N_i"], calculation_surfaces_withname["N_o"], calculation_surfaces_withname["NE"],
                              calculation_surfaces_withname["E_i"], calculation_surfaces_withname["E_o"], calculation_surfaces_withname["SE"], calculation_surfaces_withname["S_i"],
                              calculation_surfaces_withname["S_o"], calculation_surfaces_withname["SW"], calculation_surfaces_withname["W_i"], calculation_surfaces_withname["W_o"]
                             ],
                             "Vertical Illuminance by Crosswalk (lux)")


    # Plot 3: Combined Semi-Cylindrical Illuminance (top-right)
    plot_surfaces_and_luminaires(axes[0, 2], calculation_surfaces, light_sources)
    # plot_combined_crosswalks(axes[0, 2],
    #                        [Esc_results_N, Esc_results_E, Esc_results_S, Esc_results_W],
    #                        calculation_surfaces[1:],  # Just crosswalks
    #                        "Semi-Cylindrical Illuminance by Crosswalk (lux)")
    # Update - same logic as vetical illuminance
    plot_combined_crosswalks(axes[0, 2],
                                # 12 illuminance results, recall the order in which we logged vertical illuminance
                                # vector: NE, N, S, SE, E, W, SW, S, N, NW, W, E
                                [Esc_results_NE, Esc_results_N, Esc_results_S, Esc_results_SE,
                                Esc_results_E, Esc_results_W, Esc_results_SW, Esc_results_S,
                                Esc_results_N, Esc_results_NW, Esc_results_W, Esc_results_E
                                ],
                                [calculation_surfaces_withname["NW"], calculation_surfaces_withname["N_i"], calculation_surfaces_withname["N_o"], calculation_surfaces_withname["NE"],
                                calculation_surfaces_withname["E_i"], calculation_surfaces_withname["E_o"], calculation_surfaces_withname["SE"], calculation_surfaces_withname["S_i"],
                                calculation_surfaces_withname["S_o"], calculation_surfaces_withname["SW"], calculation_surfaces_withname["W_i"], calculation_surfaces_withname["W_o"]
                                ],
                                "Semi-Cylindrical Illuminance by Crosswalk (lux)")

    # # Plot 4: Luminance (bottom-left)
    # plot_surfaces_and_luminaires(axes[1, 0], calculation_surfaces, light_sources)
    # plot_heatmap_with_surfaces(axes[1, 0], x, y, lu, calculation_surfaces,
    #                          "Luminance (cd/m²)")
    # Sorry I change my mind, let's do a glare rating pointing to the light source
    # Plot 4: Glare Rating (From anywhere to light) (bottom-left)
    plot_surfaces_and_luminaires(axes[1, 0], calculation_surfaces, light_sources)
    plot_heatmap_with_surfaces(axes[1, 0], x, y, glare_anywhere_value, calculation_surfaces,
                             "Glare Rating from any most glaring angles (GR)")

    # Plot 5: Glare Rating (Car) (bottom-middle)
    plot_surfaces_and_luminaires(axes[1, 1], calculation_surfaces, light_sources)
    # plot_combined_crosswalks(axes[1, 1],
    #                        [glare_results_N, glare_results_E, glare_results_S, glare_results_W],
    #                        calculation_surfaces[1:],  # Just crosswalks
    #                        "Glare Rating (GR)")
    # Update -  "kinda" same logic as vetical illuminance. We instead cut the corner of intersection.
    # vector order same as glare for car that we logged
    # GR_NW_m_min = extracts_min(glare_results_SW, calculation_surfaces_withname["NW_m"])
    # GR_NE_m_min = extracts_min(glare_results_NW, calculation_surfaces_withname["NE_m"])
    # GR_SW_m_min = extracts_min(glare_results_SE, calculation_surfaces_withname["SW_m"])
    # GR_SE_m_min = extracts_min(glare_results_NE, calculation_surfaces_withname["SE_m"])
    # GR_NW_min = None
    # GR_N_i_min = extracts_min(glare_results_S, calculation_surfaces_withname["N_i"])
    # GR_N_o_min = extracts_min(glare_results_N, calculation_surfaces_withname["N_o"])
    # GR_NE_min = None
    # GR_E_i_min = extracts_min(glare_results_W, calculation_surfaces_withname["E_i"])
    # GR_E_o_min = extracts_min(glare_results_E, calculation_surfaces_withname["E_o"])
    # GR_SE_min = None
    # GR_S_i_min = extracts_min(glare_results_N, calculation_surfaces_withname["S_i"])
    # GR_S_o_min = extracts_min(glare_results_S, calculation_surfaces_withname["S_o"])
    # GR_SW_min = None
    # GR_W_i_min = extracts_min(glare_results_E, calculation_surfaces_withname["W_i"])
    # GR_W_o_min = extracts_min(glare_results_W, calculation_surfaces_withname["W_o"])
    plot_combined_crosswalks(axes[1, 1],
                             [glare_results_SW, glare_results_NW, glare_results_SE, glare_results_NE,
                              glare_results_S, glare_results_N, glare_results_W, glare_results_E,
                              glare_results_N, glare_results_S, glare_results_E, glare_results_W
                             ],
                             [calculation_surfaces_withname["NW_m"], calculation_surfaces_withname["NE_m"], calculation_surfaces_withname["SW_m"], calculation_surfaces_withname["SE_m"],
                              calculation_surfaces_withname["N_i"], calculation_surfaces_withname["N_o"], calculation_surfaces_withname["E_i"], calculation_surfaces_withname["E_o"],
                              calculation_surfaces_withname["S_i"], calculation_surfaces_withname["S_o"], calculation_surfaces_withname["W_i"], calculation_surfaces_withname["W_o"]
                             ],
                                "Glare Rating for drivers (GR)")

    # Plot 6: Glare Rating (Ped) (bottom-right)
    plot_surfaces_and_luminaires(axes[1, 2], calculation_surfaces, light_sources)
    # plot_combined_crosswalks(axes[1, 2],
    #                        [glare_results_N, glare_results_E, glare_results_S, glare_results_W],
    #                        calculation_surfaces[1:],  # Just crosswalks
    #                        "Glare Rating (GR)")
    # Update -  "kinda" same logic as vetical illuminance. We instead cut the corner of intersection.
    # vector order: SE, E, W, SW, S, N, NW, W, E, NE, N, S
    # GR_ped_NW_m_min = None
    # GR_ped_NE_m_min = None
    # GR_ped_SW_m_min = None
    # GR_ped_SE_m_min = None
    # GR_ped_NW_min = extracts_min(glare_results_SE, calculation_surfaces_withname["NW"])
    # GR_ped_N_i_min = extracts_min(glare_results_E, calculation_surfaces_withname["N_i"])
    # GR_ped_N_o_min = extracts_min(glare_results_W, calculation_surfaces_withname["N_o"])
    # GR_ped_NE_min = extracts_min(glare_results_SW, calculation_surfaces_withname["NE"])
    # GR_ped_E_i_min = extracts_min(glare_results_S, calculation_surfaces_withname["E_i"])
    # GR_ped_E_o_min = extracts_min(glare_results_N, calculation_surfaces_withname["E_o"])
    # GR_ped_SE_min = extracts_min(glare_results_NW, calculation_surfaces_withname["SE"])
    # GR_ped_S_i_min = extracts_min(glare_results_W, calculation_surfaces_withname["S_i"])
    # GR_ped_S_o_min = extracts_min(glare_results_E, calculation_surfaces_withname["S_o"])
    # GR_ped_SW_min = extracts_min(glare_results_NE, calculation_surfaces_withname["SW"])
    # GR_ped_W_i_min = extracts_min(glare_results_N, calculation_surfaces_withname["W_i"])
    # GR_ped_W_o_min = extracts_min(glare_results_S, calculation_surfaces_withname["W_o"])
    plot_combined_crosswalks(axes[1, 2],
                             [glare_results_SE, glare_results_E, glare_results_W, glare_results_SW,
                              glare_results_S, glare_results_N, glare_results_NW, glare_results_W,
                              glare_results_E, glare_results_NE, glare_results_N, glare_results_S
                             ],
                             [calculation_surfaces_withname["NW"], calculation_surfaces_withname["N_i"], calculation_surfaces_withname["N_o"], calculation_surfaces_withname["NE"],
                              calculation_surfaces_withname["E_i"], calculation_surfaces_withname["E_o"], calculation_surfaces_withname["SE"], calculation_surfaces_withname["S_i"],
                              calculation_surfaces_withname["S_o"], calculation_surfaces_withname["SW"], calculation_surfaces_withname["W_i"], calculation_surfaces_withname["W_o"]
                             ],
                                "Glare Rating for pedestrians (GR)")

    plt.tight_layout() # Adjust layout to prevent overlap

    if saveplot:
        # check if directory exists, if not create it
        if not os.path.exists("visualization"):
            os.makedirs("visualization")
        plt.savefig(f"visualization/{scenario_name}_results.png", dpi=150, bbox_inches='tight')
        print(f"{scenario_name}_results.png is saved")
    if showplot:
        plt.show()
    plt.close()

def plot_combined_crosswalks(ax, results_list, crosswalk_bounds, title):
    """
    Plot combined crosswalk results in one subplot with different colors.
    results_list: [N, E, S, W] results
    crosswalk_bounds: [N, E, S, W] bounds
    """
    # Use predefined colormaps for each direction
    # cmaps = ['viridis', 'viridis', 'viridis', 'viridis']
    # colors = ['blue', 'green', 'red', 'purple']
    # labels = ['North', 'East', 'South', 'West']

    # Find global min/max for consistent color scaling
    global_min = min(min(p[2] for p in results) for results in results_list if results)
    global_max = max(max(p[2] for p in results) for results in results_list if results)

    for i, (results, bounds) in enumerate(zip(results_list, crosswalk_bounds)):
        # Filter points within this crosswalk
        x_min, x_max, y_min, y_max = bounds
        points_in_crosswalk = [
            (p[0], p[1], p[2])
            for p in results
            if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max
        ]

        if not points_in_crosswalk:
            continue

        x_cw = [p[0] for p in points_in_crosswalk]
        y_cw = [p[1] for p in points_in_crosswalk]
        v_cw = [p[2] for p in points_in_crosswalk]

        # Create grid for heatmap
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        zi = griddata((x_cw, y_cw), v_cw, (xi[None,:], yi[:,None]), method='linear')

        # Plot heatmap with semi-transparency
        im = ax.pcolormesh(xi, yi, zi, shading='auto', alpha=0.7,
                          cmap='viridis', vmin=global_min, vmax=global_max)

        # # Add outline
        # rect = patches.Rectangle(
        #     (x_min, y_min), x_max-x_min, y_max-y_min,
        #     linewidth=1, edgecolor=colors[i], facecolor='none',
        #     label=labels[i]
        # )
        # ax.add_patch(rect)

    # Add colorbar (representative of all crosswalks)
    norm = plt.Normalize(
        vmin=min(min(p[2] for p in results) for results in results_list if results),
        vmax=max(max(p[2] for p in results) for results in results_list if results)
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(title.split('(')[-1].split(')')[0].strip(), rotation=270, labelpad=15)

    # Add legend for crosswalks
    # ax.legend(loc='upper right')

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True, alpha=0.3)

    # Add calculation surface outlines
    plot_surface_outlines(ax, crosswalk_bounds)

def plot_surface_outlines(ax, calculation_surfaces):
    """
    Draw outlines of all calculation surfaces on a plot.
    """
    # colors = ['red', 'blue', 'green', 'orange', 'purple']
    # labels = ['Intersection', 'N Crosswalk', 'E Crosswalk', 'S Crosswalk', 'W Crosswalk']

    for i, (x_min, x_max, y_min, y_max) in enumerate(calculation_surfaces):
        rect = patches.Rectangle(
            (x_min, y_min), x_max-x_min, y_max-y_min,
            linewidth=1, edgecolor='blue', facecolor='none', linestyle='--',
            # label=labels[i]
        )
        ax.add_patch(rect)

def plot_surfaces_and_luminaires(ax, calculation_surfaces, light_sources, glarezone=False):
    """
    Plot calculation surfaces and luminaire locations with orientations.
    """
    # Plot calculation surfaces
    # colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    # labels = ['Intersection', 'N Crosswalk', 'E Crosswalk', 'S Crosswalk', 'W Crosswalk']

    for i, (x_min, x_max, y_min, y_max) in enumerate(calculation_surfaces):
        rect = patches.Rectangle(
            (x_min, y_min), x_max-x_min, y_max-y_min,
            linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.5,
            # label=labels[i]
        )
        ax.add_patch(rect)

    # Plot luminaire locations and orientations
    for ls in light_sources:
        # Plot pole location
        ax.plot(ls.x_p, ls.y_p, 'ko', markersize=8)

        # Plot light head location and orientation
        ax.plot(ls.x, ls.y, 'ro', markersize=6)
        ax.arrow(
            ls.x_p, ls.y_p,
            ls.x - ls.x_p, ls.y - ls.y_p,
            head_width=0.5, head_length=0.7, fc='r', ec='r'
        )

        # Add height label
        ax.text(ls.x_p, ls.y_p, f'{ls.h_p}m', ha='right', va='bottom')
        # --- Add Glare Rings ---
        observer_height = 1.5  # Assume fixed eye height
        h_diff = ls.h - observer_height # h_diff should just be a single number, not a list of numbers

        if h_diff > 0 and glarezone:
            r_53 = h_diff * math.tan(math.radians(53))
            r_75 = h_diff * math.tan(math.radians(75))

            # Draw Discomfort Zone Ring (53°)
            discomfort_ring = patches.Circle((ls.x, ls.y), r_53,
                                              edgecolor='orange', linestyle='--',
                                              linewidth=1.0, fill=False, alpha=0.5)
            ax.add_patch(discomfort_ring)
            # Draw Glare Zone Ring (75°)
            glare_ring = patches.Circle((ls.x, ls.y), r_75,
                                         edgecolor='red', linestyle='-',
                                         linewidth=1.0, fill=False, alpha=0.5)
            ax.add_patch(glare_ring)

    # Set plot properties
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True)
    ax.axis('equal')
    # ax.legend(loc='upper right')

def plot_heatmap_with_surfaces(ax, x, y, values, calculation_surfaces, title, log_scale=False):
    """
    Create a heatmap with calculation surfaces overlaid.
    """
    # Create grid for heatmap
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    zi = griddata((x, y), values, (xi[None,:], yi[:,None]), method='linear')

    # Plot heatmap
    if log_scale:
        norm = LogNorm(vmin=max(0.1, min(values)), vmax=max(values))
        im = ax.pcolormesh(xi, yi, zi, shading='auto', norm=norm)
    else:
        im = ax.pcolormesh(xi, yi, zi, shading='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title.split('(')[-1].split(')')[0].strip(), rotation=270, labelpad=15)

    # Overlay calculation surfaces
    plot_surface_outlines(ax, calculation_surfaces)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True, alpha=0.3)
