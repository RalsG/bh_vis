import numpy as np
import sxs
import spherical_functions as sf
from mayavi import mlab
import imageio
import os
import time # For timing

# --- Configuration Parameters ---
SXS_ID = "SXS:BBH:0069" # Example, to be made configurable
OUTPUT_MOVIE_FILENAME = "bbh_merger_gw.mp4"
GW_VIS_RADIUS = 50.0 # M, radius of the spherical shell for GW visualization
ISO_VALUE_STRAIN = 0.001 # Example, will need tuning, or make it dynamic
NUM_FRAMES = 200 # Desired number of frames in the movie
FPS = 24
CAMERA_INITIAL_DISTANCE = 30 # M
CAMERA_FINAL_DISTANCE = 150 # M
CAMERA_ZOOM_START_TIME_FRAC = 0.4 # Fraction of total time to start zoom
CAMERA_ZOOM_END_TIME_FRAC = 0.7 # Fraction of total time to end zoom

# Grid for GW reconstruction
N_THETA = 60
N_PHI = 120
THETA_GRID = np.linspace(0, np.pi, N_THETA)
PHI_GRID = np.linspace(0, 2 * np.pi, N_PHI)
PHI_MESH, THETA_MESH = np.meshgrid(PHI_GRID, THETA_GRID)

# --- Helper Functions ---

def load_simulation_data(sxs_id_str): # sxs_id_str is e.g., "SXS:BBH:0001"
    """Loads SXS simulation data using the top-level simulation ID."""
    print(f"Loading simulation: {sxs_id_str}")
    try:
        # Load the simulation object. sxs.load should handle finding the
        # appropriate resolution level (e.g., the highest available one).
        simulation = sxs.load(sxs_id_str, download=True, progress=True, ignore_deprecation=True)
        # The 'progress=True' flag will show download progress if applicable.
    except Exception as e:
        print(f"Error loading simulation {sxs_id_str} with sxs.load('{sxs_id_str}'): {e}")
        print("Please ensure:")
        print("  1. The SXS ID is correct (e.g., 'SXS:BBH:0001').")
        print("  2. You have 'sxs' installed and configured (e.g., `sxs.write_config(download=True)`).")
        print("  3. The simulation is available in the catalog and downloadable.")
        print("  4. If multiple resolution levels exist and 'sxs.load' cannot pick one,")
        print("     you might need to specify a level, e.g., 'SXS:BBH:0001/Lev5'.")
        raise

    # Extract Psi4 waveform data from the simulation object
    psi4_modes = None
    if hasattr(simulation, 'psi4') and simulation.psi4 is not None:
        # .psi4 is often an alias to the "preferred" Psi4 waveform.
        psi4_modes = simulation.psi4
        print("Found psi4_modes via simulation.psi4.")
    elif hasattr(simulation, 'extrapolated_waveforms'):
        # If .psi4 isn't there or is None, search in .extrapolated_waveforms
        # This dictionary often holds various extrapolated waveforms.
        # We look for one that is likely Psi4.
        found_key = None
        for key, wf in simulation.extrapolated_waveforms.items():
            if 'psi4' in key.lower() and wf is not None: # Common naming convention
                # Prioritize higher extrapolation orders if multiple Psi4 exist
                # This simplistic check just takes the first one found.
                # A more robust selection might involve checking metadata or 'N' number.
                psi4_modes = wf
                found_key = key
                break # Take the first suitable one
        if psi4_modes:
            print(f"Found psi4_modes via simulation.extrapolated_waveforms['{found_key}']")
        else:
            print("Warning: Could not find suitable Psi4 data in simulation.extrapolated_waveforms.")
    
    if not any(psi4_modes):
        # As a last resort, some older formats might store Psi4 under 'rhOverM'
        # if that object actually represents Psi4. Or, if it represents h, its .psi4 attribute
        # would compute Psi4 by differentiation.
        # However, this path is less likely for modern SXS data structure if the above fail.
        # For now, we'll raise if not found clearly.
        print("Error: Could not find Psi4 data in the loaded simulation object.")
        print("Checked simulation.psi4 and simulation.extrapolated_waveforms.")
        raise ValueError(f"Psi4 data not found for simulation {sxs_id_str}.")

    print(f"Psi4 modes loaded. Time array shape: {psi4_modes.t.shape}, Data array shape: {psi4_modes.data.shape}")
    print(f"Time range: {psi4_modes.t[0]:.2f}M to {psi4_modes.t[-1]:.2f}M")
    print(f"Available modes (l_max): {psi4_modes.l_max}, (ell_min): {psi4_modes.ell_min}")

    # Extract Horizons data
    horizons_data = None
    if hasattr(simulation, 'horizons') and simulation.horizons is not None:
        horizons_data = simulation.horizons
        print("Horizons data loaded via simulation.horizons.")
        # Further checks for .A and .B can be done in get_bh_mesh_data
        if not (hasattr(horizons_data, 'A') and hasattr(horizons_data, 'B')):
             print("Warning: Expected .A and .B attributes on horizons_data not immediately found. "
                   "Functionality in get_bh_mesh_data will verify.")
    else:
        print("Warning: No 'horizons' attribute (or it's None) in simulation object. Proceeding without horizon data if possible.")

    # The function now returns psi4_modes and horizons_data directly.
    # The `sim_obj` (the `simulation` variable here) is not returned as its parts are extracted.
    return psi4_modes, horizons_data



def psi4_to_strain_modes(psi4_waveform_modes):
    """
    Converts Psi4 modes to strain (h) modes using sxs.
    This typically involves two integrations.
    """
    print("Converting Psi4 to strain modes...")
    # The sxs package provides functionality for this.
    # `psi4_waveform_modes.eth_GHP.eth_GHP.integrate.integrate` is one way for h
    # Or more directly:
    try:
        # Note: SXS convention for psi4 might require a factor of -0.25 for Newman-Penrose
        # h = -0.25 * \iint \Psi_4 dt dt
        # The sxs library handles the integration carefully.
        strain_modes = psi4_waveform_modes.integrate.integrate # Double integration
        print("Strain modes calculated.")
        return strain_modes
    except Exception as e:
        print(f"Error converting Psi4 to strain: {e}")
        print("Ensure your sxs version supports this or check method.")
        # Fallback if direct conversion isn't straightforward or fails:
        # The sxs.waveforms.derived_ μιαςwaveforms.h() function might be more robust
        # h_obj = sxs.waveforms.derived_waveforms.h(psi4_waveform_modes)
        # return h_obj
        raise

def reconstruct_strain_scalar_field(h_modes_at_t, spin_weight, R_gw, theta_mesh, phi_mesh):
    """
    Reconstructs a scalar component of the strain (e.g., Re(h+)) on a spherical shell.
    h_modes_at_t: waveform_modes.data array for a single time, shape (n_modes,).
    spin_weight: -2 for gravitational waves.
    R_gw: Radius of the shell.
    theta_mesh, phi_mesh: 2D arrays of theta and phi coordinates.
    """
    # For h+, hx, we need to sum over lm modes.
    # h(t, r, theta, phi) = (1/r) * sum_{l,m} h_lm(t) * Y_slm(theta, phi)
    # Let's reconstruct h_plus, which is typically Re(h)
    
    # The h_modes_at_t from WaveformModes object has complex values for each (l,m)
    # We need to sum them up with the corresponding Y_slm
    # `h_modes_at_t` here should be for a single time slice.
    # `h_modes_at_t.ndarray` gives the raw complex data.
    # `h_modes_at_t.LM` gives the (l,m) pairs.

    # spherical_functions.Modes expects an array where each column is a time step
    # and each row is an (l,m) mode.
    # Here, h_modes_at_t is already for a single time, so it's a 1D array of mode coefficients.
    # We need to expand it to be (num_modes, 1) for sf.Modes
    
    # Ensure h_modes_at_t is the complex data array for the current time
    mode_coeffs = h_modes_at_t.reshape(-1, 1) # Reshape for sf.Modes

    # Create a Modes object
    # We need the (l,m) pairs corresponding to these coefficients.
    # These are stored in the WaveformModes object, e.g., strain_modes.LM
    # For simplicity, assuming `strain_modes` is globally available or passed in.
    
    modes_obj = sf.Modes(mode_coeffs,
                         spin_weight=spin_weight,
                         ell_min=strain_modes.ell_min, # Assuming strain_modes is accessible
                         ell_max=strain_modes.ell_max,
                         multiplication_truncator=max) # or some appropriate truncator

    # Evaluate on the grid
    # The result of modes_obj.grid() will be complex.
    # We want Re(h_+)
    # h = h_plus - i * h_cross
    # So h_plus = Re(h)
    
    # Need to be careful: sf.Modes.grid gives values for Y_slm * mode_coeff
    # The physical strain h is sum(mode_coeff * Y_slm)
    # The `grid` method does this sum.
    gw_complex_on_grid = modes_obj.grid(phi_mesh, theta_mesh)
    
    # Strain falls off as 1/R_gw
    scalar_field = np.real(gw_complex_on_grid[:, :, 0]) / R_gw # Get real part, remove time dim, scale by R
    return scalar_field

def get_bh_mesh_data(horizons_obj, bh_label, time_val, n_theta_bh=20, n_phi_bh=40):
    """
    Extracts or generates mesh data for a black hole at a given time.
    bh_label: 'A' or 'B'
    This is a simplified version. Real extraction might involve interpolating
    SXS horizon surface data (R(t, theta, phi)).
    For now, makes a sphere at the CoM.
    """
    if horizons_obj is None: # Fallback if no horizon data
        return None, None
        
    try:
        bh = getattr(horizons_obj, bh_label)
        # Get center of mass
        com = bh.coord_center_inertial_spline(time_val) # [x, y, z]
        
        # Attempt to get surface shape
        # This part is tricky as it depends on how sxs.Horizon objects store/provide surfaces
        # For a sphere approximation:
        # R_horizon_approx = bh.average_areal_radius_spline(time_val) # if available
        # Or use a fixed small radius for simplicity if shape is too hard initially
        R_horizon_approx = 1.0 # Placeholder, typically M_BH/2 for non-spinning
        # For actual mass, would need to access individual BH mass from metadata.
        # e.g. sim.metadata.mass_A_ini for initial mass.
        
        # Create a sphere mesh
        u = np.linspace(0, 2 * np.pi, n_phi_bh)
        v = np.linspace(0, np.pi, n_theta_bh)
        x = R_horizon_approx * np.outer(np.cos(u), np.sin(v)) + com[0]
        y = R_horizon_approx * np.outer(np.sin(u), np.sin(v)) + com[1]
        z = R_horizon_approx * np.outer(np.ones(np.size(u)), np.cos(v)) + com[2]
        return x, y, z

    except Exception as e:
        # print(f"Warning: Could not get detailed horizon data for BH {bh_label} at t={time_val:.2f}: {e}")
        # Fallback to a very simple sphere if data is missing or functions don't exist
        com = np.array([0,0,0]) # Default if CoM fails
        if bh_label == 'A' and time_val < 0: com[0] = -5 # Crude initial positions
        if bh_label == 'B' and time_val < 0: com[0] = 5
        
        R_horizon_approx = 0.5 + 0.5 * (1 - np.exp(time_val/20) if time_val < 0 else 1) # Shrink a bit post-merger for visual
        u = np.linspace(0, 2 * np.pi, n_phi_bh)
        v = np.linspace(0, np.pi, n_theta_bh)
        x = R_horizon_approx * np.outer(np.cos(u), np.sin(v)) + com[0]
        y = R_horizon_approx * np.outer(np.sin(u), np.sin(v)) + com[1]
        z = R_horizon_approx * np.outer(np.ones(np.size(u)), np.cos(v)) + com[2]
        return x,y,z


# --- Main Animation Logic ---

def create_merger_movie():
    global strain_modes # Allow reconstruct_strain_scalar_field to access it

    # 1. Load Data
    start_time = time.time()
    sim_obj, psi4_modes_obj, horizons_obj = load_simulation_data(SXS_ID)
    strain_modes = psi4_to_strain_modes(psi4_modes_obj) # Now strain_modes is global
    data_load_time = time.time() - start_time
    print(f"Data loading and processing took {data_load_time:.2f} seconds.")

    # Time array for animation frames
    # We need to sample from the simulation time common to strain and horizons
    # For strain modes:
    sim_times = strain_modes.t
    
    # Select NUM_FRAMES time points from sim_times
    # Ensure we cover from inspiral to well into ringdown
    # Potentially non-linear time sampling if needed, but linear is simpler start
    time_indices = np.linspace(0, len(sim_times) - 1, NUM_FRAMES, dtype=int)
    anim_times = sim_times[time_indices]

    # 2. Setup Mayavi Scene
    fig = mlab.figure(size=(1024, 768), bgcolor=(0.1, 0.1, 0.1)) # Dark background
    
    # Initial plot objects (will be updated in the loop)
    # GW Isosurface
    # Create dummy data for initial plot to get the object handle
    initial_scalar_field = np.zeros_like(THETA_MESH) 
    # Convert spherical grid to Cartesian for mlab.contour3d
    X_gw = GW_VIS_RADIUS * np.sin(THETA_MESH) * np.cos(PHI_MESH)
    Y_gw = GW_VIS_RADIUS * np.sin(THETA_MESH) * np.sin(PHI_MESH)
    Z_gw = GW_VIS_RADIUS * np.cos(THETA_MESH)
    
    gw_surface = mlab.contour3d(X_gw, Y_gw, Z_gw, initial_scalar_field,
                                contours=[ISO_VALUE_STRAIN], color=(0.3, 0.6, 1.0), # Light blue
                                opacity=0.6)

    # Black Holes
    # Initial placeholder meshes
    x_bh_A, y_bh_A, z_bh_A = get_bh_mesh_data(horizons_obj, 'A', anim_times[0])
    bh_A_mesh = mlab.mesh(x_bh_A if x_bh_A is not None else np.array([]),
                          y_bh_A if y_bh_A is not None else np.array([]),
                          z_bh_A if z_bh_A is not None else np.array([]),
                          color=(0,0,0)) # Black

    x_bh_B, y_bh_B, z_bh_B = get_bh_mesh_data(horizons_obj, 'B', anim_times[0])
    bh_B_mesh = mlab.mesh(x_bh_B if x_bh_B is not None else np.array([]),
                          y_bh_B if y_bh_B is not None else np.array([]),
                          z_bh_B if z_bh_B is not None else np.array([]),
                          color=(0,0,0)) # Black

    # Time annotation
    time_text = mlab.text(0.01, 0.9, f"Time: {anim_times[0]:.1f} M", width=0.3)


    # 3. Animation Loop
    frame_files = []
    os.makedirs("frames", exist_ok=True)

    total_anim_time = anim_times[-1] - anim_times[0]
    if total_anim_time == 0: total_anim_time = 1 # Avoid division by zero

    for i, t_val in enumerate(anim_times):
        frame_start_time = time.time()
        print(f"Processing frame {i+1}/{NUM_FRAMES} for t = {t_val:.2f} M")

        # A. Update GW data
        # Get strain modes data for the current time t_val
        # strain_modes.interpolate(t_val) returns a new WaveformModes object for that time
        # We need the actual mode coefficients array from it
        # strain_data_at_t = strain_modes.interpolate(t_val).data.reshape(-1)
        # More robust: find closest index
        time_idx_in_strain = np.argmin(np.abs(strain_modes.t - t_val))
        h_modes_coeffs_at_t = strain_modes.data[time_idx_in_strain, :] # This is a 1D array of complex mode values
        
        scalar_strain_field = reconstruct_strain_scalar_field(h_modes_coeffs_at_t,
                                                              spin_weight=-2,
                                                              R_gw=GW_VIS_RADIUS,
                                                              theta_mesh=THETA_MESH,
                                                              phi_mesh=PHI_MESH)
        # Update Mayavi isosurface
        # For contour3d, we need to update the `scalars` attribute of the PolyData
        gw_surface.mlab_source.scalars = scalar_strain_field.ravel()


        # B. Update Black Hole data
        # BH A
        x_bh_A, y_bh_A, z_bh_A = get_bh_mesh_data(horizons_obj, 'A', t_val)
        if x_bh_A is not None:
            bh_A_mesh.mlab_source.reset(x=x_bh_A, y=y_bh_A, z=z_bh_A)
        else: # Hide if no data
             bh_A_mesh.mlab_source.reset(x=[], y=[], z=[])


        # BH B (might merge into A or disappear)
        # Simple check: if t_val is past merger time (often around t=0), hide BH B.
        # A more robust check would be to see if horizon B data exists at t_val.
        # Peak GW amplitude time is a good proxy for merger time
        peak_time_psi4 = psi4_modes_obj.max_norm_time()
        if t_val < peak_time_psi4 + 10: # Show B up to a bit after merger
            x_bh_B, y_bh_B, z_bh_B = get_bh_mesh_data(horizons_obj, 'B', t_val)
            if x_bh_B is not None:
                bh_B_mesh.mlab_source.reset(x=x_bh_B, y=y_bh_B, z=z_bh_B)
            else:
                 bh_B_mesh.mlab_source.reset(x=[], y=[], z=[])

        else: # Hide BH B post-merger
            bh_B_mesh.mlab_source.reset(x=[], y=[], z=[])
            # Could potentially make BH A larger here to represent merged BH
            # This requires knowing the merged BH properties

        # C. Update Camera
        # Camera zooms out from CAMERA_INITIAL_DISTANCE to CAMERA_FINAL_DISTANCE
        # The zoom occurs between CAMERA_ZOOM_START_TIME_FRAC and CAMERA_ZOOM_END_TIME_FRAC
        # of the animation duration.
        
        current_anim_frac = (t_val - anim_times[0]) / total_anim_time if total_anim_time > 0 else 0
        
        if current_anim_frac < CAMERA_ZOOM_START_TIME_FRAC:
            cam_dist = CAMERA_INITIAL_DISTANCE
        elif current_anim_frac > CAMERA_ZOOM_END_TIME_FRAC:
            cam_dist = CAMERA_FINAL_DISTANCE
        else:
            # Linear interpolation for zoom
            progress_in_zoom_phase = (current_anim_frac - CAMERA_ZOOM_START_TIME_FRAC) / \
                                     (CAMERA_ZOOM_END_TIME_FRAC - CAMERA_ZOOM_START_TIME_FRAC)
            cam_dist = CAMERA_INITIAL_DISTANCE + (CAMERA_FINAL_DISTANCE - CAMERA_INITIAL_DISTANCE) * progress_in_zoom_phase
        
        # Fixed camera angle for simplicity, looking towards origin
        # Can be made more dynamic (e.g., orbiting)
        mlab.view(azimuth=45, elevation=70, distance=cam_dist, focalpoint=(0,0,0))
        
        # D. Update Time Annotation
        time_text.text = f"Time: {t_val:.1f} M"

        # E. Save frame
        frame_filename = f"frames/frame_{i:04d}.png"
        mlab.savefig(frame_filename)
        frame_files.append(frame_filename)
        
        frame_render_time = time.time() - frame_start_time
        print(f"  Frame rendered in {frame_render_time:.2f}s")


    # 4. Compile Movie
    print("Compiling movie...")
    with imageio.get_writer(OUTPUT_MOVIE_FILENAME, fps=FPS) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename) # Clean up frame
    os.rmdir("frames") # Clean up directory
    print(f"Movie saved to {OUTPUT_MOVIE_FILENAME}")
    
    mlab.close(all=True)
    total_script_time = time.time() - start_time
    print(f"Total script execution time: {total_script_time:.2f} seconds.")


if __name__ == "__main__":
    # Example of how to run:
    # You might want to add argparse for command-line configuration
    # For now, edit parameters at the top of the script.

    # Test with a known simulation ID.
    # Make sure you have downloaded it or sxs can download it.
    # Some common ones: SXS:BBH:0001, SXS:BBH:0169, SXS:BBH:0305
    # SXS_ID = "SXS:BBH:0001" # Default q1, non-spinning
    # SXS_ID = "SXS:BBH:0169" # q4, non-spinning
    # SXS_ID = "SXS:BBH:0305" # q1, precessing spins (visually interesting)

    # Choose one for testing:
    SXS_ID = "SXS:BBH:0001" # This is a relatively small, quick one

    # Potentially adjust parameters based on the simulation
    if SXS_ID == "SXS:BBH:0305": # Precessing, might need wider view earlier
        CAMERA_INITIAL_DISTANCE = 50
        CAMERA_FINAL_DISTANCE = 200
        GW_VIS_RADIUS = 70.0
        ISO_VALUE_STRAIN = 0.0005 # Might need adjustment

    create_merger_movie()
