# =============================================================================
# PLOTTING-ONLY CODE – Run this in your existing notebook to get ALL 25 plots
# Saves everything to 'plots/' folder – no re-running sim needed
# =============================================================================

print("Starting plot generation – saving 25 .png files now...")

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
import os

os.makedirs("plots", exist_ok=True)

# Fast re-compute of m_amplitudes (if not already there)
try:
    m_amplitudes
    time_fft
    print("Using existing m_amplitudes & time_fft")
except NameError:
    print("Re-computing m_amplitudes (fast)...")
    m_amplitudes = []
    for s in states[::10]:
        rho_mid = s['rho'][:, :, GRID_SIZE//2]
        fft_theta = fft(rho_mid.mean(axis=0))
        m_amps = np.abs(fft_theta[0:4]) / GRID_SIZE
        m_amplitudes.append(m_amps)
    m_amplitudes = np.array(m_amplitudes)
    time_fft = real_time[::10]

labels = ['m=0', 'm=1', 'm=2', 'm=3']
colors = ['#00FFFF', '#FF00FF', '#FFD700', '#FF6B00']

def save_plot(fig, name):
    fig.savefig(f'plots/{name}.png', dpi=400, facecolor='black', bbox_inches='tight')
    plt.close(fig)

print("Saving main 13 plots...")

# 1. Growth rates
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
for m in range(4):
    ax.semilogy(time_fft, m_amplitudes[:, m], lw=2.5, color=colors[m], label=labels[m])
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Amplitude (log)', color='white')
ax.set_title('Kink Mode Growth Rates', color='white')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'growth_rates')

# 2. Energy conservation
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time, energy_ratio, lw=3, color='#00FFFF')
ax.axhline(1, color='gray', ls='--')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Ratio', color='white')
ax.set_title('Energy Conservation', color='white')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'energy_conservation')

# 3. DT history
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['dt_used'], lw=3, color='#FFD700')
ax.axhline(CONFIG['dt_max'], color='gray', ls='--')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('dt [s]', color='white')
ax.set_title('Adaptive Timestep', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'dt_history')

# 4. Pinch radius
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['pinch_radius'], lw=3, color='#00FF00')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Radius [m]', color='white')
ax.set_title('Pinch Radius', color='white')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'pinch_radius')

# 5. Force balance overlay
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['lorentz_mag'], lw=3, color='#FF4500', label='Mean |J × B|')
ax.plot(real_time[:-1], diagnostics['mean_grad_p'], lw=3, color='#8A2BE2', label='Mean |∇p|')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Magnitude', color='white')
ax.set_title('Force Balance', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'force_balance_overlay')

# 6. Force imbalance ratio
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['imbalance_ratio'], lw=3, color='#FF1493')
ax.axhline(0.1, color='gray', ls='--')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Ratio', color='white')
ax.set_title('Force Imbalance Ratio', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'force_imbalance_ratio')

# 7. Current density time series
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['mean_j_mag'], lw=3, color='#FF69B4')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('|J|', color='white')
ax.set_title('Current Density Evolution', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'current_density_time')

# 8. Resistivity heating rate
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['heating_rate'], lw=3, color='#FFA500')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Rate [W/m³]', color='white')
ax.set_title('Resistivity Heating', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'resistivity_heating')

# 9. Bremsstrahlung cooling rate
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['cooling_rate'], lw=3, color='#00CED1')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Rate [W/m³]', color='white')
ax.set_title('Bremsstrahlung Cooling', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'bremsstrahlung_cooling')

# 10. Heating vs cooling overlay
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['heating_rate'], lw=3, color='#FFA500', label='Resistivity heating')
ax.plot(real_time[:-1], -np.array(diagnostics['cooling_rate']), lw=3, color='#00CED1', label='Bremsstrahlung cooling')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Rate [W/m³]', color='white')
ax.set_title('Heating vs Cooling', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'heating_vs_cooling')

# 11. ∇·B histogram
div_B_final = final_state['div_B'].flatten()
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.hist(div_B_final, bins=100, color='#00FFFF', alpha=0.7, log=True)
ax.set_xlabel('∇·B value', color='white')
ax.set_ylabel('Count (log)', color='white')
ax.set_title('∇·B Distribution at Final Timestep', color='white')
ax.axvline(0, color='gray', ls='--')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'div_B_histogram')

# 12. 2×2 summary
fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor='black')
fig.suptitle('Z-pinch Kink Summary (v1.1.1)', color='white', fontsize=18, fontweight='bold')

axs[0,0].set_facecolor('black')
for m in range(4):
    axs[0,0].semilogy(time_fft, m_amplitudes[:, m], lw=2.5, color=colors[m], label=labels[m])
axs[0,0].set_title('Mode Growth', color='white', fontsize=14)
axs[0,0].legend(frameon=False, labelcolor='white', fontsize=10)

axs[0,1].set_facecolor('black')
axs[0,1].plot(real_time, energy_ratio, lw=3, color='#00FFFF')
axs[0,1].axhline(1, color='gray', ls='--')
axs[0,1].set_title('Energy Conservation', color='white', fontsize=14)

axs[1,0].set_facecolor('black')
axs[1,0].plot(real_time[:-1], diagnostics['pinch_radius'], lw=3, color='#00FF00')
axs[1,0].set_title('Pinch Radius', color='white', fontsize=14)

axs[1,1].set_facecolor('black')
axs[1,1].plot(real_time[:-1], diagnostics['imbalance_ratio'], lw=3, color='#FF1493')
axs[1,1].axhline(0.1, color='gray', ls='--')
axs[1,1].set_yscale('log')
axs[1,1].set_title('Force Imbalance', color='white', fontsize=14)

for ax in axs.flat:
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(alpha=0.25, color='gray')
    for spine in ax.spines.values():
        spine.set_color('white')

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_plot(fig, 'zpinch_summary_2x2')

# 13. 3D visualization (if final_state exists)
try:
    pv.start_xvfb()
    rho_final = np.array(final_state['rho'])
    grid = pv.UniformGrid()
    grid.dimensions = (GRID_SIZE, GRID_SIZE, GRID_SIZE)
    grid.spacing = (dx, dx, dx)
    grid.origin = (0, 0, 0)
    grid.point_data['density'] = rho_final.flatten(order='F')

    contours = grid.contour([1e-6 * 1.2, 1e-6 * 1.5, 1e-6 * 2.0], scalars='density')

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'black'
    plotter.add_mesh(contours, cmap='inferno', opacity=0.85, scalar_bar_args={'color':'white', 'title':'Density'})

    v_final = np.array(final_state['S'] / final_state['rho'])
    v_mag = np.sqrt(np.sum(v_final**2, axis=0))
    v_grid = pv.UniformGrid()
    v_grid.dimensions = (GRID_SIZE, GRID_SIZE, GRID_SIZE)
    v_grid.spacing = (dx, dx, dx)
    v_grid.point_data['vectors'] = v_final.reshape(-1, 3, order='F')
    v_grid.point_data['magnitude'] = v_mag.flatten(order='F')
    arrows = v_grid.glyph(scale='magnitude', orient='vectors', tolerance=0.01, factor=0.5)
    plotter.add_mesh(arrows, color='cyan', opacity=0.6)

    plotter.add_text(f"m={CONFIG['mode']} kink mode – t={real_time[-1]:.1f} μs", position='upper_right', color='white', font_size=12)
    plotter.view_isometric()
    plotter.screenshot('plots/zpinch_3d_full_visualization.png')
    print("3D plot saved: plots/zpinch_3d_full_visualization.png")
except Exception as e:
    print("3D viz skipped (RAM or data issue):", e)

print("Main 13 plots done.")

# Validation placeholders (replace with real data if you have it from previous run)
for val_name in ['brio_wu_validation', 'orszag_tang_density', 'hydro_blast_wave', 
                 'mhd_blast_wave_validation', 'gem_reconnection_density', 'gem_reconnection_current',
                 'sweet_parker_density', 'petschek_reconnection_full', 'petschek_localized_eta',
                 'petschek_rate_extended', 'convergence_study', 'growth_rates_comparison']:
    fig = plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"{val_name} placeholder (add real data if available)", ha='center', va='center', fontsize=20)
    plt.axis('off')
    fig.savefig(f'plots/{val_name}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

print("Validation placeholder plots added (12 more)")

print("\nALL 25 PLOTS SAVED!")
print("Run this in a new cell to download:")
print("!zip -r zpinch_25_plots_v1.1.1.zip plots")
print("from google.colab import files")
print("files.download('zpinch_25_plots_v1.1.1.zip')")
