import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button 

def run_simulation(delta_w, K_max, beta, delta, r, alpha, G, thresh_cos, tau_sacc,
                   T_sync_duration, noise_strength_base, noise_strength_high,
                   T=400, dt=0.02):
    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps)
    phi, x, K, S_acc = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    phi[0], x[0], K[0], S_acc[0] = np.pi * 0.1, 0.01, 0.1, 0.0
    sync_time_accumulator, exit_triggered, exit_time = 0.0, False, -1.0
    current_noise_strength = noise_strength_base

    for t in range(n_steps - 1):
        if np.cos(phi[t]) > thresh_cos:
            sync_time_accumulator += dt
        else:
            sync_time_accumulator = 0

        if sync_time_accumulator >= T_sync_duration and not exit_triggered:
            exit_triggered = True
            current_noise_strength = noise_strength_high
            exit_time = time[t]

        dphi_dt = delta_w - 2 * K[t] * np.sin(phi[t])
        dx_dt = r * S_acc[t] * x[t] * (1 - x[t]) * (1 + alpha * np.cos(phi[t]))
        dK_dt = beta * x[t] * np.cos(phi[t]) * (K_max - K[t]) - delta * K[t]
        max_term = np.maximum(0, np.cos(phi[t]) - thresh_cos)
        dS_acc_dt = G * max_term * x[t] - (S_acc[t] / tau_sacc)

        noise = current_noise_strength * np.random.randn() * np.sqrt(dt)
        phi[t+1] = phi[t] + dt * dphi_dt + noise
        phi[t+1] = (phi[t+1] + np.pi) % (2 * np.pi) - np.pi

        x[t+1] = x[t] + dt * dx_dt
        K[t+1] = K[t] + dt * dK_dt
        S_acc[t+1] = S_acc[t] + dt * dS_acc_dt

        x[t+1] = np.clip(x[t+1], 0, 1)
        K[t+1] = np.clip(K[t+1], 0, K_max)
        S_acc[t+1] = np.maximum(0, S_acc[t+1])

    return time, phi, x, K, S_acc, exit_time

initial_params = {
    'delta_w': 0.4, 'K_max': 3.0, 'beta': 0.5, 'delta': 0.1, 'r': 1.0, 'alpha': 1.0,
    'G': 1.0, 'thresh_cos': 0.5, 'tau_sacc': 10.0, 'T_sync_duration': 100.0,
    'noise_strength_base': 0.5, 'noise_strength_high': 3.0
}
fig, axs = plt.subplots(4, 1, figsize=(12, 9))
plt.subplots_adjust(left=0.1, bottom=0.50) 

time_init, phi_init, x_init, K_init, S_acc_init, exit_time_init = run_simulation(**initial_params)

line_phi, = axs[0].plot(time_init, phi_init, label='$\phi(t)$')
axs[0].set_ylabel('Phase $\phi$'); axs[0].set_ylim(-np.pi * 1.1, np.pi * 1.1); axs[0].grid(True)
axs[0].axhline(0, color='r', linestyle='--', linewidth=0.8, label='In-phase sync')
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)

line_x, = axs[1].plot(time_init, x_init, label='x(t)', color='g')
axs[1].set_ylabel('Task \n Progression \n $x$'); axs[1].set_ylim(-0.1, 1.1); axs[1].grid(True)
axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)

line_K, = axs[2].plot(time_init, K_init, label='K(t)', color='purple')
axs[2].set_ylabel('Coupling \n $K$'); axs[2].grid(True)
axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)

line_S_acc, = axs[3].plot(time_init, S_acc_init, label='$S_{acc}(t)$', color='orange')
axs[3].set_ylabel('Accumulated \n Synchrony \n $S_{acc}$'); axs[3].set_xlabel('Time (s)'); axs[3].grid(True)
axs[3].legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)


axcolor = 'lightgoldenrodyellow'
param_keys = list(initial_params.keys())
param_ranges = {
    'delta_w': (0, 2), 'K_max': (0.5, 5), 'beta': (0, 2), 'delta': (0, 1), 'r': (0, 2),
    'alpha': (0, 2), 'G': (0, 2), 'thresh_cos': (0, 1), 'tau_sacc': (1, 50),
    'T_sync_duration': (10, 350), 'noise_strength_base': (0, 2), 'noise_strength_high': (0, 10)
}
sliders = []
for i, key in enumerate(param_keys):
    ax = fig.add_axes([0.20, 0.40 - i * 0.028, 0.65, 0.02], facecolor=axcolor)
    slider = Slider(
        ax=ax,
        label=key,
        valmin=param_ranges[key][0],
        valmax=param_ranges[key][1],
        valinit=initial_params[key]
    )
    sliders.append(slider)

update_button_ax = fig.add_axes([0.8, 0.015, 0.1, 0.04])
update_button = Button(update_button_ax, 'Update Plot', color=axcolor, hovercolor='0.975')

def update_on_click(event):
    updated_params = {key: s.val for key, s in zip(param_keys, sliders)}
    
    time_new, phi_new, x_new, K_new, S_acc_new, exit_time_new = run_simulation(**updated_params, dt=0.02)
    
    line_phi.set_data(time_new, phi_new)
    line_x.set_data(time_new, x_new)
    line_K.set_data(time_new, K_new)
    line_S_acc.set_data(time_new, S_acc_new)

    for ax in axs:
        ax.relim()
        ax.autoscale_view()
    
    fig.canvas.draw_idle()

update_button.on_clicked(update_on_click)

plt.show()