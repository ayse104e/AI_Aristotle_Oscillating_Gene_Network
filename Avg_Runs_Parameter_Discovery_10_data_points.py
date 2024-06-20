import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import time
from scipy.integrate import odeint

# Constants
k1 = 0.5
k2 = 0.5
k3 = 0.5
gamma1 = 0.1 
gamma2 = 0.01 
gamma3 = 0.1 
c = 1 
n = 9
tmin, tmax = 0, 50

def drug_model(t, k1=0.5, k2=0.5, k3=0.5, gamma1=0.1, gamma2=0.01, gamma3=0.1, c=1, n=9):
    def func(y, t):
        G, B, U = y[0], y[1], y[2]
        return [
            ((c**n)/((c**n) + (U**n)) * k1) - gamma1 * G,
            ((G**n)/((c**n) + (G**n)) * k2) - gamma2 * B,
            ((B**n)/((c**n) + (B**n)) * k3) - gamma3 * U,
        ]
    y0 = [0, 0, 0]
    return odeint(func, y0, t)

def init_params(layers, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        W = jax.random.normal(key, shape=(n_in, n_out)) / jnp.sqrt(n_in)
        B = jax.random.normal(key, shape=(n_out,))
        k1 = jax.random.uniform(key, minval=0.0, maxval=1.0)
        k2 = jax.random.uniform(key, minval=0.0, maxval=1.0)
        k3 = jax.random.uniform(key, minval=0.0, maxval=1.0)
        gamma1 = jax.random.uniform(key, minval=0.0, maxval=0.5)
        gamma2 = jax.random.uniform(key, minval=0.0, maxval=0.1)
        gamma3 = jax.random.uniform(key, minval=0.0, maxval=0.5)
        params.append({'W': W, 'B': B, 'k1': k1, 'k2': k2, 'k3': k3, 'gamma1': gamma1, 'gamma2': gamma2, 'gamma3': gamma3})
    return params

def fwd(params, t):
    X = jnp.concatenate([t], axis=1)
    *hidden, last = params
    for layer in hidden:
        X = jax.nn.tanh(X @ layer['W'] + layer['B'])
    return X @ last['W'] + last['B']

@jax.jit
def MSE(true, pred):
    return jnp.mean((true - pred) ** 2)

def ODE_loss(t, params, y1, y2, y3):
    k1, k2, k3 = params[-1]['k1'], params[-1]['k2'], params[-1]['k3']
    gamma1, gamma2, gamma3 = params[-1]['gamma1'], params[-1]['gamma2'], params[-1]['gamma3']
    c, n = 1, 9

    y1_t = jax.grad(lambda t: jnp.sum(y1(t)), argnums=0)
    y2_t = jax.grad(lambda t: jnp.sum(y2(t)), argnums=0)
    y3_t = jax.grad(lambda t: jnp.sum(y3(t)), argnums=0)
    
    ode1 = y1_t(t) - ((c**n / (c**n + y3(t)**n) * k1) - gamma1 * y1(t))
    ode2 = y2_t(t) - ((y1(t)**n / (c**n + y1(t)**n) * k2) - gamma2 * y2(t))
    ode3 = y3_t(t) - ((y2(t)**n / (c**n + y2(t)**n) * k3) - gamma3 * y3(t))

    return ode1, ode2, ode3

def loss_fun(params, t_i, t_d, t_c, data_IC, data):
    y1_func = lambda t: fwd(params, t)[:, [0]]
    y2_func = lambda t: fwd(params, t)[:, [1]]
    y3_func = lambda t: fwd(params, t)[:, [2]]

    loss_y1, loss_y2, loss_y3 = ODE_loss(t_c, params, y1_func, y2_func, y3_func)

    loss_ode1 = jnp.mean(loss_y1 ** 2)
    loss_ode2 = jnp.mean(loss_y2 ** 2)
    loss_ode3 = jnp.mean(loss_y3 ** 2)

    # Compute the loss for the initial conditions
    t_i = t_i.flatten()[:, None]
    pred_IC = jnp.concatenate([y1_func(t_i), y2_func(t_i), y3_func(t_i)], axis=1)
    loss_IC = MSE(data_IC, pred_IC)

    # Compute the loss for Y_data
    t_d = t_d.flatten()[:, None]
    pred_d = jnp.concatenate([y1_func(t_d), y2_func(t_d), y3_func(t_d)], axis=1)
    loss_data = MSE(data, pred_d)

    return loss_IC + loss_data + loss_ode1 + loss_ode2 + loss_ode3

@jax.jit
def update(opt_state, params, t_i, t_data, t_c, IC, data):
    grads = jax.grad(loss_fun)(params, t_i, t_data, t_c, IC, data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

def run_simulation(seed):
    # Initialize lists to store values
    loss_values = []
    k1_values_list = []
    k2_values_list = []
    k3_values_list = []
    gamma1_values_list = []
    gamma2_values_list = []
    gamma3_values_list = []
    epoch_numbers = []

    params = init_params([1] + [20] * 4 + [3], seed)
    opt_state = optimizer.init(params)

    epochs = 50000
    start_time = time.time()

    t_i  = jnp.array([[0]])

    for ep in range(epochs):
        opt_state, params = update(opt_state, params, t_i, t_data, t_c, IC, data)

        if ep % 1000 == 0:
            loss = loss_fun(params, t_i, t_data, t_c, IC, data)
            loss_values.append(loss)
            k1_updated = params[-1]['k1']
            k2_updated = params[-1]['k2']
            k3_updated = params[-1]['k3']
            gamma1_updated = params[-1]['gamma1']
            gamma2_updated = params[-1]['gamma2']
            gamma3_updated = params[-1]['gamma3']

            print(f'Epoch={ep} \t loss={loss:.3e} \t k1= {k1_updated} \t k2={k2_updated} \t k3= {k3_updated} \t gamma1={gamma1_updated}\t gamma2= {gamma2_updated} \t gamma3={gamma3_updated}')

            k1_values_list.append(k1_updated)
            k2_values_list.append(k2_updated)
            k3_values_list.append(k3_updated)
            gamma1_values_list.append(gamma1_updated)
            gamma2_values_list.append(gamma2_updated)
            gamma3_values_list.append(gamma3_updated)
            epoch_numbers.append(ep)

            end_time = time.time()
            running_time = end_time - start_time
            print(f"Total running time: {running_time:.4f} seconds")
    
    return k1_updated, k2_updated, k3_updated, gamma1_updated, gamma2_updated, gamma3_updated

# Set up initial conditions and data
t_dense = jnp.linspace(0, 50, 501)[:, None]
y_dense = drug_model(np.ravel(t_dense))
sample_rate = 50
t_data = t_dense[::sample_rate, 0:1]
G_data = y_dense[::sample_rate, 0:1]
B_data = y_dense[::sample_rate, 1:2]
U_data = y_dense[::sample_rate, 2:3]
data = jnp.concatenate([G_data, B_data, U_data], axis=1)

t_IC = jnp.array([[0]])
IC = jnp.array([[0, 0, 0]])

t_c = jnp.linspace(tmin, tmax, 501)[:, None]

optimizer = optax.adam(1e-4)

# Run the simulation 10 times with different seeds
final_k1_values = []
final_k2_values = []
final_k3_values = []
final_gamma1_values = []
final_gamma2_values = []
final_gamma3_values = []
used_seeds = []

for i in range(10):
    seed = np.random.randint(0, 10000)
    used_seeds.append(seed)
    print(f'################Run the simulation {i} time with different seed: {seed} ##################')
    k1, k2, k3, gamma1, gamma2, gamma3 = run_simulation(seed)
    final_k1_values.append(np.abs(0.5 - k1))
    final_k2_values.append(np.abs(0.5 - k2))
    final_k3_values.append(np.abs(0.5 - k3))
    final_gamma1_values.append(np.abs(0.1 - gamma1))
    final_gamma2_values.append(np.abs(0.01 - gamma2))
    final_gamma3_values.append(np.abs(0.1 - gamma3))
    print(f'k1 absolute error: {final_k1_values}')
    print(f'k2 absolute error: {final_k2_values}')
    print(f'k3 absolute error: {final_k3_values}')
    print(f'gamma1 absolute error: {final_gamma1_values}')
    print(f'gamma2 absolute error: {final_gamma2_values}')
    print(f'gamma3 absolute error: {final_gamma3_values}')



final_k1_values = np.array([value for value in final_k1_values if not np.isnan(value)])
final_k2_values = np.array([value for value in final_k2_values if not np.isnan(value)])
final_k3_values = np.array([value for value in final_k3_values if not np.isnan(value)])
final_gamma1_values = np.array([value for value in final_gamma1_values if not np.isnan(value)])
final_gamma2_values = np.array([value for value in final_gamma2_values if not np.isnan(value)])
final_gamma3_values = np.array([value for value in final_gamma3_values if not np.isnan(value)])

# Calculate and print the average
average_k1 = np.mean(final_k1_values)
average_k2 = np.mean(final_k2_values)
average_k3 = np.mean(final_k3_values)
average_gamma1 = np.mean(final_gamma1_values)
average_gamma2 = np.mean(final_gamma2_values)
average_gamma3 = np.mean(final_gamma3_values)

print(f'Average k1 absolute error: {average_k1}')
print(f'Average k2 absolute error: {average_k2}')
print(f'Average k3 absolute error: {average_k3}')
print(f'Average gamma1 absolute error: {average_gamma1}')
print(f'Average gamma2 absolute error: {average_gamma2}')
print(f'Average gamma3 absolute error: {average_gamma3}')
