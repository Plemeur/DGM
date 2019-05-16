from first_net import *
from sampler import Sampler


# PDE parameters
r = torch.tensor(0.05)            # Interest rate
sigma = torch.tensor(0.25)        # Volatility
mu = torch.tensor(0.2)            # Mean
lambd = (mu-r)/sigma
gamma = torch.tensor(1)           # Utility decay

# Time limits
T0 = 0.0 + 1e-10    # Initial time
T  = 1.0            # Terminal time

# Space limits
S1 = 0.0 + 1e-10    # Low boundary
S2 = 1              # High boundary


# Merton's analytical known solution
def analytical_solution(t, x):
    return -np.exp(-x*gamma*np.exp(r*(T-t)) - (T-t)*0.5*lambd**2)


def analytical_dVdx(t,x):
    return gamma*np.exp(-0.5*(T-t)*lambd**2 + r*(T-t))*np.exp(-x*gamma*np.exp(r*(T-t)))


def analytical_dV2dxx(t,x):
    return -gamma**2*np.exp(-0.5*(T-t)*lambd**2 + 2*r*(T-t))*np.exp(-x*gamma*np.exp(r*(T-t)))


# Merton's final utility function
def utility(x):
    return -torch.exp(-gamma*x)


s1 = Sampler([T0, T, S1, S2], [0.5, 0, 0.5, 0.5])
s2 = Sampler([T, T, S1, S2], [0, 0, 0.5, 0.5])

model = Net(2, 1, 50, 3)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.99)

# Number of samples
NS_1 = 1000
NS_2 = 0
NS_3 = 100

# Training parameters
steps_per_sample = 10
sampling_stages = 800


def print_graph(g, level=0):
    if g == None: return
    print('*'*level, g)
    for subg in g.next_functions:
        print_graph(subg[0], level+1)


def Loss(model, sample1, sample2):
    # Loss term #1: PDE
    V = model(sample1)
    V_p = torch.autograd.grad(V.sum(), sample1, create_graph=True, retain_graph=True)[0]
    V_s = torch.autograd.grad(V_p[:, 1].sum(), sample1, create_graph=True, retain_graph=True)[0]

    f = -0.5 * torch.pow(lambd, 2) * V_p[:, 1] ** 2 + (V_p[:, 0] + r * sample1[:, 1] * V_p[:, 1]) * V_s[:, 1]

    #print(V_t, V_x, V_xx)
    L1 = torch.mean(torch.pow(f, 2))
    #print_graph(L1.grad_fn, 0)
    # Loss term #2: boundary condition
    L2 = 0.0

    # Loss term #3: initial/terminal condition
    u = model(sample2)
    g = u - utility(sample2[:, 1]).reshape(u.shape)
    L3 = torch.mean(torch.pow(g, 2))
    #print_graph(L3.grad_fn, 0)

    return L1, L2, L3


def train_model(model, optimizer, scheduler, num_epochs=100):
    since = time.time()
    model.train()
    # Set model to training mode

    for epoch in range(num_epochs):
        sample1 = torch.tensor(s1.get_sample(NS_1), requires_grad=True)
        sample2 = torch.tensor(s2.get_sample(NS_3))
        scheduler.step()

        for _ in range(steps_per_sample):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            L1, L2, L3 = Loss(model, sample1, sample2)

            loss = L1 + L2 + L3
            # backward + optimize
            #print_graph(loss.grad_fn,0)
            loss.backward()
            optimizer.step()

        epoch += 1
        if epoch % (num_epochs//num_epochs) == 0: print(f'epoch {epoch}, loss {loss.data}, L1 : {L1.data}, L3 : {L3.data}')
    time_elapsed = time.time() - since
    print(f"Training finished in {time_elapsed:.2f} for {num_epochs}.")
    print(f"The final loss value is {loss.data}")

train_model(model, opt, scheduler, sampling_stages)

# Plot results
N = 41  # Points on plot grid

times_to_plot = [0 * T, 0.33 * T, 0.66 * T, T]
tplot = np.linspace(T0, T, N)
xplot = np.linspace(S1, S2, N)

test = torch.tensor([0.,0.])
print(model(test))