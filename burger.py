from first_net import *
from sampler import Sampler


# Time limits
T0 = 0.0 + 1e-10    # Initial time
T = 1.0            # Terminal time

# Space limits
S_1 = 0.0 + 1e-10    # Low boundary
S_2 = 1              # High boundary

# viscosity limits
V1 = 1e-2
V2 = 1e-1

# alpha ??
al1 = 1e-2
al2 = 1

# Boundary condition for x = 0
a1 = -1
a2 = 1

# Boundary condition for x = 1
b1 = -1
b2 = 1


# initial condition
def g(S): return S[:, 4] + S[:, 1] * (S[:, 5]-S[:, 4])


s1 = Sampler([T0, T, S_1, S_2, V1, V2, al1, al2, a1, a2, b1, b2],
             [0.5, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0])
s2 = Sampler([T0, T, S_1, S_1, V1, V2, al1, al2, a1, a2, b1, b2],
             [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
s3 = Sampler([T0, T, S_2, S_2, V1, V2, al1, al2, a1, a2, b1, b2],
             [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
s4 = Sampler([T0, T0, S_1, S_2, V1, V2, al1, al2, a1, a2, b1, b2],
             [0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0])


def Loss(model, S1, S2, S3, S4):
    # Each set contain [t,x,nu,alpha,a,b]

    # Loss term #1: PDE
    U = model(S1)
    DU = torch.autograd.grad(U.sum(), S1, create_graph=True, retain_graph=True)[0]
    U_xx = torch.autograd.grad(DU[:, 1].sum(), S1, create_graph=True, retain_graph=True)[0]

    f = DU[:, 0] - S1[:, 2] * U_xx[:, 1] + S1[:, 3] * U[:, 0] * DU[:, 1]
    L1 = torch.mean(torch.pow(f, 2))

    # Loss term #2: boundary condition x=0
    Ub1 = model(S2)
    L2 = torch.mean(torch.pow(Ub1 - S2[:, 4].reshape(Ub1.shape), 2))

    # Loss term #2: boundary condition x=1
    Ub2 = model(S3)
    L3 = torch.mean(torch.pow(Ub2 - S3[:, 5].reshape(Ub2.shape), 2))

    # Loss term #3: initial/terminal condition
    Ui = model(S4)
    CI = g(S4)
    L4 = torch.mean(torch.pow((Ui - CI.reshape(Ui.shape)), 2))

    return L1, L2, L3, L4


model = Net(6, 1, 200, 6)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.99)

# Number of samples
NS_1 = 1000
NS_2 = 0
NS_3 = 100

# Training parameters
steps_per_sample = 10
sampling_stages = 800


def train_model(model, optimizer, scheduler, num_epochs=100):
    since = time.time()
    model.train()
    # Set model to training mode

    for epoch in range(num_epochs):
        sample1 = torch.tensor(s1.get_sample(NS_1), requires_grad=True)
        sample2 = torch.tensor(s2.get_sample(NS_3))
        sample3 = torch.tensor(s3.get_sample(NS_3))
        sample4 = torch.tensor(s4.get_sample(NS_3))
        scheduler.step()

        for _ in range(steps_per_sample):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            L1, L2, L3, L4 = Loss(model, sample1, sample2, sample3, sample4)

            loss = L1 + L2 + L3 + L4
            # backward + optimize
            #print_graph(loss.grad_fn,0)
            loss.backward()
            optimizer.step()

        epoch += 1
        if epoch % (num_epochs//num_epochs) == 0: print(f'epoch {epoch}, loss {loss.data}, L1 : {L1.data}, L2 : {L2.data},'
                                                        f' L3 : {L3.data}, L4 : {L4.data}')
    time_elapsed = time.time() - since
    print(f"Training finished in {time_elapsed:.2f} for {num_epochs}.")
    print(f"The final loss value is {loss.data}")

train_model(model, opt, scheduler, sampling_stages)