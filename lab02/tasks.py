"""Lab 2 tasks. Inputs and returned values are torch tensors (returned ones on the CPU, without gradients)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def categorical_1d(train_data, test_data, d):
    """
    Categorical distribution over d values, trained by minibatch gradient descent on NLL.

    train_data: (n_train,) integer tensor (uint8), values in {0, ..., d-1}
    test_data: (n_test,) integer tensor (uint8), values in {0, ..., d-1}
    d: number of values

    Returns
    - train_losses: (# iterations,) float tensor, training NLL of every minibatch
    - test_losses: (# epochs + 1,) float tensor, test NLL at initialisation and after every epoch
    - probs: (d,) float tensor, probs[k] = p(x = k), sums to 1

    Losses: mean NLL, nats per example
    """
    # TODO: YOUR CODE HERE

    lr = 0.05
    batch_size = 128
    epochs = 30

    phi = nn.Parameter(torch.zeros(d))
    optimizer = optim.Adam([phi], lr=lr)

    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)

    train_losses = []
    test_losses = []

    def compute_loss(data_batch, logits):
        expanded_logits = logits.unsqueeze(0).expand(data_batch.size(0), -1)
        return F.cross_entropy(expanded_logits, data_batch)

    def eval_nll(data, logits):
        with torch.no_grad():
            return compute_loss(data, logits).item()

    test_losses.append(eval_nll(test_data, phi))

    for epoch in range(epochs):
        epoch_train_losses = []

        for (batch,) in train_loader:
            loss = compute_loss(batch, phi)
            epoch_train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.extend(epoch_train_losses)
        test_losses.append(eval_nll(test_data, phi))

    with torch.no_grad():
        probs = torch.softmax(phi, dim=-1).cpu()

    return torch.tensor(train_losses), torch.tensor(test_losses), probs

        #raise NotImplementedError


def independent_2d(train_data, test_data, d, dset_id):
    """
    Independent variables, p(x0, x1) = p(x0) p(x1), trained by minibatch gradient descent on NLL.

    train_data: (n_train, 2) integer tensor (uint8), values in {0, ..., d-1}
    test_data: (n_test, 2) integer tensor (uint8), values in {0, ..., d-1}
    d: number of values of each variable
    dset_id: dataset index (1 or 2), e.g. for dataset-specific hyperparameters

    Returns
    - train_losses: (# iterations,) float tensor, training NLL of every minibatch
    - test_losses: (# epochs + 1,) float tensor, test NLL at initialisation and after every epoch
    - probs: (d, d) float tensor, probs[i, j] = p(x0 = i, x1 = j), sums to 1

    Losses: mean NLL, nats per example
    """
    # TODO: YOUR CODE HERE

    lr = 0.05
    batch_size = 256
    epochs = 30

    phi = nn.Parameter(torch.zeros(2, d))
    optimizer = optim.Adam([phi], lr=lr)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)

    train_losses = []
    test_losses =[]

    def compute_loss(data_batch, logits):
        x0 = data_batch[:, 0]
        x1 = data_batch[:, 1]
        B = x0.size(0)
        loss0 = F.cross_entropy(logits[0].unsqueeze(0).expand(B, -1), x0)
        loss1 = F.cross_entropy(logits[1].unsqueeze(0).expand(B, -1), x1)
        return loss0 + loss1

    def eval_nll(data, logits):
        with torch.no_grad():
            return compute_loss(data, logits).item()

    test_losses.append(eval_nll(test_data, phi))

    for epoch in range(epochs):
        epoch_losses = []
        for (batch, ) in train_loader:
            loss = compute_loss(batch, phi)
            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.extend(epoch_losses)
        test_losses.append(eval_nll(test_data, phi))


    with torch.no_grad():
        p = torch.softmax(phi, dim=-1)
        probs = p[0][:, None] * p[1][None, :]

    return torch.tensor(train_losses), torch.tensor(test_losses), probs.cpu()
    #raise NotImplementedError


def made_2d(train_data, test_data, d, dset_id):
    """
    Autoregressive model, p(x0, x1) = p(x0) p(x1 | x0), MADE (Germain et al., 2015), trained by minibatch gradient descent on NLL.

    train_data: (n_train, 2) integer tensor (uint8), values in {0, ..., d-1}
    test_data: (n_test, 2) integer tensor (uint8), values in {0, ..., d-1}
    d: number of values of each variable
    dset_id: dataset index (1 or 2), e.g. for dataset-specific hyperparameters

    Returns
    - train_losses: (# iterations,) float tensor, training NLL of every minibatch
    - test_losses: (# epochs + 1,) float tensor, test NLL at initialisation and after every epoch
    - probs: (d, d) float tensor, probs[i, j] = p(x0 = i, x1 = j), sums to 1

    Losses: mean NLL, nats per example
    """
    # TODO: YOUR CODE HERE
    #raise NotImplementedError

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 0.01
    batch_size = 256
    epochs = 20

    train_data = train_data.to(device).long()
    test_data  = test_data.to(device).long()

    phi0 = nn.Parameter(torch.zeros(d, device=device))

    cond_net = nn.Sequential(
        nn.Linear(d, 4 * d),
        nn.ReLU(),
        nn.Linear(4 * d, d), #25
    ).to(device)

    optimizer = optim.Adam([phi0] + list(cond_net.parameters()), lr=lr)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)

    train_losses, test_losses = [], []

    def compute_loss(data_batch):
        x0 = data_batch[:, 0]
        x1 = data_batch[:, 1]
        B = x0.size(0)
        logits_x0 = phi0.unsqueeze(0).expand(B, -1)        # (B, d), broadcasting trick
        logits_x1 = cond_net(F.one_hot(x0, d).float())     # (B, d), conditional branch
        return F.cross_entropy(logits_x0, x0) + F.cross_entropy(logits_x1, x1)

    def eval_nll(data):
        with torch.no_grad():
            return compute_loss(data).item()

    test_losses.append(eval_nll(test_data))

    for epoch in range(epochs):
        epoch_losses = []
        for (batch,) in train_loader:
            loss = compute_loss(batch)          # tensor with grad graph
            epoch_losses.append(loss.item())    # store as plain number
            optimizer.zero_grad()               # 1) clear
            loss.backward()                     # 2) compute
            optimizer.step()                    # 3) APPLY  (don't forget me!)
        train_losses.extend(epoch_losses)
        test_losses.append(eval_nll(test_data))

    # ---------- Assemble the (d, d) joint table via the chain rule ----------
    with torch.no_grad():
        p0 = torch.softmax(phi0, dim=-1)                        # (d,)
        all_x0 = torch.arange(d, device=device)
        all_onehot = F.one_hot(all_x0, d).float()               # (d, d): every possible x0
        p1_given = torch.softmax(cond_net(all_onehot), dim=-1)  # row i = p(x1 | x0=i)
        probs = (p0[:, None] * p1_given).cpu()                  # (d, d), sums to 1

    return torch.tensor(train_losses), torch.tensor(test_losses), probs



    


def made_images(train_data, test_data):
    """
    Autoregressive model of binary images, pixels in raster order, MADE (Germain et al., 2015), trained by minibatch gradient descent on NLL.

    train_data: (n_train, H, W) integer tensor (uint8), values in {0, 1}
    test_data: (n_test, H, W) integer tensor (uint8), values in {0, 1}

    Returns
    - train_losses: (# iterations,) float tensor, training NLL of every minibatch
    - test_losses: (# epochs + 1,) float tensor, test NLL at initialisation and after every epoch
    - samples: (100, H, W) integer tensor, new images sampled from trained model

    Losses: mean NLL, bits per pixel
    """
    # TODO: YOUR CODE HERE

    class MaskedLinear(nn.Linear):
        def __init__(self, in_features, out_features, mask=None, bias=True):
            super().__init__(in_features, out_features, bias)
            if mask is None:
                mask = torch.ones(out_features, in_features)
            self.register_buffer('mask', mask.float())

        def forward(self, x):
            return (F.linear(x, self.weight * self.mask, self.bias))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N, H, W = train_data.shape
    D = H * W

    if D == 400:
        H1, H2 = 256, 256
        lr, bs, epochs = 1e-3, 128, 50
    else:
        H1, H2 = 512, 512
        lr, bs, epochs = 1e-3, 128, 20

    d_in = torch.arange(D, device=device)
    d_out = torch.arange(D, device=device)

    d_h1 = torch.randint(0, D-1, (H1,), device=device)
    d_h2 = torch.randint(0, D-1, (H2,), device=device)

    m1 = (d_h1[:, None] >= d_in[None, :]).float()
    m2 = (d_h2[:, None] >= d_h1[None, :]).float()
    m3 = (d_out[:, None] > d_h2[None, :]).float()


    model = nn.Sequential(
        MaskedLinear(D, H1, mask=m1),
        nn.ReLU(),
        MaskedLinear(H1, H2, mask=m2),
        nn.ReLU(),
        MaskedLinear(H2, D, mask=m3),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_data = train_data.to(device)
    test_data = test_data.to(device)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=bs, shuffle=True)

    train_losses, test_losses = [], []

    def compute_nll_bpp(batch):
        B = batch.size(0)
        x = batch.view(B, D).float()
        logits = model(x)

        nll_nats = F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum(dim=1).mean()
        return nll_nats / (D * math.log(2))  # convert to bits per pixel

    def eval_loss(data):
        with torch.no_grad():
            return compute_nll_bpp(data).item()

    test_losses.append(eval_loss(test_data))

    for epoch in range(epochs):
        epoch_losses = []
        for (batch,) in train_loader:
            loss = compute_nll_bpp(batch)
            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.extend(epoch_losses)
        test_losses.append(eval_loss(test_data))
        print(f"Epoch {epoch+1}/{epochs} | Test BPP: {test_losses[-1]:.4f}")

    model.eval()
    num_samples = 100
    samples = torch.zeros(num_samples, D, device=device)

    with torch.no_grad():
        for i in range(D):
            logits = model(samples)
            prob_i = torch.sigmoid(logits[:, i])
            samples[:, i] = torch.bernoulli(prob_i)

    samples = samples.view(num_samples, H, W).to(device='cpu', dtype=torch.uint8)
    return torch.tensor(train_losses), torch.tensor(test_losses), samples

    
    #raise NotImplementedError


# TODO: your own helper functions and classes, if you want any
pass
