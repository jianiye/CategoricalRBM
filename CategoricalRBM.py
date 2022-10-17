import torch
from torch import nn

class CategoricalRBM(nn.Module):

    def __init__(self, D, M, K):
        '''
        Categorical RBM has a visible part of D features, each feature is a K size one-hot tensor.
        Hidden part has size M, each is a Bernoulli variable.
        '''
        super(CategoricalRBM, self).__init__()
        self.W = nn.Parameter(torch.randn(D, K, M))
        self.c = nn.Parameter(torch.randn(M))
        self.b = nn.Parameter(torch.randn(D, K))
        self.hact = nn.Sigmoid()
        self.vact = nn.Softmax(dim=-1)
        self.splus = nn.Softplus()
        self.refloss = nn.CrossEntropyLoss()


    def free_energy(self, V):
        f1 = torch.mean(torch.tensordot(V, self.b, dims=([1,2],[0,1])))
        f2 = torch.mean(torch.sum(self.splus(torch.tensordot(V, self.W, dims=([1,2],[0,1])) + self.c), dim = 1))
        return f1 + f2

    def forward(self, Visible, Mask):
        '''
        Visible is an N*D*K tensor, N: sample size; D:feature size(i.e., if Visible is user, then feature is movie or item);
        K: Category size.
        '''
        # from visible to hidden
        HofV = torch.tensordot(Visible, self.W, dims=([1,2],[0,1])) + self.c
        HofV = self.hact(HofV)
        HofV = torch.bernoulli(HofV)

        # from hidden to Visible
        VofH = torch.tensordot(HofV, self.W, dims = ([1],[2])) + self.b
        VofH = self.vact(VofH)
        Vcate = torch.distributions.categorical.Categorical(VofH)
        VofH = Vcate.sample()
        VofH = nn.functional.one_hot(VofH)
        Mask = torch.unsqueeze(Mask, dim=-1)
        refloss = self.refloss(VofH.detach()*Mask.detach(), Visible.detach()) # cross entropy loss for reference
        VofH = VofH*Mask

        return VofH, refloss


def train_loop(model, train_dataloader, epoch, lr=0.05, eval_dataloader=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = 0.0
    ref_loss = 0.0
    model.train()

    for i, data in enumerate(train_dataloader):
        visible, mask = data
        if torch.cuda.is_available():
            visible, mask = visible.cuda(), mask.cuda()

        optimizer.zero_grad()
        visible, mask = visible.float(), mask.float()
        output, refloss = model(visible, mask)
        loss = torch.abs(model.free_energy(visible) - model.free_energy(output))
        #loss = loss_func(target, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        ref_loss += refloss

    if eval_dataloader:
        valid_loss = 0.0
        model.eval()     # Optional when not using model Specific layer
        for data in eval_dataloader:
            visible, mask = data
            if torch.cuda.is_available():
                visible, mask = visible.cuda(), mask.cuda()

            output, refloss = model(visible, mask)
            loss = model.free_energy(visible) - model.free_energy(output)
            valid_loss += loss.item() * len(inputs)

    if epoch%1==0:
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss, train_loss / len(train_dataloader)} \t\t Cross Entropy Loss: {ref_loss, ref_loss / len(train_dataloader)}')
        if eval_dataloader:
            print(f'\t\t Validation Loss: {valid_loss, valid_loss / len(eval_dataloader)}')
            if min_valid_loss > valid_loss:
                print(f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})")
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(model.state_dict(), 'saved_model.pth')
    return model
