from torch import nn


class Adapter(nn.Module):
    def __init__(self, dim=8, embed_dim=768, act_layer=nn.GELU):
        super().__init__()

        self.adapter_down = nn.Linear(embed_dim, dim)
        self.adapter_down_prompt = nn.Linear(embed_dim, dim)
        self.adapter_up = nn.Linear(dim, embed_dim)

        self.act = act_layer()

        self.dim = dim

    def forward(self, x, prompt):
        x_down = self.adapter_down(x) + self.adapter_down_prompt(prompt)
        x_down = self.act(x_down)
        x_up = self.adapter_up(x_down)
        return x_up
