# cat_embed_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CatEmbeddingMLP(nn.Module):
    def __init__(self, df, cat_cols, embed_dims, num_cols, hidden_dim=16):
        super().__init__()
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for c in cat_cols:
            # In production, you might pass the cardinalities directly, 
            # or infer from some known metadata instead of using df
            cardinality = df[c].nunique()  
            emb_dim = embed_dims.get(c, 4)
            emb = nn.Embedding(cardinality, emb_dim)
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings.append(emb)
            total_emb_dim += emb_dim
        
        self.total_emb_dim = total_emb_dim
        self.num_numeric = len(num_cols)
        
        input_dim = self.total_emb_dim + self.num_numeric
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, X_cat, X_num):
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_out = emb_layer(X_cat[:, i])
            emb_list.append(emb_out)
        cat_emb = torch.cat(emb_list, dim=1)
        
        x = torch.cat([cat_emb, X_num], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)

    def extract_row_embedding(self, X_cat, X_num):
        """Get the final hidden layer as a row-level embedding."""
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_out = emb_layer(X_cat[:, i])
            emb_list.append(emb_out)
        cat_emb = torch.cat(emb_list, dim=1)
        
        x = torch.cat([cat_emb, X_num], dim=1)
        x = F.relu(self.fc1(x))
        return x