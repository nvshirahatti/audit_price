# main.py
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
# Import your custom embedding model definition
from cat_embed_model import CatEmbeddingMLP
import xgboost as xgb
import uvicorn
app = FastAPI()

address_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Define a request schema
class AuditRequest(BaseModel):
    address: str
    jurisdiction: str
    customer: str
    primary_use_type: str
    building_area: float

######################################
# 3. Global variables to store models/artifacts
######################################
model_embed = None     # PyTorch embedding model
final_model = None     # e.g. XGBoost or any other final regressor
label_encoders = None
cat_cols2 = ["Customer","Primary Use Type"]
num_cols = ["Building Area"]

######################################
# 4. On startup, load all models/artifacts
######################################
@app.on_event("startup")
def load_artifacts():
    global model_embed, final_model, label_encoders
    
    print("Loading label encoders")
    # A) Load label encoders
    label_encoders = joblib.load("model/label_encoders.pkl")  
    # e.g. label_encoders["customer"], label_encoders["primary_use_type"]
    print("Finished Loading label encoders")
    # B) Re-instantiate the embedding model
    #    We need a df or a known cardinality to init. 
    #    Typically you'd have a reference data or metadata for cardinalities.
    #    For a quick hack, let's pass a minimal "df" just for shape (but carefully).
    print("Loading model embedding")
    import pandas as pd
    df = pd.read_csv("data/audit_price.tsv", sep="\t", header=0, on_bad_lines='skip')
    df = df.dropna()

    df["Audit Price"] = df['Audit Price'].str.replace(',', '')
    df["Audit Price"] = df['Audit Price'].str.replace('$', '')
    df["Audit Price"] = df['Audit Price'].astype(float)
    df["Building Area"] = df['Building Area'].str.replace(',', '')
    df["Building Area"] = df['Building Area'].str.replace('sq ft', '')
    df["Building Area"] = df['Building Area'].astype(float)
    df['Address'] = df['Jurisdiction']  + " " + df['Address'] 
    from sentence_transformers import SentenceTransformer
    print("DF created")
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    df['Address_Embed'] = df['Address'].apply(lambda x: model_st.encode(x))
    print("Sentence embedding added")
    cat_cols = ['Customer', 'Audit Level', 'Primary Use Type']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    print("Loading cat embedding")
    embed_dims = {'Customer': 4, 'Primary Use Type': 4}
    model_embed = CatEmbeddingMLP(
        df=df, 
        cat_cols=cat_cols2, 
        embed_dims=embed_dims, 
        num_cols=num_cols, 
        hidden_dim=16
    )
    # Load state dict (the weights)
    state_dict = torch.load("model/embedding_cat_audit.pt", weights_only=True)
    model_embed.load_state_dict(state_dict)
    model_embed.eval()
    
    print("Finished Loading model embedding")
    # C) Load final regression model
    #    E.g. an XGBoost model saved with joblib
    final_model = joblib.load("model/final_regressor.pkl")  
    print("All artifacts loaded successfully!")


@app.post("/predict")
def predict(request: AuditRequest):
    # 1. Address embedding
    addr_text = request.address + " " + request.jurisdiction
    addr_emb = address_model.encode([addr_text])  # shape (1, 384)
    
    # 2. Label encode the cat columns
    default_cust_id, default_put_id = 0, 0
    if request.customer in label_encoders["Customer"].classes_:
        cust_id = label_encoders["Customer"].transform([request.customer])[0]
    else:
        cust_id = default_cust_id  # Assign a default or unknown class value
    if request.primary_use_type in label_encoders["Primary Use Type"].classes_:
        put_id = label_encoders["Primary Use Type"].transform([request.primary_use_type])[0]
    else:
        put_id = default_put_id  
    
    # 3. PyTorch embedding model forward pass
    X_cat = torch.tensor([[cust_id, put_id]], dtype=torch.long)  # shape (1,2)
    X_num = torch.tensor([[request.building_area]], dtype=torch.float)
    row_vec = model_embed.extract_row_embedding(X_cat, X_num).detach().numpy()  # shape (1, hidden_dim)
    
    # 4. Combine row_vec + addr_emb + numeric if needed
    X_final = np.concatenate([row_vec, addr_emb, [[request.building_area]]], axis=1)
    
    # 5. Predict with final regressor
    pred = final_model.predict(X_final)  # shape: (1,)
    return {"audit_price_pred": float(pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")