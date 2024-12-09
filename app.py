from flask import Flask, render_template, request
import os
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from sklearn.decomposition import PCA
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and preprocessing tools
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# Load image embeddings and perform PCA
EMBEDDINGS_FILE = "image_embeddings.pickle"
IMAGES_FOLDER = "static/coco_images_resized"
df = pd.read_pickle(EMBEDDINGS_FILE)
original_embeddings = np.vstack(df["embedding"].to_numpy())  # Original 512-dim embeddings

# Precompute PCA embeddings with the maximum number of components
pca_model = PCA(n_components=512)  # Precompute full PCA
pca_embeddings = pca_model.fit_transform(original_embeddings)  # Store PCA-transformed embeddings
df["embedding_pca"] = list(pca_embeddings)  # Store precomputed PCA embeddings in the dataframe

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        # Query parameters
        query_type = request.form.get("query_type")  # text, image, or hybrid
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5))  # Weight for hybrid queries
        use_pca = "use_pca" in request.form  # Checkbox for PCA
        k_principal_components = int(request.form.get("k_principal_components", 50))  # Default PCA components
        text_query = request.form.get("text_query")  # Text query
        image_query = request.files.get("image_query")  # Image query

        # Validate PCA component count
        if use_pca and k_principal_components > 512:
            k_principal_components = 512  # Limit to precomputed maximum components

        # Initialize query_embedding
        query_embedding = None

        # Process text query
        if query_type in ["text", "hybrid"]:
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            text_tokens = tokenizer([text_query])
            with torch.no_grad():
                text_embedding = F.normalize(model.encode_text(text_tokens).to(device), p=2, dim=-1)

        # Process image query
        if query_type in ["image", "hybrid"]:
            if image_query:
                image = preprocess(Image.open(image_query).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_embedding = F.normalize(model.encode_image(image), p=2, dim=-1)

                # Apply PCA to the image embedding if requested
                if use_pca:
                    image_embedding_pca = pca_model.transform(image_embedding.cpu().numpy())[:, :k_principal_components]
                    image_embedding = torch.tensor(image_embedding_pca, device=device, dtype=torch.float32)

        # Handle hybrid query
        if query_type == "hybrid" and "text_embedding" in locals() and "image_embedding" in locals():
            query_embedding = F.normalize(hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding, p=2, dim=-1)
        elif query_type == "text":
            query_embedding = text_embedding
        elif query_type == "image":
            query_embedding = image_embedding

        # Use appropriate database embeddings
        if use_pca:
            database_embeddings = np.vstack(df["embedding_pca"].to_numpy())[:, :k_principal_components]
            database_embeddings = torch.tensor(database_embeddings, device=device, dtype=torch.float32)
        else:
            database_embeddings = torch.tensor(original_embeddings, device=device, dtype=torch.float32)

        # Normalize database embeddings
        database_embeddings = F.normalize(database_embeddings, p=2, dim=1)  # Normalize row-wise

        # Compute cosine similarities
        query_embedding = query_embedding.squeeze(0)  # Ensure proper shape
        cos_similarities = torch.matmul(database_embeddings, query_embedding.T).squeeze().tolist()

        # Retrieve top 5 results
        top_indices = np.argsort(-np.array(cos_similarities))[:5]
        results = [{"file_name": os.path.join(IMAGES_FOLDER, df.iloc[idx]["file_name"]), "similarity": cos_similarities[idx]} for idx in top_indices]

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
