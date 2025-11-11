# --- Imports ---
import os
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI

# --- Step 1: Initialize OpenAI Client ---
# Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- Step 2: Create Sample Product Dataset ---
df = pd.DataFrame({
    "product_id": [1, 2, 3, 4],
    "product_name": [
        "Classic Leather Bag",
        "Urban Street Sneakers",
        "Elegant Silk Scarf",
        "Rugged Hiking Boots"
    ],
    "desc": [
        "A timeless brown leather handbag perfect for daily use.",
        "Trendy sneakers designed for energetic urban style.",
        "Luxury silk scarf adding elegance to any outfit.",
        "Durable boots built for mountain trails and outdoor adventure."
    ]
})

# --- Step 3: Create Embeddings for Product Descriptions ---
print(" Creating embeddings for product descriptions...")
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=df["desc"].tolist()
)
embeddings = np.array([item.embedding for item in response.data]).astype("float32")
print(f" Created embeddings with shape: {embeddings.shape}")

# --- Step 4: Build FAISS Index ---
dimension = embeddings.shape[1]  # Should be 1536 for ada-002
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f" FAISS index built with {index.ntotal} items")

# Step 5: Define Semantic Search Function ---
def search_products(query, top_k=3):
    """Search for the most semantically similar products."""
    # Create embedding for query
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    ).data[0].embedding

    query_vector = np.array([query_embedding]).astype("float32")
     
    # Search in FAISS index
    distances, indices = index.search(query_vector, top_k)

    # Collect and format results
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "product_name": df.iloc[idx]["product_name"],
            "description": df.iloc[idx]["desc"],
            "distance": float(distances[0][rank])
        })
    return results

# --- Step 6: Example Query ---
query = "lightweight outdoor travel shoes"
print(f"\n🔍 Searching for: '{query}' ...\n")
results = search_products(query, top_k=3)

# --- Step 7: Display Results ---
for r in results:
    print(f"{r['rank']}. {r['product_name']} (distance={r['distance']:.4f})")
    print(f"   {r['description']}\n")

# --- Step 8: Save Data and Index for Future Use ---
faiss.write_index(index, "product_index.faiss")
np.save("product_embeddings.npy", embeddings)
df.to_csv("products.csv", index=False)
print("Index, embeddings, and dataset saved successfully!")
