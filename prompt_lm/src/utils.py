from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nn_project(curr_embeds, embedding_layer):

    
    seq_len,emb_dim = curr_embeds.shape
    
    # Using the sentence transformers semantic search which is 
    # a dot product exact kNN search between a set of 
    # query vectors and a corpus of vectors
    curr_embeds = curr_embeds.reshape((-1,emb_dim))
    curr_embeds = normalize_embeddings(curr_embeds) # queries

    embedding_matrix = embedding_layer.weight
    embedding_matrix = normalize_embeddings(embedding_matrix) # corpus
    
    hits = semantic_search(curr_embeds, embedding_matrix, 
                            query_chunk_size=curr_embeds.shape[0], 
                            top_k=3,
                            score_function=dot_score)

    nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=device)
    projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices