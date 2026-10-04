from langchain_community.vectorstores import Chroma

class EnhancedRetriever:
    """
    Implements advanced retrieval strategies to improve RAG quality.
    """
    
    def retrieve_and_rerank(self, vector_store, query, k, llm=None):
        """
        Uses Max Marginal Relevance (MMR) to fetch diverse documents
        instead of just the most similar ones (which might be repetitive).
        """
        # MMR tries to find documents that are relevant to the query 
        # BUT also different from each other.
        # This prevents the AI from getting 5 copies of the same paragraph.
        
        # We fetch 2x more candidates than needed, then select the top k diverse ones
        retrieved_docs = vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=k*2, 
            lambda_mult=0.7 # 0.7 = Focus more on relevance, 0.3 on diversity
        )
        
        return retrieved_docs