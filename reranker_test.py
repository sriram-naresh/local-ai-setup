from sentence_transformers import CrossEncoder


# Load reranker model
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


question = "What technologies do I have experience with?"


documents = [
    "JIRA is used for project and issue tracking.",
    "Kubernetes, Amazon EKS, Terraform, Jenkins, Helm, Docker and Argo CD are technologies used for platform engineering.",
    "Terraform is an Infrastructure as Code tool.",
    "Amazon EKS is a managed Kubernetes service provided by AWS.",
]


pairs = [
    [question, document]
    for document in documents
]


scores = model.predict(pairs)


print("\nQuestion:")
print(question)


print("\nReranked results:")


results = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)


for rank, (document, score) in enumerate(results, start=1):

    print(f"\n--- Rank {rank} ---")
    print("Score:", score)
    print("Document:", document)