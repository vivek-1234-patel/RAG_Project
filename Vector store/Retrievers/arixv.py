import arxiv

# Create the ArXiv client
client = arxiv.Client()

# Search for research papers
search = arxiv.Search(
    query='all:"large language models"',
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance
)

# Fetch and print results
for i, paper in enumerate(client.results(search), start=1):
    print(f"\nPaper {i}:")
    print(f"Title: {paper.title}")
    print(f"Authors: {', '.join(author.name for author in paper.authors)}")
    print(f"Abstract: {paper.summary}")
    print(f"URL: {paper.entry_id}")