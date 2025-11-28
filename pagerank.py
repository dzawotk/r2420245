import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    total_pages = len(corpus)
    probabilities = dict()

    links = corpus[page]
    if links:
        for p in corpus:
            if p in links:
                probabilities[p] = damping_factor / len(links)
            else:
                probabilities[p] = (1 - damping_factor) / total_pages
    else:
        # If no links, treat it as linking to all pages
        for p in corpus:
            probabilities[p] = 1 / total_pages

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    page_counts = {page: 0 for page in corpus}
    pages = list(corpus.keys())

    # Start from a random page
    current_page = random.choice(pages)

    for _ in range(n):
        page_counts[current_page] += 1
        model = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(
            population=list(model.keys()),
            weights=list(model.values()),
            k=1
        )[0]

    # Normalize counts to get probabilities
    pagerank = {page: count / n for page, count in page_counts.items()}
    return pagerank

def iterate_pagerank(corpus, damping_factor):
    N = len(corpus)
    pagerank = {page: 1 / N for page in corpus}
    threshold = 0.001
    converged = False

    while not converged:
        new_ranks = {}
        converged = True

        for page in corpus:
            total = 0
            for possible_page in corpus:
                links = corpus[possible_page]
                if page in links:
                    total += pagerank[possible_page] / len(links)
                elif not links:
                    total += pagerank[possible_page] / N

            new_rank = (1 - damping_factor) / N + damping_factor * total
            if abs(new_rank - pagerank[page]) > threshold:
                converged = False
            new_ranks[page] = new_rank

        pagerank = new_ranks

    return pagerank



if __name__ == "__main__":
    main()
