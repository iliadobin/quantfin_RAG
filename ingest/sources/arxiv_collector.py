"""
arXiv document collector using arXiv API.
Builds allowlist of papers matching search criteria.
"""
import arxiv
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class ArxivCollector:
    """Collect arXiv papers matching search criteria."""
    
    def __init__(self, config_path: str):
        """Initialize with search configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.categories = self.config['categories']
        self.search_groups = self.config['search_groups']
        self.filters = self.config['filters']
        self.target_count = self.config['target_count']
        self.max_total_pages = self.config['max_total_pages']
    
    def build_query(self, group_name: str, group_config: dict) -> str:
        """Build arXiv API query for a search group."""
        # Category filter
        cat_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
        
        # Keywords (OR within group)
        kw_parts = [f'(ti:"{kw}" OR abs:"{kw}")' for kw in group_config["keywords"]]
        kw_query = " OR ".join(kw_parts)
        
        # Additional filters (AND)
        add_filters = group_config.get('additional_filters', [])
        if add_filters:
            add_query = " AND ".join([f'(ti:"{f}" OR abs:"{f}")' for f in add_filters])
            kw_query = f"({kw_query}) AND ({add_query})"
        
        # Final query
        query = f"({cat_query}) AND ({kw_query})"
        return query
    
    def is_valid_paper(self, result: arxiv.Result) -> tuple[bool, Optional[str]]:
        """Check if paper matches cheap filters. Page count is validated after download."""
        # Exclude keywords (fast)
        exclude_kw = self.filters.get('exclude_keywords', [])
        title_abs = (result.title + " " + result.summary).lower()
        for kw in exclude_kw:
            if kw.lower() in title_abs:
                return False, f"excluded_keyword_{kw}"
        
        return True, None

    def _score_paper(self, result: arxiv.Result, group_name: str) -> int:
        """Heuristic score: prefer tutorial/lecture/survey + topic coverage."""
        text = (result.title + " " + result.summary).lower()
        score = 0

        # Prefer "tutorial/lecture/survey" but do NOT require
        for kw in self.filters.get("prefer_keywords", []):
            if kw.lower() in text:
                score += 3

        # Reward matches of group keywords
        for kw in self.search_groups[group_name]["keywords"]:
            if kw.lower() in text:
                score += 1

        # Slight penalty for very short abstracts (often low-signal)
        if len(result.summary) < 800:
            score -= 1
        return score
    
    def search_group(self, group_name: str, group_config: dict) -> List[Dict]:
        """Search arXiv for a specific topic group."""
        query = self.build_query(group_name, group_config)
        max_results = group_config.get('max_results', 5)
        
        print(f"\n[{group_name}] Searching with query (first {max_results} valid)...")
        print(f"  Query: {query[:150]}...")
        
        client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
        search = arxiv.Search(
            query=query,
            max_results=max(50, max_results * 15),  # fetch enough to score
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        candidates = []
        for result in client.results(search):
            valid, reason = self.is_valid_paper(result)
            if not valid:
                continue
            
            # Extract metadata
            score = self._score_paper(result, group_name)
            paper = {
                'arxiv_id': result.entry_id.split('/abs/')[-1],
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'primary_category': result.primary_category,
                'categories': result.categories,
                'published': result.published.isoformat(),
                'summary': result.summary[:500],  # truncate for storage
                'pdf_url': result.pdf_url,
                'group': group_name,
                'score': score,
            }

            candidates.append(paper)
        
        # Sort by score and pick top-N
        candidates.sort(key=lambda p: p.get("score", 0), reverse=True)
        selected = candidates[:max_results]
        for paper in selected:
            print(f"  ✓ {paper['arxiv_id']} (score={paper['score']}): {paper['title'][:60]}...")

        print(f"  Found {len(selected)} selected papers for {group_name} (from {len(candidates)} candidates)")
        return selected
    
    def collect_allowlist(self) -> List[Dict]:
        """Collect papers from all search groups."""
        print("=" * 80)
        print("arXiv Allowlist Collection")
        print("=" * 80)
        
        all_papers = []
        for group_name, group_config in self.search_groups.items():
            papers = self.search_group(group_name, group_config)
            all_papers.extend(papers)
        
        # Deduplicate by arxiv_id
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper['arxiv_id'] not in seen_ids:
                seen_ids.add(paper['arxiv_id'])
                unique_papers.append(paper)
        
        print(f"\n{'='*80}")
        print(f"Total unique papers: {len(unique_papers)}")
        
        # Limit to target count
        if len(unique_papers) > self.target_count:
            print(f"Trimming to target count: {self.target_count}")
            unique_papers = unique_papers[:self.target_count]
        
        return unique_papers
    
    def save_allowlist(self, papers: List[Dict], output_path: str):
        """Save allowlist to YAML."""
        allowlist = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'total_count': len(papers),
            'papers': papers
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(allowlist, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nAllowlist saved to: {output_file}")


if __name__ == "__main__":
    # Run collection
    collector = ArxivCollector("configs/arxiv_search_config.yaml")
    papers = collector.collect_allowlist()
    collector.save_allowlist(papers, "configs/arxiv_allowlist.yaml")

