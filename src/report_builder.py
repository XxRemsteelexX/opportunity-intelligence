"""
report builder service -- constructs evidence packages adn compiles briefings
handles citation tagging, source indexing, and instrumentation tracking
"""

import time
from typing import Optional


class EvidenceBuilder:
    """builds citation-tagged evidence packages for llm synthesis"""

    def __init__(self):
        self.sources = []
        self.citations = []
        self._tag_counter = {}

    def add_source(self, source_name: str, source_type: str, description: str):
        """register a data source"""
        self.sources.append({
            'name': source_name,
            'type': source_type,
            'description': description,
        })

    def add_citation(self, tag: str, fact: str):
        """add a citeable fact with its tag"""
        self.citations.append({'tag': tag, 'fact': fact})

    def add_tagged_facts(self, prefix: str, facts: dict):
        """add multiple facts with auto-incrementing tags like Census-1, Census-2 etc"""
        for i, (key, value) in enumerate(facts.items(), 1):
            tag = f'{prefix}-{i}'
            self.citations.append({'tag': tag, 'fact': f'{key}: {value}'})

    def build_evidence_package(self, market_name: str) -> str:
        """compile all citations into a formatted evidnece package"""
        lines = [
            f'EVIDENCE PACKAGE - {market_name}',
            f'Generated: {time.strftime("%Y-%m-%d")}',
            '',
        ]

        # group citations by prefix
        groups = {}
        for c in self.citations:
            prefix = c['tag'].split('-')[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(c)

        # format each group
        for i, (prefix, cites) in enumerate(groups.items()):
            source = self.sources[i] if i < len(self.sources) else None
            header = f'=== SOURCE {i+1}: {source["name"]} ===' if source else f'=== {prefix} ==='
            lines.append(header)
            for c in cites:
                lines.append(f'[{c["tag"]}] {c["fact"]}')
            lines.append('')

        return '\n'.join(lines)

    def build_source_index(self) -> list:
        """generate the source index table for the report footer"""
        index = []
        # group tags by prefix
        groups = {}
        for c in self.citations:
            prefix = c['tag'].split('-')[0]
            if prefix not in groups:
                groups[prefix] = {'min': c['tag'], 'max': c['tag'], 'count': 0}
            groups[prefix]['max'] = c['tag']
            groups[prefix]['count'] += 1

        for i, (prefix, info) in enumerate(groups.items()):
            source = self.sources[i] if i < len(self.sources) else {'name': prefix, 'description': 'Derived'}
            tag_range = f'{info["min"]} through {info["max"]}' if info['count'] > 1 else info['min']
            index.append({
                'tag_range': tag_range,
                'source': source['name'],
                'description': source.get('description', ''),
            })

        return index


class ReportCompiler:
    """compiles final briefing reports with instrumentaiton"""

    def __init__(self):
        self.timings = {}
        self.token_usage = {'prompt': 0, 'completion': 0, 'total': 0}
        self.cost_per_1k_input = 0.003
        self.cost_per_1k_output = 0.015

    def track_timing(self, label: str, elapsed: float):
        """record a timing measurement"""
        self.timings[label] = round(elapsed, 2)

    def track_tokens(self, prompt: int, completion: int):
        """record token usage from llm call"""
        self.token_usage['prompt'] += prompt or 0
        self.token_usage['completion'] += completion or 0
        self.token_usage['total'] += (prompt or 0) + (completion or 0)

    def estimated_cost(self) -> float:
        """compute estimated llm cost"""
        return (
            (self.token_usage['prompt'] / 1000) * self.cost_per_1k_input +
            (self.token_usage['completion'] / 1000) * self.cost_per_1k_output
        )

    def compile_briefing(
        self,
        market_name: str,
        briefing_text: str,
        source_index: list,
        model_name: str = 'gpt-5',
    ) -> str:
        """compile the full briefing with source index adn instrumentation"""
        est_cost = self.estimated_cost()

        # source index table
        source_rows = '\n'.join(
            f'| {s["tag_range"]} | {s["source"]} | {s["description"]} |'
            for s in source_index
        )

        # timing rows
        timing_rows = '\n'.join(
            f'| {label} | {value}s |'
            for label, value in self.timings.items()
        )

        output = f"""# Opportunity Intelligence Briefing
## {market_name}
**Generated**: {time.strftime('%Y-%m-%d %H:%M')}

---

{briefing_text}

---

## Source Index

| Tag | Source | Description |
|-----|--------|-------------|
{source_rows}

---

## Instrumentation

| Metric | Value |
|--------|-------|
{timing_rows}
| Total tokens | {self.token_usage['total']:,} |
| Prompt tokens | {self.token_usage['prompt']:,} |
| Completion tokens | {self.token_usage['completion']:,} |
| Estimated cost | ${est_cost:.4f} |
| Model | {model_name} |
"""
        return output

    def build_system_prompt(self, role: str = 'market_analyst') -> str:
        """Get the system prompt for a given analyst role"""
        prompts = {
            'market_analyst': """You are a senior market analyst at a senior living company.
You produce concise, data-driven briefings for corporate decision-makers
evaluating new community acquisition opportunities.

Rules:
- Every factual claim MUST include a citation tag like [Census-1] or [CMS-3]
- Use plain business english, no jargon
- Be direct about risks and opportunities
- Include a clear recommendation at the end
- Keep it under 600 words""",

            'financial_analyst': """You are a financial analyst evaluating senior living
market opportunities. Focus on revenue potential, payor mix risks,
and competitive pricing dynamics.

Rules:
- Cite all data with source tags
- Include financial risk assessment
- Recommend pricing strategy
- Keep it under 500 words""",
        }
        return prompts.get(role, prompts['market_analyst'])
