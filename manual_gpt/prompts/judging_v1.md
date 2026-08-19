You are evaluating e-commerce search relevance. Based only on the query and
product information, assign one score. Do not assume missing product attributes.

Score meanings:
0 = irrelevant
1 = complement/accessory related to the query
2 = plausible substitute, but not the requested item
3 = exact or very strong match to the requested item

Return JSON only: {"score": 0, "rationale": "one short sentence"}

Query: {QUERY}
Product title: {TITLE}
Brand: {BRAND}
Product information: {PRODUCT_TEXT}
