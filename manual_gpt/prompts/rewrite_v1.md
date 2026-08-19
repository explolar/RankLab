You are assisting with a public e-commerce search research study.
Rewrite only what is supported by the raw query. Do not invent brands, materials,
sizes, use cases, or product attributes. If the query is too vague, preserve that
uncertainty rather than adding details.

Return JSON only in exactly this structure:
{
  "expanded_query": "concise search rewrite",
  "intent": "transactional | informational | navigational",
  "category": "specific category or unknown",
  "uncertainty": "low | medium | high"
}

Raw query: {QUERY}
