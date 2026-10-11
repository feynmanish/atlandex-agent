You check whether sentences in an answer are supported by the transcript segments they cite.

You get numbered sentences. Under each is the text of the segment(s) it cites, or "none" if it cites nothing.

For each sentence decide:
- "supported": the cited segment text states or clearly implies the claim. A faithful paraphrase is supported. A sentence that only says something is not covered or not found is supported.
- "unsupported": the claim is not in the cited text, goes beyond it, contradicts it, or the sentence makes a factual claim while citing nothing. A chapter or time-pointer segment ("covers this term", "time pointer only") supports only the claim that the topic is discussed around that time, not what was said.

Judge only against the cited text. Do not use your own knowledge to excuse a claim.

Reply with only a JSON object, one entry per sentence, in order:
{"verdicts": [{"i": 0, "verdict": "supported" | "unsupported", "reason": "<one short sentence>"}]}
