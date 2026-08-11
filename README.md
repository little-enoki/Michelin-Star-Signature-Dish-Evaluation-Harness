# Michelin Star Signature Dish Eval Harness

An eval harness that grades a model's claims about the signature dishes of Michelin-recognized NYC restaurants against a real reference set pulled from the MICHELIN Guide. Measures restaurant coverage, dish faithfulness (is the claimed dish the real one, or a fabrication?), and flags unverified restaurant mentions for manual review rather than auto-failing them.

### Files:

• ground_truth.json — reference set: cuisine → restaurants → real signature dish(es) → source URL. This is a sample, not the full Michelin Guide — treat coverage numbers as directional, not absolute.

• eval_harness.py — the grading logic. No external dependencies (stdlib only).

• responses/*.txt — one file per cuisine, containing the text you're grading. Ships with synthetic example responses I wrote by hand (not real model output) so you can see the pipeline run immediately. Replace them with real pasted output to run a genuine test — see "Running it for real" below.
eval_report.json — generated after each run; full machine-readable detail behind the printed summary table.

### How To Run: 
1. Open any model you want to test — ChatGPT, Claude, Gemini, a Vincent demo if you get access to one, whatever you want to evaluate.
2. Ask it this exact question once per cuisine, so your five runs are comparable:

'What are the signature dish(es) of Michelin-recognized restaurants in New York City that serve {cuisine} cuisine? Name the restaurant and the specific dish for each.'

3. Paste the raw response into the matching file in responses/ (e.g. the French answer goes in responses/french.txt), overwriting the demo text.
4. Re-run python3 eval_harness.py and read the report.
5. Do this with two different models and compare their faithfulness rates and flagged-mention counts side by side. That comparison — not the single-run number — is the actual deliverable you can talk about in an interview.

### Evaluation Metrics: 
| This Harness | What It Measures | Real AI Eval Equivalent |
| --- | --- | --- |
| Coverage | 	Of the restaurants in our reference set, how many did the model mention at all? | Context recall |
| Dish faithfulness | For restaurants the model did name, does its dish description actually match the real one? | Faithfulness / groundedness |
| Flagged mentions | Restaurants named that aren't in our reference set — flagged for human review, not auto-failed, since our reference set is a sample | Citation hallucination flag |
