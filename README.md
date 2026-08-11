# Michelin-Star-Signature-Dish-Evaluation-Harness

What the Eval Harness Does: Input a cuisine type (French, Italian, Contemporary, Korean, Japanese), output the signature dish(es) of Michelin-recognized NYC restaurants in that cuisine. Ground truth: dish descriptions pulled from the restaurants' actual MICHELIN Guide pages (ground_truth.json, with source URLs).

Files:
ground_truth.json — reference set: cuisine → restaurants → real signature dish(es) → source URL. This is a sample, not the full Michelin Guide — treat coverage numbers as directional, not absolute.
eval_harness.py — the grading logic. No external dependencies (stdlib only).
responses/*.txt — one file per cuisine, containing the text you're grading. Ships with synthetic example responses I wrote by hand (not real model output) so you can see the pipeline run immediately. Replace them with real pasted output to run a genuine test — see "Running it for real" below.
eval_report.json — generated after each run; full machine-readable detail behind the printed summary table.

Evaluation Metrics: 
This harness	What it measures	Real AI-eval equivalent
Coverage	Of the restaurants in our reference set, how many did the model mention at all?	Context recall
Dish faithfulness	For restaurants the model did name, does its dish description actually match the real one?	Faithfulness / groundedness
Flagged mentions	Restaurants named that aren't in our reference set — flagged for human review, not auto-failed, since our reference set is a sample	Citation hallucination flag
