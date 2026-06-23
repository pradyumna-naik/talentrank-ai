# TalentRank AI — 2-Minute Hackathon Demo Script

## 0:00–0:15 — Hook

“Recruiters often have hundreds of applications, but keyword filters are brittle and opaque. TalentRank AI is an explainable candidate discovery engine that produces a shortlist with evidence—not just a score.”

## 0:15–0:35 — Home and JD Intelligence

“On the Home page, I select the role and choose whether to use optional CrossEncoder reranking. The dashboard summarizes the job, candidate pool, seniority, and ranking mode. Next, JD Intelligence extracts the role, domain, experience range, responsibilities, and must-have versus nice-to-have skills from the description.”

## 0:35–0:55 — Run ranking

“I click Run Ranking. TalentRank AI combines semantic similarity from MiniLM, TF-IDF keyword relevance, skill coverage, experience fit, and activity. It first retrieves the top 50 candidates, then optionally applies a pairwise CrossEncoder for a more precise final ordering.”

## 0:55–1:15 — Ranked Candidates

“Here is the ranked shortlist. Instead of a black-box result, I can see final score, semantic score, keyword score, skill coverage, experience fit, activity, and CrossEncoder score when enabled.”

## 1:15–1:35 — Candidate Explanation

“Selecting a candidate opens an explanation card. It tells the recruiter exactly which required and nice-to-have skills matched, which skills are missing, how experience fits the role, what project evidence was found, any gaps or risks, and the confidence level.”

## 1:35–1:50 — Responsible AI and Evaluation

“Responsible AI is built into the workflow. Sensitive fields like name, gender, religion, caste, age, photo, and address are explicitly excluded from profile text and scoring. The audit report shows which fields were used. The Evaluation page supports Precision@K, Recall, NDCG, and MRR when labels exist, or top-10 diagnostics and ablation analysis when they do not.”

## 1:50–2:00 — Close

“Finally, Export Output downloads a clean five-column shortlist ready for recruiter review. TalentRank AI makes candidate discovery faster, more explainable, and safer to use as a human-in-the-loop decision-support tool.”
