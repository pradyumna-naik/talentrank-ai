import pandas as pd

from src.jd_parser import SkillTaxonomy, parse_jd, parse_job


def test_skill_taxonomy_extracts_common_skills():
    skills = SkillTaxonomy.extract("Build FastAPI services with Python, SQL, Docker and AWS.")
    assert skills == ["Python", "SQL", "FastAPI", "Docker", "AWS"]


def test_jd_intelligence_extracts_rule_based_fields():
    row = pd.Series({
        "job_id": "J1",
        "title": "Senior Machine Learning Engineer",
        "description": (
            "Develop and deploy NLP models. Must have Python, SQL, and Machine Learning. "
            "AWS and Docker are nice to have. Requires 5-7 years of experience."
        ),
        "required_skills": "Python, SQL",
        "preferred_skills": "AWS",
        "min_experience": None,
        "max_experience": None,
    })
    intelligence = parse_jd(row)
    assert intelligence.role_title == "Senior Machine Learning Engineer"
    assert intelligence.seniority_level == "senior"
    assert {"Python", "SQL", "Machine Learning"}.issubset(intelligence.must_have_skills)
    assert {"AWS", "Docker"}.issubset(intelligence.nice_to_have_skills)
    assert intelligence.domain == "Machine Learning / AI"
    assert intelligence.min_experience == 5
    assert intelligence.max_experience == 7
    assert any("Develop and deploy" in item for item in intelligence.responsibilities)


def test_parse_job_uses_intelligence_required_skills():
    row = pd.Series({
        "job_id": "J1", "title": "Data Scientist", "description": "Build models",
        "required_skills": "Python, SQL", "preferred_skills": "AWS",
        "min_experience": 2, "max_experience": 5,
    })
    job = parse_job(row)
    assert job.required_skills == ["python", "sql"]
    assert job.intelligence.seniority_level == "junior"
    assert "data scientist" in job.profile_text
