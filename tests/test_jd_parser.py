import pandas as pd

from src.jd_parser import parse_job


def test_parser_builds_profile_and_required_skills():
    row = pd.Series({
        "job_id": "J1", "title": "Data Scientist", "description": "Build models",
        "required_skills": "Python, SQL", "preferred_skills": "AWS",
        "min_experience": 2, "max_experience": 5,
    })
    job = parse_job(row)
    assert job.required_skills == ["python", "sql"]
    assert "data scientist" in job.profile_text
