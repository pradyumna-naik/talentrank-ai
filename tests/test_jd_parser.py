from src.jd_parser import parse_job_description


def test_parser_extracts_skills_and_experience():
    job = parse_job_description("Data Scientist", "Need 3+ years of Python, SQL and machine learning.")
    assert job.min_experience_years == 3
    assert set(job.skills) == {"python", "sql", "machine learning"}
