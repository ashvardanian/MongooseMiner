# MongooseMiner

![Preview](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/MongooseMiner.jpg?raw=true)

MongooseMiner is a search system that pushes LLM-based code generation beyond average human performance. Most LLMs for code generation write code like humans:

- They use the most common packages rather than the most appropriate and powerful ones,
- They use the most common functions instead of the most appropriate and powerful ones,
- They do not memorize all available function arguments and perform many operations where one is sufficient. This is because LLMs learn from the average developer's code, while MongooseMiner learns from the experts who wrote the libraries and documentation.

By evaluating the documentation strings of the most common PyPI projects and retrieving them as needed to guide LLM autocompletion, MongooseMiner can deliver the most appropriate and performant code.

## Dataset

To enable MongooseMiner, we needed both PyPi and GitHub data.
BigQuery hosts both:

- [PyPi downloads](https://console.cloud.google.com/marketplace/product/gcp-public-data-pypi/pypi)
  - `distribution_metadata` table contains other tables we need to fetch:
    - `name` mapped to `pypi_name`
    - `version` mapped to `pypi_version`
    - `summary` & `description` combined into a single `pypi_description` string
    - `home_page` string & `download_url` string & `project_urls` array of strings where we can find the source code links and check if it leads to GitHub export to `github_url`
    - `requires` for dependencies
  - `file_downloads` table contains columns:
    - `project` like `a8`
- [GitHub activity](https://console.cloud.google.com/marketplace/product/github/github-repos)
  - `sample_repos` table contains:
    - `repo_name` string like `FreeCodeCamp/FreeCodeCamp`
    - `watch_count` integer for the number of people watching the repo
  - `languages` table contains:
    - `repo_name` string like `FreeCodeCamp/FreeCodeCamp`
    - `language.name` string like `C`
    - `language.bytes` integer containing the amount of code written in that language

We use that data to aggregate information into one table:

1. Sample the mentioned columns from the PyPi table
2. Check if any of the links leads to GitHub
3. Extract the name of the repo name from the GitHub URL
4. Join it with the watch-count from the GitHub table

For details and the code check [`bigquery.sql`](bigquery.sql).
