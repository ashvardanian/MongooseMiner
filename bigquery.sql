-- PyPiData CTE:
-- Extracts relevant metadata from the PyPi distribution metadata table. This includes the package name,
-- version, a concatenated description (if summary is present, it's combined with the description),
-- download URLs, project URLs (which may contain GitHub links), and the upload time for version sorting.
WITH PyPiData AS (
    SELECT
        name AS pypi_name,
        version AS pypi_version,
        IF(summary IS NULL, description, CONCAT(summary, ' ', description)) AS pypi_description,
        home_page,
        download_url,
        project_urls,
        requires,
        upload_time
    FROM
        `bigquery-public-data.pypi.distribution_metadata`
),

-- GitHubURLs CTE:
-- Filters and processes the project_urls from PyPiData to extract clean GitHub repository URLs.
-- It removes unnecessary URL fragments like '/issues' or '/tree' to standardize the GitHub repository identification.
GitHubURLs AS (
    SELECT
        pypi_name,
        pypi_version,
        pypi_description,
        REGEXP_REPLACE(REGEXP_EXTRACT(url, r'https?://github\.com/([^/]+/[^/?#]+)'), r'(/issues.*)|(blob/.*)|(/pull/.*)|(tree/.*)', '') AS github_url,
        upload_time
    FROM
        PyPiData,
        UNNEST(project_urls) AS url
    WHERE
        url LIKE '%github.com%'
        AND (url LIKE '%https://github.com/%' OR url LIKE '%http://github.com/%')
),

-- MostRecentVersions CTE:
-- Determines the most recent version of each PyPi package based on the upload_time.
-- This ensures that only the latest data per package is considered for analysis.
MostRecentVersions AS (
    SELECT
        pypi_name,
        pypi_version,
        pypi_description,
        github_url
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY pypi_name ORDER BY upload_time DESC) AS rn
        FROM GitHubURLs
    )
    WHERE rn = 1
),

-- DownloadMetrics CTE:
-- Aggregates download data from the PyPi file_downloads table. It computes total downloads, yearly downloads for the past three years,
-- and captures the earliest and latest download timestamps to track the download activity over time.
DownloadMetrics AS (
    SELECT
        project,
        SUM(CASE WHEN EXTRACT(YEAR FROM timestamp) = 2021 THEN 1 ELSE 0 END) AS downloads_2021,
        SUM(CASE WHEN EXTRACT(YEAR FROM timestamp) = 2022 THEN 1 ELSE 0 END) AS downloads_2022,
        SUM(CASE WHEN EXTRACT(YEAR FROM timestamp) = 2023 THEN 1 ELSE 0 END) AS downloads_2023,
        COUNT(*) AS pypi_downloads,
        MAX(timestamp) AS pypi_downloads_date_latest,
        MIN(timestamp) AS pypi_downloads_date_earliest
    FROM
        `bigquery-public-data.pypi.file_downloads`
    GROUP BY
        project
),

-- WatcherData CTE:
-- Retrieves the number of watchers for each GitHub repository from the sample_repos table.
-- This metric indicates the level of interest and engagement a repository has garnered.
WatcherData AS (
    SELECT
        repo_name,
        watch_count
    FROM
        `bigquery-public-data.github_repos.sample_repos`
),

-- LanguageData CTE:
-- Aggregates the programming language data from the GitHub languages table. It specifically calculates the total bytes of code written in Python
-- and the total bytes of code across all languages, providing insights into the language usage within each repository.
LanguageData AS (
    SELECT
        repo_name,
        SUM(CASE WHEN name = 'Python' THEN bytes ELSE 0 END) AS code_bytes_python,
        SUM(bytes) AS code_bytes_total
    FROM
        `bigquery-public-data.github_repos.languages`,
        UNNEST(language) AS language
    GROUP BY
        repo_name
)

-- Final SELECT statement:
-- Combines all the data collected in previous CTEs into a comprehensive table.
-- This table includes PyPi package details, GitHub repository metrics, download statistics, and programming language usage,
-- providing a holistic view of each package's popularity, usage, and development activity.
SELECT
    mv.pypi_name,
    mv.pypi_version,
    mv.pypi_description,
    mv.github_url,
    dm.pypi_downloads,
    dm.downloads_2021,
    dm.downloads_2022,
    dm.downloads_2023,
    dm.pypi_downloads_date_latest,
    dm.pypi_downloads_date_earliest,
    wd.watch_count,
    ld.code_bytes_python,
    ld.code_bytes_total
FROM
    MostRecentVersions mv
LEFT JOIN
    DownloadMetrics dm ON mv.pypi_name = dm.project
LEFT JOIN
    WatcherData wd ON mv.github_url = wd.repo_name
LEFT JOIN
    LanguageData ld ON mv.github_url = ld.repo_name
WHERE
    mv.github_url IS NOT NULL
LIMIT 100;
