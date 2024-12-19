Run the scripts in the following order to generate the data for the September 2024 issue of the Quarterly Arxiv.
1. ``arxiv_lib_failsafe__init__.py``: replace the __init__.py file code of your [arxiv library](https://github.com/lukasschwab/arxiv.py/blob/master/arxiv/__init__.py) with this code to avoid the error that occurs when the library fails to retrieve a paper and skip to the next few papers. Otherwise, the error can be unrecoverable with their current implementation, as we will run into max recursion depth errors. 
2. ``scrape.py``: script that uses the bulk API of Semantic Scholar to query the citation counts of recently released Arxiv papers. The raw data is stored in a separate files. Whenever you want to regenerate, these file needs to be deleted: papers.shelf.\* . Occasionally, 
the arxiv library may skip a few papers due to an error. If the script is restarted after a successful run, it will skip the papers that are already in the shelf file and add new ones. It is recommended to run multiple times to find such papers.
3. ``apply_citation_counts.py``: applies z-scores to the output file of scrape.py


For the analysis of generated text, the following scripts are used:
1. ``download_random_papers.py``: Downloads n random arxiv papers per day. We do not control for categories
2. : Extracts the text of the downloaded papers into markdown files.
3. ``analysis_and_plots.py``: Recreate the analysis of llm detectors. Requires the installation of [Binoculars](https://github.com/ahans30/Binoculars)