# INVALSI Benchmark for Language Model Evaluation

This module provides a standardized and reproducible benchmark for the evaluation of Large Language Models (LLMs) on Italian language comprehension tasks. The benchmark is predicated on the official INVALSI (Istituto nazionale per la valutazione del sistema educativo di istruzione e di formazione) tests, which are administered nationwide in Italy to assess students' proficiency. Specifically, this benchmark utilizes the Italian language comprehension sections from tests designed for middle school students, offering a robust framework for quantifying the linguistic capabilities of LLMs in a formal educational context.

## Module Structure

The module is organized into the following components:

*   `invalsi_test.py`: This is the core script that orchestrates the entire benchmarking process. It is responsible for loading the language models, processing the test data from the JSON files, generating model responses to the test questions, and calculating the ROUGE-1 precision score for evaluation.

*   `json_tests/`: This directory contains the INVALSI test questions in a structured JSON format. These files were generated from the original PDF documents to facilitate programmatic access and processing.

*   `gemini_prompt.txt`: This file contains the precise prompt that was used with the Google Gemini Flash model to convert the original INVALSI test PDFs into the structured JSON format used in this benchmark.

*   `pdf_oringal/`: This directory archives the source material for the benchmark, including the original INVALSI tests in PDF format and their corresponding correction grids.

*   `results/`: This directory is the designated output location for the benchmark results. It stores the performance metrics of each model tested in JSON files, organized into subdirectories named after the respective models.

## Methodology

The benchmarking process follows a rigorous methodology:

1.  **Data Preparation**: The INVALSI tests, originally in PDF format, were converted into a structured JSON format using the Google Gemini Flash model. The prompt used for this conversion is provided in `gemini_prompt.txt`. To ensure a focus on a specific type of language understanding, non-multiple-choice questions were programmatically excluded from the final dataset.

2.  **Evaluation Metric**: The performance of the LLMs is quantified using the ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation) precision score. This metric was chosen for its suitability in this context, as it measures the unigram overlap between the model-generated answer and the ground-truth correct answer, which is effective for evaluating performance on multiple-choice questions where the expected output is a single character.

3.  **Few-Shot Learning**: The benchmark script incorporates a few-shot learning approach. Prior to presenting the actual test question, the model is provided with a small number of example questions and answers. This technique is employed to guide the model in generating responses that adhere to the expected format (i.e., a single letter corresponding to the correct option).

## Usage

To execute the benchmark, run the `invalsi_test.py` script from the terminal:

```bash
python invalsi_test.py
```

To evaluate a different language model, modify the `base_model` and `model_name` variables within the `invalsi_test.py` script accordingly.

## Results

The `results/` directory contains the detailed outcomes of the benchmark tests for various models, facilitating a comparative analysis of their performance on the INVALSI language comprehension tasks. Each JSON file in this directory provides a summary of the model's performance, including the mean ROUGE-1 score and the configuration used for the test run.
