# Project Title

## Overview

Modern messaging platforms generate millions of messages per day. Group chats, while highly engaging, often become noisy and difficult to follow, especially after time away. User feedback indicates that this dynamic is causing: users to miss key information from portions of the chat during which they were not present; users to experience information overload; and users to spend large amounts of time catching up on missed chats.
These consequences collectively create a poor user experience, discourage platform engagement, and risk customer churn. This churn is emphasized as competitors integrate intelligent summarization and AI-assisted messaging features.

We propose a machine-learning based summarization feature embedded in Acme’s messaging interface. The model will read recent chat transcripts and generate short, fluent summaries capturing the conversation’s essential details: who said what, major decisions, and action items. This will directly improve the user experience of catching up on missed messages and in turn address the aforementioned issues regarding engagement and retention as well as Acme’s positioning with competitors.

In this project we compare two different model architectures: BERT based and GPT based.

## Technical Approach and Methodology

This project explores text summarization using the SAMSum dataset and two different approaches:

1.  **NLTK based Token Frequency Analysis:** Initial exploratory data analysis was performed using the NLTK library to understand token, bigram, and trigram distributions after custom preprocessing (lowercasing, punctuation removal, stop word removal, and lemmatization).
2.  **Transformer-based Models:** We experimented with two transformer models from the Hugging Face `transformers` library:
    *   **BERT (as an Encoder-Decoder Model):** A pre-trained BERT model was fine-tuned for the summarization task. The model architecture was configured as an encoder-decoder setup, with BERT serving as both the encoder and decoder.
    *   **GPT-2 (with Prompt Engineering):** Due to resource limitations during fine-tuning, a different approach was taken with GPT-2. The base `distilgpt2` model was loaded, and prompt engineering was used to guide the model towards generating summaries. Different prompt structures were experimented with to evaluate their impact on performance.

Data preprocessing and tokenization for the transformer models were handled using the respective model's tokenizer from the `transformers` library. Training and evaluation were managed using the `Trainer` class from `transformers`.

## Usage

The fine-tuned model weights are not included do to file size contraints on Github. In order to generate the fine-tuned BERT model, simply follow along with the documented notebook cell by cell through the BERT training cell. The model can then be saved externally as documented in the following cell, and loaded from those external files as documented in the cell following that.

## Results and Evaluation

**BERT Fine-tuning:**

*   The BERT encoder-decoder model was fine-tuned on the training dataset.
*   Evaluation on the test set was performed using the ROUGE metric (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum).

'eval_rouge1': 0.35323447674261305, 'eval_rouge2': 0.21749473987693324, 'eval_rougeL': 0.2569865197802464, 'eval_rougeLsum': 0.3532853192619417

**GPT-2 Prompt Engineering:**

*   Different prompt structures were tested on the validation set to assess their effectiveness in guiding the base GPT-2 model for summarization.
*   ROUGE scores were computed for each prompt type to compare their performance.
*   The best performing GPT-2 prompt structure was then evaluated on the test set.

{'rouge1': np.float64(0.12352745552688203), 'rouge2': np.float64(0.008809929787377655), 'rougeL': np.float64(0.08855891531950373), 'rougeLsum': np.float64(0.08989666564804391)}

Overall, the fine-tuned BERT model is expected to perform better than the base GPT-2 model with prompt engineering for this task, as fine-tuning adapts the model specifically to the summarization dataset.

## Limitations and Future Work

**Limitations:**

*   **Resource Constraints:** Fine-tuning the GPT-2 model was limited by computational resources, preventing full fine-tuning. This necessitated the prompt engineering approach, which is generally less performant than fine-tuning for downstream tasks.
*   **Prompt Engineering Simplicity:** The prompt structures used for the GPT-2 experiments were basic. More complex and tailored prompts could potentially yield better results.
*   **Model Size:** The models used (bert-base-uncased, distilgpt2) are relatively small compared to state-of-the-art summarization models.
*   **Evaluation Metrics:** Only ROUGE metrics were used for evaluation. Other metrics like BLEU or human evaluation could provide a more comprehensive assessment.

**Future Work:**

*   **Fine-tune Larger Models:** Fine-tune larger or more suitable models for summarization (e.g., BART, T5) if computational resources become available.
*   **Advanced Prompt Engineering:** Explore more sophisticated prompt engineering techniques for GPT-2, including few-shot learning or prompt tuning.
*   **Hyperparameter Tuning:** Optimize hyperparameters (particulary on the decoder end for BERT) for both the BERT fine-tuning and GPT-2 generation to potentially improve performance.
*   **Data Preprocessing:** Experiment with reformatting the original dialogue to determine which formats are most easily parable and 'understandable' by the differnt models.
*   **Explore Other Architectures:** Investigate other transformer architectures specifically designed for sequence-to-sequence tasks like summarization.
*   **Qualitative Analysis:** Perform a deeper qualitative analysis of the generated summaries to understand common errors and areas for improvement.
*   **Implement other evaluation metrics:** Include other metrics like BLEU to get a more rounded understanding of model performance.