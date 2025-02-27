from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from codecarbon import EmissionsTracker

from src.models.model_utils import query_ollama
from src.prompts.sentiment import get_sentiment_prompt
from src.utils.clean_outputs import clean_llm_output_to_int
from src.data.process_datasets import get_processed_dataset

def evaluate_performance_sentiment(model, runs):
    _, test_sentences, _, test_labels = get_processed_dataset("sentiment")
    
    pred_labels = []
    prompts = [get_sentiment_prompt(test_sentences[i]) for i in range(runs)]

    with EmissionsTracker(
        project_name="model-distillation",
        experiment_id="123",
        tracking_mode="process",
        output_dir="results/metrics/emissions",
        allow_multiple_runs=True,
        log_level="warning"
    ) as tracker:
        for prompt in tqdm(prompts):
            predicted_label = query_ollama(model=model, prompt=prompt)
            pred_labels.append(predicted_label)

    pred_labels = [clean_llm_output_to_int(pred_labels[i]) for i in range(runs)]
    
    accuracy = accuracy_score(test_labels[:runs], pred_labels)
    confusion = confusion_matrix(test_labels[:runs], pred_labels)

    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(f'Confusion matrix for {model} on sentiment analysis')
    plt.xticks(ticks=[0.5, 1.5, 2.5], labels=['Negative', 'Neutral', 'Positive'])
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=['Negative', 'Neutral', 'Positive'])
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.2f}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    print(f"Energy Consumption: {tracker._total_energy.kWh:.5f} kWh")
    print(f"CO2 Emissions: {tracker.final_emissions:.5f} kgCO2")