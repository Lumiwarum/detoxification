import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import numpy as np
from datasets import load_dataset, load_metric

import matplotlib.pyplot as plt
from wordcloud import WordCloud


data_path = "../../data/interim/"


#read the csv files
df_orig = pd.read_csv(data_path+"origin_test.csv", index_col=[0])
df_flan = pd.read_csv(data_path+"flan_test.csv", index_col=[0])
df_skl = pd.read_csv(data_path+"skolkovo_test.csv", index_col=[0])
df_t5 = pd.read_csv(data_path+"t5_test.csv", index_col=[0])

# download the metrics
clf_name = 'SkolkovoInstitute/roberta_toxicity_classifier_v1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = RobertaForSequenceClassification.from_pretrained(clf_name).to(device);
clf_tokenizer = RobertaTokenizer.from_pretrained(clf_name)

def predict_toxicity(texts):
    """
    This function uses the sckolkovo metrics to get the toxicity of a text
    args:
        texts: a list of str - sentences to get the toxicity score for
    """
    with torch.inference_mode():
        inputs = clf_tokenizer(texts, return_tensors='pt', padding=True).to(clf.device)
        out = torch.softmax(clf(**inputs).logits, -1)[:, 1].cpu().numpy()
    return out
    
# Load the BLUE metric
metric = load_metric("sacrebleu")



orig_toxicity = []
for i in range(len(df_orig)):
    sentence = df_orig.iloc[i]['0']
    orig_toxicity.append(predict_toxicity(sentence))

orig_toxicity = np.array(orig_toxicity)

def get_metrics(df: pd.DataFrame):
    """
    A function to calculate the toxicitiy and sacreblue scores for the test datasets
    """
    toxicity = []
    bleu = []
    for i in range(len(df)):
        sentence = df.iloc[i]['0']
        toxicity.append(predict_toxicity(sentence))
        bleu.append(metric.compute(predictions=[sentence], references=[[df.iloc[i]['0']]])['score'])
    return np.array(toxicity), np.array(bleu)
    
flan_toxicity, flan_bleu = get_metrics(df_flan)
t5_toxicity, t5_bleu = get_metrics(df_t5)
skl_toxicity, skl_bleu = get_metrics(df_skl)


# plot toxicity scores
plt.plot(np.cumsum(orig_toxicity), color = 'red', label = "original")
plt.plot(np.cumsum(t5_toxicity), color = 'yellow', label ="t5_small")
plt.plot(np.cumsum(flan_toxicity), color = 'g', label = "flan")
plt.plot(np.cumsum(skl_toxicity), color = 'b', label ="skolkovo")
plt.legend()
plt.xlabel("N sentences")
plt.ylabel("Toxicity cumsum")
plt.title("Toxicity of models output")
plt.savefig("../../reports/figures/toxicity.png")
plt.show()
plt.close()

# plot sacreblue score
plt.plot(np.cumsum(t5_bleu), color = 'yellow', label ="t5_small")
plt.plot(np.cumsum(flan_bleu), color = 'g', label = "flan")
plt.plot(np.cumsum(skl_bleu), color = 'b', label ="skolkovo")
plt.legend()
plt.xlabel("N sentences")
plt.ylabel("BLEU score cumsum")
plt.title("BLEU score of models output")
plt.savefig("../../reports/figures/blue.png")
plt.show()

def plot_cloud(df: pd.DataFrame, name: str):
    """
    A function to plot word clouds
    args:
        df: pandas dataframe from which to take the sentences
        name: a name of the dataframe to save the png correctly
    """
    ref_text = ' '.join(df['0'])
    wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(ref_text)


    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("../../reports/figures/"+name+".png")
    plt.show()


plot_cloud(df_orig,"original")
plot_cloud(df_flan,"flan")
plot_cloud(df_t5,"t5-small")
plot_cloud(df_skl,"skolkovo")

