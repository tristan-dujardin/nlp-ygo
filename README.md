# NLP Yi-Gi-Oh

This project is an NLP Classifier where the goal is to classify Yu-Gi-Oh cards into their respective archetypes and banlist state based on their descriptions. We will use word embeddings and logistic regression to accomplish this task.

## Dataset
- **Yu-Gi-Oh Dataset**: [GitHub - Yu-Gi-Oh dataset](https://github.com/fferegrino/yu-gi-oh)

## Objective

Two models are trained in this dataset:

- **archetype_model**: A classifier aimed at identifying a card's archetype based on its effect.
- **banlist_model**: A classifier predicting whether a card should be banned or not.

## Install

- To download the data, at the root of the project run: git clone https://github.com/fferegrino/yu-gi-oh/
- Install the necessary libraries: pip install -r requirements.txt
- Create a models firectory: mkdir models
- To load nltk_data, run the init.py file: python init.py
- Train the model with: python train.py
- Finally you can test the model with the following command:  
    python predict.py [-h] [-n [NAMES ...]] [-i [IDS ...]]

## Example
python predict.py -i 14558127 84815190 41999284 90241276
| Name                                   | Expected Banlist         | Predicted Banlist        | Expected Archetype | Predicted Archetype      |
|----------------------------------------|--------------------------|--------------------------|---------------------|--------------------------|
| Ash Blossom & Joyous Spring            | expected BL: nan         | pred BL: Unlimited       | expected arch: N/A   | pred arch: []            |
| Baronne de Fleur                       | expected BL: Banned      | pred BL: Unlimited       | expected arch: Fleur | pred arch: ['Borrel']    |
| Linkuriboh                             | expected BL: Banned      | pred BL: Unlimited       | expected arch: Kuriboh| pred arch: ['Kuriboh']  |
| Snake-Eyes Poplar                      | expected BL: nan         | pred BL: Unlimited       | expected arch: Snake-Eye | pred arch: ['Snake-Eye'] |
