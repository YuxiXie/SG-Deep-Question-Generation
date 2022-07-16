# Method of Building Semantic Graphs (DP-based)

These files define the rules to build semantic graphs for source text depending on the results of **Dependency Parsing**. 
The 3 crucial parts of this method is:

* Extract _[entity, relation, entity]_ pattern from each sentence, like **Semantic Role Labeling**

* Merge and Prune nodes in the graph to adrress the problems of _(1)_ sparsity brought by fine-grained nodes _(2)_ graph noise caused by redundant and meaningless punctuation, conjunction, etc.

* Connect _SIMILAR_ nodes to help to connect sub-graphs of all sentences and get a unified graph for each evidence 

---

* Requirement

    ```
    allennlp==0.9.0
    overrides==3.1.0
    ```

* Predictors for dependency parsing and coreference resolution

    - The links to the predictors in our code may not be up-to-date, you may need to check the availability before running the code.

---

To run the codes, execute the commands below:

* Get the raw json files from [HotpotQA](https://hotpotqa.github.io/) [`training set` & `dev set(distractor)`] preprocessed

    ```bash
    python preprocess/preprocess_raw_data.py train.json valid.json data
    ```

* Get the results of dependency parsing and coreference resolution

    - To initialize the predictors, you need to download the models of dependency parsing and coreference resolution, _e.g._, the latest models released from [AllenNLP](https://demo.allennlp.org/).

    ```bash
    python preprocess/get_coref_and_dep_data.py data.train.json data.valid.json dp.json crf_rsltn.json
    ```

    - Since it will take long time to get these files finished, we provide the final data --- [dp.json](https://drive.google.com/file/d/1hdwS5nC86Jrss7HLQt1eds-RjSZjNJBC/view?usp=sharing) and [crf_rsltn.json](https://drive.google.com/file/d/1U9dNzAmNx1TyQ2BjVBYJ-oJ17Lws0dYE/view?usp=sharing).

* Merge data file (train or valid) with the result files from **Coreference Resolution** and **Dependency Parsing**

    ```bash
    python merge.py data.json dp.json crf_rsltn.json merged_data.json
    ```

* Build Semantic Graphs with _Question Tags_ (i.e., whether a node contains span(s) in the question) as the groundtruth of **Context Selection** and also _Answer Tags_ (i.e., whether a node contains span(s) in the answer)

    - Here you also need to provide the corresponding tokenized `questions.txt` and `answers.txt` files (cf., [`text-data`](https://drive.google.com/drive/folders/1nhBfk2EvOHGDRq6vPCf8Pk8wZFL0dqbf?usp=sharing))

    - This script will also generate the corresponding tokenized `source.txt`, so you need to provide the directory to dump the data as well.

    ```bash
    python build_semantic_graph.py merged_data.json questions.txt answers.txt source.txt graph_with_tags.json
    ```
