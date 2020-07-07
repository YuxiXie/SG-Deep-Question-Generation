# Method of Building Semantic Graphs (DP-based)

These files define the rules to build semantic graphs for source text depending on the results of **Dependency Parsing**. 
The 3 crucial parts of this method is:

* Extract _[entity, relation, entity]_ pattern from each sentence, like **Semantic Role Labeling**

* Merge and Prune nodes in the graph to adrress the problems of _(1)_ sparsity brought by fine-grained nodes _(2)_ graph noise caused by redundant and meaningless punctuation, conjunction, etc.

* Connect _SIMILAR_ nodes to help to connect sub-graphs of all sentences and get a unified graph for each evidence 

---

To run the codes, execute the commands below:

* Get the raw json files from [HotpotQA](https://hotpotqa.github.io/) [`training set` & `dev set(distractor)`] preprocessed

    ```bash
    python preprocess/preprocess_raw_data.py train.json valid.json data
    ```

* Get the results of dependency parsing and coreference resolution

    *** To initialize the predictors, you need to download the models of dependency parsing and coreference resoluation [here](https://drive.google.com/drive/folders/1Q2K5pOkASsr_R7JeeEIebCHaHfYQ9XS_?usp=sharing). Or you could just use the latest models released from [AllenNLP](https://demo.allennlp.org/).

    ```bash
    python preprocess/get_coref_and_dep_data.py data.train.json data.valid.json dp.json crf_rsltn.json
    ```

    Since it will take long time to get these files finished, we provide the final data --- [dp.json](https://drive.google.com/file/d/1KnZXqchvHqMZnTh_7tuE57cd934aMBIF/view?usp=sharing) and [crf_rsltn.json](https://drive.google.com/file/d/1I8xTvhkEXpiq4D25Dr7XRUIoe779Ytve/view?usp=sharing).

* Merge data file (train or valid) with the result files from **Coreference Resolution** and **Dependency Parsing**

    ```bash
    python merge.py data.json dp.json crf_rsltn.json merged_data.json
    ```

* Build Semantic Graphs with _Question Tags_ (i.e., whether a node contains span(s) in the question) as the groundtruth of **Context Selection**

    ```bash
    python build_semantic_graph.py merged_data.json graph_with_tags.json
    ```
