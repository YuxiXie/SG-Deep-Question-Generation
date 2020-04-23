# Method of Building Semantic Graphs (DP-based)

These files define the rules to build semantic graphs for source text depending on the results of **Dependency Parsing**. 
The 3 crucial parts of this method is:

* Extract _[entity, relation, entity]_ pattern from each sentence, like **Semantic Role Labeling**

* Merge and Prune nodes in the graph to adrress the problems of _(1)_ sparsity brought by fine-grained nodes _(2)_ graph noise caused by redundant and meaningless punctuation, conjunction, etc.

* Connect _SIMILAR_ nodes to help to connect sub-graphs of all sentences and get a unified graph for each evidence 

---

To run the codes, execute the commands below:

* Merge data file with the result files from **Coreference Resolution** and **Dependency Parsing**

```bash
python merge.py data.json crf_rsltn.json dp.json merged_data.json
```

* Build Semantic Graphs with _Question Tags_ (i.e., whether a node contains span(s) in the question) as the groundtruth of **Context Selection**

```bash
python build_semantic_graph.py merged_data.json graph_with_tags.json
```
