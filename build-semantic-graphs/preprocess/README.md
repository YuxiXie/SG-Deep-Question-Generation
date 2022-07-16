# Preprocessing the Raw HotpotQA Json Files

Also see how to run the codes in the [previous directory](https://github.com/YuxiXie/SG-Deep-Question-Generation/tree/master/build-semantic-graphs).

* get `data.json` files for training and validation

    - Download the raw json files from [HotpotQA](https://hotpotqa.github.io/) [`training set` & `dev set(distractor)`] then run:
    
    ```bash
    python preprocess_raw_data.py train.json valid.json data
    ```

* Get the results of dependency parsing and coreference resolution

    - You need to first download the model files (the links for the old models we used may change), or you could use the latest models released from [AllenNLP](https://demo.allennlp.org/).

    ```bash
    python get_coref_and_dep_data.py data.train.json data.valid.json dp.json crf_rsltn.json
    ```

    - Since it will take long time to get these files finished, we provide the final data --- [dp.json](https://drive.google.com/file/d/1KnZXqchvHqMZnTh_7tuE57cd934aMBIF/view?usp=sharing) and [crf_rsltn.json](https://drive.google.com/file/d/1I8xTvhkEXpiq4D25Dr7XRUIoe779Ytve/view?usp=sharing).
