import argparse
import json
import os
import re

import en_vectors_web_lg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm


def preprocess_text(text):
    return (
        re.sub(r"([.,'!?\"()*#:;])", "", text.lower())
        .replace("-", " ")
        .replace("/", " ")
    )


def tokenize(stat_ques_list, use_glove):
    token_to_ix = {
        "PAD": 0,
        "UNK": 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool("PAD").vector)
        pretrained_emb.append(spacy_tool("UNK").vector)

    for ques in stat_ques_list:
        words = preprocess_text(ques["question"]).split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    for ques in stat_ques_list:
        words = preprocess_text(ques["caption"]).split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb


def txt2vec(sentence, token_to_ix, pretrained_emb):
    text = []

    words = preprocess_text(sentence).split()

    for word in words:
        if word in token_to_ix:
            num = token_to_ix[word]
            text.append(pretrained_emb[num])
        else:
            text.append(pretrained_emb[1])
    return text


def cos_sim(A, B):
    return np.matmul(A, np.transpose(B)) / (norm(A) * norm(B))


def word_sim(w1, w2):
    s = 0.5 * (1 + cos_sim(w1, w2))
    return s


def sent_sim(sent1, sent2, token_to_ix, pretrained_emb):
    sent2vec1 = txt2vec(sent1, token_to_ix, pretrained_emb)
    sent2vec2 = txt2vec(sent2, token_to_ix, pretrained_emb)
    sent_tmp = []

    for i in sent2vec1:
        vec_tmp = []
        for j in sent2vec2:
            tmp_sim = word_sim(i, j)
            vec_tmp.append(tmp_sim)
        sent_tmp.append(max(vec_tmp))

    sent_similarity = sum(sent_tmp) / len(sent2vec1)

    return sent_similarity


def calculate_similarity(train_cap_path, val_cap_path, test_cap_path, val_qacap_path):
    with open(train_cap_path) as train_cap:
        train_cap = json.load(train_cap)

    with open(val_cap_path) as val_cap:
        val_cap = json.load(val_cap)

    with open(test_cap_path) as test_cap:
        test_cap = json.load(test_cap)

    val_qacap = json.load(open(val_qacap_path))

    for i in val_qacap["data"]:
        i["q_similarity"] = sent_sim(
            i["question"], i["caption"], token_to_ix, pretrained_emb
        )

    for i in val_qacap["data"]:
        i["a_similarity"] = sent_sim(
            i["multiple_choice_answer"], i["caption"], token_to_ix, pretrained_emb
        )

    for i in val_qacap["data"]:
        i["total_similarity"] = (i["a_similarity"] + i["q_similarity"]) / 2

    train_qacap = json.load(open("./datasets/caption/train_qacap_sim.json"))

    df_sim = pd.DataFrame(val_qacap["data"])

    df_sim2 = df_sim.sort_values(by="total_similarity", ascending=False)

    del df_sim2["index"]
    del df_sim2["image_id"]
    del df_sim2["question"]
    del df_sim2["question_id"]
    del df_sim2["caption"]
    del df_sim2["multiple_choice_answer"]

    for i in train_cap["data"]:
        i["q_similarity"] = sent_sim(
            i["question"], i["caption"], token_to_ix, pretrained_emb
        )

    for i in val_cap["data"]:
        i["similarity"] = sent_sim(
            i["question"], i["caption"], token_to_ix, pretrained_emb
        )

    for i in test_cap["data"]:
        i["similarity"] = sent_sim(
            i["question"], i["caption"], token_to_ix, pretrained_emb
        )

    for i in train_cap["data"]:
        del i["index"]

    for i in val_cap["data"]:
        del i["index"]

    for i in test_cap["data"]:
        del i["index"]

    with open("datasets/caption/train_cap.json", "w") as f:
        json.dump(train_cap, f)

    with open("datasets/caption/val_cap.json", "w") as f2:
        json.dump(val_cap, f2)

    with open("datasets/caption/test_cap.json", "w") as f3:
        json.dump(test_cap, f3)

    df_train = pd.DataFrame(train_cap["data"])
    df_val = pd.DataFrame(val_cap["data"])
    df_test = pd.DataFrame(test_cap["data"])

    plt.hist(
        [df_train["similarity"], df_val["similarity"], df_test["similarity"]],
        label=["train", "val", "test"],
    )

    plt.legend(loc="upper right")
    plt.show()


def make_similarity(limit, sim):
    with open("datasets/caption/val_qacap_sim.json") as val_qacap_sim:
        val_qacap_sim = json.load(val_qacap_sim)

    for i in val_qacap_sim["data"]:
        if i[sim] < limit:
            i["caption"] = ""

    for i in val_qacap_sim["data"]:
        i["question"] = i["question"] + " " + i["caption"]

    df_val_t = pd.DataFrame(val_qacap_sim["data"])

    del df_val_t["caption"]
    del df_val_t["q_similarity"]
    del df_val_t["a_similarity"]
    del df_val_t["total_similarity"]
    del df_val_t["multiple_choice_answer"]

    if not (os.path.isdir(os.path.join("datasets/caption/under", str(limit)))):
        os.makedirs(os.path.join("datasets/caption/under", str(limit)))

    df_val_t.to_json(
        os.path.join("datasets/caption/under", str(limit), "val_" + sim + ".json"),
        orient="table",
    )

    with open(
        os.path.join("datasets/caption/under", str(limit), "val_" + sim + ".json")
    ) as val_cap:
        val_cap = json.load(val_cap)

    for i in val_cap["data"]:
        del i["level_0"]
        del i["index"]

    with open(
        os.path.join("datasets/caption/under", str(limit), "val_" + sim + ".json"), "w"
    ) as f2:
        json.dump(val_cap, f2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate similarity and make similarity files.")
    parser.add_argument("train_cap_path", type=str, help="Path to the train caption file")
    parser.add_argument("val_cap_path", type=str, help="Path to the validation caption file")
    parser.add_argument("test_cap_path", type=str, help="Path to the test caption file")
    parser.add_argument("val_qacap_path", type=str, help="Path to the validation Q&A caption file")
    parser.add_argument("--limit", type=float, default=0.78, help="Similarity limit")
    parser.add_argument("--sim", type=str, default="q_similarity", help="Similarity type")

    args = parser.parse_args()

    token_to_ix, pretrained_emb = tokenize([], False)

    calculate_similarity(
        args.train_cap_path,
        args.val_cap_path,
        args.test_cap_path,
        args.val_qacap_path,
    )

    make_similarity(args.limit, args.sim)