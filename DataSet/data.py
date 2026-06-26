# import ir_datasets
#
# datasets = [
#     "beir/arguana",
#     "beir/scifact",
#     "beir/fiqa",
#     "beir/dbpedia-entity",
#     "beir/fever",
#     "beir/quora",
#     "beir/climate-fever"
# ]
#
# for ds in datasets:
#     d = ir_datasets.load(ds)
#
#     print("\n")
#     print("=" * 80)
#     print(ds)
#     print(d)

# import ir_datasets
#
# dataset = ir_datasets.load("beir/quora")
#
# print(dataset)
# print(dir(dataset))

import ir_datasets

datasets = [
    "beir/quora/test",
    "beir/fiqa/test",
    "beir/scifact/test",
    "beir/dbpedia-entity/test",
    "beir/climate-fever/test",
    "beir/fever/test",
]

for ds in datasets:
    print("\n" + "=" * 80)
    print(ds)

    try:
        d = ir_datasets.load(ds)

        print("docs:", d.docs_count())

        if hasattr(d, "queries_count"):
            print("queries:", d.queries_count())

        print("has qrels:", hasattr(d, "qrels_iter"))

    except Exception as e:
        print("ERROR:", e)