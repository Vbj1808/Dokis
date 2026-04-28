# Claimify Selection Benchmark Summary

Dataset/source: https://huggingface.co/datasets/microsoft/claimify-dataset/resolve/main/data.csv
Rows: 6490

| extractor | accuracy | precision | recall | f1 | pred+ | gold+ | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regex | 0.668 | 0.645 | 0.975 | 0.776 | 5799 | 3833 | 3738 | 2061 | 95 |
| nltk | 0.668 | 0.645 | 0.976 | 0.776 | 5801 | 3833 | 3740 | 2061 | 93 |
| claimify | 0.748 | 0.742 | 0.881 | 0.805 | 4550 | 3833 | 3375 | 1175 | 458 |

This benchmark measures Selection-stage factual-claim detection only. It does not measure full Claimify reproduction, element-level coverage, citation faithfulness, or answer correctness.
