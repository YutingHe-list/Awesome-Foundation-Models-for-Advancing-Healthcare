# Awesome-Foundation-Models-for-Advancing-Healthcare

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo is a collection of AWESOME things about Foundation models in healthcare, including language foundation models (LFMs), vision foundation models (VFMs), bioinformatics foundation models (BFMs), and multimodal foundation models (MFMs). Feel free to star and fork.

# Contents
- [Awesome-Foundation-Models-for-Advancing-Healthcare](#Awesome-Foundation-Models-for-Advancing-Healthcare)
- [Related Survery](#related-survey)
- [Methods](#methods)
  - [LFM methods](#lfm-methods)
  - [VFM methods](#vfm-methods)
  - [BFM methods](#bfm-methods)
  - [MFM methods](#mfm-methods)
- [Datasets](#datasets)
- - [LFM data](#lfm-data)
  - [VFM data](#vfm-data)
  - [BFM data](#bfm-data)
  - [MFM data](#mfm-data)
- [Lectures and Tutorials](#lectures-and-tutorials)
- [Other Resources](#other-resources)

# Related Survey


# Methods
## LFM methods in healthcare

**2023**
- [Bioinformatics] MedCPT: A method for zero-shot biomedical information retrieval using contrastive learning with PubMedBERT. [[Paper]](https://academic.oup.com/bioinformatics/article-abstract/39/11/btad651/7335842) [[Code]](https://github.com/ncbi/MedCPT)

- [arXiv] Pmc-llama: Towards building open-source language models for medicine. [[Paper]](https://arxiv.org/abs/2304.14454) [[Code]](https://github.com/chaoyi-wu/PMC-LLaMA)

- [arXiv] Zhongjing: Enhancing the chinese medical capabilities of large language model through expert feedback and realworld multi-turn dialogue. [[Paper]](https://arxiv.org/abs/2308.03549) [[Code]](https://github.com/SupritYoung/Zhongjing)

- [arXiv] Meditron-70b: Scaling medical pretraining for large language models. [[Paper]](https://arxiv.org/abs/2311.16079) [[Code]](https://github.com/epfLLM/meditron)

- [arXiv] Qilin-med: Multi-stage knowledge injection advanced medical large language model. [[Paper]](https://arxiv.org/abs/2310.09089) [[Code]](https://github.com/epfLLM/meditron)

- [arXiv] Huatuogpt-ii, one-stage training for medical adaption of llms. [[Paper]](https://arxiv.org/abs/2311.09774) [[Code]](https://github.com/freedomintelligence/huatuogpt-ii)

- [NPJ Digit. Med.] A study of generative large language model for medical research and healthcare. [[Paper]](https://www.nature.com/articles/s41746-023-00958-w) [[Code]](https://github.com/uf-hobi-informatics-lab/gatortrongpt)

- [arXiv] From beginner to expert: Modeling medical knowledge into general llms. [[Paper]](https://arxiv.org/abs/2312.01040)

- [arXiv] Huatuo: Tuning llama model with chinese medical knowledge. [[Paper]](https://arxiv.org/abs/2304.06975) [[Code]](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)

- [arXiv] Chatdoctor: A medical chat model fine-tuned on a large language model meta-ai (llama) using medical domain knowledge. [[Paper]](https://arxiv.org/abs/2303.14070) [[Code]](https://github.com/Kent0n-Li/ChatDoctor)

- [arXiv] Medalpaca–an open-source collection of medical conversational ai models and training data. [[Paper]](https://arxiv.org/abs/2304.08247) [[Code]](https://github.com/kbressem/medAlpaca)

- [arXiv] Alpacare: Instruction-tuned large language models for medical application. [[Paper]](https://arxiv.org/abs/2310.14558) [[Code]](https://github.com/xzhang97666/alpacare)

- [arXiv] Huatuogpt, towards taming language model to be a doctor. [[Paper]](https://arxiv.org/abs/2305.15075) [[Code]](https://github.com/FreedomIntelligence/HuatuoGPT)

- [arXiv] Doctorglm: Fine-tuning your chinese doctor is not a herculean task. [[Paper]](https://arxiv.org/abs/2304.01097) [[Code]](https://github.com/xionghonglin/DoctorGLM)

- [arXiv] Bianque: Balancing the questioning and suggestion ability of health llms with multi-turn health conversations polished by chatgpt. [[Paper]](https://arxiv.org/abs/2310.15896) [[Code]](https://github.com/scutcyr/BianQue)

- [arXiv] Taiyi: A bilingual fine-tuned large language model for diverse biomedical tasks. [[Paper]](https://arxiv.org/abs/2311.11608) [[Code]](https://github.com/DUTIR-BioNLP/Taiyi-LLM)

- [Github] Visual med-alpaca: A parameter-efficient biomedical llm with visual capabilities. [[Code]](https://github.com/cambridgeltl/visual-med-alpaca)

- [arXiv] Ophglm: Training an ophthalmology large languageand-vision assistant based on instructions and dialogue. [[Paper]](https://arxiv.org/abs/2306.12174) [[Code]](https://github.com/ML-AILab/OphGLM)

- [arXiv] Chatcad: Interactive computer-aided diagnosis on medical image using large language models. [[Paper]](https://arxiv.org/abs/2302.07257) [[Code]](https://github.com/zhaozh10/ChatCAD)

- [arXiv] Chatcad+: Towards a universal and reliable interactive cad using llms. [[Paper]](https://arxiv.org/abs/2305.15964) [[Code]](https://github.com/zhaozh10/ChatCAD)

- [arXiv] Deid-gpt: Zero-shot medical text de-identification by gpt-4. [[Paper]](https://arxiv.org/abs/2303.11032) [[Code]](https://github.com/yhydhx/ChatGPT-API)

- [arXiv] Can generalist foundation models outcompete special-purpose tuning? case study in medicine. [[Paper]](https://arxiv.org/abs/2311.16452) [[Code]](https://github.com/microsoft/promptbase)

- [arXiv] Medagents: Large language models as collaborators for zero-shot medical reasoning. [[Paper]](https://arxiv.org/abs/2311.10537) [[Code]](https://github.com/gersteinlab/MedAgents)

- [AIME] Soft-prompt tuning to predict lung cancer using primary care free-text dutch medical notes. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-34344-5_23) [[Code]](https://bitbucket.org/aumc-kik/prompt_tuning_cancer_prediction/src/master/)

- [arXiv] Clinical decision transformer: Intended treatment recommendation through goal prompting. [[Paper]](https://arxiv.org/abs/2302.00612) [[Code]](https://clinical-decision-transformer.github.io/)

- [Nature] Large language models encode clinical knowledge [[Paper]](https://www.nature.com/articles/s41586-023-06291-2) 

- [arXiv] Towards expert-level medical question answering with large language models [[Paper]](https://arxiv.org/abs/2305.09617) 

- [arXiv] Gpt-doctor: Customizing large language models for medical consultation [[Paper]](https://arxiv.org/abs/2312.10225) 

- [arXiv] Clinicalgpt: Large language models finetuned with diverse medical data and comprehensive evaluation [[Paper]](https://arxiv.org/abs/2306.09968) 

- [arXiv] Leveraging a medical knowledge graph into large language models for diagnosis prediction [[Paper]](https://arxiv.org/abs/2308.14321) 


**2022**
- [NPJ Digit. Med.] A large language model for electronic health records. [[Paper]](https://www.nature.com/articles/s41746-022-00742-2) [[Code]](https://github.com/uf-hobi-informatics-lab/GatorTron)

- [AMIA Annu. Symp. Proc.] Healthprompt: A zero-shot learning paradigm for clinical natural language processing. [[Paper]](https://pubmed.ncbi.nlm.nih.gov/37128372/)

- [BioNLP] Position-based prompting for health outcome generation [[Paper]](https://aclanthology.org/2022.bionlp-1.3/) 


**2021**
- [ACM Trans. Comput. Healthc.] Domain-specific language model pretraining for biomedical natural language processing. [[Paper]](https://dl.acm.org/doi/10.1145/3458754) [[Code]](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)

**2020**
- [JMIR Med. Info.] Modified bidirectional encoder representations from transformers extractive summarization model for hospital information systems based on character-level tokens (alphabert): development and performance evaluation. [[Paper]](https://medinform.jmir.org/2020/4/e17787/) [[Code]](https://github.com/wicebing/AlphaBERT)

- [Scientific reports] Behrt: transformer for electronic health records. [[Paper]](https://www.nature.com/articles/s41598-020-62922-y) [[Code]](https://github.com/deepmedicine/BEHRT)

- [BioNLP] BioBART: Pretraining and evaluation of a biomedical generative language model. [[Paper]](https://aclanthology.org/2022.bionlp-1.9/) [[Code]](https://github.com/GanjinZero/BioBART)

- [Method. Biochem. Anal.] Biobert: a pre-trained biomedical language representation model for biomedical text mining. [[Paper]](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) [[Code]](https://github.com/dmis-lab/biobert)

**2019**
- [NPJ Digit. Med.] ClinicalBERT: A hybrid learning model for natural language inference in healthcare using BERT. [[Paper]](https://www.nature.com/articles/s41746-022-00742-2) [[Code]](https://github.com/EmilyAlsentzer/clinicalBERT)

## VFM methods
## BFM methods
**2024**
  - [Nucleic Acids Research] Multiple sequence alignment-based RNA language model and its application to structural inference. [[Paper]](https://academic.oup.com/nar/article/52/1/e3/7369930), [[Code]](https://github.com/yikunpku/RNA-MSM)
  - [Nature Methods] scGPT: toward building a foundation model for single-cell multi-omics using generative AI. [[Paper]](https://www.nature.com/articles/s41592-024-02201-0), [[Code]](https://github.com/bowang-lab/scGPT)


**2023**
  - [arXiv] DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks. [[Paper]](https://arxiv.org/abs/2307.05628), [[Code]](https://github.com/TencentAILabHealthcare/DNAGPT)
  - [arXiv] HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. [[Paper]](https://arxiv.org/abs/2306.15794), [[Code]](https://github.com/HazyResearch/hyena-dna)
  - [Nature Biotechnology] Large language models generate functional protein sequences across diverse families. [[Paper]](https://www.nature.com/articles/s41587-022-01618-2), [[Code]](https://github.com/salesforce/progen)
  - [Cell Systems] ProGen2: Exploring the boundaries of protein language models. [[Paper]](https://www.cell.com/cell-systems/abstract/S2405-4712(23)00272-7), [[Code]](https://github.com/salesforce/progen/tree/main/progen2)
  - [Nature] Transfer learning enables predictions in network biology. [[Paper]](https://www.nature.com/articles/s41586-023-06139-9), [[Code]](https://huggingface.co/ctheodoris/Geneformer)
  - [arXiv] DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. [[Paper]](https://arxiv.org/abs/2306.15006), [[Code]](https://github.com/Zhihan1996/DNABERT_2)
  - [bioRxiv] The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1), [[Code]](https://github.com/instadeepai/nucleotide-transformer)
  - [bioRxiv] GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v2), [[Code]](https://github.com/AIRI-Institute/GENA_LM)
  - [bioRxiv] Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.01.31.526427v2), [[Code]](https://github.com/biomed-AI/SpliceBERT)
  - [bioRxiv] A 5’ UTR Language Model for Decoding Untranslated Regions of mRNA and Function Predictions. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.10.11.561938v1.full), [[Code]](https://github.com/a96123155/UTR-LM)
  - [bioRxiv] Deciphering 3’ UTR mediated gene regulation using interpretable deep representation learning. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.09.08.556883v1), [[Code]](https://github.com/yangyn533/3UTRBERT)
  - [Science] Evolutionary-scale prediction of atomic-level protein structure with a language model. [[Paper]](https://www.science.org/doi/10.1126/science.ade2574), [[Code]](https://github.com/facebookresearch/esm)
  - [bioRxiv] Universal Cell Embeddings: A Foundation Model for Cell Biology. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1), [[Code]](https://github.com/snap-stanford/UCE)
  - [bioRxiv] Large Scale Foundation Model on Single-cell Transcriptomics. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.05.29.542705v4), [[Code]](https://github.com/biomap-research/scFoundation)

  - [arXiv] Large-Scale Cell Representation Learning via Divide-and-Conquer Contrastive Learning. [[Paper]](https://arxiv.org/abs/2306.04371), [[Code]](https://github.com/PharMolix/OpenBioMed)

  - [bioRxiv] CodonBERT: Large Language Models for mRNA design and optimization. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v2), [[Code]](https://github.com/Sanofi-Public/CodonBERT)
  - [bioRxiv] xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v4)
  - [bioRxiv] GenePT: A Simple But Effective Foundation Model for Genes and Cells Built From ChatGPT. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.10.16.562533v2), [[Code]](https://github.com/yiqunchen/GenePT)


**2022**
  - [Nature Machine Intelligence] scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data. [[Paper]](https://www.nature.com/articles/s42256-022-00534-z), [[Code]](https://github.com/TencentAILabHealthcare/scBERT)
  - [bioRxiv] Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions. [[Paper]](https://www.biorxiv.org/content/10.1101/2022.08.06.503062v2), [[Code]](https://github.com/ml4bio/RNA-FM)
  - [NAR Genomics & Bioinformatics] Informative RNA base embedding for RNA structural alignment and clustering by deep representation learning. [[Paper]](https://academic.oup.com/nargab/article/4/1/lqac012/6534363), [[Code]](https://github.com/mana438/RNABERT)
  - [Nature Biotechnology] Single-sequence protein structure prediction using language models and deep learning. [[Paper]](https://www.nature.com/articles/s41587-022-01432-w), [[Code]](https://github.com/aqlaboratory/rgn2)


**2021**
  - [Bioinformatics] DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. [[Paper]](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680), [[Code]](https://github.com/jerryji1993/DNABERT)
  - [IEEE TPAMI] ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. [[Paper]](https://ieeexplore.ieee.org/document/9477085), [[Code]](https://github.com/agemagician/ProtTrans)
  - [ICML 2021] MSA Transformer. [[Paper]](https://proceedings.mlr.press/v139/rao21a.html), [[Code]](https://github.com/rmrao/msa-transformer)
  - [PNAS] Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. [[Paper]](https://www.pnas.org/doi/10.1073/pnas.2016239118), [[Code]](https://github.com/facebookresearch/esm)
  - [Nature] Highly accurate protein structure prediction with AlphaFold. [[Paper]](https://www.nature.com/articles/s41586-021-03819-2), [[Code]](https://github.com/google-deepmind/alphafold)
  - [arXiv] Multi-modal Self-supervised Pre-training for Regulatory Genome Across Cell Types. [[Paper]](https://arxiv.org/abs/2110.05231), [[Code]](https://github.com/ZovcIfzm/GeneBERT)



## MFM methods


- MMBERT: Multimodal BERT Pretraining for Improved Medical VQA. [[Paper]](https://arxiv.org/abs/2104.01394) [[Code]](https://github.com/VirajBagal/MMBERT)
- Advancing radiograph representation learning with masked record modeling.[[Paper]](https://arxiv.org/abs/2301.13155) [[Code]](https://github.com/RL4M/MRM-pytorch)
- Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data. [[Paper]](https://arxiv.org/pdf/2308.02463.pdf) [[Code]](https://github.com/chaoyi-wu/RadFM)
- BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks. [[Paper]](https://arxiv.org/abs/2305.17100) [[Code]](https://github.com/taokz/BiomedGPT) 

- [EMNLP] Translation between Molecules and Natural Language. [[Paper]](https://aclanthology.org/2022.emnlp-main.26.pdf) [[Code]](https://github.com/blender-nlp/MolT5)


# Datasets
## LFM data

## VFM data
|                           Dataset  Name                               | Modality  |            Scale           |    Task    |                       Link                             |
| :-------------------------------------------------------------------: | :-------: | :------------------------: | :--------: | :----------------------------------------------------: |
|[LIMUC](https://academic.oup.com/ibdjournal/article/29/9/1431/6830946) | Endoscopy | 1043 videos (11276 frames) |  Detection |[*](https://zenodo.org/records/5827695#.Yi8GJ3pByUk)|
|[SUN](https://www.sciencedirect.com/science/article/pii/S0016510720346551)| Endoscopy | 1018 videos (158,690 frames) |  Detection |[*](http://amed8k.sundatabase.org/)|
|[Kvasir-Capsule](https://www.nature.com/articles/s41597-021-00920-z)| Endoscopy | 117 videos (4,741,504 frames) |  Detection |[*](https://datasets.simula.no/kvasir-capsule/)|
|[EndoSLAM](https://www.sciencedirect.com/science/article/pii/S1361841521001043) | Endoscopy | 1020 videos (158,690 frames) |  Detection, Registration |[*](https://github.com/CapsuleEndoscope/EndoSLAM)|
|[LDPolypVideo](https://link.springer.com/chapter/10.1007/978-3-030-87240-3_37) | Endoscopy | 263 videos (895,284 frames) |  Detection |[*](https://github.com/dashishi/LDPolypVideo-Benchmark)|
|[HyperKvasir](https://www.nature.com/articles/s41597-020-00622-y)| Endoscopy | 374 videos (1,059,519 frames) |  Detection |[*](https://datasets.simula.no/hyper-kvasir)|
|[CholecT45](https://arxiv.org/pdf/2204.05235.pdf)| Endoscopy |  45 videos (90489 frames)  |  Segmentation, Detection |[*](https://github.com/CAMMA-public/cholect45)|
|[DeepLesion]()| CT slices (2D) | 32,735 images |  Segmentation, Registration |[*](nihcc.app.box.com)|
|[]()| Endoscopy | 1,018 volumes	 |  Detection |[*]()|
|[   ]()| Endoscopy |  |  Detection |[*]()|
|[   ]()| Endoscopy |  |  Detection |[*]()|



## BFM data
|                           Dataset  Name                               | Modality  |            Scale           |    Task    |                       Link                             |
| :-------------------------------------------------------------------: | :-------: | :------------------------: | :--------: | :----------------------------------------------------: |
|[CellxGene Corpus]([https://academic.oup.com/ibdjournal/article/29/9/1431/6830946](https://www.biorxiv.org/content/10.1101/2023.10.30.563174v1)) | scRNA-seq | over 72M scRNA-seq data |  Single cell omics study |[*](https://cellxgene.cziscience.com)|
|[NCBI GenBank](https://doi.org/10.1093/nar/gks1195)| DNA | 3.7B sequences |  Genomics study |[*](https://www.ncbi.nlm.nih.gov/genbank/)|
|[SCP](https://www.biorxiv.org/content/10.1101/2023.07.13.548886v1)| scRNA-seq | over 40M scRNA-seq data | Single cell omics study |[*](https://singlecell.broadinstitute.org/single_cell)|
|[Gencode](https://doi.org/10.1093/nar/gky955)| DNA |  |  Genomics study |[*](https://www.gencodegenes.org/)|
|10x Genomics| scRNA-seq, DNA |  |  Single cell omics and genomics study |[*](https://support.10xgenomics.com/single-cell-gene-expression/datasets)|
|ABC Atlas| scRNA-seq | over 15M scRNA-seq data | Single cell omics study |[*](https://portal.brain-map.org)|
|[Human Cell Atlas](https://doi.org/10.7554/eLife.27041)| scRNA-seq | over 50M scRNA-seq data | Single cell omics study |[*](https://www.humancellatlas.org/)|
|[UCSC Genome Browser](https://doi.org/10.1093/nar/gkad987)| DNA |  |  Genomics study |[*](https://hgdownload.soe.ucsc.edu/downloads.html)|
|[Ensembl Project](https://academic.oup.com/nar/article/51/D1/D933/6786199)| Protein |  |  Proteomics study |[*](https://ensembl.org/index.html)|
|[RNAcentral database](https://doi.org/10.1093/nar/gky1034)| RNA | 36M sequences |  Transcriptomics study |[*](https://rfam.org)|
|[AlphaFold DB](https://www.nature.com/articles/s41586-021-03819-2)| Protein | 214M structures |  Proteomics study |[*](https://alphafold.ebi.ac.uk/download)|
|[PDBe](https://europepmc.org/articles/PMC7145656)| Protein |  |  Proteomics study |[*](https://www.ebi.ac.uk/pdbe/)|
|[UniProt](https://doi.org/10.1093/nar/gkac1052)| Protein | over 250M sequences |  Proteomics study |[*](https://www.uniprot.org)|
|[LINCS L1000](http://lincsportal.ccs.miami.edu/datasets/#/view/LDS-1398)| Small molecules | 1,000 genes with 41k small molecules |  Disease research, drug response |[*](https://lincsportal.ccs.miami.edu/dcic-portal/)|
|[GDSC](https://doi.org/10.1093/nar/gks1111)| Small molecules | 1,000 cancer cells with 400 compounds |  Disease research, drug response |[*](https://www.cancerrxgene.org)|
|[CCLE](https://doi.org/10.1038/s41586-019-1186-3)|  |  |  Bioinformatics study |[*](https://sites.broadinstitute.org/ccle/datasets)|
|[TCGA]()|  |  |  Bioinformatics study |[*](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)|
|[CGGA](https://www.sciencedirect.com/science/article/pii/S1672022921000450)|  |  |  Bioinformatics study |[*](http://www.cgga.org.cn)|
|[UK Biobank](https://www.nature.com/articles/s41586-018-0579-z)|  |  |  Bioinformatics study |[*](https://www.ukbiobank.ac.uk)|


## MFM data

# Lectures and Tutorials

[Bioinformatics - Geneformer Tutorial (unofficial)](https://wang-lab.hkust.edu.hk/others/Tutorials/geneformer/readme.html)

# Other Resources



